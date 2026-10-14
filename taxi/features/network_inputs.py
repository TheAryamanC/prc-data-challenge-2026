"""
Neural networks learn best when every input number is on a similar scale and there are no blanks, but the raw features are messy
This file prepares a tidy 'network input cache' for them in one folder

This is how it works:
-pick the most useful number features (250 by default), ranked by how much they helped an XGBoost value model (its total gain)
-for each of these features:
    -skip it if it carries no information on the 2025 rows
    -squeeze it onto a bell-curve scale fitted on its 2025 values, clipped to between -5 and 5, with blanks turned into 0
    -add a yes/no 'this was missing' flag when the feature is sometimes missing, keeping only one flag per different missing pattern
-turn the 14 label columns (airport, runway, stand, aircraft type, airline etc.) into whole-number codes, with rare values and blanks sharing code 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from scipy.special import ndtri

from taxi.main import FEATURE_MATRIX, MODEL_WEIGHTS, recording_regimes

# Categorical columns of base_features
CATEGORICAL_COLUMNS = ['ADEP_mvt', 'airport_runway', 'RUNWAY_mvt', 'airport_stand', 'airport_runway_stand', 'stand_prefix', 'route', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'callsign_prefix', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']

# The tables the numerical features are read from
BASE_TABLE = 'base_features'
OTHER_SOURCE_TABLES = ('arrival_proxy_features', 'xplane_geometry_features')
COMPUTED_SOURCE = 'computed'   # msched / sched_daydiff: computed from the challenge files, not read from a table

# Features of the importance model that are computed from the challenge files rather than read from a table
COMPUTED_FEATURES = ('msched', 'sched_daydiff')

# Features of the importance model that never become numerical inputs
EXCLUDED_FEATURES = ('operator3', 'vbase')
FEATURES_PER_READ = 30           # numerical features read from a parquet table at a time (bounds the memory)
ROWS_PER_COPY = 200_000          # rows copied at a time from the temporary matrix into x_num.npy
MIN_LEVEL_COUNT_2025 = 20        # a categorical level gets its own code when it occurs at least this often on the 2025 rows

# Quantile-normal transform of the numerical features
QUANTILE_LEVELS = 1000      # the default of sklearn's QuantileTransformer, which this transform copies
MAX_QUANTILE_FIT_VALUES = 1_000_000   # a random-sample cap that keeps the fit quick - a million values pin down 1,000 quantiles closely
QUANTILE_SUBSAMPLE_SEED = 0 # just for repeatability but not needed
NORMAL_SCORE_LIMIT = 5      # anything above is overkill for normalized values - 0.0000287% chance of happening

# Missing-value flags
MIN_MISSING_SHARE = 0.005
MAX_MISSING_SHARE = 0.995

# Command line
DEFAULT_TOP_FEATURES = 250 # chosen empirically by cross validation (down from 527 total features)


def quantile_normal_transform(values: np.ndarray, fit_rows: np.ndarray, n_quantiles: int = QUANTILE_LEVELS) -> np.ndarray:
    """
    This function converts the data to a standard normal distribution
    """
    fit_values = values[fit_rows]
    fit_values = fit_values[np.isfinite(fit_values)]
    if len(fit_values) > MAX_QUANTILE_FIT_VALUES:
        fit_values = np.random.default_rng(QUANTILE_SUBSAMPLE_SEED).choice(fit_values, MAX_QUANTILE_FIT_VALUES, replace=False)
    probabilities = np.linspace(0, 1, n_quantiles)
    quantiles = np.quantile(fit_values, probabilities)
    is_finite = np.isfinite(values)
    probability_from_below = np.interp(values[is_finite], quantiles, probabilities)
    minus_probability_from_above = np.interp(-values[is_finite], -quantiles[::-1], -probabilities[::-1])
    uniform = np.full(len(values), np.nan)
    uniform[is_finite] = 0.5 * (probability_from_below - minus_probability_from_above)
    # ndtri: the inverse of the standard normal distribution function (probability -> z-score)
    normal_scores = ndtri(np.clip(uniform, 1e-7, 1 - 1e-7))   # 1e-7 keeps probabilities off 0 and 1 (infinite z-scores), the same bound as sklearn's QuantileTransformer
    return np.clip(normal_scores, -NORMAL_SCORE_LIMIT, NORMAL_SCORE_LIMIT)


class MissingValueFlags:
    """
    This is a class that collects the 'was this missing?' columns of x_num, keeping one flag per different missing pattern
    A feature's pattern is only kept when 0.5 % to 99.5 % of the rows are missing and no earlier feature had exactly the same pattern
    """

    def __init__(self) -> None:
        """
        This function starts the collector empty
        """
        self.flags: dict[str, np.ndarray] = {}            # 'miss:<feature>' -> boolean mask (True = missing)
        self.seen_pattern_hashes: set[str] = set()

    def add(self, column: str, is_missing: np.ndarray) -> None:
        """
        This function offers one feature's missing pattern to the collector, which keeps it as 'miss:<feature>' when 0.5 % to 99.5 % of the rows are missing and the pattern is new
        """
        name = f'miss:{column}'
        if MIN_MISSING_SHARE < is_missing.mean() < MAX_MISSING_SHARE:
            pattern_hash = hashlib.sha1(np.packbits(is_missing).tobytes()).hexdigest()
            if pattern_hash not in self.seen_pattern_hashes:
                self.seen_pattern_hashes.add(pattern_hash)
                self.flags[name] = is_missing

    def names(self) -> list[str]:
        """
        This function lists the names of the kept flags, in the order they were added
        """
        return list(self.flags)

    def matrix(self, row_count: int) -> np.ndarray:
        """
        This function stacks the kept flags side by side into a table of 0s and 1s (an empty table when there are no flags)
        """
        names = self.names()
        if not names:
            return np.zeros((row_count, 0), np.float16)
        return np.stack([self.flags[name] for name in names], 1).astype(np.float16)


def column_source_tables(adsb_table: str) -> dict[str, str]:
    """
    This function works out which table each column should be read from
    """
    column_source: dict[str, str] = {}
    for table_name in (BASE_TABLE, adsb_table) + OTHER_SOURCE_TABLES:
        for column in pq.read_schema(FEATURE_MATRIX / f'{table_name}.parquet').names:
            column_source.setdefault(column, table_name)
    return column_source


def categorical_model_features(model_features: list[str], column_source: dict[str, str], base_schema: pa.Schema) -> set[str]:
    """
    This function lists the label features that should be treated as categories
    """
    stored_as_category = {feature for feature in model_features if column_source.get(feature) == BASE_TABLE and str(base_schema.field(feature).type).startswith('dictionary')}
    return set(CATEGORICAL_COLUMNS) | stored_as_category


def is_numerical_input(feature: str, categorical: set[str], column_source: dict[str, str]) -> bool:
    """
    This function decides whether a feature may become a number input
    """
    if feature in categorical or feature in EXCLUDED_FEATURES:
        return False
    return feature in column_source or feature in COMPUTED_FEATURES


def select_numerical_features(importance_from: str, column_source: dict[str, str], top_features: int) -> list[str]:
    """
    This function picks the number features - it loads the XGBoost value model, ranks the features by how much they helped it (total gain) and keeps the best ones
    """
    booster = xgb.Booster()
    booster.load_model(MODEL_WEIGHTS / importance_from / 'value.ubj')
    gain_by_feature = booster.get_score(importance_type='total_gain')
    model_features = booster.feature_names
    del booster             # the importance model takes about 3 GB of memory once loaded
    base_schema = pq.read_schema(FEATURE_MATRIX / f'{BASE_TABLE}.parquet')
    categorical = categorical_model_features(model_features, column_source, base_schema)
    # sorted() is stable: features with equal gain (e.g. never used: gain 0) keep the model's feature order.
    by_gain = sorted(model_features, key=lambda feature: -gain_by_feature.get(feature, 0.0))
    numerical = [feature for feature in by_gain if is_numerical_input(feature, categorical, column_source)][:top_features]
    selected_gain = sum(gain_by_feature.get(feature, 0) for feature in numerical)
    print('numerical features', len(numerical), 'gain share %.4f' % (selected_gain / sum(gain_by_feature.values())), flush=True)
    return numerical


def first_adsb_rows_by_departure(adsb_table: str, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    This function lines up the ADS-B table with the master list of departures
    """
    adsb_ids = pd.read_parquet(FEATURE_MATRIX / f'{adsb_table}.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt
    is_first_adsb_row = ~adsb_ids.duplicated().to_numpy()
    position = pd.Index(adsb_ids[is_first_adsb_row]).get_indexer(metadata.MVT_ID_mvt)
    first_rows = np.flatnonzero(is_first_adsb_row)
    return first_rows, position


def features_by_source_table(numerical: list[str], column_source: dict[str, str]) -> dict[str, list[str]]:
    """
    This function groups the chosen features by the table they come from
    """
    features_by_source: dict[str, list[str]] = {}
    for feature in numerical:
        features_by_source.setdefault(column_source.get(feature, COMPUTED_SOURCE), []).append(feature)
    return features_by_source


def read_feature_chunk(source: str, chunk: list[str], seconds_after_schedule: np.ndarray, day_difference: np.ndarray) -> pd.DataFrame:
    """
    This function reads/computes a batch of columns from one table
    """
    if source == COMPUTED_SOURCE:
        return pd.DataFrame({'msched': seconds_after_schedule, 'sched_daydiff': day_difference})[chunk]
    return pq.ParquetFile(FEATURE_MATRIX / f'{source}.parquet').read(columns=chunk).to_pandas()


def normalise_features(unfiltered: np.ndarray, numerical: list[str], column_source: dict[str, str], adsb_table: str, adsb_alignment: tuple[np.ndarray, np.ndarray], is_2025: np.ndarray, seconds_after_schedule: np.ndarray, day_difference: np.ndarray, missing_flags: MissingValueFlags) -> set[str]:
    """
    This is the main tidying loop - it reads each table 30 columns at a time, lines the ADS-B columns up with the departures and skips features that carry no information on the 2025 rows
    Every other feature has its missing pattern offered to the flag collector, is squeezed onto the bell-curve scale fitted on 2025 with blanks turned into 0, and is written into its column of the big matrix
    """
    first_adsb_rows, adsb_position = adsb_alignment
    column_index = {feature: index for index, feature in enumerate(numerical)}
    uninformative: set[str] = set()
    for source, source_features in features_by_source_table(numerical, column_source).items():
        for start in range(0, len(source_features), FEATURES_PER_READ):
            chunk = source_features[start:start + FEATURES_PER_READ]
            chunk_table = read_feature_chunk(source, chunk, seconds_after_schedule, day_difference)
            for feature in chunk:
                values = chunk_table[feature].to_numpy(dtype=np.float64)
                if source == adsb_table:
                    values = np.where(adsb_position >= 0, values[first_adsb_rows][np.maximum(adsb_position, 0)], np.nan)
                is_missing = ~np.isfinite(values)
                fit_values = values[is_2025 & ~is_missing]
                if len(fit_values) < 2 or fit_values.min() == fit_values.max():
                    uninformative.add(feature)    # no information on the 2025 rows (all missing or constant)
                    continue
                missing_flags.add(feature, is_missing)
                normalised = quantile_normal_transform(values, is_2025)
                normalised[~np.isfinite(normalised)] = 0.0
                unfiltered[:, column_index[feature]] = normalised.astype(np.float16)
            del chunk_table
            print(f'  {source}: {min(start + FEATURES_PER_READ, len(source_features))}/{len(source_features)}', flush=True)
    return uninformative


def write_final_matrix(output_dir: Path, unfiltered: np.ndarray, kept_columns: np.ndarray, flag_matrix: np.ndarray, row_count: int, feature_count: int) -> None:
    """
    This function writes the kept feature columns followed by the missing flags
    """
    shape = (row_count, feature_count + flag_matrix.shape[1])
    network_matrix = np.lib.format.open_memmap(output_dir / 'x_num.npy', mode='w+', dtype=np.float16, shape=shape)
    for start in range(0, row_count, ROWS_PER_COPY):
        network_matrix[start:start + ROWS_PER_COPY, :feature_count] = unfiltered[start:start + ROWS_PER_COPY][:, kept_columns]
        network_matrix[start:start + ROWS_PER_COPY, feature_count:] = flag_matrix[start:start + ROWS_PER_COPY]
    network_matrix.flush()
    del network_matrix


def write_numerical_inputs(output_dir: Path, numerical: list[str], column_source: dict[str, str], adsb_table: str, metadata: pd.DataFrame, is_2025: np.ndarray, seconds_after_schedule: np.ndarray, day_difference: np.ndarray, missing_flags: MissingValueFlags) -> tuple[list[str], list[str]]:
    """
    This function writes the file
    """
    row_count = len(metadata)
    adsb_alignment = first_adsb_rows_by_departure(adsb_table, metadata)
    # one column per feature of `numerical`; written to disk (a memory map) because the full matrix is about 1.4 GB
    unfiltered = np.lib.format.open_memmap(output_dir / 'x_num_tmp.npy', mode='w+', dtype=np.float16, shape=(row_count, len(numerical)))
    uninformative = normalise_features(unfiltered, numerical, column_source, adsb_table, adsb_alignment, is_2025, seconds_after_schedule, day_difference, missing_flags)
    unfiltered.flush()
    flag_names = missing_flags.names()
    flag_matrix = missing_flags.matrix(row_count)
    column_index = {feature: index for index, feature in enumerate(numerical)}
    kept_columns = np.array([column_index[feature] for feature in numerical if feature not in uninformative])
    kept_features = [feature for feature in numerical if feature not in uninformative]
    print(f'dropped {len(uninformative)} features without information on 2025 rows: {sorted(uninformative)}', flush=True)
    write_final_matrix(output_dir, unfiltered, kept_columns, flag_matrix, row_count, len(kept_features))
    del unfiltered, flag_matrix
    (output_dir / 'x_num_tmp.npy').unlink()
    return kept_features, flag_names


def frequent_level_codes(column: pd.Series, is_2025: np.ndarray) -> tuple[np.ndarray, int]:
    """
    This function turns one label column into whole-number codes - values seen at least 20 times in 2025 get codes 1, 2, 3 and so on, starting with the most common, and all rarer values and blanks share code 0
    """
    level_codes = pd.Series(pd.Categorical(column).codes)   # -1 = missing
    level_counts_2025 = level_codes[is_2025 & (level_codes.to_numpy() >= 0)].value_counts()
    kept_levels = level_counts_2025.index[level_counts_2025.to_numpy() >= MIN_LEVEL_COUNT_2025]
    new_code = pd.Series(np.arange(1, len(kept_levels) + 1), index=kept_levels)
    codes = level_codes.map(new_code).fillna(0).to_numpy(dtype=np.int32)
    return codes, len(kept_levels) + 1


def write_categorical_inputs(output_dir: Path, row_count: int, is_2025: np.ndarray, operator_code: np.ndarray) -> tuple[list[str], list[int]]:
    """
    This function reads the 13 label columns of base_features, adds the first three characters of the flight number and turns each into codes with frequent_level_codes
    """
    categorical_table = pq.ParquetFile(FEATURE_MATRIX / f'{BASE_TABLE}.parquet').read(columns=CATEGORICAL_COLUMNS).to_pandas()
    categorical_table['operator3'] = operator_code
    codes_matrix = np.zeros((row_count, len(categorical_table.columns)), np.int32)
    cardinalities: list[int] = []
    for column_number, column in enumerate(categorical_table.columns):
        codes, cardinality = frequent_level_codes(categorical_table[column], is_2025)
        codes_matrix[:, column_number] = codes
        cardinalities.append(cardinality)
    np.save(output_dir / 'x_cat.npy', codes_matrix)
    return list(categorical_table.columns), cardinalities


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build neural network input caches.')
    parser.add_argument('--top-features', type=int, default=DEFAULT_TOP_FEATURES)
    parser.add_argument('--importance-from', default='xgboost_value_seed42')
    parser.add_argument('--adsb-table', default='adsb_pass1')
    parser.add_argument('--out', default='network_inputs_pass1')
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    missing_flags = MissingValueFlags()
    output_dir = FEATURE_MATRIX / args.out
    output_dir.mkdir(exist_ok=True)
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet')
    row_count = len(metadata)
    seconds_after_schedule, day_difference, _, _, operator_code = recording_regimes(metadata)
    is_2025 = (metadata.year.to_numpy() == 2025)
    column_source = column_source_tables(args.adsb_table)
    numerical = select_numerical_features(args.importance_from, column_source, args.top_features)
    kept_features, flag_names = write_numerical_inputs(output_dir, numerical, column_source, args.adsb_table, metadata, is_2025, seconds_after_schedule, day_difference, missing_flags)
    categorical_columns, cardinalities = write_categorical_inputs(output_dir, row_count, is_2025, operator_code)
    # Key names are read by taxi.models.neural_network (num, cat, card).
    with open(output_dir / 'columns.json', 'w') as columns_file:
        json.dump({'num': kept_features + flag_names,
                   'n_value_features': len(kept_features),
                   'cat': categorical_columns,
                   'card': cardinalities,
                   'importance_from': f'{args.importance_from}/value.ubj'}, columns_file, indent=1)
    print('done: x_num', (row_count, len(kept_features) + len(flag_names)), 'x_cat', (row_count, len(categorical_columns)), 'cardinalities', dict(zip(categorical_columns, cardinalities)))


if __name__ == '__main__':
    main()
