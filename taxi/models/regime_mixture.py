"""
Every departing plane has a taxi-out time - the seconds from leaving its stand to lifting off the runway
The problem is that the 'left the stand' time was written down in three different ways, and the number to guess changes with it:
-P (schedule copy): the timetable time was just copied, so the number is really how late the plane took off compared with its timetable, which can be hours
-S (previous date, only at Rome): the right clock time was written with yesterday's date, so the number is about one day (86,400 seconds) too big
-N (normal): the time was written down properly, so the number really is the taxi time
This file trains the XGBoost and CatBoost models (big teams of small decision trees) that guess the chance of each way (pP, pS) and the normal taxi time v, which is only ever learned from N planes
One formula then mixes them into the answer: pred = pP x (seconds after the timetable time) + pS x (86,400 + v) + (1 - pP - pS) x v

This is how it works:
-train makes one full set of models from all the 2025 departures:
    -it makes vbase, a rough taxi-time guess that never looks at the timetable - take-off minus the 'left the stand' time of the NM (EUROCONTROL's Network Manager, the European flight-plan office), or the airport's middle value when the flight has no NM link
    -the XGBoost value model learns the correction on top of vbase to give v - if asked, two extra value models (two-part and CatBoost) are trained too and can be averaged in
    -three 'copied from the timetable?' guessers (all rows, Rome rows, rows with no NM link) give pP
    -for Rome planes with no NM link that take off on a later day than their timetable day, a small formula that only looks at the delay gives pP and pS
-stake-classifiers trains a second opinion on pP with CatBoost and XGBoost, where rows whose mistake would cost a lot count more
-nextday nudges the P vs S split of those Rome next-day planes toward what happened before to the same airline and the same flight
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.linear_model import LogisticRegression

from taxi.main import FEATURE_MATRIX, MODEL_WEIGHTS, RUN_PREDICTIONS, competition_columns, departure_frame

# XGBoost settings shared by the value models and the schedule-copy classifiers (each model overrides a few of them).
# device='cuda' trains on the GPU; without a visible GPU XGBoost warns and trains on the CPU.
XGBOOST_BASE_PARAMETERS: dict[str, Any] = dict(objective='reg:squarederror', tree_method='hist', device='cuda', max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, min_child_weight=20, reg_lambda=5.0, max_cat_to_onehot=12, max_cat_threshold=64, max_bin=256, seed=42)
VALUE_RESIDUAL_CLIP = 6000.0        # the value model's 2025 training residuals are clipped at +-6000 s
FULL_FIT_ITERATION_SCALE = 1.1      # validation iteration counts x1.1 for the fit on all 2025 rows
TOP_FEATURE_COUNT = 200             # the CatBoost models use the 200 columns of highest total gain of an XGBoost model
PREDICTION_CHUNK_ROWS = 400000      # rows per prediction batch: bounds the memory of the DMatrix / Pool copies
LONG_TAXI_SECONDS = 2000            # two_part value variant: label above which a taxi time counts as long
NEXT_DAY_PRIOR_MIN_HOURS = 6.0      # the nextday prior acts above this delay (hours after schedule), where regime N does not occur
VALUE_VARIANTS = ('two_part', 'catboost')
DEFAULT_EXTRA_TABLES = 'arrival_proxy_features,xplane_geometry_features'
# Boosting rounds chosen by early stopping on July 2025 held out (training on the other 2025 months); the full fits use them x1.1.
VALIDATION_ITERATIONS: dict[str, dict[str, int]] = {
    'regime': {'value': 3666, 'v_tp_cls': 634, 'v_tp_long': 129, 'v_tp_short': 3371, 'v_cb': 5999, 'pclass': 1012, 'pclass_lirf': 480, 'pclass_na': 418},
    'xgboost_value': {'value': 3666, 'pclass': 368, 'pclass_lirf': 311, 'pclass_na': 801},
    'stake_classifiers': {'pclass': 390, 'pclass_lirf': 456, 'pclass_na': 764, 'pclass_cb': 1780, 'pclass_lirf_cb': 2021, 'pclass_na_cb': 1502},
}
# The mixture prediction and the XGBoost training matrices
PREDICTION_FLOOR_SECONDS = -12      # no prediction below this: no 2025 label is lower (-12 s is the lowest 2025 label)
XGBOOST_DEFAULT_MAX_BIN = 256       # matrix bins when the parameters name no max_bin; XGBoost's own default (checked in xgboost 3.2.0)
# XGBoost settings of the global schedule-copy classifier pclass (on top of XGBOOST_BASE_PARAMETERS)
GLOBAL_CLASSIFIER_SETTINGS = {
    'objective': 'binary:logistic',     # logistic loss; the target is the 0/1 schedule-copy flag
    'eval_metric': 'logloss',
    'max_depth': 7,
}
# XGBoost settings of the Rome schedule-copy classifier pclass_lirf (on top of the global classifier's)
ROME_CLASSIFIER_SETTINGS = {
    'max_depth': 6,
    'min_child_weight': 5,
    'learning_rate': 0.03,
}
# XGBoost settings of the no-link schedule-copy classifier pclass_na (on top of the global classifier's)
NO_LINK_CLASSIFIER_SETTINGS = {
    'max_depth': 5,
    'min_child_weight': 3,
    'learning_rate': 0.03,
    'colsample_bytree': 0.5,
}
# XGBoost settings of the two_part value variant: the long-taxi classifier v_tp_cls (on top of XGBOOST_BASE_PARAMETERS) and the
# long-taxi regression v_tp_long (on top of the value model's)
LONG_TAXI_CLASSIFIER_SETTINGS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 8,
}
LONG_TAXI_REGRESSION_SETTINGS = {     # shallower trees, as only the few rows with labels above 2,000 s train it
    'max_depth': 6,
}
# CatBoost settings of the catboost value variant v_cb
CATBOOST_VALUE_SETTINGS = {
    'loss_function': 'RMSE',
    'depth': 8,
    'learning_rate': 0.08,
    'task_type': 'GPU',
    'border_count': 128,
}
CATBOOST_VALUE_RUNTIME_SETTINGS = {   # logging and memory only
    'verbose': 500,                    # print the loss every 500 rounds
    'gpu_ram_part': 0.8,               # share of the 8 GB GPU memory CatBoost may take (its default is 0.95)
    'allow_writing_files': False,
}
# Rome next-day no-link group: the P / S / N logistic regression and the nextday prior
NEXT_DAY_LOGISTIC_SETTINGS = {
    'C': 1.0,
    'max_iter': 1000,
}
NEXT_DAY_PRIOR_PSEUDO_COUNT = 3.0       # the model's chances count like 3 past flights - better on held-out July 2025, same on January / December
# stake-classifiers: row weights from the stake |msched - vbase| (a wrong copy guess costs about (msched - v)^2, README 'Regime Runs')
STAKE_FLOOR_SECONDS = 300               # 5 minutes so that rows with a small stake still count
STAKE_CAP_SECONDS = 14400               # 4 hours so that few very late flights cannot take over the training
STAKE_WEIGHT_UNIT_SECONDS = 300.0       # stakes counted in 5-minute units, so a row at the floor weighs 1
# CatBoost settings of the stake-classifiers' schedule-copy classifiers (depth and learning rate per classifier below)
CATBOOST_CLASSIFIER_SETTINGS = {
    'loss_function': 'Logloss',
    'task_type': 'GPU',
    'border_count': 128,
}
CATBOOST_CLASSIFIER_RUNTIME_SETTINGS = {  # logging and memory only
    'verbose': 1000,                      # print the loss every 1000 rounds
    'gpu_ram_part': 0.8,                  # share of the 8 GB GPU memory CatBoost may take (its default is 0.95)
    'allow_writing_files': False,
}
# (depth, learning rate) of each CatBoost schedule-copy classifier, by the key of the XGBoost classifier of the same rows
CATBOOST_CLASSIFIER_DEPTH_AND_LEARNING_RATE = {
    'pclass': (8, 0.08),
    'pclass_lirf': (6, 0.05),
    'pclass_na': (5, 0.03),
}
STAKE_CLASSIFIER_BLEND_WEIGHT = 0.5     # a plain average of the XGBoost and CatBoost chances
# Command-line defaults
DEFAULT_VALUE_MAX_DEPTH = 10            # chosen on held-out July 2025, where depth 10 did a little better than the earlier depth 8
DEFAULT_VALUE_LEARNING_RATE = 0.03      # chosen on held-out July 2025, where it did a little better than 0.05 (273.8 vs 274.5 RMSE)
DEFAULT_SEED = 42                       # the usual fixed seed
DEFAULT_STAKE_WEIGHT_POWER = 0.0        # no stake weights unless asked; the pipeline passes 1 (STAKE_WEIGHT_POWER in taxi/main.py)


# data and shared model helpers
def comma_separated(names: str) -> list[str]:
    return [name for name in names.split(',') if name]


def rows_in_departure_order(table: pd.DataFrame, movement_ids: pd.Series) -> pd.DataFrame:
    first_row_per_id = table.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
    return first_row_per_id.reindex(movement_ids).reset_index(drop=True)


def load_feature_matrix(adsb_table: str, extra_tables: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function builds the big table of clues the models learn from, one row per departure - the base features, the number columns of the chosen ADS-B table and every column of the extra tables
    """
    features = pd.read_parquet(FEATURE_MATRIX / 'base_features.parquet')
    departures = departure_frame()
    adsb = pd.read_parquet(FEATURE_MATRIX / f'{adsb_table}.parquet')
    # the ADS-B table is keyed by movement id rather than row-aligned: first row per id, in departure order (NaN when no track)
    adsb = rows_in_departure_order(adsb, departures.MVT_ID_mvt)
    adsb = adsb.drop(columns=[column for column in adsb if not pd.api.types.is_numeric_dtype(adsb[column])])
    features = pd.concat([features, adsb.astype('float32')], axis=1)
    for table_name in comma_separated(extra_tables):
        table = pd.read_parquet(FEATURE_MATRIX / f'{table_name}.parquet')
        assert (table.MVT_ID_mvt.to_numpy() == departures.MVT_ID_mvt.to_numpy()).all()
        features = pd.concat([features, table.drop(columns='MVT_ID_mvt').astype('float32')], axis=1)
    return features, departures


def schedule_independent_baseline(departures: pd.DataFrame, median_rows: np.ndarray) -> np.ndarray:
    """
    This function makes vbase, a rough first guess of each plane's taxi time that never looks at the timetable
    """
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    airports = departures.ADEP_mvt.to_numpy()
    airport_median = pd.Series(labels[median_rows]).groupby(airports[median_rows]).median()
    airport_median_per_row = departures.ADEP_mvt.map(airport_median).to_numpy(dtype=float)
    return np.where(departures.no_nm_link.to_numpy(), airport_median_per_row, departures.baseline.to_numpy())


def add_regime_features(features: pd.DataFrame, departures: pd.DataFrame, value_baseline: np.ndarray) -> None:
    """
    This function adds four clues to the clue table
    """
    features['operator3'] = pd.Categorical(departures.operator_code.to_numpy())
    features['msched'] = departures.seconds_after_schedule.to_numpy().astype('float32')
    features['sched_daydiff'] = departures.day_difference.to_numpy().astype('float32')
    features['vbase'] = value_baseline.astype('float32')


def release_freed_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def features_by_total_gain(booster: xgb.Booster) -> pd.Index:
    return pd.Series(booster.get_score(importance_type='total_gain')).sort_values(ascending=False).index


def catboost_columns(gain_ranking: pd.Index, features: pd.DataFrame) -> tuple[list[str], list[str]]:
    columns = [column for column in gain_ranking[:TOP_FEATURE_COUNT] if column in features]
    categorical_columns = [column for column in columns if str(features[column].dtype) == 'category']
    return columns, categorical_columns


def schedule_copy_classifier_parameters(seed: int) -> dict[str, dict[str, Any]]:
    """
    This function gives the settings of the three 'was the time copied from the timetable?' guessers
    """
    global_parameters = dict(XGBOOST_BASE_PARAMETERS, **GLOBAL_CLASSIFIER_SETTINGS, seed=seed)
    rome_parameters = dict(global_parameters, **ROME_CLASSIFIER_SETTINGS)
    no_link_parameters = dict(global_parameters, **NO_LINK_CLASSIFIER_SETTINGS)
    return {'pclass': global_parameters, 'pclass_lirf': rome_parameters, 'pclass_na': no_link_parameters}


def full_fit_iterations(iteration_set: str) -> dict[str, int]:
    """
    This function gives the number of tree-building rounds for each model of a full fit
    """
    validation_iterations = VALIDATION_ITERATIONS[iteration_set]
    return {key: int(round(count * FULL_FIT_ITERATION_SCALE)) for key, count in validation_iterations.items()}


def train_booster(parameters: dict[str, Any], features: pd.DataFrame, label: np.ndarray, training_rows: np.ndarray, boosting_rounds: int, row_weights: np.ndarray | None = None) -> xgb.Booster:
    """
    This function trains one XGBoost model on the chosen rows for a fixed number of rounds (no early stopping), and can make some rows count more than others
    """
    training_matrix = xgb.QuantileDMatrix(features[training_rows], label=label[training_rows], enable_categorical=True, max_bin=parameters.get('max_bin', XGBOOST_DEFAULT_MAX_BIN), weight=None if row_weights is None else row_weights[training_rows])
    booster = xgb.train(parameters, training_matrix, boosting_rounds, verbose_eval=False)
    del training_matrix
    gc.collect()
    return booster


def predict_in_chunks(booster: xgb.Booster, features: pd.DataFrame, row_mask: np.ndarray | None = None, chunk_size: int = PREDICTION_CHUNK_ROWS) -> np.ndarray:
    """
    This function asks a trained XGBoost model for its answers on every row
    """
    row_positions = np.arange(len(features)) if row_mask is None else np.flatnonzero(row_mask)
    predictions = np.empty(len(row_positions))
    for start in range(0, len(row_positions), chunk_size):
        chunk_positions = row_positions[start:start + chunk_size]
        predictions[start:start + chunk_size] = booster.predict(xgb.DMatrix(features.iloc[chunk_positions], enable_categorical=True))
    return predictions


def chunk_masks(row_count: int, chunk_size: int) -> Iterator[np.ndarray]:
    """
    This function hands out, one after another, selections of the next block of rows in order (the last block may be smaller) - used to ask CatBoost models for answers piece by piece
    """
    for start in range(0, row_count, chunk_size):
        chunk_rows = np.zeros(row_count, bool)
        chunk_rows[start:start + chunk_size] = True
        yield chunk_rows


def catboost_pool(features: pd.DataFrame, columns: list[str], categorical_columns: list[str], row_mask: np.ndarray, label: np.ndarray | None = None, row_weights: np.ndarray | None = None) -> Pool:
    """
    This function packs the chosen rows and columns into a CatBoost Pool (CatBoost's input format), turning category columns into whole-number codes with -1 for missing
    """
    subset = features.loc[row_mask, columns].copy()
    for column in categorical_columns:
        subset[column] = subset[column].cat.codes.astype('int32')
    return Pool(subset, label=label, cat_features=categorical_columns, weight=row_weights)


def mixture_prediction(probability_schedule_copy: np.ndarray, probability_previous_date: np.ndarray, seconds_after_schedule: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    This function is the "brain" of the file - it mixes the three ways into one answer: pred = pP x msched + pS x (86,400 + v) + (1 - pP - pS) x v
    """
    prediction = (probability_schedule_copy * seconds_after_schedule + probability_previous_date * (86400 + value) + (1 - probability_schedule_copy - probability_previous_date) * value)
    prediction = np.where(np.isfinite(prediction), prediction, value)
    return np.maximum(prediction, PREDICTION_FLOOR_SECONDS)


def run_metrics(iterations_used: dict[str, int], arguments: argparse.Namespace) -> dict[str, Any]:
    """
    This function builds what goes in a run's metrics.json: the number of rounds each model used and every command-line setting of the run
    """
    return {'iters': iterations_used, 'args': {option: setting for option, setting in vars(arguments).items() if option != 'func'}}


# Rome next-day no-link group
def rome_next_day_no_link_group(departures: pd.DataFrame) -> np.ndarray:
    """
    This function picks out the Rome planes with no NM link that take off on a later date than their timetable date
    Only these planes can be in way S (yesterday's date written down), so they get their own treatment
    """
    return departures.is_rome.to_numpy() & departures.no_nm_link.to_numpy() & (departures.day_difference.to_numpy() >= 1)


def regime_class_labels(departures: pd.DataFrame) -> np.ndarray:
    """
    This function numbers the way each row's time was written down: 0 for P (timetable copy), 1 for S (previous date), 2 for N (normal)
    """
    return np.where(departures.is_schedule_copy.to_numpy(), 0, np.where(departures.is_previous_date.to_numpy(), 1, 2))


def delay_hour_features(delay_hours: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    This function makes the two inputs of the small P / S / N formula: the hours after the timetable time, and a squashed version of it (log of 1 + hours) so very long delays do not swamp everything
    """
    return np.c_[delay_hours[rows], np.log1p(np.maximum(delay_hours[rows], 0))]


def fit_next_day_logistic(delay_hours: np.ndarray, class_labels: np.ndarray, fit_rows: np.ndarray, group_rows: np.ndarray) -> tuple[LogisticRegression, np.ndarray]:
    """
    This function fits a logistic regression that tells P, S and N apart from the delay alone, and gives the chances for the group rows
    """
    logistic = LogisticRegression(**NEXT_DAY_LOGISTIC_SETTINGS).fit(delay_hour_features(delay_hours, fit_rows), class_labels[fit_rows])
    fitted_class_probabilities = logistic.predict_proba(delay_hour_features(delay_hours, group_rows))
    probabilities = np.zeros((group_rows.sum(), 3))
    for fitted_column, class_label in enumerate([int(fitted_class) for fitted_class in logistic.classes_]):
        probabilities[:, class_label] = fitted_class_probabilities[:, fitted_column]
    return logistic, probabilities


def save_logistic_regression(logistic: LogisticRegression, path: Path) -> None:
    """
    This function saves the numbers inside a fitted logistic regression to a JSON file so they can be read later
    """
    path.write_text(json.dumps({'classes': [int(class_label) for class_label in logistic.classes_], 
                                'coef': logistic.coef_.tolist(), 
                                'intercept': logistic.intercept_.tolist()}, indent=1))


def flight_and_operator_keys(departures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    This function reads each departure's flight number and tidies it
    """
    flight_numbers = competition_columns(['FLIGHT_mvt'], departures.MVT_ID_mvt).FLIGHT_mvt.fillna('')
    flight_numbers = flight_numbers.str.replace(r'^([A-Z]{3})([A-Z]{2})(?=\w)', r'\1', regex=True)
    return flight_numbers.to_numpy(), flight_numbers.str[:3].to_numpy()     # the first three characters stand for the airline (the operator key of the next-day prior)


def shrink_toward_group_counts(probabilities: np.ndarray, class_labels: np.ndarray, fit_rows: np.ndarray, group_keys: np.ndarray, pseudo_count: float = NEXT_DAY_PRIOR_PSEUDO_COUNT) -> np.ndarray:
    """
    This function nudges a model's chances toward what really happened before to planes with the same key
    """
    fit_table = pd.DataFrame({'key': group_keys[fit_rows], 'label': class_labels[fit_rows]})
    counts_per_key = pd.crosstab(fit_table.key, fit_table.label).reindex(columns=range(probabilities.shape[1]), fill_value=0)
    counts = counts_per_key.reindex(group_keys).fillna(0).to_numpy(dtype=float)
    return (counts + pseudo_count * probabilities) / (counts.sum(1, keepdims=True) + pseudo_count)


def next_day_group_prior(probabilities: np.ndarray, class_labels: np.ndarray, fit_rows: np.ndarray, delay_hours: np.ndarray, key_arrays: tuple[np.ndarray, ...], min_hours: float = NEXT_DAY_PRIOR_MIN_HOURS) -> np.ndarray:
    """
    This function fine-tunes the P / S / N chances of the Rome next-day planes that are more than 6 hours late
    It keeps the N chance and the total of P plus S as they were, and only moves the split between P and S toward the past counts of the airline and then of the flight
    """
    probabilities = probabilities.copy()
    long_delay = delay_hours > min_hours
    schedule_copy_or_previous_date = probabilities[:, :2].sum(1)
    # P and S rescaled to sum to 1: the split that the prior shrinks
    schedule_copy_vs_previous_date = probabilities[:, :2] / np.maximum(probabilities[:, :2].sum(1, keepdims=True), 1e-9)     # 1e-9 only avoids dividing by zero
    labels_without_normal = np.where(class_labels == 2, -1, class_labels)      # N rows are not counted
    prior_fit_rows = fit_rows & (labels_without_normal >= 0) & long_delay
    for keys in key_arrays:
        schedule_copy_vs_previous_date = shrink_toward_group_counts(schedule_copy_vs_previous_date, labels_without_normal, prior_fit_rows, keys)
    probabilities[long_delay, 0] = schedule_copy_vs_previous_date[long_delay, 0] * schedule_copy_or_previous_date[long_delay]
    probabilities[long_delay, 1] = schedule_copy_vs_previous_date[long_delay, 1] * schedule_copy_or_previous_date[long_delay]
    return probabilities


#  train
@dataclass
class TrainRun:
    """
    This data structure holds what all the models of one train run share
    """
    features: pd.DataFrame              # the model matrix (load_feature_matrix + add_regime_features), one row per departure
    fixed_iterations: dict[str, int]    # boosting rounds per model key (full_fit_iterations)
    run_weights: Path                   # taxi/models/weights/<run>/: every model of the run
    iterations_used: dict[str, int]     # metrics.json 'iters': the rounds of every model trained so far, in training order

    def train_xgboost(self, key: str, parameters: dict[str, Any], label: np.ndarray, training_rows: np.ndarray) -> xgb.Booster:
        booster = train_booster(parameters, self.features, label, training_rows, self.fixed_iterations[key])
        self.iterations_used[key] = self.fixed_iterations[key]
        return booster


def requested_value_variants(value_variants_option: str) -> list[str]:
    """
    This function reads the extra value models asked for with --value-variants and stops the program if one is not 'two_part' or 'catboost'
    """
    value_variants = comma_separated(value_variants_option)
    unknown_variants = [variant for variant in value_variants if variant not in VALUE_VARIANTS]
    if unknown_variants:
        raise SystemExit(f'unknown --value-variants {unknown_variants}: choose from {",".join(VALUE_VARIANTS)}')
    return value_variants


def fit_value_model(run: TrainRun, value_parameters: dict[str, Any], value_label: np.ndarray, fit_rows: np.ndarray, value_baseline: np.ndarray) -> tuple[np.ndarray, pd.Index]:
    """
    This function trains the main XGBoost value model
    """
    value_model = run.train_xgboost('value', value_parameters, value_label, fit_rows)
    value = value_baseline + predict_in_chunks(value_model, run.features)
    value_gain_ranking = features_by_total_gain(value_model)
    value_model.save_model(run.run_weights / 'value.ubj')
    del value_model
    release_freed_memory()
    return value, value_gain_ranking


def fit_two_part_value(run: TrainRun, value_parameters: dict[str, Any], labels: np.ndarray, value_baseline: np.ndarray, training_rows: np.ndarray, is_normal: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    This function builds the two-part value model: a guesser for the chance that the taxi time is over 2,000 seconds (pLong), plus one correction model for long taxis and one for short ones
    The answer is v_tp = vbase + pLong x long answer + (1 - pLong) x short answer - all three learn from normal 2025 rows, and the correction is not cut here
    """
    is_long_taxi = (labels > LONG_TAXI_SECONDS).astype(float)
    long_classifier_parameters = dict(XGBOOST_BASE_PARAMETERS, **LONG_TAXI_CLASSIFIER_SETTINGS, seed=seed)
    long_classifier = run.train_xgboost('v_tp_cls', long_classifier_parameters, is_long_taxi, training_rows & is_normal)
    probability_long = predict_in_chunks(long_classifier, run.features)
    long_rows = is_normal & (labels > LONG_TAXI_SECONDS)
    short_rows = is_normal & (labels <= LONG_TAXI_SECONDS)
    residual = labels - value_baseline
    long_regression = run.train_xgboost('v_tp_long', dict(value_parameters, **LONG_TAXI_REGRESSION_SETTINGS), residual, training_rows & long_rows)
    short_regression = run.train_xgboost('v_tp_short', value_parameters, residual, training_rows & short_rows)
    value_two_part = (value_baseline + probability_long * predict_in_chunks(long_regression, run.features) + (1 - probability_long) * predict_in_chunks(short_regression, run.features))
    long_classifier.save_model(run.run_weights / 'v_tp_cls.ubj')
    long_regression.save_model(run.run_weights / 'v_tp_long.ubj')
    short_regression.save_model(run.run_weights / 'v_tp_short.ubj')
    del long_classifier, long_regression, short_regression
    release_freed_memory()
    return value_two_part, probability_long


def fit_catboost_value(run: TrainRun, value_gain_ranking: pd.Index, value_label: np.ndarray, fit_rows: np.ndarray, value_baseline: np.ndarray, seed: int) -> np.ndarray:
    """
    This function trains the CatBoost value model
    """
    columns, categorical_columns = catboost_columns(value_gain_ranking, run.features)
    catboost_parameters = dict(**CATBOOST_VALUE_SETTINGS, random_seed=seed, **CATBOOST_VALUE_RUNTIME_SETTINGS)
    catboost_model = CatBoostRegressor(iterations=run.fixed_iterations['v_cb'], **catboost_parameters).fit(catboost_pool(run.features, columns, categorical_columns, fit_rows, value_label[fit_rows]))
    run.iterations_used['v_cb'] = run.fixed_iterations['v_cb']
    catboost_residual = np.empty(len(value_baseline))
    for chunk_rows in chunk_masks(len(value_baseline), PREDICTION_CHUNK_ROWS):
        catboost_residual[chunk_rows] = catboost_model.predict(catboost_pool(run.features, columns, categorical_columns, chunk_rows))
    value_catboost = value_baseline + catboost_residual
    catboost_model.save_model(str(run.run_weights / 'v_cb.cbm'))
    del catboost_model
    release_freed_memory()
    return value_catboost


def fit_value_models(run: TrainRun, arguments: argparse.Namespace, value_variants: list[str], labels: np.ndarray, value_baseline: np.ndarray, training_rows: np.ndarray, is_normal: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    This function organizes the value models: it makes the correction to learn (cut to at most 6,000 seconds up or down on the 2025 rows), trains the XGBoost value model and then the extra models that were asked for
    Only normal (N) rows are used for learning
    """
    value_parameters = dict(XGBOOST_BASE_PARAMETERS, max_depth=arguments.depth, seed=arguments.seed, learning_rate=arguments.learning_rate)
    value_label = np.where(training_rows, np.clip(labels - value_baseline, -VALUE_RESIDUAL_CLIP, VALUE_RESIDUAL_CLIP), labels - value_baseline)
    value, value_gain_ranking = fit_value_model(run, value_parameters, value_label, training_rows & is_normal, value_baseline)
    value_variant_columns: dict[str, np.ndarray] = {}
    if 'two_part' in value_variants:
        value_two_part, probability_long = fit_two_part_value(run, value_parameters, labels, value_baseline, training_rows, is_normal, arguments.seed)
        value_variant_columns['v_tp'] = value_two_part
        value_variant_columns['pLong'] = probability_long
    if 'catboost' in value_variants:
        value_variant_columns['v_cb'] = fit_catboost_value(run, value_gain_ranking, value_label, training_rows & is_normal, value_baseline, arguments.seed)
    return value, value_variant_columns


def fit_schedule_copy_classifiers(run: TrainRun, seed: int, is_schedule_copy: np.ndarray, training_rows: np.ndarray, is_rome: np.ndarray, no_link: np.ndarray) -> tuple[np.ndarray, xgb.Booster, xgb.Booster]:
    """
    This function trains the three 'was the time copied from the timetable?' guessers on 2025 rows: one on all rows, one on Rome rows and one on rows with no NM link
    Each specialist then overwrites pP on its own rows, the no-link one last, so a Rome row with no NM link gets the no-link answer
    """
    classifier_parameters = schedule_copy_classifier_parameters(seed)
    schedule_copy_target = is_schedule_copy.astype(float)
    global_classifier = run.train_xgboost('pclass', classifier_parameters['pclass'], schedule_copy_target, training_rows)
    probability_schedule_copy = predict_in_chunks(global_classifier, run.features)
    rome_classifier = run.train_xgboost('pclass_lirf', classifier_parameters['pclass_lirf'], schedule_copy_target, training_rows & is_rome)
    probability_schedule_copy[is_rome] = predict_in_chunks(rome_classifier, run.features, is_rome)
    no_link_classifier = run.train_xgboost('pclass_na', classifier_parameters['pclass_na'], schedule_copy_target, training_rows & no_link)
    probability_schedule_copy[no_link] = predict_in_chunks(no_link_classifier, run.features, no_link)
    no_link_classifier.save_model(run.run_weights / 'pclass_na.ubj')
    return probability_schedule_copy, global_classifier, rome_classifier


def train(arguments: argparse.Namespace) -> None:
    start_time = time.time()
    value_variants = requested_value_variants(arguments.value_variants)
    features, departures = load_feature_matrix(arguments.adsb_table, arguments.extra_tables)
    seconds_after_schedule = departures.seconds_after_schedule.to_numpy()
    is_schedule_copy = departures.is_schedule_copy.to_numpy()
    is_normal = ~is_schedule_copy & ~departures.is_previous_date.to_numpy()
    no_link = departures.no_nm_link.to_numpy()
    is_rome = departures.is_rome.to_numpy()
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    training_rows = departures.year.to_numpy() == 2025       # all twelve months of 2025
    ranking_rows = departures.TAXITIME_SEC_mvt.isna().to_numpy()
    fixed_iterations = full_fit_iterations(arguments.iteration_set)
    value_baseline = schedule_independent_baseline(departures, training_rows & is_normal & ~no_link)
    add_regime_features(features, departures, value_baseline)
    run_predictions = RUN_PREDICTIONS / arguments.name
    run_predictions.mkdir(parents=True, exist_ok=True)
    run = TrainRun(features, fixed_iterations, MODEL_WEIGHTS / arguments.name, {})
    run.run_weights.mkdir(parents=True, exist_ok=True)

    value, value_variant_columns = fit_value_models(run, arguments, value_variants, labels, value_baseline, training_rows, is_normal)
    probability_schedule_copy, global_classifier, rome_classifier = fit_schedule_copy_classifiers(run, arguments.seed, is_schedule_copy, training_rows, is_rome, no_link)
    # regime S occurs only in the Rome next-day no-link group: there P / S / N come from the delay-hours logistic regression
    delay_hours = seconds_after_schedule / 3600.0
    next_day_group = rome_next_day_no_link_group(departures)
    next_day_logistic, group_probabilities = fit_next_day_logistic(delay_hours, regime_class_labels(departures), training_rows & next_day_group, next_day_group)
    probability_previous_date = np.zeros(len(labels))
    probability_schedule_copy[next_day_group] = group_probabilities[:, 0]
    probability_previous_date[next_day_group] = group_probabilities[:, 1]
    if arguments.average_value_variants:
        value_parts = [value] + [value_variant_columns[key] for key in ('v_tp', 'v_cb') if key in value_variant_columns]
        value_variant_columns['v_xgb'] = value
        value = sum(value_parts) / len(value_parts)
    prediction = mixture_prediction(probability_schedule_copy, probability_previous_date, seconds_after_schedule, value)
    # no validation rows in a full fit: the column 'valid' is kept for the common pred.parquet layout
    pd.DataFrame({'MVT_ID_mvt': departures.MVT_ID_mvt, 
                  'pred': prediction, 
                  'v': value, 
                  'pP': probability_schedule_copy,
                  'pS': probability_previous_date, 
                  'msched': seconds_after_schedule, 
                  **value_variant_columns,
                  'valid': np.zeros(len(labels), bool), 
                  'train': training_rows, 
                  'rank': ranking_rows}).to_parquet(run_predictions / 'pred.parquet')
    global_classifier.save_model(run.run_weights / 'pclass.ubj')
    rome_classifier.save_model(run.run_weights / 'pclass_lirf.ubj')
    save_logistic_regression(next_day_logistic, run.run_weights / 'nextday_logistic.json')
    metrics = run_metrics(run.iterations_used, arguments)
    (run_predictions / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print('time', time.time() - start_time)


# stake-classifiers
@dataclass
class StakeClassifierInputs:
    """
    This data structure holds what all the copy guessers of one stake-classifiers job share
    """
    features: pd.DataFrame              # the model matrix (load_feature_matrix + add_regime_features)
    is_schedule_copy: np.ndarray        # the target: regime P
    stake_weights: np.ndarray | None    # row weights (stake_row_weights), None = unweighted
    fixed_iterations: dict[str, int]    # boosting rounds per model key (stake_classifiers validation rounds x1.1)
    seed: int
    catboost_columns: list[str]         # the CatBoost models' columns, and those of them that are categorical
    categorical_columns: list[str]
    classifier_weights: Path            # taxi/models/weights/regime_stake_classifiers/


def stake_row_weights(seconds_after_schedule: np.ndarray, value_baseline: np.ndarray, training_rows: np.ndarray, stake_weight_power: float) -> np.ndarray | None:
    """
    This function decides how much each row counts when learning, from its stake - the gap between the timetable answer and vbase, which is how much a wrong copy guess would cost (cut to between 300 seconds and 4 hours)
    """
    stake_weights: np.ndarray | None = None
    if stake_weight_power > 0:
        stake = np.clip(np.abs(np.nan_to_num(seconds_after_schedule, nan=0.0) - value_baseline), STAKE_FLOOR_SECONDS, STAKE_CAP_SECONDS)
        stake_weights = (stake / STAKE_WEIGHT_UNIT_SECONDS) ** stake_weight_power
        stake_weights = stake_weights / stake_weights[training_rows].mean()
    return stake_weights


def fit_and_predict_catboost_classifier(inputs: StakeClassifierInputs, key: str, fit_rows: np.ndarray, prediction_rows: np.ndarray, depth: int, learning_rate: float) -> np.ndarray:
    """
    This function trains one CatBoost 'copied from the timetable?' guesser
    """
    iteration_count = inputs.fixed_iterations[key]
    model = CatBoostClassifier(iterations=iteration_count, **CATBOOST_CLASSIFIER_SETTINGS, depth=depth, learning_rate=learning_rate, random_seed=inputs.seed, **CATBOOST_CLASSIFIER_RUNTIME_SETTINGS)
    model.fit(catboost_pool(inputs.features, inputs.catboost_columns, inputs.categorical_columns, fit_rows, inputs.is_schedule_copy[fit_rows].astype(int), None if inputs.stake_weights is None else inputs.stake_weights[fit_rows]))
    prediction_pool = catboost_pool(inputs.features, inputs.catboost_columns, inputs.categorical_columns, prediction_rows)
    probabilities = model.predict_proba(prediction_pool)[:, 1]
    del prediction_pool
    model.save_model(str(inputs.classifier_weights / f'{key}.cbm'))
    del model
    gc.collect()
    print(key, iteration_count, 'done', flush=True)
    return probabilities


def fit_and_predict_xgboost_classifier(inputs: StakeClassifierInputs, key: str, fit_rows: np.ndarray, prediction_rows: np.ndarray) -> np.ndarray:
    """
    This function trains one of the train job's XGBoost copy guessers again
    """
    model = train_booster(schedule_copy_classifier_parameters(inputs.seed)[key], inputs.features, inputs.is_schedule_copy.astype(float), fit_rows, inputs.fixed_iterations[key], row_weights=inputs.stake_weights)
    probabilities = predict_in_chunks(model, inputs.features, prediction_rows)
    model.save_model(inputs.classifier_weights / f'{key}.ubj')
    del model
    gc.collect()
    print(key, 'xgb done', flush=True)
    return probabilities


def update_run_with_stake_classifiers(run_name: str, suffix: str, movement_ids: np.ndarray, ranking_rows: np.ndarray, next_day_group: np.ndarray, xgboost_probability: np.ndarray, catboost_probability: np.ndarray, seconds_after_schedule: np.ndarray) -> None:
    """
    This function takes a finished train run and, on the ranking rows (the ones the challenge scores) outside the Rome next-day group, makes pP the average of the new XGBoost and CatBoost chances
    """
    run = pd.read_parquet(RUN_PREDICTIONS / run_name / 'pred.parquet')
    assert (run.MVT_ID_mvt.to_numpy() == movement_ids).all()
    updated_rows = ranking_rows & ~next_day_group        # the next-day group keeps the delay-hours logistic regression
    probability_schedule_copy = run.pP.to_numpy().copy()
    probability_schedule_copy[updated_rows] = xgboost_probability[updated_rows]
    run['pP_xgb'] = probability_schedule_copy.copy()
    run['pP_cb'] = catboost_probability
    probability_schedule_copy[updated_rows] = STAKE_CLASSIFIER_BLEND_WEIGHT * (probability_schedule_copy[updated_rows] + catboost_probability[updated_rows])
    value = run.v.to_numpy()
    probability_previous_date = run.pS.to_numpy()
    prediction = mixture_prediction(probability_schedule_copy, probability_previous_date, seconds_after_schedule, value)
    previous_prediction = run.pred.to_numpy().copy()
    run['pP'] = probability_schedule_copy
    run.loc[ranking_rows, 'pred'] = prediction[ranking_rows]
    output = RUN_PREDICTIONS / (run_name + suffix)
    output.mkdir(exist_ok=True)
    run.to_parquet(output / 'pred.parquet', index=False)
    mean_change = np.abs(prediction[ranking_rows] - previous_prediction[ranking_rows]).mean()
    print(run_name, '->', output.name, 'mean |dpred| on ranking %.1f' % mean_change, flush=True)


def stake_classifiers(arguments: argparse.Namespace) -> None:
    """
    This function is the whole stake-classifiers job: it trains three CatBoost and three weighted XGBoost copy guessers (all rows, Rome rows, rows with no NM link) once, and uses them to update every named run
    """
    run_names = arguments.runs.split(',')
    fixed_iterations = full_fit_iterations('stake_classifiers')
    features, departures = load_feature_matrix(arguments.adsb_table, arguments.extra_tables)
    seconds_after_schedule = departures.seconds_after_schedule.to_numpy()
    is_schedule_copy = departures.is_schedule_copy.to_numpy()
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    training_rows = ~np.isnan(labels) & (departures.year.to_numpy() == 2025)
    ranking_rows = np.isnan(labels)
    is_normal = ~is_schedule_copy & ~departures.is_previous_date.to_numpy()
    no_link = departures.no_nm_link.to_numpy()
    is_rome = departures.is_rome.to_numpy()
    value_baseline = schedule_independent_baseline(departures, training_rows & is_normal & ~no_link)
    add_regime_features(features, departures, value_baseline)
    stake_weights = stake_row_weights(seconds_after_schedule, value_baseline, training_rows, arguments.stake_weight_power)
    # the CatBoost columns: top gain of the first run's global XGBoost classifier
    importance_model = xgb.Booster()
    importance_model.load_model(MODEL_WEIGHTS / run_names[0] / 'pclass.ubj')
    classifier_weights = MODEL_WEIGHTS / 'regime_stake_classifiers'
    classifier_weights.mkdir(parents=True, exist_ok=True)
    columns, categorical_columns = catboost_columns(features_by_total_gain(importance_model), features)
    print('features', len(columns), 'cats', len(categorical_columns), flush=True)
    inputs = StakeClassifierInputs(features, is_schedule_copy, stake_weights, fixed_iterations, arguments.seed, columns, categorical_columns, classifier_weights)
    # the three classifiers of train: global, Rome and no-link (fit rows, predicted rows); only ranking rows are predicted
    classifier_rows = {'pclass': (training_rows, ranking_rows), 
                       'pclass_lirf': (training_rows & is_rome, ranking_rows & is_rome),
                       'pclass_na': (training_rows & no_link, ranking_rows & no_link)}
    catboost_depth_and_learning_rate = {**CATBOOST_CLASSIFIER_DEPTH_AND_LEARNING_RATE}
    catboost_probability = np.full(len(labels), np.nan)
    for key, (fit_rows, prediction_rows) in classifier_rows.items():
        depth, learning_rate = catboost_depth_and_learning_rate[key]
        catboost_probability[prediction_rows] = fit_and_predict_catboost_classifier(inputs, f'{key}_cb', fit_rows, prediction_rows, depth, learning_rate)
    next_day_group = rome_next_day_no_link_group(departures)
    xgboost_probability = np.full(len(labels), np.nan)
    for key, (fit_rows, prediction_rows) in classifier_rows.items():
        xgboost_probability[prediction_rows] = fit_and_predict_xgboost_classifier(inputs, key, fit_rows, prediction_rows)
    for run_name in run_names:
        update_run_with_stake_classifiers(run_name, arguments.suffix, departures.MVT_ID_mvt.to_numpy(), ranking_rows, next_day_group, xgboost_probability, catboost_probability, seconds_after_schedule)


# nextday
def apply_next_day_prior(run_name: str, suffix: str, movement_ids: np.ndarray, seconds_after_schedule: np.ndarray, delay_hours: np.ndarray, class_labels: np.ndarray, next_day_group: np.ndarray, operator_keys: np.ndarray, flight_keys: np.ndarray) -> None:
    """
    This function does the nextday fix for one run: it refits the delay-only P / S / N formula on the run's training rows of the Rome next-day group and nudges its P vs S split toward the airline's and the flight's past counts
    """
    run = pd.read_parquet(RUN_PREDICTIONS / run_name / 'pred.parquet')
    assert (run.MVT_ID_mvt.to_numpy() == movement_ids).all()
    fit_rows = run.train.to_numpy() & next_day_group
    logistic, group_probabilities = fit_next_day_logistic(delay_hours, class_labels, fit_rows, next_day_group)
    group_probabilities = next_day_group_prior(group_probabilities, class_labels[next_day_group], fit_rows[next_day_group], delay_hours[next_day_group], (operator_keys[next_day_group], flight_keys[next_day_group]))
    probability_schedule_copy = run.pP.to_numpy().copy()
    probability_previous_date = run.pS.to_numpy().copy()
    value = run.v.to_numpy()
    probability_schedule_copy[next_day_group] = group_probabilities[:, 0]
    probability_previous_date[next_day_group] = group_probabilities[:, 1]
    prediction = mixture_prediction(probability_schedule_copy, probability_previous_date, seconds_after_schedule, value)
    new_prediction = run.pred.to_numpy().copy()
    new_prediction[next_day_group] = prediction[next_day_group]
    run['pP'] = probability_schedule_copy
    run['pS'] = probability_previous_date
    run['pred'] = new_prediction
    output = RUN_PREDICTIONS / (run_name + suffix)
    output.mkdir(exist_ok=True)
    run.to_parquet(output / 'pred.parquet', index=False)
    run_weights = MODEL_WEIGHTS / (run_name + suffix)
    run_weights.mkdir(parents=True, exist_ok=True)
    save_logistic_regression(logistic, run_weights / 'nextday_logistic.json')
    print(run_name, '->', output.name, 'group rows ranked:', (next_day_group & run['rank'].to_numpy()).sum())


def next_day_prior_subcommand(arguments: argparse.Namespace) -> None:
    """
    This function is the whole nextday job
    """
    departures = departure_frame()
    seconds_after_schedule = departures.seconds_after_schedule.to_numpy()
    flight_keys, operator_keys = flight_and_operator_keys(departures)
    class_labels = regime_class_labels(departures)
    next_day_group = rome_next_day_no_link_group(departures)
    delay_hours = seconds_after_schedule / 3600.0
    for run_name in arguments.runs:
        apply_next_day_prior(run_name, arguments.suffix, departures.MVT_ID_mvt.to_numpy(), seconds_after_schedule, delay_hours, class_labels, next_day_group, operator_keys, flight_keys)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the regime mixture models.')
    subcommands = parser.add_subparsers(required=True)
    train_parser = subcommands.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--name', required=True)
    train_parser.add_argument('--iteration-set', required=True, choices=['regime', 'xgboost_value'])
    train_parser.add_argument('--depth', type=int, default=DEFAULT_VALUE_MAX_DEPTH)
    train_parser.add_argument('--learning-rate', type=float, default=DEFAULT_VALUE_LEARNING_RATE)
    train_parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    train_parser.add_argument('--adsb-table', default='adsb_pass1')
    train_parser.add_argument('--extra-tables', default=DEFAULT_EXTRA_TABLES)
    train_parser.add_argument('--value-variants', default='')
    train_parser.add_argument('--average-value-variants', action='store_true')
    stake_parser = subcommands.add_parser('stake-classifiers')
    stake_parser.set_defaults(func=stake_classifiers)
    stake_parser.add_argument('--runs', required=True)
    stake_parser.add_argument('--adsb-table', default='adsb_pass1')
    stake_parser.add_argument('--extra-tables', default=DEFAULT_EXTRA_TABLES)
    stake_parser.add_argument('--stake-weight-power', type=float, default=DEFAULT_STAKE_WEIGHT_POWER)
    stake_parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    stake_parser.add_argument('--suffix', default='_stake_classifiers')
    nextday_parser = subcommands.add_parser('nextday')
    nextday_parser.set_defaults(func=next_day_prior_subcommand)
    nextday_parser.add_argument('runs', nargs='+')
    nextday_parser.add_argument('--suffix', default='_nextday')
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.func(arguments)


if __name__ == '__main__':
    main()
