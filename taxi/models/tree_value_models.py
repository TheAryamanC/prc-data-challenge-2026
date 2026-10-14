"""
This file trains two more value models (XGBoost and CatBoost tree models), which guess the normal taxi time of a plane - way N, where the 'left the stand' time was written down properly
Both start from the same base - the rough first guess and only learn the correction on top
The two-part model exists because a few planes take far longer than the NM time suggests, for example when the airport crew is deicing the runway or the plane has to wait in a long line
Fewer than 1% of linked rows go over by more than 20 minutes, but they cause over a quarter of the rough guess's squared error

This is how it works:
-two-part:
    -a classifier q guesses the chance that the correction is above a threshold (1,200 seconds by default)
    -a 'hi' model learns the big corrections and a 'lo' model learns the rest
    -the answer is v_tpr = vbase + q x hi + (1 - q) x lo
-catboost: one CatBoost model learns the correction from the 200 most useful clues plus the extra tables (airport_state by default), giving v_cb
-each model is trained in five ways:
    -'full' learns from all of 2025, with 1.1 times the rounds found on the July 2025 test runs
    -'wk0' to 'wk3' each hide every fourth week of 2025, so the answers on the hidden weeks are generalizable
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from taxi.models.regime_mixture import (chunk_masks, comma_separated, features_by_total_gain, rows_in_departure_order, run_metrics, schedule_independent_baseline)
from taxi.main import FEATURE_MATRIX, MODEL_WEIGHTS, RUN_PREDICTIONS, WEEK_FOLDS, departure_frame, held_out_week_fold_rows

RESIDUAL_CLIP = 6000                # regression targets (residuals on vbase) are clipped at +-6000 s
PREDICTION_CHUNK_ROWS = 400000      # rows per prediction batch: bounds the memory of the DMatrix / Pool copies
CATBOOST_TOP_FEATURE_COUNT = 200    # the CatBoost value model uses the 200 features of highest total gain of --importance-from
FOLD_CHOICES = ('full',) + WEEK_FOLDS
# Boosting rounds chosen by early stopping on July 2025 held out: x1.1 for the full fit, x1.0 for a week-fold run (iteration_scale)
TWO_PART_VALIDATION_ITERATIONS = {'q': 765, 'hi': 113, 'lo': 3991}     # classifier q, regressions hi / lo
CATBOOST_VALIDATION_ITERATIONS = 5996       # the CatBoost value model's best round on July 2025 held out (training on the other 2025 months)
# Factor on the validation rounds (iteration_scale)
FULL_FIT_ITERATION_SCALE = 1.1      # fold 'full': the full fit has one more month of training rows than the July-held-out run
WEEK_FOLD_ITERATION_SCALE = 1.0     # a week-fold run reuses the July 2025 counts as they are, so it never early-stops on its hidden weeks
# XGBoost settings of the two-part regressions hi / lo
TWO_PART_REGRESSION_SETTINGS = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 20,
    'reg_lambda': 5.0,
    'max_bin': 256,
}
TWO_PART_CLASSIFIER_SETTINGS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}
TWO_PART_HIGH_RESIDUAL_SETTINGS = {     # shallower trees and smaller leaves, as only the few rows above the threshold train 'hi'
    'max_depth': 6,
    'min_child_weight': 10,
}
# CatBoost settings of the catboost value model
CATBOOST_VALUE_SETTINGS = {
    'loss_function': 'RMSE',
    'depth': 8,
    'learning_rate': 0.08,
    'task_type': 'GPU',
    'border_count': 128,
}
CATBOOST_LOG_PERIOD_ROUNDS = 500    # print the loss every 500 rounds - logging only
# Command-line defaults
DEFAULT_SEED = 42                           # the usual fixed seed, only so a rerun gives the same model
DEFAULT_TWO_PART_TOP_FEATURES = 200         # a round number, the same 200 as the CatBoost value model
DEFAULT_TWO_PART_THRESHOLD_SECONDS = 1200   # 20 minutes over vbase counts as a long wait
DEFAULT_CATBOOST_GPU_RAM_PART = 0.6         # share of the 8 GB GPU memory CatBoost may take (its default is 0.95)


def iteration_scale(fold: str) -> float:
    """
    This function gives the factor on the round counts: 1.1 when learning from all of 2025 ('full'), 1.0 for a week-fold run
    """
    return FULL_FIT_ITERATION_SCALE if fold == 'full' else WEEK_FOLD_ITERATION_SCALE


def held_out_fold_rows(departures: pd.DataFrame, fold: str) -> np.ndarray:
    """
    This function marks the rows hidden from training: none for 'full', otherwise the 2025 departures of that week fold (every fourth week)
    """
    return np.zeros(len(departures), bool) if fold == 'full' else held_out_week_fold_rows(departures, fold)


def feature_table_names(adsb_table: str) -> tuple[str, str, str, str]:
    """
    This function lists, in order, the four tables the clues are looked up in: base_features, the chosen ADS-B table, arrival_proxy_features and xplane_geometry_features
    """
    return ('base_features', adsb_table, 'arrival_proxy_features', 'xplane_geometry_features')


def table_column_names(table_name: str) -> set[str]:
    return set(pq.read_schema(FEATURE_MATRIX / f'{table_name}.parquet').names)


def read_table_columns(table_name: str, columns: list[str], is_adsb_table: bool, movement_ids: pd.Series) -> pd.DataFrame:
    """
    This function reads only the wanted columns of a table, one row per departure
    """
    table = pd.read_parquet(FEATURE_MATRIX / f'{table_name}.parquet', columns=(['MVT_ID_mvt'] if is_adsb_table else []) + columns)
    if is_adsb_table:
        table = rows_in_departure_order(table, movement_ids)
    return table


def training_and_baseline(departures: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function sets up the rows of a run: the hidden rows, the training rows (labeled 2025 rows that are not hidden), the normal (N) rows, and vbase
    vbase takes the airport middle values only from normal, NM-linked training rows, so the hidden rows do not help build it
    """
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    no_link = departures.no_nm_link.to_numpy()
    held_out_rows = held_out_fold_rows(departures, fold)
    training_rows = np.isfinite(labels) & ~held_out_rows & (departures.year.to_numpy() == 2025)
    is_normal = (~departures.is_schedule_copy & ~departures.is_previous_date).to_numpy()
    value_baseline = schedule_independent_baseline(departures, training_rows & is_normal & ~no_link)
    return held_out_rows, training_rows, is_normal, value_baseline


def write_run_predictions(run_name: str, columns: dict[str, Any], iterations_used: dict[str, int], arguments: argparse.Namespace) -> Path:
    """
    This function saves a run's answers as pred.parquet and its report card as metrics.json in the run's folder
    """
    run_predictions = RUN_PREDICTIONS / run_name
    run_predictions.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_parquet(run_predictions / 'pred.parquet', index=False)
    (run_predictions / 'metrics.json').write_text(json.dumps(run_metrics(iterations_used, arguments), indent=1))
    return run_predictions


# two-part
def load_top_gain_features(top_feature_count: int, importance_run: str, extra_tables: list[str], adsb_table: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function builds the clue table of the two-part model: it takes the most useful clues of a chosen XGBoost value model, reads them from the four clue tables and adds every column of the extra tables
    """
    importance_model = xgb.Booster()
    importance_model.load_model(MODEL_WEIGHTS / importance_run / 'value.ubj')
    wanted_features = features_by_total_gain(importance_model)[:top_feature_count]
    del importance_model
    departures = departure_frame()
    parts: list[pd.DataFrame] = []
    for table_name in feature_table_names(adsb_table):
        column_names = table_column_names(table_name)
        selected_columns = [column for column in wanted_features if column in column_names]
        if not selected_columns:
            continue
        table = read_table_columns(table_name, selected_columns, table_name == adsb_table, departures.MVT_ID_mvt)
        parts.append(table[selected_columns].apply(lambda column: column.cat.codes if str(column.dtype) == 'category' else column).astype('float32'))
    for table_name in extra_tables:
        parts.append(pd.read_parquet(FEATURE_MATRIX / f'{table_name}.parquet').drop(columns='MVT_ID_mvt').astype('float32'))
    return pd.concat(parts, axis=1), departures


def two_part_training_rows(fit_rows: np.ndarray, residual: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """
    This function decides which rows each part learns from: q learns from all fit rows, 'hi' only from rows whose correction is above the threshold and 'lo' only from the rest
    """
    above = residual > threshold
    return {'q': fit_rows, 'hi': fit_rows & above, 'lo': fit_rows & ~above}


def train_two_part_submodel(features: pd.DataFrame, parameters: dict[str, Any], label: np.ndarray, training_rows: np.ndarray, boosting_rounds: int, model_path: Path) -> tuple[np.ndarray, int]:
    """
    This function trains one part of the two-part model with XGBoost for a fixed number of rounds
    """
    training_matrix = xgb.QuantileDMatrix(features[training_rows], label=label[training_rows])
    booster = xgb.train(parameters, training_matrix, boosting_rounds)
    boosted_rounds = booster.num_boosted_rounds()
    booster.save_model(model_path)
    predictions = np.empty(len(label))
    for start in range(0, len(label), PREDICTION_CHUNK_ROWS):
        chunk = features.iloc[start:start + PREDICTION_CHUNK_ROWS]
        predictions[start:start + PREDICTION_CHUNK_ROWS] = booster.predict(xgb.DMatrix(chunk), iteration_range=(0, boosted_rounds))
    return predictions, boosted_rounds


def two_part_subcommand(arguments: argparse.Namespace) -> None:
    """
    This function is the whole two-part job
    """
    features, departures = load_top_gain_features(arguments.top_features, arguments.importance_from, comma_separated(arguments.extra_tables), arguments.adsb_table)
    features = features.drop(columns=[column for column in arguments.drop_columns.split(',') if column in features])
    features['msched'] = departures.seconds_after_schedule.to_numpy().astype('float32')
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    held_out_rows, training_rows, is_normal, value_baseline = training_and_baseline(departures, arguments.fold)
    features['vbase'] = value_baseline.astype('float32')
    residual = labels - value_baseline
    clipped_residual = np.clip(residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    fit_rows = training_rows & is_normal & np.isfinite(clipped_residual)
    submodel_rows = two_part_training_rows(fit_rows, residual, arguments.threshold)
    boosting_rounds = {key: int(round(count * iteration_scale(arguments.fold))) for key, count in TWO_PART_VALIDATION_ITERATIONS.items()}
    regression_parameters = dict(**TWO_PART_REGRESSION_SETTINGS, seed=arguments.seed)
    submodels = {'q': (dict(regression_parameters, **TWO_PART_CLASSIFIER_SETTINGS), (residual > arguments.threshold).astype(float)),
                 'hi': (dict(regression_parameters, **TWO_PART_HIGH_RESIDUAL_SETTINGS), clipped_residual),
                 'lo': (regression_parameters, clipped_residual)}
    run_weights = MODEL_WEIGHTS / arguments.name
    run_weights.mkdir(parents=True, exist_ok=True)
    submodel_predictions: dict[str, np.ndarray] = {}
    iterations_used: dict[str, int] = {}       # metrics.json 'iters', keyed q / hi / lo as in the iterations file
    for key, (parameters, label) in submodels.items():
        submodel_predictions[key], iterations_used[key] = train_two_part_submodel(features, parameters, label, submodel_rows[key], boosting_rounds[key], run_weights / f'{key}.ubj')
    exceed_probability = submodel_predictions['q']
    value_two_part = (value_baseline + exceed_probability * submodel_predictions['hi'] + (1 - exceed_probability) * submodel_predictions['lo'])
    run_predictions = write_run_predictions(arguments.name, {'MVT_ID_mvt': departures.MVT_ID_mvt, 
                                                             'v_tpr': value_two_part,
                                                             'q_exceed': exceed_probability, 
                                                             'valid': held_out_rows,
                                                             'rank': ~np.isfinite(labels)}, iterations_used, arguments)
    print('saved', run_predictions, iterations_used)


# catboost
def catboost_top_features(importance_run: str, drop_columns: str) -> list[str]:
    """
    This function takes the 200 most useful clues of a chosen XGBoost value model
    """
    importance_model = xgb.Booster()
    importance_model.load_model(MODEL_WEIGHTS / importance_run / 'value.ubj')
    top_features = list(features_by_total_gain(importance_model)[:CATBOOST_TOP_FEATURE_COUNT])
    del importance_model
    return [column for column in top_features if column not in drop_columns.split(',')]


def read_top_feature_columns(top_features: list[str], departures: pd.DataFrame, adsb_table: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    This function goes through the four clue tables in order and reads each wanted clue the first time it finds it
    Category columns become whole-number codes (-1 for missing), and the memory is freed after each table
    """
    columns: dict[str, np.ndarray] = {}
    categorical_columns: list[str] = []
    for table_name in feature_table_names(adsb_table):
        column_names = table_column_names(table_name)
        selected_columns = [column for column in top_features if column in column_names and column not in columns]
        if not selected_columns:
            continue
        table = read_table_columns(table_name, selected_columns, table_name == adsb_table, departures.MVT_ID_mvt)
        for column in selected_columns:
            if str(table[column].dtype) == 'category':
                columns[column] = table[column].cat.codes.to_numpy().astype('int32')
                categorical_columns.append(column)
            else:
                columns[column] = table[column].to_numpy(dtype='float32')
        del table
        gc.collect()
    return columns, categorical_columns


def add_derived_columns(columns: dict[str, np.ndarray], categorical_columns: list[str], top_features: list[str], departures: pd.DataFrame, value_baseline: np.ndarray) -> None:
    """
    This function adds the clues that are in no table but are made from the departure details (msched, sched_daydiff, vbase and the airline operator3), but only the ones among the top clues
    """
    derived_features = {'msched': departures.seconds_after_schedule.to_numpy(dtype='float32'),
                        'sched_daydiff': departures.day_difference.to_numpy(dtype='float32'), 
                        'vbase': value_baseline.astype('float32')}
    for feature_name, feature_values in derived_features.items():
        if feature_name in top_features:
            columns[feature_name] = feature_values
    if 'operator3' in top_features:
        columns['operator3'] = pd.Categorical(departures.operator_code.to_numpy()).codes.astype('int32')
        categorical_columns.append('operator3')


def add_extra_table_columns(columns: dict[str, np.ndarray], extra_tables: str, movement_ids: np.ndarray) -> None:
    """
    This function adds every column of the extra tables that is not there yet
    """
    for table_name in comma_separated(extra_tables):
        table = pd.read_parquet(FEATURE_MATRIX / f'{table_name}.parquet')
        assert (table.MVT_ID_mvt.to_numpy() == movement_ids).all()
        for column in table.columns:
            if column != 'MVT_ID_mvt' and column not in columns:
                columns[column] = table[column].to_numpy(dtype='float32')
        del table
        gc.collect()


def catboost_matrix(top_features: list[str], departures: pd.DataFrame, value_baseline: np.ndarray, adsb_table: str, extra_tables: str) -> tuple[pd.DataFrame, list[str]]:
    """
    This function puts the CatBoost clue table together
    """
    columns, categorical_columns = read_top_feature_columns(top_features, departures, adsb_table)
    add_derived_columns(columns, categorical_columns, top_features, departures, value_baseline)
    add_extra_table_columns(columns, extra_tables, departures.MVT_ID_mvt.to_numpy())
    column_order = [column for column in top_features if column in columns] + [column for column in columns if column not in top_features]
    matrix = pd.DataFrame({column: columns[column] for column in column_order}, copy=False)
    del columns
    gc.collect()
    return matrix, categorical_columns


def catboost_pool_of_rows(matrix: pd.DataFrame, categorical_columns: list[str], row_mask: np.ndarray, label: np.ndarray | None = None) -> Pool:
    """
    This function packs the chosen rows of the clue table into a CatBoost Pool
    """
    return Pool(matrix.loc[row_mask], label=label, cat_features=categorical_columns)


def predict_catboost_residual(model: CatBoostRegressor, matrix: pd.DataFrame, categorical_columns: list[str]) -> np.ndarray:
    """
    This function asks the trained CatBoost model for its correction on every row
    """
    catboost_residual = np.empty(len(matrix))
    for chunk_rows in chunk_masks(len(matrix), PREDICTION_CHUNK_ROWS):
        catboost_residual[chunk_rows] = model.predict(catboost_pool_of_rows(matrix, categorical_columns, chunk_rows))
    return catboost_residual


def catboost_value_subcommand(arguments: argparse.Namespace) -> None:
    """
    This function is the whole catboost job
    """
    start_time = time.time()
    departures = departure_frame()
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    held_out_rows, training_rows, is_normal, value_baseline = training_and_baseline(departures, arguments.fold)
    value_label = np.where(training_rows, np.clip(labels - value_baseline, -RESIDUAL_CLIP, RESIDUAL_CLIP), labels - value_baseline)
    top_features = catboost_top_features(arguments.importance_from, arguments.drop_columns)
    matrix, categorical_columns = catboost_matrix(top_features, departures, value_baseline, arguments.adsb_table, arguments.extra_tables)
    print('catboost columns', len(matrix.columns), 'of which top-200', sum(column in top_features for column in matrix.columns), 'cats', categorical_columns, '%.0fs' % (time.time() - start_time), flush=True)

    catboost_parameters = dict(**CATBOOST_VALUE_SETTINGS, random_seed=arguments.seed, verbose=CATBOOST_LOG_PERIOD_ROUNDS, gpu_ram_part=arguments.gpu_ram_part, allow_writing_files=False)
    fit_rows = training_rows & is_normal
    iteration_count = int(round(CATBOOST_VALIDATION_ITERATIONS * iteration_scale(arguments.fold)))
    model = CatBoostRegressor(iterations=iteration_count, **catboost_parameters).fit(catboost_pool_of_rows(matrix, categorical_columns, fit_rows, value_label[fit_rows]))
    iterations_used = {'v_cb': iteration_count}
    (MODEL_WEIGHTS / arguments.name).mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_WEIGHTS / arguments.name / 'v_cb.cbm'))
    gc.collect()
    value = value_baseline + predict_catboost_residual(model, matrix, categorical_columns)
    write_run_predictions(arguments.name, {'MVT_ID_mvt': departures.MVT_ID_mvt, 
                                           'v_cb': value, 
                                           'valid': held_out_rows,
                                           'rank': ~np.isfinite(labels)}, iterations_used, arguments)
    print(json.dumps(run_metrics(iterations_used, arguments), indent=1))
    print('time %.0f' % (time.time() - start_time))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the tree value models.')
    subcommands = parser.add_subparsers(required=True)
    for subcommand_name, function in (('two-part', two_part_subcommand), ('catboost', catboost_value_subcommand)):
        subparser = subcommands.add_parser(subcommand_name)
        subparser.set_defaults(func=function)
        subparser.add_argument('--name', required=True)
        subparser.add_argument('--importance-from', required=True)
        subparser.add_argument('--extra-tables', default='airport_state')
        subparser.add_argument('--seed', type=int, default=DEFAULT_SEED)
        subparser.add_argument('--drop-columns', default='')
        subparser.add_argument('--adsb-table', default='adsb_pass1')
        subparser.add_argument('--fold', required=True, choices=FOLD_CHOICES)
    two_part_parser = subcommands.choices['two-part']
    two_part_parser.add_argument('--top-features', type=int, default=DEFAULT_TWO_PART_TOP_FEATURES)
    two_part_parser.add_argument('--threshold', type=float, default=DEFAULT_TWO_PART_THRESHOLD_SECONDS)
    subcommands.choices['catboost'].add_argument('--gpu-ram-part', type=float, default=DEFAULT_CATBOOST_GPU_RAM_PART)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.func(arguments)


if __name__ == '__main__':
    main()
