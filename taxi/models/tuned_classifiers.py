"""
At some airports, Rome most of all, the recorded time a plane left its stand is sometimes just a copy of the timetable time - a schedule copy, or regime P
For such a flight the true answer is take-off minus timetable time, which can be hours, so the main models need the chance that a record is a copy
This file trains two special yes/no XGBoost guessers (classifiers) for the two groups where a wrong chance costs the most: Rome flights with an NM link (a matching Network Manager flight-plan record) and flights with no NM link at all

This is how it works:
-take the 200 most helpful facts of the earlier classifier of the same group (from the regime run regime_seed42)
-give each flight a stake: how far apart its copy answer and its normal-time guess are, capped at 4 hours
-tune (the 'tune' command):
    -let Optuna try many settings, starting with the hand-set ones, and make high-stake flights count more in training
    -score each setting with 4-fold month-group cross-validation, by the average of (stake times (chance minus truth)) squared
-final (the 'final' command):
    -take the best settings, drop the calendar columns (month, day, day of year) and retrain on all labeled 2025 flights of the group
    -save the model and each 2026 flight's chance of being a schedule copy
-a third version of the Rome classifier also gets the reference network's taxi time and the push delay it implies
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from taxi.models.regime_mixture import features_by_total_gain, schedule_independent_baseline
from taxi.main import (FEATURE_MATRIX, MODEL_WEIGHTS, RUN_PREDICTIONS, TUNED_CLASSIFIERS, WEEK_FOLDS, departure_frame, held_out_week_fold_rows)

MONTH_GROUP_FOLDS = [(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]     # held-out months of the four cross-validation folds
# taxi.models.out_of_fold changes device and nthread in place before cross-validating, so the functions read this dict when they run
CLASSIFIER_BASE_PARAMETERS: dict[str, str | int | float] = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda', subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, max_cat_to_onehot=12, max_cat_threshold=64, max_bin=256, seed=42)
# trial 0 of each study: the configuration of the regime runs' specialist (pclass_lirf / pclass_na) with stake-weight power 1
HAND_SET_PARAMETERS: dict[str, dict[str, float]] = {
    'rome_linked': dict(max_depth=6, min_child_weight=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, alpha_stake=1.0),
    'no_link': dict(max_depth=5, min_child_weight=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, alpha_stake=1.0)}
CALENDAR_COLUMNS = ['month', 'day', 'dayofyear']      # the calendar position, removed by final
IMPORTANCE_RUN = 'regime_seed42'                      # the regime run whose classifier ranks the input features
IMPORTANCE_MODEL_FILE = {'rome_linked': 'pclass_lirf.ubj', 'no_link': 'pclass_na.ubj'}
TOP_FEATURE_COUNT = 200                               # a round number that keeps the loaded table small in memory
ROW_ALIGNED_TABLES = ('base_features', 'arrival_proxy_features', 'xplane_geometry_features')   # one row per departure, departure order
ADSB_TABLE = 'adsb_pass1'                             # ADS-B track features (aircraft position broadcasts), keyed by movement id
REFERENCE_NETWORK_FOLD_RUN = 'network_pass3_clip15000'   # week-fold runs <name>_wk0 .. _wk3 of the reference network
FULL_FIT_ITERATION_SCALE = 1.1                        # cross-validated rounds x1.1 for the fit on all four month groups
READ_BATCH_ROWS = 200000                              # rows per parquet batch when only some rows of a table are needed
STAKE_CAP_SECONDS = 14400                             # stakes are capped at 4 hours
STAKE_WEIGHT_FLOOR_SECONDS = 300.0                    # training weights count stakes below 5 minutes as 5 minutes
# XGBoost training of the cross-validation folds and of the final fit
QUANTILE_MATRIX_MAX_BIN = 256                         # XGBoost's own default number of bins
CROSS_VALIDATION_MAX_ROUNDS = 4000                    # only a ceiling: early stopping ends each fold far sooner (about 270-520 rounds in a past study)
EARLY_STOPPING_ROUNDS = 300                           # stop a fold after 300 rounds without a better validation score
# Optuna search space of tune: the low and high end of each suggest call
TUNE_MAX_DEPTH_LOW = 3                                # every range is picked by hand, wide, and holds the hand-set value of trial 0
TUNE_MAX_DEPTH_HIGH = 10                              # trial 0 uses depth 6 (Rome) and 5 (no link)
TUNE_MIN_CHILD_WEIGHT_LOW = 1                         # searched on a log scale
TUNE_MIN_CHILD_WEIGHT_HIGH = 200                      # trial 0 uses 5 (Rome) and 3 (no link)
TUNE_LEARNING_RATE_LOW = 0.01                         # searched on a log scale
TUNE_LEARNING_RATE_HIGH = 0.2                         # trial 0 uses 0.03
TUNE_SUBSAMPLE_LOW = 0.5                              # share of the rows each tree sees
TUNE_SUBSAMPLE_HIGH = 1.0                             # all rows
TUNE_COLSAMPLE_BYTREE_LOW = 0.2                       # share of the columns each tree sees
TUNE_COLSAMPLE_BYTREE_HIGH = 1.0                      # all columns
TUNE_REG_LAMBDA_LOW = 0.1                             # searched on a log scale
TUNE_REG_LAMBDA_HIGH = 100                            # trial 0 uses 5
TUNE_STAKE_WEIGHT_POWER_LOW = 0.0                     # 0: every row counts the same
TUNE_STAKE_WEIGHT_POWER_HIGH = 2.0                    # 2: weights grow with the squared stake, like the score being minimised
# Optuna TPE sampler of tune
OPTUNA_SAMPLER_SETTINGS = {
    'seed': 42,                         # make the search repeatable
    'n_startup_trials': 10,             # Optuna's own default
}
DEFAULT_TRIALS = 40


def read_rows_in_batches(table_path: Path, row_mask: np.ndarray, columns: list[str]) -> pd.DataFrame:
    selected_parts = []
    offset = 0
    for batch in pq.ParquetFile(table_path).iter_batches(batch_size=READ_BATCH_ROWS, columns=columns):
        batch_rows = row_mask[offset:offset + batch.num_rows]
        offset += batch.num_rows
        if batch_rows.any():
            selected_parts.append(batch.to_pandas()[batch_rows])
    return pd.concat(selected_parts, ignore_index=True)


def network_run_value(run_name: str, metadata: pd.DataFrame) -> np.ndarray:
    """
    This function reads a saved neural network run's taxi-time guesses
    """
    run = pd.read_parquet(RUN_PREDICTIONS / run_name / 'pred.parquet', columns=['MVT_ID_mvt', 'v_nn'])
    assert (run.MVT_ID_mvt.to_numpy() == metadata.MVT_ID_mvt.to_numpy()).all(), run_name
    return run.v_nn.to_numpy()


def reference_network_value(metadata: pd.DataFrame, full_fit_run: str) -> np.ndarray:
    """
    This function gets a taxi-time guess from the reference network for every flight
    """
    value = np.full(len(metadata), np.nan)
    for fold in WEEK_FOLDS:
        held_out_rows = held_out_week_fold_rows(metadata, fold)
        value[held_out_rows] = network_run_value(f'{REFERENCE_NETWORK_FOLD_RUN}_{fold}', metadata)[held_out_rows]
    ranking_rows = metadata.year.to_numpy() == 2026
    value[ranking_rows] = network_run_value(full_fit_run, metadata)[ranking_rows]
    return value


# the classifier inputs
def top_features(model: str) -> list[str]:
    """
    This function asks the earlier regime classifier of the group which facts helped it most and returns the top 200 names, best first
    """
    importance_model = xgb.Booster()
    importance_model.load_model(MODEL_WEIGHTS / IMPORTANCE_RUN / IMPORTANCE_MODEL_FILE[model])
    return list(features_by_total_gain(importance_model)[:TOP_FEATURE_COUNT])


def read_adsb_features(adsb_path: Path, adsb_columns: list[str], movement_ids: pd.Series) -> pd.DataFrame:
    """
    This function reads the wanted ADS-B columns for a list of flights
    """
    wanted_ids = set(movement_ids)
    kept_parts = []
    for batch in pq.ParquetFile(adsb_path).iter_batches(batch_size=READ_BATCH_ROWS, columns=['MVT_ID_mvt'] + adsb_columns):
        batch_frame = batch.to_pandas()
        kept_parts.append(batch_frame[batch_frame.MVT_ID_mvt.isin(wanted_ids)])
    adsb = pd.concat(kept_parts, ignore_index=True).drop_duplicates('MVT_ID_mvt')
    return adsb.set_index('MVT_ID_mvt').reindex(movement_ids).reset_index(drop=True).astype('float32')


def read_wanted_features(departures: pd.DataFrame, rows: np.ndarray, wanted_features: list[str]) -> pd.DataFrame:
    """
    This function collects the wanted facts for the chosen flights from the three row-aligned tables and the ADS-B table and glues them side by side
    """
    parts: list[pd.DataFrame] = []
    for table_name in ROW_ALIGNED_TABLES:
        table_path = FEATURE_MATRIX / f'{table_name}.parquet'
        table_column_names = set(pq.read_schema(table_path).names)
        table_columns = [column for column in wanted_features if column in table_column_names]
        if table_columns:
            parts.append(read_rows_in_batches(table_path, rows, table_columns))
    adsb_path = FEATURE_MATRIX / f'{ADSB_TABLE}.parquet'
    adsb_column_names = set(pq.read_schema(adsb_path).names)
    adsb_columns = [column for column in wanted_features if column in adsb_column_names]
    if adsb_columns:
        parts.append(read_adsb_features(adsb_path, adsb_columns, departures.MVT_ID_mvt[rows]))
    return pd.concat(parts, axis=1)


def add_derived_features(features: pd.DataFrame, departures: pd.DataFrame, rows: np.ndarray, value_baseline: np.ndarray, wanted_features: list[str]) -> None:
    """
    This function adds the wanted facts that no table stores and that are worked out on the spot
    """
    derived_features = {'operator3': pd.Categorical(departures.operator_code.to_numpy()[rows]),
                        'msched': departures.seconds_after_schedule.to_numpy()[rows],
                        'sched_daydiff': departures.day_difference.to_numpy()[rows],
                        'vbase': value_baseline[rows]}
    for feature_name, feature_values in derived_features.items():
        if feature_name in wanted_features:
            features[feature_name] = feature_values


def add_push_delay_features(features: pd.DataFrame, departures: pd.DataFrame, rows: np.ndarray, push_delay_network: str) -> None:
    """
    This function adds two extra facts for the special Rome classifier
    """
    network_value = reference_network_value(departures, push_delay_network)[rows]
    features['oof_v'] = network_value
    features['oof_push'] = departures.seconds_after_schedule.to_numpy()[rows] - network_value


def use_classifier_column_types(features: pd.DataFrame) -> None:
    """
    This function tidies the column types the way XGBoost wants them
    """
    for column in features.columns:
        if features[column].dtype == object or str(features[column].dtype) == 'category':
            features[column] = features[column].astype(str).astype('category')
        elif features[column].dtype != 'float32':
            features[column] = features[column].astype('float32')


def loss_information(departures: pd.DataFrame, rows: np.ndarray, value_baseline: np.ndarray, model: str) -> pd.DataFrame:
    """
    This function builds a small scoring sheet for each chosen flight: whether it really was a schedule copy, its month, airport group, stake (at most 4 hours) and whether it is scored
    """
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    seconds_after_schedule = departures.seconds_after_schedule.to_numpy()
    stake = np.minimum(np.abs(np.nan_to_num(seconds_after_schedule[rows] - value_baseline[rows])), STAKE_CAP_SECONDS)
    row_information = pd.DataFrame({'MVT_ID_mvt': departures.MVT_ID_mvt[rows].to_numpy(),
                                    'is_schedule_copy': departures.is_schedule_copy.to_numpy()[rows].astype(float),
                                    'is_ranking': ~np.isfinite(labels[rows]),
                                    'month': departures.month.to_numpy()[rows],
                                    'is_rome': departures.is_rome.to_numpy()[rows],
                                    'no_nm_link': departures.no_nm_link.to_numpy()[rows],
                                    'day_difference': departures.day_difference.to_numpy()[rows],
                                    'stake': stake})
    if model == 'rome_linked':
        row_information['scored'] = row_information.is_rome & ~row_information.no_nm_link
    else:       # every no-link row except the Rome next-day group (take-off at least one calendar day after the scheduled date)
        row_information['scored'] = ~(row_information.is_rome & (row_information.day_difference >= 1))
    return row_information


def build_classifier_inputs(model: str, include_ranking_rows: bool = False, push_delay_network: str = '') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function builds everything a classifier needs for its group
    """
    departures = departure_frame()
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    no_link = departures.no_nm_link.to_numpy()
    is_rome = departures.is_rome.to_numpy()
    labelled_2025 = np.isfinite(labels) & (departures.year.to_numpy() == 2025)
    is_unlabelled = ~np.isfinite(labels)
    rows = (labelled_2025 | (include_ranking_rows & is_unlabelled)) & (is_rome if model == 'rome_linked' else no_link)
    # the value baseline's airport medians (rows without an NM link) come from labelled 2025 regime-N rows with an NM link
    is_normal = ~departures.is_schedule_copy.to_numpy() & ~departures.is_previous_date.to_numpy()
    value_baseline = schedule_independent_baseline(departures, labelled_2025 & is_normal & ~no_link)
    wanted_features = top_features(model)
    features = read_wanted_features(departures, rows, wanted_features)
    add_derived_features(features, departures, rows, value_baseline, wanted_features)
    if push_delay_network:
        add_push_delay_features(features, departures, rows, push_delay_network)
    use_classifier_column_types(features)
    return features, loss_information(departures, rows, value_baseline, model)


# cross-validation, study and final fit
def stake_weights(stake: np.ndarray, stake_weight_power: float) -> np.ndarray:
    """
    This function turns each flight's stake into how much it counts in training
    """
    return (np.clip(stake, STAKE_WEIGHT_FLOOR_SECONDS, STAKE_CAP_SECONDS) / STAKE_WEIGHT_FLOOR_SECONDS) ** stake_weight_power


def fit_fold(parameters: dict[str, int | float], features: pd.DataFrame, target: np.ndarray, weights: np.ndarray, validation_rows: np.ndarray) -> tuple[np.ndarray, int]:
    """
    This function runs one practice test
    """
    fit_rows = ~validation_rows
    fit_matrix = xgb.QuantileDMatrix(features[fit_rows], label=target[fit_rows], weight=weights[fit_rows], enable_categorical=True, max_bin=QUANTILE_MATRIX_MAX_BIN)
    validation_matrix = xgb.QuantileDMatrix(features[validation_rows], label=target[validation_rows], weight=weights[validation_rows], enable_categorical=True, ref=fit_matrix)
    booster = xgb.train(dict(CLASSIFIER_BASE_PARAMETERS, **parameters), fit_matrix, CROSS_VALIDATION_MAX_ROUNDS, evals=[(validation_matrix, 'va')], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False)
    fold_rounds = booster.best_iteration + 1
    probability = booster.predict(validation_matrix, iteration_range=(0, fold_rounds))
    del fit_matrix, validation_matrix, booster
    gc.collect()
    return probability, fold_rounds


def month_group_cross_validation(parameters: dict[str, int | float], features: pd.DataFrame, row_information: pd.DataFrame) -> tuple[float, np.ndarray, list[int]]:
    """
    This function runs the four practice tests for one set of settings - each month group is hidden once and gets its chances from a model trained on the other three
    It returns the cost (the average of (stake times (chance minus truth)) squared over the scored flights)
    """
    parameters = dict(parameters)
    stake_weight_power = parameters.pop('alpha_stake')
    weights = stake_weights(row_information.stake.to_numpy(), stake_weight_power)
    weights = weights / weights.mean()
    out_of_fold = np.full(len(row_information), np.nan)
    best_rounds: list[int] = []
    target = row_information.is_schedule_copy.to_numpy()
    for held_out_months in MONTH_GROUP_FOLDS:
        validation_rows = row_information.month.isin(held_out_months).to_numpy()
        fold_probability, fold_rounds = fit_fold(parameters, features, target, weights, validation_rows)
        best_rounds.append(fold_rounds)
        out_of_fold[validation_rows] = fold_probability
    scored = row_information.scored.to_numpy()
    stake = row_information.stake.to_numpy()
    loss = float(np.mean((stake[scored] * (out_of_fold[scored] - target[scored])) ** 2))
    return loss, out_of_fold, best_rounds


def write_study(study: optuna.Study, model: str) -> None:
    best_trial = study.best_trial
    hand_set_loss = study.trials[0].value
    relative_change = 100 * (best_trial.value / hand_set_loss - 1)
    print('baseline loss %.0f  best loss %.0f (%.1f%%)  best params %s'
          % (hand_set_loss, best_trial.value, relative_change, best_trial.params))
    with open(TUNED_CLASSIFIERS / f'study_{model}.json', 'w') as study_file:
        trials = [{'n': trial.number, 'loss': trial.value, 'params': trial.params} for trial in study.trials]
        json.dump({'baseline_loss': hand_set_loss, 
                   'best_loss': best_trial.value, 
                   'best_params': best_trial.params,
                   'best_iters': best_trial.user_attrs['iters'], 
                   'baseline_iters': study.trials[0].user_attrs['iters'],
                   'trials': trials}, study_file, indent=1)


def tune(arguments: argparse.Namespace) -> None:
    """
    This function is the tunes the models using Optuna
    """
    TUNED_CLASSIFIERS.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    features, row_information = build_classifier_inputs(arguments.model)
    print('rows', len(features), 'features', features.shape[1], 'P rate %.3f' % row_information.is_schedule_copy.mean(), 'scored', int(row_information.scored.sum()), 'build %.0fs' % (time.time() - start_time), flush=True)

    def objective(trial: optuna.Trial) -> float:
        """
        This function scores one set of settings that Optuna suggests by running the four practice tests and returning the cost (smaller is better)
        """
        parameters = dict(max_depth=trial.suggest_int('max_depth', TUNE_MAX_DEPTH_LOW, TUNE_MAX_DEPTH_HIGH),
                          min_child_weight=trial.suggest_float('min_child_weight', TUNE_MIN_CHILD_WEIGHT_LOW, TUNE_MIN_CHILD_WEIGHT_HIGH, log=True),
                          learning_rate=trial.suggest_float('learning_rate', TUNE_LEARNING_RATE_LOW, TUNE_LEARNING_RATE_HIGH, log=True),
                          subsample=trial.suggest_float('subsample', TUNE_SUBSAMPLE_LOW, TUNE_SUBSAMPLE_HIGH),
                          colsample_bytree=trial.suggest_float('colsample_bytree', TUNE_COLSAMPLE_BYTREE_LOW, TUNE_COLSAMPLE_BYTREE_HIGH),
                          reg_lambda=trial.suggest_float('reg_lambda', TUNE_REG_LAMBDA_LOW, TUNE_REG_LAMBDA_HIGH, log=True),
                          alpha_stake=trial.suggest_float('alpha_stake', TUNE_STAKE_WEIGHT_POWER_LOW, TUNE_STAKE_WEIGHT_POWER_HIGH))
        loss, _, best_rounds = month_group_cross_validation(parameters, features, row_information)
        trial.set_user_attr('iters', best_rounds)
        print('trial %d loss %.0f iters %s %.0fs' % (trial.number, loss, best_rounds, time.time() - start_time), flush=True)
        return loss

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(**OPTUNA_SAMPLER_SETTINGS))
    study.enqueue_trial(HAND_SET_PARAMETERS[arguments.model])
    study.optimize(objective, n_trials=arguments.trials)
    write_study(study, arguments.model)


def fit_final_classifier(model: str, push_delay_network: str, suffix: str) -> None:
    best_parameters = dict(json.loads((TUNED_CLASSIFIERS / f'study_{model}.json').read_text())['best_params'])
    features, row_information = build_classifier_inputs(model, include_ranking_rows=True, push_delay_network=push_delay_network)
    features = features.drop(columns=[column for column in CALENDAR_COLUMNS if column in features])
    labelled = ~row_information['is_ranking'].to_numpy()
    loss, _, best_rounds = month_group_cross_validation(best_parameters, features[labelled].reset_index(drop=True), row_information[labelled].reset_index(drop=True))
    print(f'{model}: cross-validated loss {loss:.0f}, iterations {best_rounds}', flush=True)
    stake_weight_power = best_parameters.pop('alpha_stake')
    boosting_rounds = int(round(np.mean(best_rounds) * FULL_FIT_ITERATION_SCALE))
    weights = stake_weights(row_information.stake.to_numpy(), stake_weight_power)
    weights = weights / weights[labelled].mean()
    training_matrix = xgb.QuantileDMatrix(features[labelled], label=row_information.is_schedule_copy.to_numpy()[labelled], weight=weights[labelled], enable_categorical=True, max_bin=QUANTILE_MATRIX_MAX_BIN)
    booster = xgb.train(dict(CLASSIFIER_BASE_PARAMETERS, **best_parameters), training_matrix, boosting_rounds)
    ranking = ~labelled
    classifier_weights = MODEL_WEIGHTS / 'tuned_classifiers'
    classifier_weights.mkdir(parents=True, exist_ok=True)
    booster.save_model(classifier_weights / f'{model}_classifier{suffix}.ubj')
    probability = booster.predict(xgb.DMatrix(features[ranking], enable_categorical=True))
    pd.DataFrame({'MVT_ID_mvt': row_information.MVT_ID_mvt[ranking].to_numpy(), 'p': probability}).to_parquet(TUNED_CLASSIFIERS / f'{model}_classifier{suffix}.parquet', index=False)
    print(model, 'final rounds', boosting_rounds, 'ranking rows', int(ranking.sum()), 'mean p %.3f' % probability.mean(), flush=True)


def final(arguments: argparse.Namespace) -> None:
    models = arguments.models.split(',')
    unknown_models = [model for model in models if model not in HAND_SET_PARAMETERS]
    if unknown_models:
        raise SystemExit(f'unknown --models {unknown_models}: choose from {",".join(HAND_SET_PARAMETERS)}')
    for model in models:
        fit_final_classifier(model, arguments.push_delay_network, arguments.suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune the schedule-copy classifiers.')
    subcommands = parser.add_subparsers(required=True)
    tune_parser = subcommands.add_parser('tune')
    tune_parser.set_defaults(func=tune)
    tune_parser.add_argument('--model', required=True, choices=list(HAND_SET_PARAMETERS))
    tune_parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    final_parser = subcommands.add_parser('final')
    final_parser.set_defaults(func=final)
    final_parser.add_argument('--models', default='rome_linked,no_link')
    final_parser.add_argument('--suffix', default='')
    final_parser.add_argument('--push-delay-network', default='')
    arguments = parser.parse_args()
    arguments.func(arguments)


if __name__ == '__main__':
    main()
