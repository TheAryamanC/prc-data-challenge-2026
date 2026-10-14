"""
This file works out what the whole pipeline would have guessed for each 2025 departure if it had never seen that departure's answer - this is called out-of-fold prediction
The last corrections need these guesses to be fitted fairly

This is how it works:
-global: the 'was the time copied from the schedule?' classifier is retrained once per week fold
-tuned: the two tuned classifiers are retrained by groups of months
-value: the networks' corrected taxi time (the second stage of network_ensembles.py) is trained on the other three week folds
-prior: the prior (the project's older, simpler model) is retrained with each pair of months hidden
-components: the parts are combined exactly as the final build does:
    1. the P / S / N chances (time copied from the schedule, recorded with the previous day's date, or recorded normally)
    2. the LOBT window rules (the flight plan's last known off-block time limits the right answer) and the recovered-LOBT rules
    3. the blend with the prior and the late-day cap
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

import taxi.models.tuned_classifiers as tuned
import taxi.prediction.build_submission as build
from taxi.main import (FEATURE_MATRIX, MODEL_WEIGHTS, RUN_PREDICTIONS, WEEK_FOLDS, competition_files, departure_frame, held_out_week_fold_rows, lobt_label_window)
from taxi.models import network_ensembles as ensembles, regime_mixture as regime
from taxi.models.regime_mixture import (VALIDATION_ITERATIONS, add_regime_features, features_by_total_gain, schedule_copy_classifier_parameters, schedule_independent_baseline)

OUT_OF_FOLD = RUN_PREDICTIONS / 'out_of_fold_2025'
GLOBAL_TOP_FEATURES = 150                       # inputs of the global classifier: the top features of regime_seed42's pclass
MONTH_PAIRS = ((1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12))     # the months held out together by each out-of-fold prior
VALUE_MIX = 'pass3_clip15000_seed_pairs'        # the network mix of the second-stage value (taxi.models.network_ensembles)
VALUE_TREE_INPUTS = ['two_part_residual_pass1_airport_state:v_tpr', 'catboost_value_pass1_airport_state:v_cb', 'two_part_residual_pass3_airport_state_extended:v_tpr']
VALUE_SECOND_STAGE = {'max_depth': 8, 'min_child_weight': 20}      # second-stage parameters of the tracked-rows variant
VALUE_ROUNDS = 165                              # the build's SECOND_STAGE_ROUNDS: the mean of the early-stopping rounds (186 and 144) of the first two 2025 week-fold checks
# the global classifier, rebuilt as the regime run regime_seed42 trained it
GLOBAL_CLASSIFIER_SEED = 42                     # the seed of regime_seed42, whose global classifier (pclass) is rebuilt here
STAKE_FLOOR_SECONDS = 300                       # s: smaller stakes count as this; the stake weights of the regime runs' classifiers, clip(|msched - vbase|, 300, 14400) / 300
STAKE_CAP_SECONDS = 14400                       # s: larger stakes count as this; the same stake weights
STAKE_UNIT_SECONDS = 300.0                      # s: stakes are counted in units of this (a floored stake weighs 1); the same stake weights
# XGBoost settings switched in for the out-of-fold refits of the global and the tuned classifiers
CPU_TRAINING_SETTINGS = {
    'device': 'cpu',
    'nthread': 12,
}
# the build's rules (taxi.prediction.build_submission), reproduced as the build applies them
LOBT_WINDOW_TOLERANCE_SECONDS = 10              # s: a schedule copy further outside the LOBT window is impossible
LABEL_FLOOR_SECONDS = -12                       # s: the lowest 2025 label; a recovered window starting below it leaves no room for a label under it (case B), one starting at or above it does (case A)
ROME_CASE_B_MARGIN_SECONDS = 1224               # s: the median distance of the label above the window end over the 30 Rome case-B departures of 2025
ISTANBUL_CASE_B_MARGIN_SECONDS = 469            # s: the same median over the 19 Istanbul case-B departures of 2025


def out_of_fold_file(name: str) -> Path:
    """
    This function gives the path of an out-of-fold file
    """
    return OUT_OF_FOLD / f'{name}.parquet'


# global classifier
def as_float32_except_categories(table: pd.DataFrame) -> pd.DataFrame:
    """
    This function turns every column of a table into float32, leaving category columns as they are
    """
    return table.apply(lambda column: column if str(column.dtype) == 'category' else column.astype('float32'))


def global_classifier_features(ranked_features: list[str], departures: pd.DataFrame) -> pd.DataFrame:
    """
    This function collects the global classifier's clues for every departure
    """
    base_path = FEATURE_MATRIX / 'base_features.parquet'
    base_columns = set(pq.read_schema(base_path).names)
    base = pd.read_parquet(base_path, columns=[column for column in ranked_features if column in base_columns])
    parts = [as_float32_except_categories(base)]
    adsb_path = FEATURE_MATRIX / 'adsb_pass1.parquet'
    adsb_columns = set(pq.read_schema(adsb_path).names)
    adsb = pd.read_parquet(adsb_path, columns=['MVT_ID_mvt'] + [column for column in ranked_features if column in adsb_columns])
    adsb = adsb.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
    adsb = adsb.reindex(departures.MVT_ID_mvt).reset_index(drop=True)
    parts.append(as_float32_except_categories(adsb))
    return pd.concat(parts, axis=1)


def global_classifier() -> None:
    """
    This function rebuilds the global 'was the time copied from the schedule?' classifier out-of-fold: for each week fold, XGBoost learns from the rest of the labeled 2025 rows and guesses the hidden week
    """
    importance = xgb.Booster()
    importance.load_model(MODEL_WEIGHTS / 'regime_seed42' / 'pclass.ubj')
    ranked = list(features_by_total_gain(importance))[:GLOBAL_TOP_FEATURES]
    del importance
    departures = departure_frame()
    features = global_classifier_features(ranked, departures)
    labels = departures.TAXITIME_SEC_mvt.to_numpy()
    is_schedule_copy = departures.is_schedule_copy.to_numpy()
    is_normal = ~is_schedule_copy & ~departures.is_previous_date.to_numpy()
    training = np.isfinite(labels) & (departures.year.to_numpy() == 2025)
    value_baseline = schedule_independent_baseline(departures, training & is_normal & ~departures.no_nm_link.to_numpy())
    add_regime_features(features, departures, value_baseline)
    # stake of a row: what a wrong regime call costs, |seconds after schedule - value baseline| clipped to 300 .. 14400 s, in
    # units of 300 s (a missing seconds-after-schedule counts as 0)
    seconds_after_schedule = np.nan_to_num(departures.seconds_after_schedule.to_numpy(), nan=0.0)
    stake = np.clip(np.abs(seconds_after_schedule - value_baseline), STAKE_FLOOR_SECONDS, STAKE_CAP_SECONDS) / STAKE_UNIT_SECONDS
    parameters = dict(schedule_copy_classifier_parameters(GLOBAL_CLASSIFIER_SEED)['pclass'], **CPU_TRAINING_SETTINGS)
    rounds = VALIDATION_ITERATIONS['stake_classifiers']['pclass']
    probability = np.full(len(departures), np.nan)
    for fold in WEEK_FOLDS:
        held_out = held_out_week_fold_rows(departures, fold) & training
        fit = training & ~held_out
        matrix = xgb.QuantileDMatrix(features[fit], label=is_schedule_copy[fit].astype(float), weight=stake[fit] / stake[fit].mean(), enable_categorical=True)
        booster = xgb.train(parameters, matrix, rounds)
        del matrix
        probability[held_out] = booster.predict(xgb.DMatrix(features[held_out], enable_categorical=True))
        print('global classifier', fold, 'done', flush=True)
    OUT_OF_FOLD.mkdir(parents=True, exist_ok=True)
    global_probabilities = pd.DataFrame({'MVT_ID_mvt': departures.MVT_ID_mvt.to_numpy()[training], 'p': probability[training]})
    global_probabilities.to_parquet(out_of_fold_file('global'), index=False)


# tuned classifiers
def tuned_classifiers() -> None:
    """
    This function rebuilds the two tuned classifiers (Rome flights with a flight-plan link, and flights without a link) with their best Optuna settings - each group of months is guessed by a model trained on the other groups
    """
    tuned.CLASSIFIER_BASE_PARAMETERS.update(**CPU_TRAINING_SETTINGS)
    for model in ('rome_linked', 'no_link'):
        study = json.loads((tuned.TUNED_CLASSIFIERS / f'study_{model}.json').read_text())
        parameters = dict(study['best_params'])
        features, information = tuned.build_classifier_inputs(model)
        features = features.drop(columns=[column for column in tuned.CALENDAR_COLUMNS if column in features])
        loss, probability, rounds = tuned.month_group_cross_validation(parameters, features, information)
        print(f'tuned {model}: month-group loss {loss:.0f}, rounds {rounds}', flush=True)
        OUT_OF_FOLD.mkdir(parents=True, exist_ok=True)
        information.assign(p=probability)[['MVT_ID_mvt', 'p']].to_parquet(out_of_fold_file(f'tuned_{model}'), index=False)


# second-stage value
def held_out_fold_value(fold: str, frame: pd.DataFrame, context: pd.DataFrame, training: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]], labels: np.ndarray) -> pd.DataFrame:
    """
    This function gives a corrected taxi time for every labeled row of one week fold: the second stage is trained on the other three folds and applied to this fold's guesses
    """
    mix = ensembles.MIXES[VALUE_MIX]
    other_folds = [other for other in WEEK_FOLDS if other != fold]
    inputs = pd.concat([training[other][0] for other in other_folds], ignore_index=True)
    model = ensembles.train_second_stage(inputs, np.concatenate([training[other][1] for other in other_folds]), np.concatenate([training[other][2] for other in other_folds]), VALUE_ROUNDS)
    member_values = {member.name: ensembles.member_fold_values(member, fold) for member in mix.members}
    rows = held_out_week_fold_rows(frame, fold) & np.isfinite(labels)
    reference_values = member_values[ensembles.reference_member_name(mix)]
    fold_inputs, fold_ensemble = ensembles.second_stage_inputs(mix, member_values, reference_values, context, rows)
    correction = model.predict(xgb.DMatrix(fold_inputs))
    deviation_limits, ensemble_limit = ensembles.training_range_limits(inputs)
    in_range = ensembles.within_training_range(fold_inputs, deviation_limits, ensemble_limit)
    return pd.DataFrame({'MVT_ID_mvt': frame.MVT_ID_mvt.to_numpy()[rows], 'v': fold_ensemble[rows] + np.where(in_range, correction, 0.0)})


def second_stage_value() -> None:
    """
    This function rebuilds the networks' corrected taxi time for all of 2025
    """
    mix = ensembles.MIXES[VALUE_MIX]
    ensembles.SECOND_STAGE_PARAMS.update(VALUE_SECOND_STAGE)
    frame = departure_frame()
    context = ensembles.row_context(frame, 'adsb_pass3' if mix.members_saw_pass3_tracks else 'adsb_pass2')
    context = pd.concat([context, ensembles.out_of_fold_inputs(VALUE_TREE_INPUTS, frame)], axis=1)
    training = {fold: ensembles.fold_training_rows(mix, frame, context, fold) for fold in WEEK_FOLDS}
    labels = frame.TAXITIME_SEC_mvt.to_numpy()
    parts = []
    for fold in WEEK_FOLDS:
        parts.append(held_out_fold_value(fold, frame, context, training, labels))
        print('second-stage value', fold, 'done', flush=True)
    OUT_OF_FOLD.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out_of_fold_file('value'), index=False)


# prior
def prior() -> None:
    """
    This function rebuilds the prior out-of-fold: for each of the six month pairs it runs prior_model.py as a separate program that trains without those two months and guesses them
    """
    OUT_OF_FOLD.mkdir(parents=True, exist_ok=True)
    for first_month, second_month in MONTH_PAIRS:
        target = out_of_fold_file(f'prior_{first_month}_{second_month}')
        if not target.exists():
            subprocess.run([sys.executable, '-u', '-m', 'taxi.models.prior_model', '--held-out-months', f'{first_month},{second_month}', '--out', str(target)], check=True)


# components
def out_of_fold_column(name: str, column: str, movement_ids: np.ndarray) -> np.ndarray:
    """
    This function reads one column of an out-of-fold file, lined up with the given ids (empty for ids the file does not have)
    """
    by_movement = pd.read_parquet(out_of_fold_file(name)).set_index('MVT_ID_mvt')
    return getattr(by_movement, column).reindex(movement_ids).to_numpy()


def blended_schedule_copy_probability(p_global: np.ndarray, movement_ids: np.ndarray, no_link: np.ndarray, next_day_group: np.ndarray) -> np.ndarray:
    """
    This function mixes the classifiers' schedule-copy chances the way the final build does
    """
    probability = p_global.copy()
    for model in ('rome_linked', 'no_link'):              # the build's order: no_link last
        tuned = out_of_fold_column(f'tuned_{model}', 'p', movement_ids)
        rows = np.isfinite(tuned) & (~no_link if model == 'rome_linked' else ~next_day_group)
        share = build.TUNED_CLASSIFIER_SHARE[model]
        probability[rows] = 0.5 * (share * tuned[rows] + (1 - share) * p_global[rows] + p_global[rows])
    return probability


def fold_numbers(frame: pd.DataFrame) -> np.ndarray:
    """
    This function gives every row its week fold number, 0 to 3 (or -1 when it is in no fold)
    """
    fold = np.full(len(frame), -1)
    for number, name in enumerate(WEEK_FOLDS):
        fold[held_out_week_fold_rows(frame, name)] = number
    return fold


def rome_next_day_probabilities(frame: pd.DataFrame, next_day_group: np.ndarray, fold: np.ndarray, schedule_copy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function handles the Rome next-day group: Rome flights without a flight-plan link that take off on a later date than scheduled, the only rows that can be recorded with the previous day's date
    For each week fold it fits a small P / S / N logistic regression on the delay in hours from the other folds' group rows, and for long delays shrinks the P-versus-S split toward what the same airline and flight number usually did
    """
    schedule_copy = schedule_copy.copy()
    previous_date = np.zeros(len(frame))
    delay_hours = frame.seconds_after_schedule.to_numpy() / 3600
    class_labels = regime.regime_class_labels(frame)
    flights, operators = regime.flight_and_operator_keys(frame)
    group_index = np.flatnonzero(next_day_group)
    for number in range(len(WEEK_FOLDS)):
        fit_rows = next_day_group & (fold != number)
        _, probability = regime.fit_next_day_logistic(delay_hours, class_labels, fit_rows, next_day_group)
        group_keys = (operators[next_day_group], flights[next_day_group])
        probability = regime.next_day_group_prior(probability, class_labels[next_day_group], fit_rows[next_day_group], delay_hours[next_day_group], group_keys)
        selected = fold[next_day_group] == number
        schedule_copy[group_index[selected]] = probability[selected, 0]
        previous_date[group_index[selected]] = probability[selected, 1]
    return schedule_copy, previous_date


def window_regime_probabilities(schedule_copy: np.ndarray, previous_date: np.ndarray, msched: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function applies the LOBT window rules: a schedule copy is impossible when the delay is more than 10 seconds outside the window or below -12 seconds, and a previous-day record is impossible on any row with a window
    """
    window = np.isfinite(lower)
    impossible = ((window & np.isfinite(msched) & ((msched < lower - LOBT_WINDOW_TOLERANCE_SECONDS) | (msched > upper + LOBT_WINDOW_TOLERANCE_SECONDS))) | (np.isfinite(msched) & (msched < LABEL_FLOOR_SECONDS)))
    return np.where(impossible, 0.0, schedule_copy), np.where(window, 0.0, previous_date)


def mixture_value(schedule_copy: np.ndarray, previous_date: np.ndarray, msched: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    This function mixes the three possible answers by their chances: the delay if the time was copied from the schedule, a day plus the taxi-time guess if it has the previous day's date, otherwise the taxi-time guess itself
    """
    mixture = schedule_copy * msched + previous_date * (86400 + value) + (1 - schedule_copy - previous_date) * value
    return np.maximum(np.where(np.isfinite(mixture), mixture, value), LABEL_FLOOR_SECONDS)


def value_in_lobt_window(value: np.ndarray, lower: np.ndarray, upper: np.ndarray, no_link: np.ndarray) -> np.ndarray:
    """
    This function moves the taxi-time guess into the LOBT window the way the build does
    """
    window = np.isfinite(lower)
    reflect = window & ~no_link                     # NM-linked lanes are built with reflected=True
    reflected_up = np.minimum(lower + build.REFLECTION_OFFSET + build.REFLECTION_SLOPE * (lower - value), upper)
    reflected_down = np.maximum(upper - build.REFLECTION_SLOPE * (value - upper), lower)
    value = np.where(reflect & (value < lower), reflected_up, np.where(reflect & (value > upper), reflected_down, value))
    return np.where(window, np.clip(value, lower, upper), value)


def recovered_lobt_cases(movement_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Some departures have no flight-plan link but their flight's arrival has one, which gives back their LOBT (a recovered LOBT) - this function finds them in the 2025 training files and sorts them into cases
    """
    training_files = [path for path in competition_files() if 'training_' in path.name]
    recovered = build.recovered_lobt_departures(training_files).set_index('MVT_ID_mvt')
    recovered = recovered[recovered.index.isin(movement_ids)]
    position = pd.Series(np.arange(len(movement_ids)), index=movement_ids)
    case = np.full(len(movement_ids), '', dtype=object)
    case_floor = np.full(len(movement_ids), np.nan)
    case_lo = np.full(len(movement_ids), np.nan)
    case_lo[position[recovered.index].to_numpy()] = recovered.lo.to_numpy()
    has_flight_number = recovered.FLIGHT_mvt.notna() & (recovered.FLIGHT_mvt.astype(str).str.strip() != '')
    case_b = recovered[(recovered.lo < LABEL_FLOOR_SECONDS) & (recovered.hi > 0) & recovered.ADEP_mvt.isin(['LIRF', 'LTFM']) & has_flight_number]
    b_position = position[case_b.index].to_numpy()
    b_msched = (case_b.MVT_TIME_UTC_mvt - case_b.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    case[b_position] = 'B'
    case_floor[b_position] = np.where(case_b.ADEP_mvt == 'LIRF', np.maximum(case_b.hi + ROME_CASE_B_MARGIN_SECONDS, b_msched), case_b.hi + ISTANBUL_CASE_B_MARGIN_SECONDS)
    case[position[recovered.index[recovered.lo >= LABEL_FLOOR_SECONDS]].to_numpy()] = 'A'
    return case, case_floor, case_lo


def components() -> None:
    """
    This function puts the out-of-fold parts together for every 2025 departure, and computes the pipeline's guess with combined_prediction
    """
    frame = departure_frame()
    frame = frame[frame.year == 2025].reset_index(drop=True)
    ids = frame.MVT_ID_mvt.to_numpy()
    value = out_of_fold_column('value', 'v', ids)
    p_global = out_of_fold_column('global', 'p', ids)
    group = regime.rome_next_day_no_link_group(frame)
    no_link = frame.no_nm_link.to_numpy()
    pP = blended_schedule_copy_probability(p_global, ids, no_link, group)
    fold = fold_numbers(frame)
    pP, pS = rome_next_day_probabilities(frame, group, fold, pP)
    msched = frame.seconds_after_schedule.to_numpy()
    lower, upper = lobt_label_window(ids)
    pP, pS = window_regime_probabilities(pP, pS, msched, lower, upper)
    value = value_in_lobt_window(value, lower, upper, no_link)
    mixture = mixture_value(pP, pS, msched, value)
    case, case_floor, case_lo = recovered_lobt_cases(ids)
    prior_values = pd.concat([pd.read_parquet(out_of_fold_file(f'prior_{a}_{b}')) for a, b in MONTH_PAIRS]).drop_duplicates('MVT_ID_mvt')
    regime_letter = np.select([frame.is_schedule_copy.to_numpy(), frame.is_previous_date.to_numpy()], ['P', 'S'], 'N')
    table = pd.DataFrame({'MVT_ID_mvt': ids, 
                          'fold': fold, 
                          'month': frame.month.to_numpy(), 
                          'airport': frame.ADEP_mvt.to_numpy(),
                          'no_link': no_link, 
                          'is_rome': frame.is_rome.to_numpy(), 
                          'next_day': group,
                          'label': frame.TAXITIME_SEC_mvt.to_numpy(),
                          'regime': regime_letter, 
                          'msched': msched, 
                          'v': value, 
                          'pP': pP, 
                          'pS': pS, 
                          'mixture': mixture, 
                          'lower': lower,
                          'upper': upper, 
                          'rec_case': case, 
                          'rec_floor': case_floor, 
                          'rec_lo': case_lo,
                          'prior': prior_values.set_index('MVT_ID_mvt').prior_oof.reindex(ids).to_numpy()})
    table['pipeline'] = combined_prediction(table)
    error = table.pipeline - table.label
    print(f'2025 out-of-fold pipeline RMSE {np.sqrt(np.mean(error ** 2)):.3f} over {len(table)} rows', flush=True)
    table.to_parquet(out_of_fold_file('components'), index=False)


def combined_prediction(table: pd.DataFrame, schedule_copy_probability: np.ndarray | None = None, previous_date_probability: np.ndarray | None = None, next_day_model_weight: float | None = None) -> np.ndarray:
    """
    This function follows the build's recipe to turn the parts into one final guess per row: the window rules and the mixture, the blend with the prior predictions, then clipping into the window
    """
    pP = table.pP.to_numpy() if schedule_copy_probability is None else schedule_copy_probability
    pS = table.pS.to_numpy() if previous_date_probability is None else previous_date_probability
    value = table.v.to_numpy()
    msched = table.msched.to_numpy()
    lower = table.lower.to_numpy()
    upper = table.upper.to_numpy()
    window = np.isfinite(lower)
    next_day = table.next_day.to_numpy()
    no_link = table.no_link.to_numpy()
    pP, pS = window_regime_probabilities(pP, pS, msched, lower, upper)
    mixture = mixture_value(pP, pS, msched, value)
    weights = build.MODEL_WEIGHT_AGAINST_PRIOR
    next_day_weight = weights[2] if next_day_model_weight is None else next_day_model_weight
    model_weight = np.where(next_day, next_day_weight, np.where(no_link, weights[1], weights[0]))
    prediction = np.maximum(model_weight * mixture + (1 - model_weight) * np.nan_to_num(table.prior.to_numpy(), nan=0.0), LABEL_FLOOR_SECONDS)
    prediction = np.maximum(np.where(window, np.clip(prediction, lower, upper), prediction), LABEL_FLOOR_SECONDS)
    case = table.rec_case.to_numpy()
    case_floor = table.rec_floor.to_numpy()
    case_lo = table.rec_lo.to_numpy()
    prediction = np.where((case == 'B') & (prediction < case_floor), case_floor, prediction)
    prediction = np.where((case == 'A') & (prediction > case_lo), np.maximum(np.minimum(value, case_lo), LABEL_FLOOR_SECONDS), prediction)
    late_day_cap = np.where(next_day, np.maximum(msched, 86400 + value), np.inf)
    return np.minimum(prediction, late_day_cap)


def load_components() -> pd.DataFrame:
    return pd.read_parquet(out_of_fold_file('components'))


# the command-line steps (python -m taxi.models.out_of_fold <step>); components reads the outputs of the four others
STEPS = {'global': global_classifier, 'tuned': tuned_classifiers, 'value': second_stage_value, 'prior': prior, 'components': components}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build 2025 out-of-fold predictions.')
    parser.add_argument('step', choices=list(STEPS))
    STEPS[parser.parse_args().step]()


if __name__ == '__main__':
    main()
