"""
This is the very last step of the project - it collects the saved guesses of all the trained models, mixes them by a fixed recipe and writes the one file that is handed in to the challenge
The tricky part is that the true times were written down in three ways: P means the timetable time was copied instead of the real leaving time, S means the real time was written with yesterday's date (so the answer is a whole day, 86,400 seconds, too big) and N is a normal record
So for every flight the file mixes the three possible answers by how likely each one is
Flights are split by their NM link (whether the organizers found the flight in Europe's Network Manager flight-plan data, which gives its planned leaving time, the LOBT) and by their ADS-B track (the planes' own position messages, matched in three passes)

This is how it works:
-build the network ensembles and second stages that the lanes read
-optionally, on airport-days when the ADS-B receivers were mostly down, mix networks trained without ADS-B into the NM-linked flights without a track
-for each group of flights, build a 'lane' (one complete set of guesses from one network run and one set of model weights):
    1. average the model families into a plain taxi-time guess v (networks90 = 90 % neural network and 10 % tree models, networks100 = the network alone) and keep it inside the flight's LOBT window
    2. mix the P / S / N answers by their chances, for two seeds (42 and 7), and average the two
    3. blend that with the prior (an older, separate model) and keep it inside the window again
-flights without an NM link take the lane of their ADS-B track, then the recovered-LOBT rules (their LOBT is found through the flight's landing record)
-NM-linked flights with a third-pass track take the pass-3 greedy lane, the other NM-linked flights a 0.4 / 0.6 blend of two second-stage lanes
-Rome flights without an NM link that take off a later UTC day than planned are capped (the late-day ceiling)
-then the optional fixes, each switched on from the command line: the airport-month calibration, the support caps, the Rome P / S / N model and, last, the late-Rome corrections
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taxi.main import (FEATURE_MATRIX, INITIAL_MONTHS_FILE, INITIAL_SUBMITTING_FILE, RANKING_FILE, RUN_PREDICTIONS, SUBMISSION_FILE, SUBMITTING_FILE, TUNED_CLASSIFIERS, competition_files, departure_frame, departures_with_later_pass_track, lobt_label_window, recording_regimes)
from taxi.models import aggregate_calibration as calibration, out_of_fold, rome_regime_stack as stack
from taxi.models.network_ensembles import (LONG_TAXI_THRESHOLD_SECONDS, build_ensemble_pass2_hand_weighted, build_ensemble_pass3_greedy_selected, write_prediction_run)
from taxi.models.prior_model import prior_ranking_predictions
from taxi.models.tuned_classifiers import reference_network_value
from taxi.prediction.late_rome_departures import LATE_ROME_TRANSFER_FACTOR, correct_late_rome_departures

# (regime run with the stake classifiers and the next-day prior, XGBoost value run of the same seed)
REGIME_RUN_PAIRS = [('regime_seed42_stake_classifiers_nextday', 'xgboost_value_seed42'), ('regime_seed7_stake_classifiers_nextday', 'xgboost_value_seed7')]
CATBOOST_VALUE_RUN = 'catboost_value_pass1_airport_state_full'
TWO_PART_RESIDUAL_RUN = 'two_part_residual_pass1_airport_state_full'
# share of the tuned classifier in the XGBoost half of P(schedule copy): Rome NM-linked rows, rows without an NM link
TUNED_CLASSIFIER_SHARE = {'rome_linked': 1.0, 'no_link': 0.5}
# model weight against the prior: NM-linked rows, rows without an NM link, Rome next-day rows without an NM link
MODEL_WEIGHT_AGAINST_PRIOR = (1.0, 0.9, 0.3) # chosen based on held-out performance
# Family weights in the order: old XGBoost value (unused), XGBoost value, regime two-part, old CatBoost value (unused),
# removed family (always 0), CatBoost value, residual two-part, network. The last weight absorbs any rounding remainder.
FAMILY_WEIGHTS_NETWORKS_90 = (0.0, 0.0024, 0.0172, 0.0, 0.0, 0.0204, 0.06, 0.90)
FAMILY_WEIGHTS_NETWORKS_100 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)      # all weight on the neural network, every tree family off
REFLECTION_SLOPE, REFLECTION_OFFSET = 1.25, 250.0      # fitted on the 2025 week folds, one fold left out at a time; beat a plain mirror (slope 1, offset 0) in all four folds
ROME_LINKED_CLASSIFIER, ROME_LINKED_CLASSIFIER_WITH_PUSH_DELAY = 'rome_linked_classifier', 'rome_linked_classifier_with_push_delay'
REFERENCE_NETWORK_FULL_FIT = 'network_pass3_clip15000_full'      # its value bounds the late-day cap
# ADS-B coverage outages (--coverage-outage-blend): an airport-day is an outage when its tracked share is below
# COVERAGE_OUTAGE_SHARE x the airport's median daily share in the same calendar month of 2025 (that median above
# COVERAGE_OUTAGE_MINIMUM_MEDIAN)
COVERAGE_OUTAGE_SHARE, COVERAGE_OUTAGE_MINIMUM_MEDIAN = 0.6, 0.3      # 0.6: the low-coverage cut of the 2025 outage study; 0.3: a round floor (no tuning recorded) so airport-months that are seldom tracked anyway never count
# networks trained without ADS-B inputs
NO_ADSB_NETWORK_RUNS = ('network_pass2_without_adsb_full', 'network_pass2_unimpeded_without_adsb_full')

# network runs (taxi/models/predictions/<run>) built here
ENSEMBLE_PASS2_HAND_WEIGHTED = 'ensemble_pass2_hand_weighted'
STACK_PASS2_HAND_WEIGHTED = 'stack_pass2_hand_weighted'
ENSEMBLE_PASS2_GREEDY = 'ensemble_pass2_greedy_selected'
ENSEMBLE_PASS3_GREEDY = 'ensemble_pass3_greedy_selected'
STACK_PASS3 = 'stack_pass3_clip15000_seed_pairs'
STACK_PASS2_NO_ADSB_CONTEXT = 'stack_pass2_greedy_selected_no_adsb_context'
COMBINED_GREEDY = 'combined_pass2_pass3_greedy_selected'
# out-of-fold tree-model inputs of the second stages: 'run base:column' (week-fold runs <base>_wk<k>, full fit <base>_full)
TREE_INPUTS_TRACKED_ROWS = ['two_part_residual_pass1_airport_state:v_tpr', 'catboost_value_pass1_airport_state:v_cb', 'two_part_residual_pass3_airport_state_extended:v_tpr']
TREE_INPUTS_UNTRACKED_ROWS = TREE_INPUTS_TRACKED_ROWS + ['two_part_residual_pass3_airport_state_extended_threshold600:v_tpr', 'two_part_residual_pass3_airport_state_extended_threshold2400:v_tpr']
# (run name suffix, tree inputs, second-stage XGBoost parameters)
# depth 8 with min_child_weight 20 / 50: picked in a capacity grid on the 2025 week folds (20 did better in all four folds on tracked and Rome rows, nothing beat 50 on untracked rows)
SECOND_STAGE_VARIANTS = [('tree_inputs_tracked_rows', TREE_INPUTS_TRACKED_ROWS, 'max_depth=8,min_child_weight=20'), ('tree_inputs_untracked_rows', TREE_INPUTS_UNTRACKED_ROWS, 'max_depth=8,min_child_weight=50')]
# the parts of a no-link lane handed to the corrections (build_lane's parts_out), in this order
NO_LINK_PART_NAMES = ('pP', 'pS', 'msched', 'mixture', 'model_weight', 'prior', 'lane_pred')
LABEL_FLOOR = -12                      # s: no label lies below it (a take-off 12 s before the recorded off-block at most)
SECONDS_PER_DAY = 86400                # a previous-date (S) record adds one day to the taxi time

RowRule = Callable[[np.ndarray], np.ndarray]
# a change of a recovered-LOBT rule: (movement id, prediction before, prediction after, window lower bound), seconds
Change = tuple[float, float, float, float]
_TUNED_ROW_RULES: dict[str, RowRule] = {}
# the LOBT window and the label floor
LOBT_WINDOW_TOLERANCE_SECONDS = 10     # s: a schedule copy further outside the LOBT window is impossible
FLOAT_LABEL_FLOOR_SECONDS = -12.0      # s: the same label floor written as a float (case A)
# recovered LOBT: no-link departures whose LOBT comes from the flight's NM-linked arrival
MINIMUM_FLIGHT_TIME_SECONDS = 1200     # s: 20 min, a round lower bound: no flight between two of the ten airports is that short
MAXIMUM_FLIGHT_TIME_SECONDS = 8 * 3600 # s: 8 h, a round upper bound: the longest flights here (Spain - Istanbul) take about 4 h
LOBT_LINK_LIMIT_SECONDS = 3606         # s: the organisers' NM link rule |block - LOBT| <= 3606 s (held for all 2,062,577 2025 departures with an LOBT)
RECOVERED_WINDOW_WIDTH_SECONDS = 7212  # s: width of the window take-off - LOBT -+ 3606 s, 2 x 3606
ROME_CASE_B_MARGIN_SECONDS = 1224      # s: median of how far the 30 Rome case-B labels of 2025 lay above the window (best of five rules tried on 2025)
ISTANBUL_CASE_B_MARGIN_SECONDS = 469   # s: the same median for the 19 Istanbul case-B labels of 2025
# network runs and blends
SECOND_STAGE_ROUNDS = '165'            # boosting rounds: mean of the early-stopping points of the two 2025 week-fold checks (186 and 144)
COMBINED_PASS2_SHARE = 0.5             # a plain equal average of the pass-2 and pass-3 ensembles
COMBINED_PASS3_SHARE = 0.5             # (the other half of that average)
NM_LINKED_COMBINED_SHARE = 0.4         # 0.4 / 0.6: the 2025 week-fold optimum (shares 0.3-0.7 tried; 0.6 for the stack was best in 3 of 4 folds)
NM_LINKED_STACK_SHARE = 0.6            # see above
# the corrections after the late-day cap
HELICOPTER_LABEL_LIMIT_SECONDS = 3600  # s: helicopter labels of an hour or more are left out of their mean
PLAIN_TOLERANCE_SECONDS = 1            # s: a guess moved less than this counts as untouched by the recovered-LOBT rules and the cap


def utc_days_after_schedule(times: pd.DataFrame) -> np.ndarray:
    """
    This function counts how many calendar days (UTC) lie between a flight's timetable date and its take-off date
    """
    return (times.MVT_TIME_UTC_mvt.dt.normalize() - times.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days.to_numpy()


def tuned_classifier_rows() -> dict[str, RowRule]:
    """
    This function builds the rules for which flights each of the two specially tuned 'was the time copied from the timetable?' detectors may help with
    """
    if not _TUNED_ROW_RULES:
        metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['MVT_ID_mvt', 'ADEP_mvt', 'AOBT_3_flt'])
        schedule_columns = ['MVT_ID_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt']
        schedule = pd.concat([pd.read_parquet(path, columns=schedule_columns) for path in competition_files()])
        schedule = schedule.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(metadata.MVT_ID_mvt)
        day_difference = utc_days_after_schedule(schedule)
        no_nm_link = metadata.AOBT_3_flt.isna().to_numpy()      # AOBT_3_flt: the NM actual off-block time, present when linked
        is_rome = (metadata.ADEP_mvt == 'LIRF').to_numpy()

        def rome_linked_rows(rows: np.ndarray) -> np.ndarray:
            """
            This function says yes for each row that has an NM link
            """
            return ~no_nm_link[rows]

        def no_link_rows(rows: np.ndarray) -> np.ndarray:
            """
            This function says no for a Rome flight that took off a later UTC day than its timetable date, and yes for every other row
            """
            return ~(is_rome[rows] & (day_difference[rows] >= 1))

        _TUNED_ROW_RULES.update({'rome_linked': rome_linked_rows, 'no_link': no_link_rows})
    return _TUNED_ROW_RULES


# one lane
def normalised_family_weights(family_weights: tuple[float, ...]) -> list[float]:
    """
    This function makes the eight model-family weights add up to exactly 1 by adding any rounding leftover to the last neural network
    """
    weights = list(family_weights)
    weights[-1] += 1 - sum(weights)
    assert len(weights) == 8 and abs(sum(weights) - 1) < 1e-9 and weights[4] == 0, weights
    return weights


def run_column(run: str, column: str) -> np.ndarray:
    return pd.read_parquet(RUN_PREDICTIONS / run / 'pred.parquet', columns=[column])[column].to_numpy()


def family_value(weights: list[float], regime: pd.DataFrame, xgboost_value: pd.DataFrame, network_run: str, is_rome_row: np.ndarray) -> np.ndarray:
    """
    This function works out v, the plain taxi-time guess (the answer if the time was recorded normally), as a weighted average of the model families: tree models and one neural network run
    On Rome flights the residual two-part family is left out, because it did worse when cross-validated, and the others are scaled up so they still count as a whole
    """
    value = (weights[0] * regime.v_xgb.to_numpy() + weights[1] * xgboost_value.v.to_numpy() + weights[2] * regime.v_tp.to_numpy() + weights[3] * regime.v_cb.to_numpy())
    if weights[5]:
        value = value + weights[5] * run_column(CATBOOST_VALUE_RUN, 'v_cb')
    if weights[7]:
        value = value + weights[7] * run_column(network_run, 'v_nn')
    if weights[6]:
        two_part = run_column(TWO_PART_RESIDUAL_RUN, 'v_tpr')
        value = np.where(is_rome_row, value / (1 - weights[6]), value + weights[6] * two_part)
    return value


def schedule_copy_probability(regime: pd.DataFrame, tuned_classifier_file: dict[str, str]) -> np.ndarray:
    """
    This function works out the chance that each flight's leaving time was copied from the timetable (P)
    """
    probability = regime.pP.to_numpy().copy()
    row_position = pd.Series(np.arange(len(regime)), index=regime.MVT_ID_mvt)
    for model in ('rome_linked', 'no_link'):
        tuned = pd.read_parquet(TUNED_CLASSIFIERS / f'{tuned_classifier_file[model]}.parquet')
        rows = row_position.reindex(tuned.MVT_ID_mvt).to_numpy()
        keep = tuned_classifier_rows()[model](rows)
        target_rows = rows[keep]
        share = TUNED_CLASSIFIER_SHARE[model]
        probability[target_rows] = 0.5 * (share * tuned.p.to_numpy()[keep] + (1 - share) * regime.pP_xgb.to_numpy()[target_rows] + regime.pP_cb.to_numpy()[target_rows])      # mean of the XGBoost half and the CatBoost classifier
    return probability


def impossible_schedule_copy(seconds_after_schedule: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    This function finds flights for which 'copied from the timetable' cannot be the answer
    That is when the time after the timetable lies more than 10 seconds outside the flight's LOBT window (never happened in 2025), or is below -12 seconds, which no answer ever is
    """
    has_window = np.isfinite(lower)
    known = np.isfinite(seconds_after_schedule)
    outside_window = (seconds_after_schedule < lower - LOBT_WINDOW_TOLERANCE_SECONDS) | (seconds_after_schedule > upper + LOBT_WINDOW_TOLERANCE_SECONDS)
    return (has_window & known & outside_window) | (known & (seconds_after_schedule < LABEL_FLOOR))


def value_in_lobt_window(value: np.ndarray, lower: np.ndarray, upper: np.ndarray, reflected: bool) -> np.ndarray:
    """
    This function keeps the taxi-time guess inside the flight's LOBT window by pushing a guess that is too low or too high to the nearest edge (flights without a window keep their guess)
    With 'reflected' on, a guess outside is first bounced back in: below the window it becomes bottom + 250 s + 1.25 times how far below it was, above it becomes top minus 1.25 times how far above it was
    """
    has_window = np.isfinite(lower)
    if reflected:
        value = np.where(has_window & (value < lower), np.minimum(lower + REFLECTION_OFFSET + REFLECTION_SLOPE * (lower - value), upper), np.where(has_window & (value > upper), np.maximum(upper - REFLECTION_SLOPE * (value - upper), lower), value))
    return np.where(has_window, np.clip(value, lower, upper), value)


def regime_mixture(probability_schedule_copy: np.ndarray, probability_previous_date: np.ndarray, seconds_after_schedule: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    This function turns the three possible answers into one expected answer: chance of P times the time after the timetable + chance of S times (86,400 + v) + chance of N times v
    """
    mixture = (probability_schedule_copy * seconds_after_schedule + probability_previous_date * (SECONDS_PER_DAY + value) + (1 - probability_schedule_copy - probability_previous_date) * value)
    return np.maximum(np.where(np.isfinite(mixture), mixture, value), LABEL_FLOOR)


def model_weight_against_prior(movement_ids: np.ndarray) -> np.ndarray:
    """
    This function decides how much the new model counts against the prior (an older, separate model) for each flight
    """
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet').set_index('MVT_ID_mvt').reindex(movement_ids)
    no_nm_link = metadata.AOBT_3_flt.isna().to_numpy()
    schedule = pd.read_parquet(RANKING_FILE, columns=['MVT_ID_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt'] ).set_index('MVT_ID_mvt').reindex(movement_ids)
    day_difference = utc_days_after_schedule(schedule)
    rome_next_day_no_link = no_nm_link & (metadata.ADEP_mvt.to_numpy() == 'LIRF') & (day_difference >= 1)
    linked_weight, no_link_weight, rome_next_day_weight = MODEL_WEIGHT_AGAINST_PRIOR
    return np.where(rome_next_day_no_link, rome_next_day_weight, np.where(no_nm_link, no_link_weight, linked_weight))


@dataclass
class RegimeRunLane:
    """
    This data structure holds what one regime run gives to a lane
    """
    prediction: np.ndarray                  # the regime mixture (s), floored at LABEL_FLOOR
    value: np.ndarray                       # the value v (s), restricted to the LOBT window
    probability_schedule_copy: np.ndarray   # pP after the LOBT-window rules
    probability_previous_date: np.ndarray   # pS after the LOBT-window rules
    seconds_after_schedule: np.ndarray      # msched: take-off minus schedule (s)
    is_ranking_row: np.ndarray              # True on the ranking (2026) departures (the run's 'rank' column)
    ranking_movement_ids: np.ndarray        # MVT_ID_mvt (movement id) of the ranking rows, in departure_metadata order


def regime_run_lane(regime_run: str, xgboost_value_run: str, weights: list[float], network_run: str, tuned_classifier_file: dict[str, str], is_rome_row: np.ndarray, reflected: bool) -> RegimeRunLane:
    """
    This function builds one regime run's part of a lane: it works out v and the chances of P and S, applies the LOBT-window rules and mixes the three answers into one guess
    """
    regime = pd.read_parquet(RUN_PREDICTIONS / regime_run / 'pred.parquet', columns=['MVT_ID_mvt', 'v_xgb', 'v_tp', 'v_cb', 'pP', 'pS', 'msched', 'rank', 'pP_cb', 'pP_xgb'])
    xgboost_value = pd.read_parquet(RUN_PREDICTIONS / xgboost_value_run / 'pred.parquet', columns=['MVT_ID_mvt', 'v'])
    assert (regime.MVT_ID_mvt.to_numpy() == xgboost_value.MVT_ID_mvt.to_numpy()).all()
    value = family_value(weights, regime, xgboost_value, network_run, is_rome_row)
    probability_schedule_copy = schedule_copy_probability(regime, tuned_classifier_file)
    probability_previous_date = regime.pS.to_numpy()
    seconds_after_schedule = regime.msched.to_numpy()
    # with an LOBT, the previous-date regime never occurs and a schedule copy only inside the window (2025: 0 exceptions)
    lower, upper = lobt_label_window(regime.MVT_ID_mvt.to_numpy())
    has_window = np.isfinite(lower)
    probability_schedule_copy = np.where(impossible_schedule_copy(seconds_after_schedule, lower, upper), 0.0, probability_schedule_copy)
    probability_previous_date = np.where(has_window, 0.0, probability_previous_date)
    value = value_in_lobt_window(value, lower, upper, reflected)
    prediction = regime_mixture(probability_schedule_copy, probability_previous_date, seconds_after_schedule, value)
    is_ranking_row = regime['rank'].to_numpy()
    return RegimeRunLane(prediction, value, probability_schedule_copy, probability_previous_date, seconds_after_schedule, is_ranking_row, regime.MVT_ID_mvt[is_ranking_row].to_numpy())


def blended_with_prior(mixture: np.ndarray, model_weight: np.ndarray, prior_values: np.ndarray, movement_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function mixes the model's guess with the prior's
    """
    before_window = np.maximum(model_weight * mixture + (1 - model_weight) * prior_values, LABEL_FLOOR)
    lower, upper = lobt_label_window(movement_ids)
    has_window = np.isfinite(lower)
    in_window = np.maximum(np.where(has_window, np.clip(before_window, lower, upper), before_window), LABEL_FLOOR)
    return before_window, in_window


def build_lane(family_weights: tuple[float, ...], network_run: str, prior: pd.Series, reflected: bool, rome_linked_classifier: str = ROME_LINKED_CLASSIFIER, value_out: np.ndarray | None = None, parts_out: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """
    This function builds one lane - a complete answer sheet for every 2026 flight, from one network run and one set of family weights
    """
    weights = normalised_family_weights(family_weights)
    tuned_classifier_file = {'rome_linked': rome_linked_classifier, 'no_link': 'no_link_classifier'}
    is_rome_row = (pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['ADEP_mvt']).ADEP_mvt == 'LIRF').to_numpy()
    runs = [regime_run_lane(regime_run, xgboost_value_run, weights, network_run, tuned_classifier_file, is_rome_row, reflected) for regime_run, xgboost_value_run in REGIME_RUN_PAIRS]
    # both regime runs share the row order, the ranking rows and msched: the last run's are used
    is_ranking_row = runs[-1].is_ranking_row
    movement_ids = runs[-1].ranking_movement_ids
    mixture_mean = (sum(run.prediction for run in runs) / len(runs))[is_ranking_row]
    # blend with the prior's ranking predictions (an independent older model, submitted as v8), then the LOBT window
    model_weight = model_weight_against_prior(movement_ids)
    prior_values = prior.reindex(movement_ids).to_numpy()
    before_window, lane = blended_with_prior(mixture_mean, model_weight, prior_values, movement_ids)
    template = pd.read_parquet(SUBMITTING_FILE)      # the organisers' template: submission row order

    def in_template(array: np.ndarray) -> np.ndarray:
        """
        This function puts a list from the pipeline's flight order into the organizers' template order, by matching flight ids
        """
        return pd.Series(array, index=movement_ids).reindex(template.MVT_ID_mvt).to_numpy()

    values = in_template(lane)
    assert np.isfinite(values).all()
    if value_out is not None:
        value_out[:] = in_template((sum(run.value for run in runs) / len(runs))[is_ranking_row])
    if parts_out is not None:
        parts_out.update(pP=in_template((sum(run.probability_schedule_copy for run in runs) / len(runs))[is_ranking_row]),
                         pS=in_template((sum(run.probability_previous_date for run in runs) / len(runs))[is_ranking_row]),
                         msched=in_template(runs[-1].seconds_after_schedule[is_ranking_row]),
                         mixture=in_template(mixture_mean),
                         model_weight=in_template(model_weight),
                         prior=in_template(prior_values),
                         lane_pred=in_template(before_window))
    return values


def print_and_build_lane(prior: pd.Series, network_run: str, family_weights: tuple[float, ...], reflected: bool, rome_linked_classifier: str = ROME_LINKED_CLASSIFIER, value_out: np.ndarray | None = None, parts_out: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """
    This function writes a log line saying which lane is being built and then builds it
    """
    weights_name = 'networks90' if family_weights == FAMILY_WEIGHTS_NETWORKS_90 else 'networks100'
    print('lane', network_run, weights_name, 'reflected' if reflected else '', flush=True)
    return build_lane(family_weights, network_run, prior, reflected, rome_linked_classifier, value_out, parts_out)


# no-link departures with a recoverable LOBT
def recovered_lobt_departures(files: list[Path]) -> pd.DataFrame:
    """
    This function finds departures without an NM link whose LOBT can still be found through the flight's NM-linked landing at another airport (same flight number or callsign, same route, landing 20 minutes to 8 hours after take-off)
    From that LOBT it builds a window of take-off minus LOBT, plus or minus 3,606 seconds - since the departure was not linked, its answer should lie outside this window
    """
    columns = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'ADEP_flt', 'CALLSIGN_flt', 'TAXITIME_SEC_mvt']
    movements = pd.concat([pd.read_parquet(path, columns=columns) for path in files])
    # LOBT_flt is only present on movements linked to their NM flight
    departures = movements[(movements.PHASE_mvt == 'DEP') & movements.LOBT_flt.isna()].copy()
    linked_arrivals = movements[(movements.PHASE_mvt == 'ARR') & movements.LOBT_flt.notna()].copy()
    departures['key'] = departures.FLIGHT_mvt.astype(str).str.strip() + '|' + departures.ADEP_mvt + '|' + departures.ADES_mvt.astype(str)
    matches_by_source = []
    for source, identifier in (('flight', 'FLIGHT_mvt'), ('callsign', 'CALLSIGN_flt')):
        # ADEP_flt: the arrival's origin in its flight plan
        linked_arrivals['key'] = (linked_arrivals[identifier].astype(str).str.strip() + '|' + linked_arrivals.ADEP_flt.astype(str) + '|' + linked_arrivals.ADES_mvt.astype(str))
        arrivals = linked_arrivals[['key', 'MVT_TIME_UTC_mvt', 'LOBT_flt']].rename(columns={'MVT_TIME_UTC_mvt': 'land', 'LOBT_flt': 'lobt'})
        matched = departures.merge(arrivals, on='key')
        flight_time = (matched.land - matched.MVT_TIME_UTC_mvt).dt.total_seconds()
        matched = matched[(flight_time > MINIMUM_FLIGHT_TIME_SECONDS) & (flight_time < MAXIMUM_FLIGHT_TIME_SECONDS)]
        # one arrival per departure (first non-missing value of each column)
        matches_by_source.append(matched.groupby('MVT_ID_mvt', as_index=False).first().assign(source=source))
    recovered = pd.concat(matches_by_source, ignore_index=True).drop_duplicates('MVT_ID_mvt')     # flight-number matches first
    recovered['lo'] = (recovered.MVT_TIME_UTC_mvt - pd.to_datetime(recovered.lobt, utc=True)).dt.total_seconds() - LOBT_LINK_LIMIT_SECONDS
    recovered['hi'] = recovered.lo + RECOVERED_WINDOW_WIDTH_SECONDS
    return recovered


def raise_case_b(predictions: pd.Series, recovered: pd.DataFrame) -> list[Change]:
    """
    This function handles case B at Rome and Istanbul: the window starts below -12 seconds and ends above 0, so the answer cannot be under it and must be above it
    """
    case_b = recovered[(recovered.lo < LABEL_FLOOR) & (recovered.hi > 0) & recovered.ADEP_mvt.isin(['LIRF', 'LTFM']) & recovered.FLIGHT_mvt.notna() & (recovered.FLIGHT_mvt.astype(str).str.strip() != '')].set_index('MVT_ID_mvt')
    seconds_after_schedule = (case_b.MVT_TIME_UTC_mvt - case_b.SCHED_TIME_UTC_mvt).dt.total_seconds()
    floor = np.where(case_b.ADEP_mvt == 'LIRF', np.maximum(case_b.hi + ROME_CASE_B_MARGIN_SECONDS, seconds_after_schedule), case_b.hi + ISTANBUL_CASE_B_MARGIN_SECONDS)
    changes: list[Change] = []
    for (movement_id, row), floor_value in zip(case_b.iterrows(), floor):
        if movement_id in predictions.index and predictions[movement_id] < floor_value:
            changes.append((movement_id, predictions[movement_id], floor_value, row.lo))
            predictions[movement_id] = floor_value
    return changes


def no_link_lane_network_value(no_link_network_runs: tuple[str, str, str]) -> Callable[[float], float]:
    """
    This function builds a small lookup that gives a no-link flight's neural network guess from the network run of its lane, which depends on its ADS-B track
    """
    pass2_ids, pass3_ids = departures_with_later_pass_track('pass2'), departures_with_later_pass_track('pass3')
    network_values = {run: pd.read_parquet(RUN_PREDICTIONS / f'{run}/pred.parquet').set_index('MVT_ID_mvt').v_nn for run in no_link_network_runs}

    def lane_run(movement_id: float) -> str:
        """
        This function picks the network run of the flight's lane: the third-pass run if the plane was found in the third matching pass, the second-pass run if it was found in the second, otherwise the first
        """
        if movement_id in pass3_ids:
            return no_link_network_runs[2]
        return no_link_network_runs[1] if movement_id in pass2_ids else no_link_network_runs[0]

    def network_value(movement_id: float) -> float:
        """
        This function looks up the flight's guess in the network run of its lane
        """
        return float(network_values[lane_run(movement_id)][movement_id])
    return network_value


def lower_case_a(predictions: pd.Series, recovered: pd.DataFrame, network_value: Callable[[float], float]) -> list[Change]:
    """
    This function handles case A: the window starts at -12 seconds or more, and in 2025, 99.3 % of such answers lay below the window
    So a guess above the window's bottom is replaced by the network's guess for the flight's lane, kept between -12 seconds and the bottom, and every change is listed
    """
    case_a = recovered[recovered.lo >= LABEL_FLOOR].set_index('MVT_ID_mvt')
    changes: list[Change] = []
    for movement_id, row in case_a.iterrows():
        if movement_id in predictions.index and predictions[movement_id] > row.lo:
            new_value = max(min(network_value(movement_id), row.lo), FLOAT_LABEL_FLOOR_SECONDS)
            changes.append((movement_id, predictions[movement_id], new_value, row.lo))
            predictions[movement_id] = new_value
    return changes


def apply_recovered_lobt_rules(predictions: pd.Series, no_link_network_runs: tuple[str, str, str]) -> tuple[list[Change], int]:
    """
    This function runs both recovered-LOBT rules on the 2026 flights - case B first, then case A
    """
    recovered = recovered_lobt_departures([RANKING_FILE])
    # every file is read before a rule changes `predictions`: a missing run raises with the argument untouched
    network_value = no_link_lane_network_value(no_link_network_runs)
    case_b_changes = raise_case_b(predictions, recovered)
    case_a_changes = lower_case_a(predictions, recovered, network_value)
    return case_b_changes + case_a_changes, len(case_b_changes)


# ADS-B coverage outages
def coverage_outage_rows() -> set[float]:
    """
    This function finds 2026 NM-linked flights with no ADS-B track on days when the airport's receivers were mostly down
    """
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'month', 'year'])
    tracked = metadata.MVT_ID_mvt.isin(pd.read_parquet(FEATURE_MATRIX / 'adsb_pass3.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt).to_numpy()
    airport = metadata.ADEP_mvt.to_numpy()
    day = pd.to_datetime(metadata.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m-%d').to_numpy()
    month = metadata.month.to_numpy()
    is_2025 = metadata.year.to_numpy() == 2025
    daily_share = pd.Series(tracked).groupby([airport, day]).transform('mean').to_numpy()      # tracked share of each airport-day
    reference = pd.Series(daily_share[is_2025]).groupby([airport[is_2025], month[is_2025]]).median()      # over the 2025 departures
    month_median = pd.Series(list(zip(airport, month))).map(reference.to_dict()).to_numpy(dtype=float)
    outage = (daily_share < COVERAGE_OUTAGE_SHARE * month_median) & (month_median > COVERAGE_OUTAGE_MINIMUM_MEDIAN)
    rows = outage & ~tracked & metadata.AOBT_3_flt.notna().to_numpy() & (metadata.year.to_numpy() == 2026)
    return set(metadata.MVT_ID_mvt[rows])


def blend_runs_toward_no_adsb(weight: float, movement_ids: set[float]) -> dict[str, str]:
    """
    This function helps the outage flights: it copies the two network runs that the Rome NM-linked lanes read and, for those flights only, mixes in networks trained without any ADS-B clues by the given weight
    """
    reference = pd.read_parquet(RUN_PREDICTIONS / f'{REFERENCE_NETWORK_FULL_FIT}/pred.parquet')
    plain, unimpeded = (pd.read_parquet(RUN_PREDICTIONS / f'{run}/pred.parquet').v_nn.to_numpy() for run in NO_ADSB_NETWORK_RUNS)
    without_adsb = np.where(reference.v_nn.to_numpy() > LONG_TAXI_THRESHOLD_SECONDS, plain, 0.5 * (plain + unimpeded))      # plain average of the two, except on long taxis where the unimpeded network is left out
    target = reference.MVT_ID_mvt.isin(list(movement_ids)).to_numpy()
    run_map: dict[str, str] = {}
    for run in (f'{COMBINED_GREEDY}_tree_inputs_tracked_rows', f'{STACK_PASS3}_tree_inputs_tracked_rows'):
        frame = pd.read_parquet(RUN_PREDICTIONS / f'{run}/pred.parquet')
        assert (frame.MVT_ID_mvt.to_numpy() == reference.MVT_ID_mvt.to_numpy()).all()
        blended = np.where(target, (1 - weight) * frame.v_nn.to_numpy() + weight * without_adsb, frame.v_nn.to_numpy())
        description = {'blend': f'{1 - weight} x {run} + {weight} x no-ADS-B networks on {int(target.sum())} coverage-outage rows'}
        write_prediction_run(f'{run}_coverage_outage', frame.MVT_ID_mvt.to_numpy(), blended, description)
        run_map[run] = f'{run}_coverage_outage'
    print(f'coverage outage: {int(target.sum())} rows blended toward the no-ADS-B networks (weight {weight})', flush=True)
    return run_map


# ensembles and second stages
def run_second_stage(*arguments: str) -> None:
    print('$ python -m taxi.models.network_ensembles', ' '.join(arguments), flush=True)
    subprocess.run([sys.executable, '-m', 'taxi.models.network_ensembles', *arguments], check=True)


def write_combined_greedy_run(variant_suffix: str) -> None:
    names = (ENSEMBLE_PASS2_GREEDY, ENSEMBLE_PASS3_GREEDY, f'{STACK_PASS2_NO_ADSB_CONTEXT}_{variant_suffix}')
    runs = {name: pd.read_parquet(RUN_PREDICTIONS / f'{name}/pred.parquet') for name in names}
    pass2_values, pass3_values, corrected_values = (runs[name].v_nn.to_numpy() for name in names)
    combined = COMBINED_PASS2_SHARE * pass2_values + COMBINED_PASS3_SHARE * pass3_values + (corrected_values - pass2_values)
    write_prediction_run(f'{COMBINED_GREEDY}_{variant_suffix}', runs[names[0]].MVT_ID_mvt.to_numpy(), combined, {'correction': f'{names[2]} - {names[0]}'})


def build_network_runs() -> None:
    build_ensemble_pass2_hand_weighted(ENSEMBLE_PASS2_HAND_WEIGHTED)
    build_ensemble_pass3_greedy_selected(ENSEMBLE_PASS3_GREEDY)
    # the pass2_hand_weighted mix corrects the hand-weighted ensemble above
    run_second_stage('--member-mix', 'pass2_hand_weighted', '--output-run', STACK_PASS2_HAND_WEIGHTED, '--rounds', SECOND_STAGE_ROUNDS)
    run_second_stage('--member-mix', 'pass2_greedy_selected', '--output-run', ENSEMBLE_PASS2_GREEDY, '--no-correction')
    for variant_suffix, tree_inputs, xgboost_params in SECOND_STAGE_VARIANTS:
        for mix, stack in (('pass3_clip15000_seed_pairs', STACK_PASS3), ('pass2_greedy_selected_no_adsb_context', STACK_PASS2_NO_ADSB_CONTEXT)):
            run_second_stage('--member-mix', mix, '--folds', 'wk0,wk1,wk2,wk3', '--output-run', f'{stack}_{variant_suffix}', '--rounds', SECOND_STAGE_ROUNDS, '--out-of-fold-inputs', ','.join(tree_inputs), '--xgboost-params', xgboost_params)
        write_combined_greedy_run(variant_suffix)


# the lanes of the submission rows
def no_link_lane_values(prior: pd.Series, no_nm_link: np.ndarray, pass2_track: np.ndarray, pass3_track: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    This function makes the guesses for 2026 flights without an NM link, with a different lane depending on their ADS-B track
    """
    row_count = len(no_nm_link)
    values = np.full(row_count, np.nan)
    other = no_nm_link & ~pass2_track & ~pass3_track
    second_pass = no_nm_link & pass2_track & ~pass3_track
    third_pass = no_nm_link & pass3_track
    no_link_value = np.full(row_count, np.nan)
    no_link_parts = {name: np.full(row_count, np.nan) for name in NO_LINK_PART_NAMES}
    for rows, network_run, weights in ((other, STACK_PASS2_HAND_WEIGHTED, FAMILY_WEIGHTS_NETWORKS_90), (second_pass, ENSEMBLE_PASS2_HAND_WEIGHTED, FAMILY_WEIGHTS_NETWORKS_100), (third_pass, ENSEMBLE_PASS3_GREEDY, FAMILY_WEIGHTS_NETWORKS_100)):
        lane_value = np.full(row_count, np.nan)
        parts: dict[str, np.ndarray] = {}
        values[rows] = print_and_build_lane(prior, network_run, weights, False, value_out=lane_value, parts_out=parts)[rows]
        no_link_value[rows] = lane_value[rows]
        for name, array in parts.items():
            no_link_parts[name][rows] = array[rows]
    return values, no_link_value, no_link_parts


def nm_linked_blend(prior: pd.Series, run_map: dict[str, str], movement_ids: pd.Series, frame: pd.DataFrame, pass2_track: np.ndarray) -> np.ndarray:
    """
    This function makes the guesses for NM-linked flights: 0.4 times the combined greedy run + 0.6 times the pass-3 second stage, using the outage copies when there are any
    Which second-stage variant and which family weights a flight gets depends on whether it is from Rome and whether it has an ADS-B track
    """
    is_rome = (frame.set_index('MVT_ID_mvt').ADEP_mvt.reindex(movement_ids) == 'LIRF').to_numpy()
    tracked = np.isin(movement_ids, pd.read_parquet(FEATURE_MATRIX / 'adsb_pass3.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt)
    variant_lanes = {}
    for network_run in (COMBINED_GREEDY, STACK_PASS3):
        for variant, _, _ in SECOND_STAGE_VARIANTS:
            for weights_name, weights in (('networks90', FAMILY_WEIGHTS_NETWORKS_90), ('networks100', FAMILY_WEIGHTS_NETWORKS_100)):
                run = run_map.get(f'{network_run}_{variant}', f'{network_run}_{variant}')
                lane = print_and_build_lane(prior, run, weights, True, ROME_LINKED_CLASSIFIER_WITH_PUSH_DELAY)
                variant_lanes[(network_run, variant, weights_name)] = lane

    def share_values(network_run: str, variant: str, networks100_everywhere: bool) -> np.ndarray:
        """
        This function picks the lane for each flight: networks100 everywhere if told so, otherwise networks100 only for flights with a second-pass track and networks90 for the rest
        """
        all_100 = variant_lanes[(network_run, variant, 'networks100')]
        return all_100 if networks100_everywhere else np.where(pass2_track, all_100, variant_lanes[(network_run, variant, 'networks90')])

    blended = np.full(len(movement_ids), np.nan)
    groups = [(is_rome, 'tree_inputs_tracked_rows', False), (~is_rome & tracked, 'tree_inputs_tracked_rows', True), (~is_rome & ~tracked, 'tree_inputs_untracked_rows', False)]
    for in_group, variant, networks100_everywhere in groups:
        combined = share_values(COMBINED_GREEDY, variant, networks100_everywhere)
        stack = share_values(STACK_PASS3, variant, networks100_everywhere)
        blended[in_group] = (NM_LINKED_COMBINED_SHARE * combined + NM_LINKED_STACK_SHARE * stack)[in_group]
    return blended


def recovered_lobt_rules_applied(values: np.ndarray, movement_ids: pd.Series) -> np.ndarray:
    """
    This function applies the recovered-LOBT rules to the guesses in template order and prints how many flights case B raised and case A lowered
    """
    predictions = pd.Series(values, index=movement_ids.to_numpy())
    no_link_network_runs = (STACK_PASS2_HAND_WEIGHTED, ENSEMBLE_PASS2_HAND_WEIGHTED, ENSEMBLE_PASS3_GREEDY)
    changes, case_b_changes = apply_recovered_lobt_rules(predictions, no_link_network_runs)
    print(f'recovered LOBT: {case_b_changes} case-B rows raised, {len(changes) - case_b_changes} case-A rows lowered')
    return predictions.to_numpy().copy()


def late_day_cap(movement_ids: pd.Series) -> np.ndarray:
    """
    This function works out a ceiling for 2026 Rome no-link flights that take off a later UTC day than planned, since their answer is either a timetable copy or a yesterday's-date record
    The ceiling is the larger of the time after the timetable and 86,400 + the reference network's taxi time - an estimate, not a hard limit
    """
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet')
    seconds_after_schedule, day_difference, _, _, _ = recording_regimes(metadata)
    late_day = ((metadata.ADEP_mvt.to_numpy() == 'LIRF') & metadata.AOBT_3_flt.isna().to_numpy() & (day_difference >= 1) & (metadata.year.to_numpy() == 2026))
    cap = np.maximum(seconds_after_schedule, SECONDS_PER_DAY + reference_network_value(metadata, REFERENCE_NETWORK_FULL_FIT))
    cap_by_id = pd.Series(cap[late_day], index=metadata.MVT_ID_mvt.to_numpy()[late_day])
    return cap_by_id.reindex(movement_ids).to_numpy()


def late_day_cap_applied(values: np.ndarray, movement_ids: pd.Series) -> np.ndarray:
    """
    This function lowers every guess above its late-day ceiling down to the ceiling and prints how many were lowered
    """
    values = values.copy()
    cap = late_day_cap(movement_ids)
    capped = np.isfinite(cap) & (values > cap)
    values[capped] = cap[capped]
    print(f'capped rows {int(capped.sum())}')
    return values


def late_rome_corrections_applied(values: np.ndarray, movement_ids: pd.Series, no_link_value: np.ndarray, late_rome_factor: float) -> np.ndarray:
    """
    This function runs the late-Rome corrections at the chosen strength, prints the table of changes and returns the new guesses
    """
    ids = movement_ids.to_numpy()
    corrected, changes = correct_late_rome_departures(pd.Series(values, index=ids), pd.Series(no_link_value, index=ids), late_rome_factor)
    print(f'late Rome departures corrected (factor {late_rome_factor}):\n' + changes.to_string(index=False))
    return corrected.to_numpy()


def write_submission_files(output: pd.DataFrame) -> None:
    SUBMISSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(SUBMISSION_FILE, index=False)
    print('wrote', SUBMISSION_FILE)
    if INITIAL_SUBMITTING_FILE.exists():
        initial = pd.read_parquet(INITIAL_SUBMITTING_FILE)[['MVT_ID_mvt']]
        initial['TAXITIME_SEC_mvt'] = output.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(initial.MVT_ID_mvt).to_numpy()
        assert initial.TAXITIME_SEC_mvt.notna().all()
        initial.to_parquet(INITIAL_MONTHS_FILE, index=False)
        print('wrote', INITIAL_MONTHS_FILE)


def build_submission(late_rome_corrections: bool = False, coverage_outage_blend: float = 0.0, aggregate_calibration: bool = False, support_caps: bool = False, next_day_regime_model: float = 0.0, late_rome_factor: float = LATE_ROME_TRANSFER_FACTOR) -> pd.DataFrame:
    """
    This is the main function that runs everything in order
    """
    prior = prior_ranking_predictions()
    build_network_runs()
    run_map = blend_runs_toward_no_adsb(coverage_outage_blend, coverage_outage_rows()) if coverage_outage_blend else {}
    output = pd.read_parquet(SUBMITTING_FILE)
    movement_ids = output.MVT_ID_mvt
    frame = departure_frame()
    # the row groups of the template: no NM link, a second-pass / third-pass ADS-B track
    no_nm_link = np.isin(movement_ids, list(set(frame.MVT_ID_mvt[frame.no_nm_link & (frame.year == 2026)])))
    pass2_track = np.isin(movement_ids, list(departures_with_later_pass_track('pass2')))
    pass3_track = np.isin(movement_ids, list(departures_with_later_pass_track('pass3')))
    # rows without an NM link, then the recovered-LOBT rules
    values, no_link_value, no_link_parts = no_link_lane_values(prior, no_nm_link, pass2_track, pass3_track)
    values = recovered_lobt_rules_applied(values, movement_ids)
    # NM-linked rows with a third-pass track, then the other NM-linked rows
    linked_third_pass = ~no_nm_link & pass3_track
    values[linked_third_pass] = print_and_build_lane(prior, ENSEMBLE_PASS3_GREEDY, FAMILY_WEIGHTS_NETWORKS_100, True)[linked_third_pass]
    linked_other = ~no_nm_link & ~pass3_track
    values[linked_other] = nm_linked_blend(prior, run_map, movement_ids, frame, pass2_track)[linked_other]
    values = late_day_cap_applied(values, movement_ids)
    before_corrections = values.copy()
    values = apply_post_cap_corrections(values, movement_ids, frame, no_nm_link, no_link_parts, no_link_value, before_corrections, aggregate_calibration, support_caps, next_day_regime_model)
    if late_rome_corrections:
        values = late_rome_corrections_applied(values, movement_ids, no_link_value, late_rome_factor)
    assert np.isfinite(values).all()
    output['TAXITIME_SEC_mvt'] = values.astype(output.TAXITIME_SEC_mvt.dtype)
    write_submission_files(output)
    return output


# ---------------------------------------------------------------- the corrections after the late-day cap
# 2025 maxima of the labels of LEMD and LEBL: 3,598 s and 3,604 s (389,000 departures)
SUPPORT_CAP_AIRPORTS, SUPPORT_CAP_SECONDS = ('LEMD', 'LEBL'), 3600.0


def support_caps_applied(values: np.ndarray, airport: np.ndarray, helicopter: np.ndarray) -> np.ndarray:
    """
    This function caps Madrid and Barcelona guesses at 3,600 seconds, because their largest 2025 answers were 3,598 and 3,604 seconds
    Helicopter departures (runway 'H') all get the average answer of the 2025 helicopter departures
    """
    values = values.copy()
    capped = np.isin(airport, SUPPORT_CAP_AIRPORTS) & (values > SUPPORT_CAP_SECONDS)
    values[capped] = SUPPORT_CAP_SECONDS
    training = pd.concat([pd.read_parquet(path, columns=['PHASE_mvt', 'RUNWAY_mvt', 'TAXITIME_SEC_mvt']) for path in competition_files() if 'training_' in path.name])
    helicopter_departures = (training.PHASE_mvt == 'DEP') & (training.RUNWAY_mvt == 'H') & (training.TAXITIME_SEC_mvt.abs() < HELICOPTER_LABEL_LIMIT_SECONDS)
    helicopter_labels = training[helicopter_departures].TAXITIME_SEC_mvt
    values[helicopter] = helicopter_labels.mean()
    print(f'support caps: {int(capped.sum())} rows at {SUPPORT_CAP_SECONDS:.0f} s; {int(helicopter.sum())} helicopter rows at {helicopter_labels.mean():.1f} s', flush=True)
    return values


def no_link_regime_parts(movement_ids: pd.Series, no_nm_link: np.ndarray, no_link_parts: dict[str, np.ndarray], no_link_value: np.ndarray, airport: np.ndarray, ranking: pd.DataFrame, before_corrections: np.ndarray) -> pd.DataFrame:
    """
    This function packs the table the Rome correction models need - one row per 2026 flight without an NM link, with its lane pieces and timing
    It also marks whether the guess is still its lane's own (no recovered-LOBT rule or ceiling moved it by 1 second or more)
    """
    rows = no_nm_link
    parts = pd.DataFrame({name: array[rows] for name, array in no_link_parts.items()})
    parts['MVT_ID_mvt'] = movement_ids.to_numpy()[rows]
    parts['v'] = no_link_value[rows]
    parts['airport'] = airport[rows]
    parts['takeoff'] = ranking.MVT_TIME_UTC_mvt.to_numpy()[rows]
    parts['day_difference'] = utc_days_after_schedule(ranking)[rows]
    parts['submitted_before_corrections'] = before_corrections[rows]
    parts['plain'] = np.abs(parts.submitted_before_corrections - parts.lane_pred) < PLAIN_TOLERANCE_SECONDS      # no recovered-LOBT rule or cap acted
    return parts


def regime_model_values_applied(values: np.ndarray, position: pd.Series, new_values: pd.Series, strength: float) -> np.ndarray:
    """
    This function puts the P / S / N model's new guesses in place of the old ones and prints how many flights changed and the typical size of the change
    """
    values = values.copy()
    rows = position[new_values.index].to_numpy()
    old = values[rows]
    values[rows] = new_values.to_numpy()
    rms_change = np.sqrt(np.mean((new_values.to_numpy() - old) ** 2))
    print(f'Rome regime model (strength {strength}): {len(new_values)} rows, rms change {rms_change:.0f} s', flush=True)
    return values


def apply_post_cap_corrections(values: np.ndarray, movement_ids: pd.Series, frame: pd.DataFrame, no_nm_link: np.ndarray, no_link_parts: dict[str, np.ndarray], no_link_value: np.ndarray, before_corrections: np.ndarray, aggregate_calibration: bool, support_caps: bool, next_day_regime_model: float) -> np.ndarray:
    """
    This function runs the optional fixes after the late-day ceiling
    """
    if not (aggregate_calibration or support_caps or next_day_regime_model):
        return values
    values = values.copy()
    components = out_of_fold.load_components()          # the 2025 out-of-fold components the corrections are fitted on
    meta = frame.set_index('MVT_ID_mvt').reindex(movement_ids)
    ranking = pd.read_parquet(RANKING_FILE, columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'RUNWAY_mvt']).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(movement_ids)
    airport, month = meta.ADEP_mvt.to_numpy(), meta.month.to_numpy()
    if aggregate_calibration:
        # one row per template row: the input of aggregate_calibration.calibrate (linked = has an NM link)
        table = pd.DataFrame({'airport': airport, 'month': month, 'linked': ~no_nm_link, 'prediction': values})
        values, shifts = calibration.calibrate(table, components)
        print('airport-month calibration (shift on NM-linked rows predicted below 7200 s):\n' + shifts.round(1).to_string(), flush=True)
    if support_caps:
        values = support_caps_applied(values, airport, (ranking.RUNWAY_mvt == 'H').to_numpy())
    if next_day_regime_model:
        parts = no_link_regime_parts(movement_ids, no_nm_link, no_link_parts, no_link_value, airport, ranking, before_corrections)
        position = pd.Series(np.arange(len(movement_ids)), index=movement_ids.to_numpy())      # template row of each movement id
        new_values = stack.regime_model_values(parts, stack.fit_regime_model(components), next_day_regime_model)
        values = regime_model_values_applied(values, position, new_values, next_day_regime_model)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the submission file.')
    parser.add_argument('--late-rome-corrections', action='store_true')
    parser.add_argument('--coverage-outage-blend', type=float, default=0.0, metavar='WEIGHT')
    parser.add_argument('--aggregate-calibration', action='store_true')
    parser.add_argument('--support-caps', action='store_true')
    parser.add_argument('--next-day-regime-model', type=float, default=0.0, metavar='STRENGTH')
    parser.add_argument('--late-rome-factor', type=float, default=LATE_ROME_TRANSFER_FACTOR, metavar='FACTOR')
    arguments = parser.parse_args()
    build_submission(arguments.late_rome_corrections, arguments.coverage_outage_blend, arguments.aggregate_calibration, arguments.support_caps, arguments.next_day_regime_model, arguments.late_rome_factor)


if __name__ == '__main__':
    main()
