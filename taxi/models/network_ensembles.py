"""
The project trains many neural networks, each built a little differently, and each one on its own is noisy
This file blends them into an ensemble - a weighted average where the more useful networks get bigger weights
Every network was trained 4 times with one week fold of 2025 hidden and once on all of 2025

This is how it works:
-pick a mix: a list of networks (members) and their weights
-average the members - where the reference network says the taxi is longer than 6,000 seconds, leave out the 'unimpeded' members
-train a second stage, an XGBoost model that learns a small correction to the ensemble:
    1. it looks at how far each member is from the ensemble, plus some details about the departure
    2. it learns from the week-fold guesses
    3. it corrects the full-fit guesses of the 2026 rows, but only where they look like what it trained on
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from taxi.main import (FEATURE_MATRIX, MODEL_WEIGHTS, RUN_PREDICTIONS, WEEK_FOLDS, departure_frame, departures_with_later_pass_track, held_out_week_fold_rows, rmse)


@dataclass(frozen=True)
class Member:
    """
    This data structure is a name card for one network in the ensemble
    """
    name: str
    fold_runs: list[str]
    full_runs: list[str]
    weight: float

    @property
    def predicts_unimpeded_offset(self) -> bool:
        """
        This function says whether the network learned the extra time on top of an unimpeded (no-traffic) taxi, which is when its name contains 'unimpeded' - those networks are left out of the average on very long taxis
        """
        return 'unimpeded' in self.name


@dataclass(frozen=True)
class Mix:
    """
    This data structure is the list of members plus whether they saw the third-pass ADS-B tracks
    """
    members: list[Member]
    members_saw_pass3_tracks: bool

    def member(self, name: str) -> Member:
        """
        This function finds a member in the list
        """
        return next(m for m in self.members if m.name == name)


# Members of each mix
# weights rounded by hand from greedy ensemble selections on the 2025 week folds wk0 / wk2, then checked on those folds; weight 0 = second-stage input only
PASS2_HAND_WEIGHTED: list[Member] = [
    Member('network_pass1_unimpeded', ['network_pass1_unimpeded'], ['network_pass2_unimpeded_seeds123'], 3),
    Member('network_pass1_periodic', ['network_pass1_periodic'], ['network_pass2_periodic'], 2),
    Member('network_pass1_context16_seeds1_2', ['network_pass1_context16_seed1', 'network_pass1_context16_seed2'], ['network_pass2_context16_seed1', 'network_pass2_context16_seed2', 'network_pass2_context16_seed3'], 2),
    Member('network_pass1_muon', ['network_pass1_muon'], ['network_pass2_muon_seeds123'], 1),
    Member('network_pass1_nm_neighbours', ['network_pass1_nm_neighbours'], ['network_pass2_nm_neighbours_seeds123'], 1),
    Member('network_pass1_twohot', ['network_pass1_twohot'], ['network_pass2_twohot'], 1),
    Member('network_pass1_realmlp_pbld', ['network_pass1_realmlp_pbld'], ['network_pass2_realmlp_pbld'], 1),
    Member('network_pass1_unimpeded_periodic', ['network_pass1_unimpeded_periodic'], ['network_pass2_unimpeded_periodic'], 3),
    Member('network_pass1_unimpeded_context16_seed1', ['network_pass1_unimpeded_context16_seed1'], ['network_pass2_unimpeded_context16_seed1', 'network_pass2_unimpeded_context16_seed2'], 2),
    Member('network_pass1_unimpeded_muon', ['network_pass1_unimpeded_muon'], ['network_pass2_unimpeded_muon'], 2),
    Member('network_pass1_unimpeded_winter_weather', ['network_pass1_unimpeded_winter_weather'], ['network_pass2_unimpeded_winter_weather'], 1),
    Member('network_pass1_unimpeded_twohot', ['network_pass1_unimpeded_twohot'], ['network_pass2_unimpeded_twohot'], 1),
    Member('network_pass1_unimpeded_nm_neighbours', ['network_pass1_unimpeded_nm_neighbours'], ['network_pass2_unimpeded_nm_neighbours'], 1),
    Member('network_pass1', ['network_pass1'], ['network_pass2_seeds123'], 0),
    Member('network_pass1_lr0.001', ['network_pass1_lr0.001'], ['network_pass2_lr0.001'], 0),
    Member('network_pass1_winter_weather', ['network_pass1_winter_weather'], ['network_pass2_winter_weather'], 0),
    Member('network_pass1_wide_windows', ['network_pass1_wide_windows'], ['network_pass2_wide_windows'], 0),
    Member('network_pass1_ordering', ['network_pass1_ordering'], ['network_pass2_ordering'], 0),
    Member('network_pass1_tabm16_seed1', ['network_pass1_tabm16_seed1'], ['network_pass2_tabm16_seed1', 'network_pass2_tabm16_seed2'], 0),
]
# weights = summed counts of two greedy ensemble selections on the 2025 week folds (one fitted on wk0, one on wk2); weight 0 = second-stage input only
PASS2_GREEDY_SELECTED: list[Member] = [
    Member('network_pass1_unimpeded_periodic_muon', ['network_pass1_unimpeded_periodic_muon'], ['network_pass2_unimpeded_periodic_muon'], 14),
    Member('network_pass1_unimpeded_context16_seed1', ['network_pass1_unimpeded_context16_seed1'], ['network_pass2_unimpeded_context16_seed1', 'network_pass2_unimpeded_context16_seed2', 'network_pass2_unimpeded_context16_seed3'], 8),
    Member('network_pass1_unimpeded_twohot_epochs24', ['network_pass1_unimpeded_twohot_epochs24'], ['network_pass2_unimpeded_twohot_epochs24'], 6),
    Member('network_pass1_periodic', ['network_pass1_periodic'], ['network_pass2_periodic', 'network_pass2_periodic_seeds34'], 6),
    Member('network_pass1_twohot_epochs24', ['network_pass1_twohot_epochs24'], ['network_pass2_twohot_epochs24'], 6),
    Member('network_pass1_unimpeded_periodic_context16_seed1', ['network_pass1_unimpeded_periodic_context16_seed1'], ['network_pass2_unimpeded_periodic_context16'], 5),
    Member('network_pass1_unimpeded_muon', ['network_pass1_unimpeded_muon'], ['network_pass2_unimpeded_muon', 'network_pass2_unimpeded_muon_seeds34'], 5),
    Member('network_pass1_context16_seeds1_2', ['network_pass1_context16_seed1', 'network_pass1_context16_seed2'], ['network_pass2_context16_seed1', 'network_pass2_context16_seed2', 'network_pass2_context16_seed3'], 5),
    Member('network_pass1_unimpeded_winter_weather', ['network_pass1_unimpeded_winter_weather'], ['network_pass2_unimpeded_winter_weather'], 4),
    Member('network_pass1_nm_neighbours', ['network_pass1_nm_neighbours'], ['network_pass2_nm_neighbours_seeds123'], 4),
    Member('network_pass1_unimpeded_periodic', ['network_pass1_unimpeded_periodic'], ['network_pass2_unimpeded_periodic', 'network_pass2_unimpeded_periodic_seeds34'], 4),
    Member('network_pass1_muon', ['network_pass1_muon'], ['network_pass2_muon_seeds123'], 3),
    Member('network_pass1_unimpeded_context32', ['network_pass1_unimpeded_context32'], ['network_pass2_unimpeded_context32'], 3),
    Member('network_pass1_winter_weather', ['network_pass1_winter_weather'], ['network_pass2_winter_weather'], 1),
    Member('network_pass1', ['network_pass1'], ['network_pass2_seeds123'], 0),
    Member('network_pass1_unimpeded', ['network_pass1_unimpeded'], ['network_pass2_unimpeded_seeds123', 'network_pass2_unimpeded_seeds45'], 0),
    Member('network_pass1_lr0.001', ['network_pass1_lr0.001'], ['network_pass2_lr0.001'], 0),
    Member('network_pass1_wide_windows', ['network_pass1_wide_windows'], ['network_pass2_wide_windows'], 0),
    Member('network_pass1_ordering', ['network_pass1_ordering'], ['network_pass2_ordering'], 0),
    Member('network_pass1_tabm16_seed1', ['network_pass1_tabm16_seed1'], ['network_pass2_tabm16_seed1', 'network_pass2_tabm16_seed2'], 0),
    Member('network_pass1_twohot', ['network_pass1_twohot'], ['network_pass2_twohot'], 0),
    Member('network_pass1_realmlp_pbld', ['network_pass1_realmlp_pbld'], ['network_pass2_realmlp_pbld'], 0),
    Member('network_pass1_unimpeded_twohot', ['network_pass1_unimpeded_twohot'], ['network_pass2_unimpeded_twohot'], 0),
    Member('network_pass1_unimpeded_nm_neighbours', ['network_pass1_unimpeded_nm_neighbours'], ['network_pass2_unimpeded_nm_neighbours'], 0),
]
# the PASS2_GREEDY_SELECTED weights, plus the two networks without ADS-B inputs as second-stage inputs only (weight 0)
PASS2_GREEDY_SELECTED_NO_ADSB_CONTEXT: list[Member] = [
    Member('network_pass1_unimpeded_periodic_muon', ['network_pass1_unimpeded_periodic_muon'], ['network_pass2_unimpeded_periodic_muon'], 14),
    Member('network_pass1_unimpeded_context16_seed1', ['network_pass1_unimpeded_context16_seed1'], ['network_pass2_unimpeded_context16_seed1', 'network_pass2_unimpeded_context16_seed2', 'network_pass2_unimpeded_context16_seed3'], 8),
    Member('network_pass1_unimpeded_twohot_epochs24', ['network_pass1_unimpeded_twohot_epochs24'], ['network_pass2_unimpeded_twohot_epochs24'], 6),
    Member('network_pass1_periodic', ['network_pass1_periodic'], ['network_pass2_periodic', 'network_pass2_periodic_seeds34'], 6),
    Member('network_pass1_twohot_epochs24', ['network_pass1_twohot_epochs24'], ['network_pass2_twohot_epochs24'], 6),
    Member('network_pass1_unimpeded_periodic_context16_seed1', ['network_pass1_unimpeded_periodic_context16_seed1'], ['network_pass2_unimpeded_periodic_context16'], 5),
    Member('network_pass1_unimpeded_muon', ['network_pass1_unimpeded_muon'], ['network_pass2_unimpeded_muon', 'network_pass2_unimpeded_muon_seeds34'], 5),
    Member('network_pass1_context16_seeds1_2', ['network_pass1_context16_seed1', 'network_pass1_context16_seed2'], ['network_pass2_context16_seed1', 'network_pass2_context16_seed2', 'network_pass2_context16_seed3'], 5),
    Member('network_pass1_unimpeded_winter_weather', ['network_pass1_unimpeded_winter_weather'], ['network_pass2_unimpeded_winter_weather'], 4),
    Member('network_pass1_nm_neighbours', ['network_pass1_nm_neighbours'], ['network_pass2_nm_neighbours_seeds123'], 4),
    Member('network_pass1_unimpeded_periodic', ['network_pass1_unimpeded_periodic'], ['network_pass2_unimpeded_periodic', 'network_pass2_unimpeded_periodic_seeds34'], 4),
    Member('network_pass1_muon', ['network_pass1_muon'], ['network_pass2_muon_seeds123'], 3),
    Member('network_pass1_unimpeded_context32', ['network_pass1_unimpeded_context32'], ['network_pass2_unimpeded_context32'], 3),
    Member('network_pass1_winter_weather', ['network_pass1_winter_weather'], ['network_pass2_winter_weather'], 1),
    Member('network_pass1', ['network_pass1'], ['network_pass2_seeds123'], 0),
    Member('network_pass1_unimpeded', ['network_pass1_unimpeded'], ['network_pass2_unimpeded_seeds123', 'network_pass2_unimpeded_seeds45'], 0),
    Member('network_pass1_lr0.001', ['network_pass1_lr0.001'], ['network_pass2_lr0.001'], 0),
    Member('network_pass1_wide_windows', ['network_pass1_wide_windows'], ['network_pass2_wide_windows'], 0),
    Member('network_pass1_ordering', ['network_pass1_ordering'], ['network_pass2_ordering'], 0),
    Member('network_pass1_tabm16_seed1', ['network_pass1_tabm16_seed1'], ['network_pass2_tabm16_seed1', 'network_pass2_tabm16_seed2'], 0),
    Member('network_pass1_twohot', ['network_pass1_twohot'], ['network_pass2_twohot'], 0),
    Member('network_pass1_realmlp_pbld', ['network_pass1_realmlp_pbld'], ['network_pass2_realmlp_pbld'], 0),
    Member('network_pass1_unimpeded_twohot', ['network_pass1_unimpeded_twohot'], ['network_pass2_unimpeded_twohot'], 0),
    Member('network_pass1_unimpeded_nm_neighbours', ['network_pass1_unimpeded_nm_neighbours'], ['network_pass2_unimpeded_nm_neighbours'], 0),
    Member('network_pass1_without_adsb', ['network_pass1_without_adsb'], ['network_pass2_without_adsb'], 0),
    Member('network_pass1_unimpeded_without_adsb', ['network_pass1_unimpeded_without_adsb'], ['network_pass2_unimpeded_without_adsb'], 0),
]
# the PASS2_GREEDY_SELECTED weights again, for the same networks retrained with --target-clip 15000
PASS3_CLIP15000_SEED_PAIRS: list[Member] = [
    Member('network_pass3_clip15000_unimpeded_periodic_muon', ['network_pass3_clip15000_unimpeded_periodic_muon'], ['network_pass3_clip15000_unimpeded_periodic_muon', 'network_pass3_clip15000_unimpeded_periodic_muon_seeds34'], 14),
    Member('network_pass3_clip15000_unimpeded_context16_seeds34', ['network_pass3_clip15000_unimpeded_context16_seeds34'], ['network_pass3_clip15000_unimpeded_context16', 'network_pass3_clip15000_unimpeded_context16_seeds34'], 8),
    Member('network_pass3_clip15000_periodic', ['network_pass3_clip15000_periodic'], ['network_pass3_clip15000_periodic', 'network_pass3_clip15000_periodic_seeds34'], 6),
    Member('network_pass3_clip15000_unimpeded_twohot_epochs24', ['network_pass3_clip15000_unimpeded_twohot_epochs24'], ['network_pass3_clip15000_unimpeded_twohot_epochs24', 'network_pass3_clip15000_unimpeded_twohot_epochs24_seeds34'], 6),
    Member('network_pass3_clip15000_twohot_epochs24', ['network_pass3_clip15000_twohot_epochs24'], ['network_pass3_clip15000_twohot_epochs24', 'network_pass3_clip15000_twohot_epochs24_seeds34'], 6),
    Member('network_pass3_clip15000_unimpeded_periodic_context16', ['network_pass3_clip15000_unimpeded_periodic_context16'], ['network_pass3_clip15000_unimpeded_periodic_context16', 'network_pass3_clip15000_unimpeded_periodic_context16_seeds34'], 5),
    Member('network_pass3_clip15000_unimpeded_muon', ['network_pass3_clip15000_unimpeded_muon'], ['network_pass3_clip15000_unimpeded_muon', 'network_pass3_clip15000_unimpeded_muon_seeds34'], 5),
    Member('network_pass3_clip15000_context16', ['network_pass3_clip15000_context16'], ['network_pass3_clip15000_context16', 'network_pass3_clip15000_context16_seeds34'], 5),
    Member('network_pass3_clip15000_unimpeded_winter_weather', ['network_pass3_clip15000_unimpeded_winter_weather'], ['network_pass3_clip15000_unimpeded_winter_weather', 'network_pass3_clip15000_unimpeded_winter_weather_seeds34'], 4),
    Member('network_pass3_clip15000_nm_neighbours', ['network_pass3_clip15000_nm_neighbours'], ['network_pass3_clip15000_nm_neighbours', 'network_pass3_clip15000_nm_neighbours_seeds34'], 4),
    Member('network_pass3_clip15000_unimpeded_periodic', ['network_pass3_clip15000_unimpeded_periodic'], ['network_pass3_clip15000_unimpeded_periodic', 'network_pass3_clip15000_unimpeded_periodic_seeds34'], 4),
    Member('network_pass3_clip15000_muon', ['network_pass3_clip15000_muon'], ['network_pass3_clip15000_muon', 'network_pass3_clip15000_muon_seeds34'], 3),
    Member('network_pass3_clip15000_unimpeded_context32', ['network_pass3_clip15000_unimpeded_context32'], ['network_pass3_clip15000_unimpeded_context32', 'network_pass3_clip15000_unimpeded_context32_seeds34'], 3),
    Member('network_pass3_clip15000', ['network_pass3_clip15000'], ['network_pass3_clip15000'], 0),
    Member('network_pass1_without_adsb', ['network_pass1_without_adsb'], ['network_pass2_without_adsb'], 0),
    Member('network_pass1_unimpeded_without_adsb', ['network_pass1_unimpeded_without_adsb'], ['network_pass2_unimpeded_without_adsb'], 0),
]
MIXES: dict[str, Mix] = {
    'pass2_hand_weighted': Mix(PASS2_HAND_WEIGHTED, members_saw_pass3_tracks=False),
    'pass2_greedy_selected': Mix(PASS2_GREEDY_SELECTED, members_saw_pass3_tracks=False),
    'pass2_greedy_selected_no_adsb_context': Mix(PASS2_GREEDY_SELECTED_NO_ADSB_CONTEXT, members_saw_pass3_tracks=False),
    'pass3_clip15000_seed_pairs': Mix(PASS3_CLIP15000_SEED_PAIRS, members_saw_pass3_tracks=True),
}

CORRECTION_CLIP_SECONDS = 3000.0          # the second-stage target (label - ensemble) is clipped at +- this
LONG_TAXI_THRESHOLD_SECONDS = 6000.0      # above this reference value, unimpeded-offset members are left out of the ensemble
ENSEMBLE_MISS_LIMIT_SECONDS = 30000       # training rows whose ensemble misses the label by this much or more are dropped
TRAINING_RANGE_QUANTILE = 0.999           # ranking rows beyond this quantile of the training inputs get no correction
# parameters of the second stage's XGBoost model (main() can override them with --xgboost-params)
SECOND_STAGE_PARAMS: dict[str, Any] = dict(objective='reg:squarederror', tree_method='hist', device='cpu', nthread=6, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=200, reg_lambda=10)
DEFAULT_SECOND_STAGE_ROUNDS = 160         # close to the early-stopping round counts (146 to 186) when the second stage was trained on one 2025 week fold and tested on the other


def full_fit_predictions(run_base_name: str) -> pd.DataFrame:
    """
    This function loads the guesses of a network's full-fit run - one guess for every departure
    """
    return pd.read_parquet(RUN_PREDICTIONS / f'{run_base_name}_full/pred.parquet')


def write_prediction_run(run_name: str, movement_ids: np.ndarray, network_values: np.ndarray, description: dict[str, Any]) -> None:
    run_folder = RUN_PREDICTIONS / run_name
    run_folder.mkdir(exist_ok=True)
    pd.DataFrame({'MVT_ID_mvt': movement_ids, 'v_nn': network_values}).to_parquet(run_folder / 'pred.parquet', index=False)
    (run_folder / 'members.json').write_text(json.dumps(description, indent=1))


# plain ensembles of full fits
# the non-zero PASS2_HAND_WEIGHTED weights
PASS2_HAND_WEIGHTED_ENSEMBLE: list[tuple[list[str], int, bool]] = [
    (['network_pass2_unimpeded_seeds123'], 3, True),
    (['network_pass2_periodic'], 2, False),
    (['network_pass2_context16_seed1', 'network_pass2_context16_seed2', 'network_pass2_context16_seed3'], 2, False),
    (['network_pass2_muon_seeds123'], 1, False),
    (['network_pass2_nm_neighbours_seeds123'], 1, False),
    (['network_pass2_twohot'], 1, False),
    (['network_pass2_realmlp_pbld'], 1, False),
    (['network_pass2_unimpeded_periodic'], 3, True),
    (['network_pass2_unimpeded_context16_seed1', 'network_pass2_unimpeded_context16_seed2'], 2, True),
    (['network_pass2_unimpeded_muon'], 2, True),
    (['network_pass2_unimpeded_winter_weather'], 1, True),
    (['network_pass2_unimpeded_twohot'], 1, True),
    (['network_pass2_unimpeded_nm_neighbours'], 1, True)]
PASS2_HAND_WEIGHTED_REFERENCE = 'network_pass2_seeds123'
# the non-zero PASS2_GREEDY_SELECTED weights without the weight-1 winter-weather network, for the 13 networks retrained on the pass-3 inputs
PASS3_GREEDY_SELECTED_ENSEMBLE: list[tuple[str, int, bool]] = [
    ('network_pass3_unimpeded_periodic_muon', 14, True),
    ('network_pass3_unimpeded_context16', 8, True),
    ('network_pass3_periodic', 6, False),
    ('network_pass3_unimpeded_twohot_epochs24', 6, True),
    ('network_pass3_twohot_epochs24', 6, False),
    ('network_pass3_unimpeded_periodic_context16', 5, True),
    ('network_pass3_unimpeded_muon', 5, True),
    ('network_pass3_context16', 5, False),
    ('network_pass3_unimpeded_winter_weather', 4, True),
    ('network_pass3_nm_neighbours', 4, False),
    ('network_pass3_unimpeded_periodic', 4, True),
    ('network_pass3_muon', 3, False),
    ('network_pass3_unimpeded_context32', 3, True)]


def mean_of_full_fits(run_base_names: list[str], movement_ids: np.ndarray) -> np.ndarray:
    """
    This function averages several full-fit runs of one network, row by row
    """
    run_predictions = []
    for run_base_name in run_base_names:
        predictions = full_fit_predictions(run_base_name)
        assert (predictions.MVT_ID_mvt.to_numpy() == movement_ids).all()
        run_predictions.append(predictions)
    return np.mean([predictions.v_nn.to_numpy() for predictions in run_predictions], 0)


def build_ensemble_pass2_hand_weighted(output_run: str) -> None:
    """
    This function builds the first ensemble
    """
    reference = full_fit_predictions(PASS2_HAND_WEIGHTED_REFERENCE)
    movement_ids = reference.MVT_ID_mvt.to_numpy()
    member_values = []
    for runs, weight, predicts_unimpeded_offset in PASS2_HAND_WEIGHTED_ENSEMBLE:
        values = mean_of_full_fits(runs, movement_ids)
        assert np.isfinite(values).all(), runs
        member_values.append((values, weight, predicts_unimpeded_offset))
    reference_is_long = reference.v_nn.to_numpy() > LONG_TAXI_THRESHOLD_SECONDS
    # "ordinary" members predict the taxi time itself (not the unimpeded offset); they alone make the value of long taxis
    weighted_sum_all = sum(weight * values for values, weight, _ in member_values)
    weight_all = sum(weight for _, weight, _ in member_values)
    weighted_sum_ordinary = sum(weight * values for values, weight, unimpeded in member_values if not unimpeded)
    weight_ordinary = sum(weight for _, weight, unimpeded in member_values if not unimpeded)
    ensemble = np.where(reference_is_long, weighted_sum_ordinary / weight_ordinary, weighted_sum_all / weight_all)
    write_prediction_run(output_run, movement_ids, ensemble, {
        'members': {'+'.join(runs): weight for runs, weight, _ in PASS2_HAND_WEIGHTED_ENSEMBLE},
        'rule': f'unimpeded-offset members left out where {PASS2_HAND_WEIGHTED_REFERENCE}_full > 6000 s'})


def build_ensemble_pass3_greedy_selected(output_run: str) -> None:
    """
    This function builds the same kind of ensemble with the greedy-selected weights and 13 networks retrained on the pass-3 inputs
    """
    reference = full_fit_predictions('network_pass3')
    movement_ids = reference.MVT_ID_mvt.to_numpy()
    reference_values = reference.v_nn.to_numpy()
    weighted_sum_all = np.zeros(len(movement_ids))
    weight_all = np.zeros(len(movement_ids))
    weighted_sum_ordinary = np.zeros(len(movement_ids))
    weight_ordinary = np.zeros(len(movement_ids))
    for run_base_name, weight, predicts_unimpeded_offset in PASS3_GREEDY_SELECTED_ENSEMBLE:
        member = full_fit_predictions(run_base_name)
        assert (member.MVT_ID_mvt.to_numpy() == movement_ids).all()
        values = member.v_nn.to_numpy()
        weighted_sum_all += weight * values
        weight_all += weight
        if not predicts_unimpeded_offset:
            weighted_sum_ordinary += weight * values
            weight_ordinary += weight
    reference_is_long = reference_values > LONG_TAXI_THRESHOLD_SECONDS
    ensemble = np.where(reference_is_long, weighted_sum_ordinary / weight_ordinary, weighted_sum_all / weight_all)
    write_prediction_run(output_run, movement_ids, ensemble, {
        'members': {run_base_name: weight for run_base_name, weight, _ in PASS3_GREEDY_SELECTED_ENSEMBLE},
        'rule': 'unimpeded-offset members left out where network_pass3_full > 6000 s', 
        'inputs': 'network_inputs_pass3'})


# second stage
def reference_member_name(mix: Mix) -> str:
    """
    This function picks the network whose guess decides the 6,000-second rule
    """
    names = [m.name for m in mix.members]
    return 'network_pass3_clip15000' if 'network_pass3_clip15000' in names else 'network_pass1'


def row_context(frame: pd.DataFrame, adsb_table: str) -> pd.DataFrame:
    """
    This function gathers the facts about each departure that the second stage reads besides the networks' guesses
    """
    adsb_columns = ['MVT_ID_mvt', 'ads_nsurf', 'ads_first_dstand', 'ads_first_rel']
    adsb = pd.read_parquet(FEATURE_MATRIX / f'{adsb_table}.parquet', columns=adsb_columns)
    # the ADS-B table is keyed by movement id: first row per id, in the order of `frame`
    adsb = adsb.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(frame.MVT_ID_mvt.to_numpy())
    # base_features.parquet is row-aligned with departure_metadata.parquet (and so with `frame`)
    base = pd.read_parquet(FEATURE_MATRIX / 'base_features.parquet', columns=['EOBT_1_flt__minus__AOBT_3_flt', 'AOBT_3_flt_second'])
    takeoff_time = pd.to_datetime(frame.MVT_TIME_UTC_mvt)
    no_nm_link = frame.no_nm_link.to_numpy()
    return pd.DataFrame({
        'airport_code': pd.Categorical(frame.ADEP_mvt).codes,
        'no_nm_link': no_nm_link.astype(float),
        'is_rome': frame.is_rome.to_numpy().astype(float),
        'nm_taxi_estimate': np.where(no_nm_link, np.nan, frame.baseline.to_numpy()),
        'push_delay': -base['EOBT_1_flt__minus__AOBT_3_flt'].to_numpy(dtype=float),
        'nm_off_block_second': base.AOBT_3_flt_second.to_numpy(dtype=float),
        'hour_of_day': takeoff_time.dt.hour.to_numpy() + takeoff_time.dt.minute.to_numpy() / 60,
        'adsb_surface_points': adsb.ads_nsurf.to_numpy(dtype=float),
        'adsb_first_distance_to_stand': adsb.ads_first_dstand.to_numpy(dtype=float),
        'adsb_first_seen_relative': adsb.ads_first_rel.to_numpy(dtype=float),
        'seconds_after_schedule': frame.seconds_after_schedule.to_numpy()})


def run_column(run_name: str, column: str) -> np.ndarray:
    """
    This function reads one column of a run's guesses, for every departure
    """
    return pd.read_parquet(RUN_PREDICTIONS / run_name / 'pred.parquet', columns=[column])[column].to_numpy()


def out_of_fold_inputs(specification: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function brings in other models' guesses (the tree models) as extra clues for the second stage
    """
    columns = {}
    ranking_rows = (frame.year.to_numpy() == 2026)
    for item in specification:
        run_base, column = item.split(':')
        values = np.full(len(frame), np.nan)
        for fold in WEEK_FOLDS:
            held_out = held_out_week_fold_rows(frame, fold)
            values[held_out] = run_column(f'{run_base}_{fold}', column)[held_out]
        values[ranking_rows] = run_column(f'{run_base}_full', column)[ranking_rows]
        columns[f'out_of_fold_{run_base}'] = values
    return pd.DataFrame(columns)


def member_ensemble(mix: Mix, member_values: dict[str, np.ndarray], reference_values: np.ndarray) -> np.ndarray:
    """
    This function computes the ensemble for every row
    """
    weights = {m.name: m.weight for m in mix.members if m.weight > 0}
    ensemble_all = sum(weights[name] * member_values[name] for name in weights) / sum(weights.values())
    ordinary = [name for name in weights if not mix.member(name).predicts_unimpeded_offset]
    ensemble_ordinary = sum(weights[name] * member_values[name] for name in ordinary) / sum(weights[name] for name in ordinary)
    return np.where(reference_values > LONG_TAXI_THRESHOLD_SECONDS, ensemble_ordinary, ensemble_all)


def second_stage_inputs(mix: Mix, member_values: dict[str, np.ndarray], reference_values: np.ndarray, context: pd.DataFrame, rows: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    This function fills in the second stage's worksheet for the chosen rows
    """
    ensemble = member_ensemble(mix, member_values, reference_values)
    inputs = context[rows].reset_index(drop=True).copy()
    inputs['ensemble'] = ensemble[rows]
    out_of_fold_columns = [c for c in inputs.columns if c.startswith('out_of_fold_')]
    for column in out_of_fold_columns:      # other model families: their gap to the ensemble
        inputs[f'gap_{column}'] = inputs[column] - inputs['ensemble']
    for member in mix.members:
        inputs[f'deviation_{member.name}'] = member_values[member.name][rows] - ensemble[rows]
    return inputs, ensemble


def member_fold_values(member: Member, fold: str) -> np.ndarray:
    """
    This function gives a member's guesses for one week fold
    """
    runs = [pd.read_parquet(RUN_PREDICTIONS / f'{run}_{fold}' / 'pred.parquet').v_nn.to_numpy() for run in member.fold_runs]
    return runs[0] if len(runs) == 1 else (runs[0] + runs[1]) / 2


FoldTrainingData = tuple[pd.DataFrame, np.ndarray, np.ndarray]     # (second-stage inputs, labels in s, ensemble values in s)


def fold_training_rows(mix: Mix, frame: pd.DataFrame, context: pd.DataFrame, fold: str) -> FoldTrainingData:
    """
    This function makes the second stage's training data for one week fold, using the members' guesses for that fold
    """
    member_values = {member.name: member_fold_values(member, fold) for member in mix.members}
    labels = frame.TAXITIME_SEC_mvt.to_numpy()
    is_held_out = held_out_week_fold_rows(frame, fold)
    is_labelled = np.isfinite(labels)
    # regime N: neither a copy of the schedule (P) nor a block time recorded with the previous day's date (S)
    is_ordinary_regime = (~frame.is_schedule_copy & ~frame.is_previous_date).to_numpy()
    rows = is_held_out & is_labelled & is_ordinary_regime
    inputs, ensemble = second_stage_inputs(mix, member_values, member_values[reference_member_name(mix)], context, rows)
    return inputs, labels[rows], ensemble[rows]


def train_second_stage(inputs: pd.DataFrame, labels: np.ndarray, ensemble: np.ndarray, rounds: int) -> xgb.Booster:
    """
    This function trains the second stage: an XGBoost model that learns how far off the ensemble is (true answer minus ensemble)
    The target is capped at plus or minus 3,000 seconds and rows where the ensemble is off by 30,000 seconds or more are dropped, so a few wild rows cannot drag it around
    """
    keep = np.abs(labels - ensemble) < ENSEMBLE_MISS_LIMIT_SECONDS
    clipped_target = np.clip(labels - ensemble, -CORRECTION_CLIP_SECONDS, CORRECTION_CLIP_SECONDS)
    training_matrix = xgb.QuantileDMatrix(inputs[keep], label=clipped_target[keep])
    return xgb.train(SECOND_STAGE_PARAMS, training_matrix, rounds)


def training_range_limits(training_inputs: pd.DataFrame) -> tuple[dict[str, float], float]:
    """
    This function measures how far the training values went, to build fences - for each member the distance from the ensemble that 999 of 1,000 training rows stay under, and the same for the ensemble value
    """
    deviation_columns = [c for c in training_inputs.columns if c.startswith('deviation_')]
    deviation_limits = {c: np.quantile(np.abs(training_inputs[c]), TRAINING_RANGE_QUANTILE) for c in deviation_columns}
    ensemble_limit = np.quantile(training_inputs.ensemble, TRAINING_RANGE_QUANTILE)
    return deviation_limits, ensemble_limit


def within_training_range(inputs: pd.DataFrame, deviation_limits: dict[str, float], ensemble_limit: float) -> np.ndarray:
    """
    This function checks which rows are inside the fences: every member close enough to the ensemble and the ensemble not above its fence (a missing value counts as outside)
    The second stage only corrects rows inside the fences, because it never trained on stranger ones
    """
    deviation_within_limit = np.stack([np.abs(inputs[c].to_numpy()) <= deviation_limits[c] for c in deviation_limits], 1)
    return np.all(deviation_within_limit, 1) & (inputs.ensemble.to_numpy() <= ensemble_limit)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Combine neural network runs.')
    parser.add_argument('--member-mix', required=True, choices=sorted(MIXES))
    parser.add_argument('--folds', default='wk0,wk2')
    parser.add_argument('--output-run', required=True)
    parser.add_argument('--rounds', type=int, default=DEFAULT_SECOND_STAGE_ROUNDS)
    parser.add_argument('--no-correction', action='store_true')
    parser.add_argument('--xgboost-params', default='')
    parser.add_argument('--out-of-fold-inputs', default='')
    return parser.parse_args()


def apply_parameter_overrides(overrides: str) -> None:
    """
    This function changes the second stage's XGBoost settings from text like 'max_depth=8,min_child_weight=50'
    """
    for assignment in [x for x in overrides.split(',') if x]:
        key, value = assignment.split('=')
        SECOND_STAGE_PARAMS[key] = type(SECOND_STAGE_PARAMS[key])(value) if key in SECOND_STAGE_PARAMS else float(value)


def second_stage_context(mix: Mix, frame: pd.DataFrame, out_of_fold_specification: str) -> pd.DataFrame:
    """
    This function builds the departure facts with the right ADS-B track table, then adds the tree models' guesses if any were asked for
    """
    context = row_context(frame, 'adsb_pass3' if mix.members_saw_pass3_tracks else 'adsb_pass2')
    if out_of_fold_specification:
        context = pd.concat([context, out_of_fold_inputs(out_of_fold_specification.split(','), frame)], axis=1)
    return context


def print_fold_ensemble_errors(member_mix: str, folds: list[str], training: dict[str, FoldTrainingData]) -> None:
    """
    This function prints the plain ensemble's RMSE on each week fold's training rows, leaving out rows it misses by 30,000 seconds or more
    """
    for fold in folds:
        _, labels, ensemble = training[fold]
        keep = np.abs(labels - ensemble) < ENSEMBLE_MISS_LIMIT_SECONDS
        print(f'mix {member_mix} ensemble on {fold}: {rmse((labels - ensemble)[keep]):.2f}', flush=True)


def full_fit_member_values(mix: Mix) -> dict[str, np.ndarray]:
    """
    This function averages each member's full-fit runs to get its guesses for every departure - these are the guesses used for the 2026 rows
    """
    return {m.name: np.mean([full_fit_predictions(run).v_nn.to_numpy() for run in m.full_runs], 0) for m in mix.members}


def save_second_stage(model: xgb.Booster, deviation_limits: dict[str, float], ensemble_limit: float, output_run: str) -> None:
    """
    This function saves files fromthe above
    """
    weights_folder = MODEL_WEIGHTS / output_run
    weights_folder.mkdir(parents=True, exist_ok=True)
    model.save_model(weights_folder / 'second_stage.ubj')
    range_limits = {'deviation_limits': {c: float(v) for c, v in deviation_limits.items()}, 'ensemble_limit': float(ensemble_limit)}
    (weights_folder / 'range.json').write_text(json.dumps(range_limits, indent=1))


def ranking_correction(mix: Mix, training: dict[str, FoldTrainingData], folds: list[str], ranking_inputs: pd.DataFrame, ranking_rows: np.ndarray, movement_ids: np.ndarray, rounds: int, output_run: str) -> np.ndarray:
    """
    This function trains the second stage on the chosen week folds together and predicts a correction for the 2026 rows (0 for every other row)
    """
    correction = np.zeros(len(movement_ids))
    training_inputs = pd.concat([training[fold][0] for fold in folds], ignore_index=True)
    training_labels = np.concatenate([training[fold][1] for fold in folds])
    training_ensemble = np.concatenate([training[fold][2] for fold in folds])
    model = train_second_stage(training_inputs, training_labels, training_ensemble, rounds)
    correction[ranking_rows] = model.predict(xgb.DMatrix(ranking_inputs))
    deviation_limits, ensemble_limit = training_range_limits(training_inputs)
    in_range = within_training_range(ranking_inputs, deviation_limits, ensemble_limit)
    out_of_range = np.flatnonzero(ranking_rows)[~in_range]
    correction[out_of_range] = 0.0
    print('ranking rows outside the fold range (no correction):', len(out_of_range))
    save_second_stage(model, deviation_limits, ensemble_limit, output_run)
    if not mix.members_saw_pass3_tracks:     # fold members trained without the second-pass tracks: no correction on those rows
        correction[np.isin(movement_ids, list(departures_with_later_pass_track('pass2')))] = 0.0
    return correction


def main() -> None:
    arguments = parse_arguments()
    folds = arguments.folds.split(',')
    apply_parameter_overrides(arguments.xgboost_params)
    mix = MIXES[arguments.member_mix]
    frame = departure_frame()
    context = second_stage_context(mix, frame, arguments.out_of_fold_inputs)
    training = {fold: fold_training_rows(mix, frame, context, fold) for fold in folds}
    print_fold_ensemble_errors(arguments.member_mix, folds, training)

    # the ranking rows: the full-fit members, their ensemble and (unless --no-correction) the second-stage correction
    movement_ids = full_fit_predictions(PASS2_HAND_WEIGHTED_REFERENCE).MVT_ID_mvt.to_numpy()
    member_values = full_fit_member_values(mix)
    ranking_rows = (frame.year.to_numpy() == 2026)
    assert (frame.MVT_ID_mvt.to_numpy() == movement_ids).all()
    ranking_inputs, ensemble = second_stage_inputs(mix, member_values, member_values[reference_member_name(mix)], context, ranking_rows)
    # the pass2_hand_weighted mix corrects the separately built hand-weighted ensemble; every other mix its own member ensemble
    corrects_hand_weighted_ensemble = arguments.member_mix == 'pass2_hand_weighted'
    if corrects_hand_weighted_ensemble:
        base_values = pd.read_parquet(RUN_PREDICTIONS / 'ensemble_pass2_hand_weighted/pred.parquet').v_nn.to_numpy()
    else:
        base_values = ensemble
    if arguments.no_correction:
        correction = np.zeros(len(movement_ids))
    else:
        correction = ranking_correction(mix, training, folds, ranking_inputs, ranking_rows, movement_ids, arguments.rounds, arguments.output_run)
    write_prediction_run(arguments.output_run, movement_ids, base_values + correction, {
        'base': 'ensemble_pass2_hand_weighted' if corrects_hand_weighted_ensemble else f'member mix {arguments.member_mix}',
        'members': {m.name: {'fold_runs': m.fold_runs, 'full_runs': m.full_runs, 'weight': m.weight} for m in mix.members},
        'correction': 'none' if arguments.no_correction else 'taxi/models/network_ensembles.py', 
        'rounds': arguments.rounds})
    ranking_correction_values = correction[ranking_rows]
    print(f'correction on ranking rows: rms {rmse(ranking_correction_values):.1f}, mean {ranking_correction_values.mean():+.2f}, max {np.abs(ranking_correction_values).max():.0f}; wrote {arguments.output_run}')


if __name__ == '__main__':
    main()
