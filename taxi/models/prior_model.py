"""
This file builds the prior model - an older, simpler model of the taxi-out time
It does not use the ADS-B position messages and does not know about the three ways the times were written down (regimes P / S / N), so it gives a second opinion that looks at the problem differently

This is how it works:
-practice on the 2025 months February to June and August to November, and test on January and July 2025 (December is used by neither)
-run eight steps one after another:
    1. an XGBoost model on the leftover
    2. a CatBoost model only for the flights with no NM link
    3. the 2026 guesses of steps 1 and 2 together
    4. a new XGBoost model that also uses weather, OPDI open flight data, queues and clock precision
    5. two new models for the no-link flights, one for Rome and one for the other airports
    6. one XGBoost model per airport, on flights with an NM link
    7. a small neural network, on flights with an NM link
    8. step 6 again with the OPDI ground events, for nine airports
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from torch import nn
from xgboost import XGBRegressor

from taxi.features.prior_tables import TARGET_COLUMN, load_prior_tables
from taxi.main import PRIOR_TABLES


GROUND_EVENT_AIRPORTS = ['LSZH', 'LIRF', 'LFPG', 'LTFM', 'EGLL', 'EHAM', 'EDDF', 'EDDM', 'LEBL']
MONTH_SPLIT = {'held_out': (1, 7), 'not_trained_on': (1, 7, 12)}     # test on Jan / Jul 2025 held out, and December added for sanity check to ensure no overfitting
PREDICTION_FLOOR_SECONDS = -12                  # the lowest taxi-out time among the 2025 labels; no guess goes below it

ERROR_REPORT_QUANTILE = .95                     # report only: the 95th percentile of the absolute error, a common size for the bad cases

# step 1: XGBoost settings of the residual model on every training row
STEP1_RESIDUAL_SETTINGS = {
    'n_estimators': 1600,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 60,
    'subsample': .85,
    'colsample_bytree': .9,
    'reg_lambda': 20,
    'reg_alpha': .1,
    'max_bin': 256,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'enable_categorical': True,
    'max_cat_to_onehot': 12,
    'max_cat_threshold': 64,
    'n_jobs': 12,
    'random_state': 42,
    'early_stopping_rounds': 80,
}

# step 2: CatBoost settings of the no-link specialist
STEP2_NO_LINK_SETTINGS = {
    'iterations': 700,
    'depth': 5,
    'learning_rate': .045,
    'l2_leaf_reg': 30,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'thread_count': 4,
    'allow_writing_files': False,
    'verbose': False,
}
# steps 2 and 5: the no-link models
NO_LINK_RESIDUAL_CLIP_SECONDS = 20000           # much wider than the network's 6000 s: a no-link label can be hours away from the baseline

# step 4: XGBoost settings of the weather / OPDI model
STEP4_WEATHER_OPDI_SETTINGS = {
    'n_estimators': 1100,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 60,
    'subsample': .85,
    'colsample_bytree': .9,
    'reg_lambda': 20,
    'reg_alpha': .1,
    'tree_method': 'hist',
    'enable_categorical': True,
    'max_cat_to_onehot': 12,
    'max_cat_threshold': 64,
    'n_jobs': 12,
    'random_state': 42,
    'early_stopping_rounds': 80,
    'disable_default_eval_metric': True,
}
STEP4_BLEND_WEIGHTS = [0., .25, .5, .75, 1.]    # shares of the new model tried; the one with the lowest error on the Jan / Jul 2025 held out months is kept

# step 5: CatBoost settings of the two no-link models, Rome and the other airports
STEP5_NO_LINK_SETTINGS = {
    'iterations': 600,
    'depth': 6,
    'learning_rate': .04,
    'l2_leaf_reg': 30,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'thread_count': 6,
    'allow_writing_files': False,
    'verbose': False,
}
STEP5_BLEND_WEIGHTS = [0, .5, 1]                # the same test with a coarser list (none, half, all), once for Rome and once for the other airports

# steps 6 and 8: XGBoost settings of the airport models
AIRPORT_MODEL_SETTINGS = {
    'n_estimators': 1200,
    'learning_rate': .045,
    'min_child_weight': 40,
    'reg_lambda': 30,
    'reg_alpha': .1,
    'subsample': .85,
    'colsample_bytree': .9,
    'enable_categorical': True,
    'max_cat_to_onehot': 12,
    'max_cat_threshold': 64,
    'tree_method': 'hist',
    'n_jobs': 8,
    'random_state': 42,
    'early_stopping_rounds': 80,
}
AIRPORT_MODEL_DEPTH = 7
AIRPORT_BLEND_WEIGHTS = [0, .25, .5, .75, 1.]   # shares tried per airport; the one with the lowest error on that airport's Jan / Jul 2025 held out rows is kept

# step 7: inputs of the neural network
MIN_CATEGORY_LEVEL_COUNT = 10                   # codes seen fewer than 10 times in training share one 'other' code
NUMERIC_STD_FLOOR = .1                          # near-constant columns are not blown up when divided by their spread
NUMERIC_INPUT_CLIP = 10                         # scaled inputs are cut at +-10 spreads so one extreme value cannot swamp the network
# step 7: shape of the neural network
EMBEDDING_WIDTH_EXPONENT = .25                  # the fourth root of the number of codes, a common rule of thumb for embedding sizes
EMBEDDING_WIDTH_FACTOR = 2                      # that root is doubled
MIN_EMBEDDING_WIDTH = 2
MAX_EMBEDDING_WIDTH = 12
FIRST_HIDDEN_WIDTH = 192
SECOND_HIDDEN_WIDTH = 96                        # half of the first layer, picked by hand the same way
NETWORK_DROPOUT = .1                            # a light dropout, a common starting value
# step 7: training of the neural network
NETWORK_RESIDUAL_CLIP_SECONDS = 6000            # the same +-6000 s cut as the value model's VALUE_RESIDUAL_CLIP in regime_mixture.py
TARGET_UNIT_SECONDS = 300                       # the network learns the leftover in 5-minute units, so its target is a number near 1
NETWORK_THREADS = 4                             # CPU threads for the network
NETWORK_SEED = 42                               # the usual fixed seed
NETWORK_OPTIMIZER_SETTINGS = {
    'lr': .001,
    'weight_decay': .001,
}
TRAINING_BATCH_SIZE = 2048                      # large batches keep a CPU epoch short
GRADIENT_NORM_CLIP = 10                         # a safety limit against a rare huge training step
PREDICTION_BATCH_SIZE = 8192                    # rows per prediction batch
NETWORK_MAX_EPOCHS = 24                         # an upper limit - training stops sooner when the Jan / Jul 2025 held out error stops improving
NETWORK_PATIENCE_EPOCHS = 5                     # stop after 5 epochs without a better Jan / Jul 2025 held out error
NETWORK_BLEND_WEIGHTS = [.1, .25, .5, .75, 1.]  # network shares tried on the linked rows - the one with the lowest Jan / Jul 2025 held out error is kept


@dataclass
class StepPredictions:
    """
    This data structure holds one step's guesses in seconds
    """
    held_out: np.ndarray
    ranking: np.ndarray


def error_metrics(labels: np.ndarray | pd.Series, predictions: np.ndarray | pd.Series) -> dict[str, float | int]:
    """
    This function measures how wrong a list of guesses is using several metrics
    """
    error = np.asarray(predictions, dtype=float) - np.asarray(labels, dtype=float)
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite values in evaluation')
    absolute_error = np.abs(error)
    return dict(n=len(error), rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(absolute_error)), bias=float(error.mean()), p95_absolute_error=float(np.quantile(absolute_error, ERROR_REPORT_QUANTILE)))


def report(step: str, labels: np.ndarray | pd.Series, predictions: np.ndarray, **details: object) -> None:
    """
    This function prints one line with a step's score on the test months, plus any extra notes such as the mixing amount it picked
    """
    print(f'prior {step}:', json.dumps(dict(**error_metrics(labels, predictions), **details)), flush=True)


def load_tree_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the table of facts about every departure that steps 4, 6, 7 and 8 learn from - basic facts, OPDI open flight data, weather, queues and clock precision
    It drops the flight number, callsign and transponder address, because they would only let a model memorize single flights
    """
    features, metadata = load_prior_tables(include_opdi_features=True)
    features = pd.concat([features, pd.read_parquet(PRIOR_TABLES / 'weather_features.parquet')], axis=1)
    return features.drop(columns=['FLIGHT_mvt', 'CALLSIGN_flt', 'opdi_icao24'], errors='ignore'), metadata


def month_split(metadata: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    This function sorts the flights into practice flights (2025 without January, July and December), test flights (January and July 2025) and the 2026 flights to predict
    """
    is_2025 = metadata.year.eq(2025)
    is_training = is_2025 & ~metadata.month.isin(MONTH_SPLIT['not_trained_on'])
    is_held_out = is_2025 & metadata.month.isin(MONTH_SPLIT['held_out'])
    is_ranking = metadata.year.eq(2026)
    return is_training, is_held_out, is_ranking


def with_string_categoricals(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    This function makes a copy of the table with every category column (like an airline code) turned into text, the way CatBoost wants them
    """
    features = features.copy()
    categorical_columns = features.select_dtypes('category').columns.tolist()
    for column in categorical_columns:
        features[column] = features[column].astype(str)
    return features, categorical_columns


# steps 1-3: residual model, no-link specialist, their ranking predictions
def fit_residual() -> tuple[XGBRegressor, np.ndarray]:
    """
    step 1 - it trains an XGBoost model on the leftover over the practice months
    """
    features, metadata = load_prior_tables(include_opdi_features=False)
    is_training, is_held_out, _ = month_split(metadata)
    model = XGBRegressor(**STEP1_RESIDUAL_SETTINGS)
    residual = metadata[TARGET_COLUMN].to_numpy(dtype=float) - metadata.baseline.to_numpy()
    model.fit(features.loc[is_training], residual[is_training], eval_set=[(features.loc[is_held_out], residual[is_held_out])], verbose=False)
    held_out_baseline = metadata.loc[is_held_out].baseline.to_numpy()
    predictions = model.predict(features.loc[is_held_out]).astype(float) + held_out_baseline
    report('step 1 residual', metadata.loc[is_held_out, TARGET_COLUMN], predictions, best_iteration=int(model.best_iteration))
    return model, predictions


def fit_no_link_specialist() -> tuple[CatBoostRegressor, pd.Series]:
    """
    step 2 - flights with no NM link only have the rough fallback baseline, so it trains a CatBoost model on those flights alone
    """
    features, metadata = load_prior_tables(include_opdi_features=True)
    is_no_link = metadata.AOBT_3_flt.isna()
    is_training, is_held_out, _ = month_split(metadata)
    training_rows = is_no_link & is_training
    held_out_rows = is_no_link & is_held_out
    training_features, categorical_columns = with_string_categoricals(features.loc[training_rows])
    held_out_features, _ = with_string_categoricals(features.loc[held_out_rows])
    training_labels = metadata.loc[training_rows, TARGET_COLUMN].to_numpy()
    clipped_training_residual = np.clip(training_labels - metadata.loc[training_rows, 'baseline'], -NO_LINK_RESIDUAL_CLIP_SECONDS, NO_LINK_RESIDUAL_CLIP_SECONDS)
    model = CatBoostRegressor(**STEP2_NO_LINK_SETTINGS)
    model.fit(training_features, clipped_training_residual, cat_features=categorical_columns)
    predictions = model.predict(held_out_features) + metadata.loc[held_out_rows, 'baseline'].to_numpy()
    report('step 2 no-link specialist', metadata.loc[held_out_rows, TARGET_COLUMN], predictions)
    return model, pd.Series(predictions, index=metadata.loc[held_out_rows, 'MVT_ID_mvt'].to_numpy())


def ranking_of_steps_1_and_2(residual_model: XGBRegressor, specialist_model: CatBoostRegressor) -> np.ndarray:
    """
    step 3 - it guesses every 2026 flight with the step-1 model and swaps in the step-2 guesses for the flights with no NM link
    """
    features, metadata = load_prior_tables(include_opdi_features=False)
    is_ranking = metadata.year.eq(2026)
    predictions = residual_model.predict(features.loc[is_ranking]).astype(float) + metadata.loc[is_ranking].baseline.to_numpy()
    features_with_opdi, _ = load_prior_tables(include_opdi_features=True)
    # Note: the specialist gets the category columns as pandas categories here (as built), not converted to str as in its fit.
    is_no_link_among_ranking = metadata.loc[is_ranking, 'AOBT_3_flt'].isna().to_numpy()     # mask over the ranking rows
    is_no_link_ranking = is_ranking & metadata.AOBT_3_flt.isna()                            # mask over all rows
    specialist_predictions = specialist_model.predict(features_with_opdi.loc[is_no_link_ranking])
    predictions[is_no_link_among_ranking] = specialist_predictions + metadata.loc[is_no_link_ranking, 'baseline'].to_numpy()
    smallest_label = float(metadata.loc[metadata.year.eq(2025), TARGET_COLUMN].min())
    return np.maximum(predictions, smallest_label)


# blending helper of steps 4, 6 and 8
def best_blend_weight(labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, weights: list[float]) -> float:
    """
    This function tries several mixing amounts of new and old guesses and returns the one with the smallest RMSE (the first one if two tie)
    """
    return min(weights, key=lambda weight: error_metrics(labels, weight * candidate + (1 - weight) * reference)['rmse'])


# step 4: weather and OPDI
def steps_1_and_2_held_out(step1_held_out: np.ndarray, step2_held_out: pd.Series, held_out_ids: np.ndarray) -> np.ndarray:
    """
    This function builds the old answer that step 4 compares itself with on the test months - step 1's guesses, replaced by step 2's on the no-link flights, never below -12 seconds
    """
    reference = pd.Series(step1_held_out, index=held_out_ids)
    reference.loc[step2_held_out.index] = step2_held_out
    return reference.reindex(held_out_ids).to_numpy().clip(PREDICTION_FLOOR_SECONDS)


def fit_weather_opdi(step1_held_out: np.ndarray, step2_held_out: pd.Series, steps12_ranking: np.ndarray) -> StepPredictions:
    """
    step 4 - it trains a new XGBoost model on the leftover that also uses weather, OPDI open flight data, queues and clock precision, judging early stopping in real seconds and blends it with the existing models
    """
    features, metadata = load_tree_features()
    is_training, is_held_out, is_ranking = month_split(metadata)
    # the reference: step 1 on the held-out months with step 2 on their no-link rows; steps 1 + 2 on the ranking rows
    reference_held_out = steps_1_and_2_held_out(step1_held_out, step2_held_out, metadata.loc[is_held_out, 'MVT_ID_mvt'].to_numpy())
    reference_ranking = steps12_ranking
    is_no_link_held_out = metadata.loc[is_held_out, 'AOBT_3_flt'].isna().to_numpy()      # mask over the held-out rows
    baseline = metadata.baseline.to_numpy()
    labels = metadata[TARGET_COLUMN].to_numpy()
    residual = labels - baseline
    held_out_labels = labels[is_held_out]

    # XGBoost names the metric after this function and infers the early-stopping direction from that name
    def seconds_rmse(held_out_residual: np.ndarray, predicted_residual: np.ndarray) -> float:
        return float(np.sqrt(np.mean((predicted_residual + baseline[is_held_out] - labels[is_held_out]) ** 2)))
    model = XGBRegressor(**STEP4_WEATHER_OPDI_SETTINGS, eval_metric=seconds_rmse)
    model.fit(features.loc[is_training], residual[is_training], eval_set=[(features.loc[is_held_out], residual[is_held_out])], verbose=False)
    held_out_predictions = (model.predict(features.loc[is_held_out]).astype(float) + baseline[is_held_out]).clip(PREDICTION_FLOOR_SECONDS)
    ranking_predictions = (model.predict(features.loc[is_ranking]).astype(float) + baseline[is_ranking]).clip(PREDICTION_FLOOR_SECONDS)
    # keep the no-link specialist (step 2) on its rows when the candidate's mean squared error there is larger
    no_link_labels = held_out_labels[is_no_link_held_out]
    candidate_error = np.mean((held_out_predictions[is_no_link_held_out] - no_link_labels) ** 2)
    specialist_error = np.mean((reference_held_out[is_no_link_held_out] - no_link_labels) ** 2)
    keep_specialist = candidate_error > specialist_error
    if keep_specialist:
        held_out_predictions[is_no_link_held_out] = reference_held_out[is_no_link_held_out]
        is_no_link_ranking = metadata.loc[is_ranking, 'AOBT_3_flt'].isna().to_numpy()
        ranking_predictions[is_no_link_ranking] = reference_ranking[is_no_link_ranking]
    chosen_weight = best_blend_weight(held_out_labels, held_out_predictions, reference_held_out, STEP4_BLEND_WEIGHTS)
    held_out_predictions = chosen_weight * held_out_predictions + (1 - chosen_weight) * reference_held_out
    ranking_predictions = chosen_weight * ranking_predictions + (1 - chosen_weight) * reference_ranking
    report('step 4 weather / OPDI', held_out_labels, held_out_predictions, best_iteration=int(model.best_iteration), weight=chosen_weight, specialist_kept=bool(keep_specialist))
    return StepPredictions(held_out_predictions, ranking_predictions)


# step 5: no-link models
def positions_within(movement_ids: pd.Series, all_movement_ids: pd.Series) -> np.ndarray:
    """
    This function finds where each wanted flight sits in a longer list of flights (if a flight is in the list twice, the later position wins)
    """
    position_of = dict(zip(all_movement_ids, np.arange(len(all_movement_ids))))
    return np.array([position_of[movement_id] for movement_id in movement_ids])


def no_link_group_candidates(no_link_features: pd.DataFrame, categorical_columns: list[str], clipped_residual: np.ndarray, baseline: np.ndarray, training_rows: pd.Series, held_out_rows: pd.Series, ranking_rows: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    This function builds step 5's CatBoost model for one group of no-link flights (Rome, or all other airports) and returns its guesses for the group's test and 2026 flights, never below -12 seconds
    """
    model = CatBoostRegressor(**STEP5_NO_LINK_SETTINGS)
    model.fit(no_link_features.loc[training_rows], clipped_residual[training_rows], cat_features=categorical_columns, sample_weight=None)
    candidate_held_out = (model.predict(no_link_features.loc[held_out_rows]) + baseline[held_out_rows]).clip(PREDICTION_FLOOR_SECONDS)
    candidate_ranking = (model.predict(no_link_features.loc[ranking_rows]) + baseline[ranking_rows]).clip(PREDICTION_FLOOR_SECONDS)
    return candidate_held_out, candidate_ranking


def fit_no_link(reference: StepPredictions) -> StepPredictions:
    """
    step 5 - it looks again only at the flights with no NM link, using a facts table made just for them
    It builds one model for Rome and one for all other airports, and for each picks the mixing amount (0, 0.5 or 1) with the old guesses that gives the smallest test mistake - flights with an NM link keep the old guesses
    """
    no_link_features = pd.read_parquet(PRIOR_TABLES / 'no_link_features.parquet')
    all_metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    metadata = all_metadata.loc[all_metadata.AOBT_3_flt.isna()].reset_index(drop=True)     # no-link rows, the rows of no_link_features
    is_training, is_held_out, is_ranking = month_split(metadata)
    labels = metadata[TARGET_COLUMN].to_numpy()
    baseline = metadata.baseline.to_numpy()
    clipped_residual = np.clip(labels - baseline, -NO_LINK_RESIDUAL_CLIP_SECONDS, NO_LINK_RESIDUAL_CLIP_SECONDS)
    categorical_columns = no_link_features.select_dtypes('category').columns.tolist()
    all_held_out = month_split(all_metadata)[1]
    # positions of the no-link held-out / ranking rows within the reference arrays (all held-out rows / all ranking rows)
    held_out_positions = positions_within(metadata.loc[is_held_out, 'MVT_ID_mvt'], all_metadata.loc[all_held_out, 'MVT_ID_mvt'])
    ranking_positions = positions_within(metadata.loc[is_ranking, 'MVT_ID_mvt'], all_metadata.loc[all_metadata.year.eq(2026), 'MVT_ID_mvt'])
    no_link_held_out = reference.held_out[held_out_positions].copy()
    no_link_ranking = reference.ranking[ranking_positions].copy()
    is_rome_row = metadata.ADEP_mvt.eq('LIRF').to_numpy()       # LIRF = Rome Fiumicino
    weights = {}
    for is_rome in [False, True]:
        in_group = is_rome_row == is_rome
        group_held_out = is_held_out & in_group
        candidate_held_out, candidate_ranking = no_link_group_candidates(no_link_features, categorical_columns, clipped_residual, baseline, is_training & in_group, group_held_out, is_ranking & in_group)
        in_group_among_held_out = in_group[is_held_out]
        in_group_among_ranking = in_group[is_ranking]
        group_reference_held_out = no_link_held_out[in_group_among_held_out]
        group_labels = labels[group_held_out]
        # only adopt a better specialist (weight 0 keeps the reference)
        weight = min(STEP5_BLEND_WEIGHTS, key=lambda candidate_weight: np.mean((candidate_weight * candidate_held_out + (1 - candidate_weight) * group_reference_held_out - group_labels) ** 2))
        no_link_held_out[in_group_among_held_out] = weight * candidate_held_out + (1 - weight) * group_reference_held_out
        no_link_ranking[in_group_among_ranking] = weight * candidate_ranking + (1 - weight) * no_link_ranking[in_group_among_ranking]
        weights['rome' if is_rome else 'other airports'] = weight
    held_out_predictions = reference.held_out.copy()
    ranking_predictions = reference.ranking.copy()
    held_out_predictions[held_out_positions] = no_link_held_out
    ranking_predictions[ranking_positions] = no_link_ranking
    report('step 5 no-link models', all_metadata.loc[all_held_out, TARGET_COLUMN], held_out_predictions, weights=weights)
    return StepPredictions(held_out_predictions, ranking_predictions)


# steps 6 and 8: airport models
def with_extra_tables(features: pd.DataFrame, extra_tables: tuple[str, ...]) -> pd.DataFrame:
    """
    This function glues the columns of other prior tables onto the facts table
    """
    for table_name in extra_tables:
        extra_table = pd.read_parquet(PRIOR_TABLES / f'{table_name}.parquet')
        assert len(extra_table) == len(features) and extra_table.index.equals(features.index)
        features = pd.concat([features, extra_table], axis=1)
    return features


def airport_residual_model(depth: int) -> XGBRegressor:
    """
    This function makes a fresh, untrained XGBoost model with the settings for one airport in steps 6 and 8
    """
    return XGBRegressor(**AIRPORT_MODEL_SETTINGS, max_depth=depth)


def fit_airport_models(reference: StepPredictions, step: str, extra_tables: tuple[str, ...] = (), airports: list[str] | None = None, depth: int = AIRPORT_MODEL_DEPTH) -> StepPredictions:
    """
    steps 6 and 8 - it trains a separate XGBoost model for each airport, only on flights with an NM link
    """
    features, metadata = load_tree_features()
    features = with_extra_tables(features, extra_tables)
    is_training, is_held_out, is_ranking = month_split(metadata)
    has_nm_link = metadata.AOBT_3_flt.notna()
    is_training = is_training & has_nm_link
    held_out_predictions = reference.held_out.copy()
    ranking_predictions = reference.ranking.copy()
    labels = metadata[TARGET_COLUMN].to_numpy()
    baseline = metadata.baseline.to_numpy()
    residual = labels - baseline
    weights = {}
    for airport in airports or sorted(metadata.ADEP_mvt.unique()):
        in_group = metadata.ADEP_mvt.eq(airport) & has_nm_link
        group_training = is_training & in_group
        group_held_out = is_held_out & in_group
        group_ranking = is_ranking & in_group
        model = airport_residual_model(depth)
        model.fit(features.loc[group_training], residual[group_training], eval_set=[(features.loc[group_held_out], residual[group_held_out])], verbose=False)
        candidate_held_out = (baseline[group_held_out] + model.predict(features.loc[group_held_out])).clip(PREDICTION_FLOOR_SECONDS)
        candidate_ranking = (baseline[group_ranking] + model.predict(features.loc[group_ranking])).clip(PREDICTION_FLOOR_SECONDS)
        in_group_among_held_out = in_group[is_held_out].to_numpy()
        in_group_among_ranking = in_group[is_ranking].to_numpy()
        group_reference_held_out = held_out_predictions[in_group_among_held_out]
        weight = best_blend_weight(labels[group_held_out], candidate_held_out, group_reference_held_out, AIRPORT_BLEND_WEIGHTS)
        held_out_predictions[in_group_among_held_out] = weight * candidate_held_out + (1 - weight) * group_reference_held_out
        group_reference_ranking = ranking_predictions[in_group_among_ranking]
        ranking_predictions[in_group_among_ranking] = weight * candidate_ranking + (1 - weight) * group_reference_ranking
        weights[airport] = weight
    report(step, labels[is_held_out], held_out_predictions, weights=weights)
    return StepPredictions(held_out_predictions, ranking_predictions)


# step 7: neural
@dataclass
class NetworkInputs:
    """
    This data structure holds step 7's inputs for the neural network
    """
    numeric: torch.Tensor          # float32: the standardised numeric features, then one missing-value flag (0 / 1) per feature
    categorical: torch.Tensor      # int64: one code per categorical column (0 = a level outside the training vocabulary)
    vocabulary_sizes: list[int]    # number of codes of each categorical column


def category_codes(features: pd.DataFrame, categorical_columns: list[str], is_training: pd.Series) -> tuple[np.ndarray, list[int]]:
    """
    This function turns category columns (like the airline) into whole-number codes for the network
    """
    codes = []
    vocabulary_sizes = []
    for column in categorical_columns:
        level_counts = features.loc[is_training, column].value_counts()
        levels = level_counts[level_counts >= MIN_CATEGORY_LEVEL_COUNT].index.astype(str).tolist()
        vocabulary_sizes.append(len(levels) + 1)
        level_code = {level: position + 1 for position, level in enumerate(levels)}
        codes.append(features[column].astype(str).map(level_code).fillna(0).to_numpy('int64'))
    return np.column_stack(codes), vocabulary_sizes


def load_network_inputs() -> tuple[NetworkInputs, pd.DataFrame]:
    """
    This function prepares step 7's inputs
    """
    features, metadata = load_tree_features()
    is_training, _, _ = month_split(metadata)
    is_training = is_training & metadata.AOBT_3_flt.notna()
    training_rows = np.flatnonzero(is_training)
    categorical_columns = features.select_dtypes('category').columns.tolist()
    numeric_columns = [column for column in features if column not in categorical_columns]
    codes, vocabulary_sizes = category_codes(features, categorical_columns, is_training)
    numeric_values = features[numeric_columns].to_numpy('float32')
    del features
    is_missing = ~np.isfinite(numeric_values)
    numeric_values = np.sign(numeric_values) * np.log1p(np.abs(numeric_values))
    numeric_values[is_missing] = np.nan
    mean = np.nanmean(numeric_values[training_rows], axis=0)
    std = np.nanstd(numeric_values[training_rows], axis=0)
    std = np.maximum(std, NUMERIC_STD_FLOOR)
    numeric_values = np.nan_to_num((numeric_values - mean) / std, nan=0, posinf=0, neginf=0).clip(-NUMERIC_INPUT_CLIP, NUMERIC_INPUT_CLIP)
    numeric_inputs = np.concatenate([numeric_values, is_missing.astype('float32')], axis=1)
    return NetworkInputs(torch.from_numpy(numeric_inputs), torch.from_numpy(codes), vocabulary_sizes), metadata


def embedding_network(numeric_width: int, vocabulary_sizes: list[int]) -> torch.nn.Module:
    """
    This function builds a brand-new, untrained step-7 network
    """

    class EmbeddingNetwork(nn.Module):
        """
        This is a class that describes the step-7 network - each category column gets an embedding (a small learned list of numbers per code), which goes with the number inputs through layers of 192 and 96 and comes out as the guessed leftover divided by 300 seconds
        While training, 10% of the first layer's signals are switched off at random (dropout)
        """

        def __init__(self, numeric_width: int, vocabulary_sizes: list[int]) -> None:
            """
            This function sets up the network's parts - each embedding is between 2 and 12 numbers long, and the last layer starts at zero so the network first guesses exactly the baseline
            """
            super().__init__()
            # embedding width: 2 x the fourth root of the vocabulary size (rounded down), between 2 and 12
            embedding_widths = [min(MAX_EMBEDDING_WIDTH, max(MIN_EMBEDDING_WIDTH, int(size ** EMBEDDING_WIDTH_EXPONENT) * EMBEDDING_WIDTH_FACTOR)) for size in vocabulary_sizes]
            self.embeddings = nn.ModuleList([nn.Embedding(size, width) for size, width in zip(vocabulary_sizes, embedding_widths)])
            self.layers = nn.Sequential(nn.Linear(numeric_width + sum(embedding_widths), FIRST_HIDDEN_WIDTH), nn.ReLU(), nn.Dropout(NETWORK_DROPOUT), nn.Linear(FIRST_HIDDEN_WIDTH, SECOND_HIDDEN_WIDTH), nn.ReLU(), nn.Linear(SECOND_HIDDEN_WIDTH, 1))
            # zero output layer: the network starts by predicting the baseline exactly
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

        def forward(self, numeric_inputs: torch.Tensor, categorical_codes: torch.Tensor) -> torch.Tensor:
            """
            This function runs a batch of flights through the network and gives one number per flight
            """
            embedded = [embedding(categorical_codes[:, column]) for column, embedding in enumerate(self.embeddings)]
            return self.layers(torch.cat([numeric_inputs] + embedded, dim=1)).squeeze(1)

    return EmbeddingNetwork(numeric_width, vocabulary_sizes)


def train_one_epoch(network: torch.nn.Module, optimizer: torch.optim.Optimizer, inputs: NetworkInputs, scaled_target: torch.Tensor, epoch_order: np.ndarray) -> None:
    network.train()
    for start in range(0, len(epoch_order), TRAINING_BATCH_SIZE):
        batch = epoch_order[start:start + TRAINING_BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        loss = ((network(inputs.numeric[batch], inputs.categorical[batch]) - scaled_target[batch]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), GRADIENT_NORM_CLIP)
        optimizer.step()


def network_predictions(network: torch.nn.Module, inputs: NetworkInputs, baseline: np.ndarray, rows: np.ndarray) -> np.ndarray:
    network.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(rows), PREDICTION_BATCH_SIZE):
            batch = rows[start:start + PREDICTION_BATCH_SIZE]
            parts.append(network(inputs.numeric[batch], inputs.categorical[batch]).numpy() * TARGET_UNIT_SECONDS + baseline[batch])
    return np.concatenate(parts).clip(PREDICTION_FLOOR_SECONDS)


def best_linked_blend(held_out_labels: np.ndarray, reference_held_out: np.ndarray, network_held_out: np.ndarray, is_linked_among_held_out: np.ndarray) -> tuple[float, float]:
    """
    This function tries mixing the network's guesses into the old guesses on the NM-linked test flights (amounts 0.1 to 1) and returns the best score and its amount
    """
    blended_held_out = reference_held_out.copy()
    candidates = []
    for weight in NETWORK_BLEND_WEIGHTS:
        blended_held_out[is_linked_among_held_out] = weight * network_held_out + (1 - weight) * reference_held_out[is_linked_among_held_out]
        candidates.append((error_metrics(held_out_labels, blended_held_out)['rmse'], weight))
    return min(candidates)


def blend_on_linked_rows(reference: StepPredictions, weight: float, network_held_out: np.ndarray, network_ranking: np.ndarray, is_linked_among_held_out: np.ndarray, is_linked_among_ranking: np.ndarray) -> StepPredictions:
    """
    This function makes step 7's answer
    """
    held_out_predictions = reference.held_out.copy()
    ranking_predictions = reference.ranking.copy()
    held_out_predictions[is_linked_among_held_out] = (weight * network_held_out + (1 - weight) * reference.held_out[is_linked_among_held_out])
    ranking_predictions[is_linked_among_ranking] = (weight * network_ranking + (1 - weight) * reference.ranking[is_linked_among_ranking])
    return StepPredictions(held_out_predictions, ranking_predictions)


def fit_neural(reference: StepPredictions, epochs: int = NETWORK_MAX_EPOCHS) -> StepPredictions:
    """
    This function is step 7 - it trains the network on flights with an NM link to guess the leftover
    """
    # seeds before anything that consumes randomness (network initialisation, dropout, epoch order)
    torch.set_num_threads(NETWORK_THREADS)
    torch.manual_seed(NETWORK_SEED)
    epoch_rng = np.random.default_rng(NETWORK_SEED)
    inputs, metadata = load_network_inputs()
    is_training, is_held_out, is_ranking = month_split(metadata)
    has_nm_link = metadata.AOBT_3_flt.notna()
    training_rows = np.flatnonzero(is_training & has_nm_link)
    linked_held_out_rows = np.flatnonzero(is_held_out & has_nm_link)
    linked_ranking_rows = np.flatnonzero(is_ranking & has_nm_link)
    is_linked_among_held_out = metadata.loc[is_held_out, 'AOBT_3_flt'].notna().to_numpy()
    baseline = metadata.baseline.to_numpy()
    labels = metadata[TARGET_COLUMN].to_numpy()
    held_out_labels = labels[is_held_out]
    scaled_target = torch.from_numpy(np.nan_to_num((labels - baseline).clip(-NETWORK_RESIDUAL_CLIP_SECONDS, NETWORK_RESIDUAL_CLIP_SECONDS) / TARGET_UNIT_SECONDS).astype('float32'))
    network = embedding_network(inputs.numeric.shape[1], inputs.vocabulary_sizes)
    optimizer = torch.optim.AdamW(network.parameters(), **NETWORK_OPTIMIZER_SETTINGS)
    best_rmse = error_metrics(held_out_labels, reference.held_out)['rmse']
    best_state = None
    best_weight = 0.
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        train_one_epoch(network, optimizer, inputs, scaled_target, epoch_rng.permutation(training_rows))
        network_held_out = network_predictions(network, inputs, baseline, linked_held_out_rows)
        epoch_rmse, weight = best_linked_blend(held_out_labels, reference.held_out, network_held_out, is_linked_among_held_out)
        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            best_weight = weight
            best_state = copy.deepcopy(network.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= NETWORK_PATIENCE_EPOCHS:
            break
    if best_state is None:
        print('prior step 7 neural: no improvement over the reference', flush=True)
        return reference
    network.load_state_dict(best_state)
    is_linked_among_ranking = metadata.loc[is_ranking, 'AOBT_3_flt'].notna().to_numpy()
    network_held_out = network_predictions(network, inputs, baseline, linked_held_out_rows)
    network_ranking = network_predictions(network, inputs, baseline, linked_ranking_rows)
    step = blend_on_linked_rows(reference, best_weight, network_held_out, network_ranking, is_linked_among_held_out, is_linked_among_ranking)
    report('step 7 neural', held_out_labels, step.held_out, weight=best_weight, epochs=epochs_run)
    return step


# the chain
def steps_1_to_3() -> tuple[np.ndarray, pd.Series, np.ndarray]:
    residual_model, step1_held_out = fit_residual()
    specialist_model, step2_held_out = fit_no_link_specialist()
    steps12_ranking = ranking_of_steps_1_and_2(residual_model, specialist_model)
    return step1_held_out, step2_held_out, steps12_ranking


def prior_ranking_predictions() -> pd.Series:
    step1_held_out, step2_held_out, steps12_ranking = steps_1_to_3()
    step = fit_weather_opdi(step1_held_out, step2_held_out, steps12_ranking)
    step = fit_no_link(step)
    step = fit_airport_models(step, 'step 6 airport models')
    step = fit_neural(step)
    step = fit_airport_models(step, 'step 8 airport models with ground events', extra_tables=('ground_event_features',), airports=GROUND_EVENT_AIRPORTS)
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet', columns=['MVT_ID_mvt', 'year'])
    return pd.Series(step.ranking, index=metadata.loc[metadata.year.eq(2026), 'MVT_ID_mvt'].to_numpy(), name='prediction')


def held_out_no_link_prior(held_out_months: tuple[int, ...]) -> pd.Series:
    """
    This function makes guesses for chosen 2025 months
    """
    saved_months = dict(MONTH_SPLIT)
    MONTH_SPLIT.update(held_out=tuple(held_out_months), not_trained_on=tuple(held_out_months))
    try:
        step1_held_out, step2_held_out, steps12_ranking = steps_1_to_3()
        step = fit_no_link(fit_weather_opdi(step1_held_out, step2_held_out, steps12_ranking))
        metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet', columns=['MVT_ID_mvt', 'year', 'month'])
        held_out = month_split(metadata)[1]
        return pd.Series(step.held_out, index=metadata.loc[held_out, 'MVT_ID_mvt'].to_numpy(), name='prior_oof')
    finally:
        MONTH_SPLIT.clear()
        MONTH_SPLIT.update(saved_months)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the prior model.')
    parser.add_argument('--held-out-months', default='', help='comma-separated 2025 months: write the out-of-fold prior of those months')
    parser.add_argument('--out', default='', help='output parquet of --held-out-months (MVT_ID_mvt, prior_oof)')
    arguments = parser.parse_args()
    if arguments.held_out_months:
        months = tuple(int(month) for month in arguments.held_out_months.split(','))
        predictions = held_out_no_link_prior(months)
        predictions.rename_axis('MVT_ID_mvt').reset_index().to_parquet(arguments.out, index=False)
        print('out-of-fold prior, months', months, ':', len(predictions), 'rows ->', arguments.out, flush=True)
        return
    predictions = prior_ranking_predictions()
    print('prior ranking predictions:', len(predictions), 'rows, mean', round(float(predictions.mean()), 1))


if __name__ == '__main__':
    main()
