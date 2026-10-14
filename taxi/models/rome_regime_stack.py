"""
This file adds a small second-opinion model for one tricky group: Rome flights without an NM link (no matching flight plan) that took off more than 30 minutes after their timetable time
For these the big question is how the time was written down - P (copied from the timetable), S (yesterday's date, so a day too long) or N (normal)
The pipeline already gives chances for these, and this file learns from 2025 how to improve them with extra clues, mainly the turnaround (how long the plane stood on the ground between its previous landing and this flight)

This is how it works:
-the P / S / N model, for every such flight:
    -a logistic regression (a simple model that turns a weighted sum of clues into a chance) gives all three chances from the pipeline's chances, the lateness, the Rome date change, the date window (14.05 h), the flight's recording habit and the turnaround
    -a plane that rushed off soon after a late arrival was a copy far more often than one that had stood for hours
-it is trained on the 2025 out-of-fold table (guesses from models that never saw that flight), and flights where a wrong answer would cost a lot count more
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from taxi.features import airframe_turnaround
from taxi.main import OPDI_FOLDER
from taxi.prediction.late_rome_departures import RecordingHabit, rome_departures

MINIMUM_DELAY_SECONDS = 1800.0            # rows more than 30 min after schedule
DATE_WINDOW_HOURS = 14.05                 # first previous-date record beyond it (taxi.prediction.late_rome_departures)
MSCHED_CAP_HOURS = (3.0, 12.45)           # same Rome-local date and this late: the label never exceeded msched (late-Rome rule)
OUTAGE_SHARE = 0.5                        # an OPDI outage day has fewer flights than this share of its month's median day
TURNAROUND_COLUMNS = slice(10, 16)        # the turnaround inputs of next_day_inputs (columns 10-15)
MISSING_TURNAROUND_COLUMN = 14            # next_day_inputs column: turnaround unknown (1.0) or known (0.0)
# turnaround inputs
UNKNOWN_TURNAROUND_SECONDS = -3600.0      # an unknown turnaround becomes -1 h, a value no real turnaround has
TURNAROUND_FLOOR_HOURS = -1               # lower clip of the turnaround (h): the -1 h of an unknown turnaround (same reason)
TURNAROUND_CEILING_HOURS = 12             # upper clip of the turnaround (h): a round half day, set when the model was built
TURNAROUND_FIRST_CAP_HOURS = 1            # bends of the turnaround curve at 1, 2 and 4 h: round doubling steps
TURNAROUND_SECOND_CAP_HOURS = 2           # (second bend, see the line above)
TURNAROUND_THIRD_CAP_HOURS = 4            # (third bend)
LONG_TURNAROUND_HOURS = 8                 # flag for a plane that stood 8 h or more, about a night stop
# delay inputs of the P / S / N model (h)
DELAY_CEILING_HOURS = 36                  # upper clip of the delay: a day and a half
DELAY_FIRST_CAP_HOURS = 3                 # bend at 3 h, the same boundary as the late-Rome msched cap (MINIMUM_DELAY_HOURS)
DELAY_SECOND_CAP_HOURS = 8                # bend at 8 h, the same boundary as the late-Rome overrides (OVERRIDE_DELAY_HOURS)
# stakes (the row weights of the fit)
STAKE_FLOOR_SECONDS = 300                 # s: the same floor as the stake weights of the regime classifiers
REGIME_MODEL_STAKE_CAP_SECONDS = 30000    # s: keeps a few one-day mix-ups (about 86,400 s) from ruling the fit
PLAIN_TOLERANCE_SECONDS = 1               # s: the same 'untouched by the rules and caps' test as in build_submission
# scikit-learn settings of the P / S / N model
REGIME_MODEL_SETTINGS = {
    'C': 1.0,             # scikit-learn's default
    'max_iter': 5000,     # far above the default of 100, so the lbfgs solver has room to converge
}


def turnaround_hours(turn_seconds: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(turn_seconds, nan=UNKNOWN_TURNAROUND_SECONDS) / 3600, TURNAROUND_FLOOR_HOURS, TURNAROUND_CEILING_HOURS)


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-4, 1 - 1e-4)      # keeps the logit finite (within about +-9.2)
    return np.log(probability / (1 - probability))


def next_day_inputs(probability_copy: np.ndarray, probability_previous: np.ndarray, delay_hours: np.ndarray, later_local_date: np.ndarray, recorded_share: np.ndarray, turn_seconds: np.ndarray) -> np.ndarray:
    """
    This function builds the 16 clues of the P / S / N model for each flight
    """
    delay = np.clip(delay_hours, 0, DELAY_CEILING_HOURS)
    hours = turnaround_hours(turn_seconds)
    beyond = (delay >= DATE_WINDOW_HOURS).astype(float)             # beyond the date window: an S record needs the recording habit
    later = later_local_date.astype(float)
    # column 1 adds 1e-3 so that log(pS) stays finite when pS is 0
    return np.column_stack([
        logit(probability_copy),                                  # 0
        np.log(np.clip(probability_previous, 0, 1) + 1e-3),       # 1
        delay,                                                    # 2  delay (h), clipped to 0..36
        np.minimum(delay, DELAY_FIRST_CAP_HOURS),                 # 3
        np.minimum(delay, DELAY_SECOND_CAP_HOURS),                # 4
        np.minimum(delay, DATE_WINDOW_HOURS),                     # 5
        later,                                                    # 6  later Rome-local date
        later * beyond,                                           # 7
        recorded_share,                                           # 8
        recorded_share * later * beyond,                          # 9
        hours,                                                    # 10 turnaround (h), -1 when unknown
        np.minimum(hours, TURNAROUND_FIRST_CAP_HOURS),            # 11
        np.minimum(hours, TURNAROUND_SECOND_CAP_HOURS),           # 12
        np.minimum(hours, TURNAROUND_THIRD_CAP_HOURS),            # 13
        np.isnan(turn_seconds).astype(float),                     # 14 turnaround unknown
        (hours >= LONG_TURNAROUND_HOURS) * later])                # 15 stood 8 h or more, later local date


def rome_context() -> tuple[pd.DataFrame, pd.Series]:
    rome = rome_departures()
    habit = RecordingHabit(rome)
    share = [habit.share(flight, operator, leave_out=(regime != 'P') if year == 2025 else None) for flight, operator, regime, year in zip(rome.flight, rome.operator, rome.regime, rome.year)]
    rome = rome.set_index('MVT_ID_mvt')
    return rome, pd.Series(share, index=rome.index)


def later_local_date_of(rome: pd.DataFrame, movement_ids: np.ndarray) -> np.ndarray:
    """
    This function looks up, for each flight id, whether it took off on a later Rome date than planned
    """
    return rome.later_local_date.reindex(movement_ids).fillna(False).to_numpy().astype(bool)


def late_no_link_rome_rows(components: pd.DataFrame) -> np.ndarray:
    """
    This function picks the 2025 rows that are Rome departures without an NM link that took off more than 30 minutes after their timetable time
    """
    msched = components.msched.to_numpy()
    return components.is_rome.to_numpy() & components.no_link.to_numpy() & np.isfinite(msched) & (msched > MINIMUM_DELAY_SECONDS)


def fit_regime_model(components: pd.DataFrame) -> LogisticRegression:
    """
    This function trains the P / S / N model on every 2025 Rome no-link flight more than 30 minutes late
    Each flight counts by its stake: the bigger of |msched - v| and, where an S is possible, the cost of a one-day mix-up, kept between 300 and 30,000 s and squared
    """
    turnaround, _ = airframe_turnaround.load_turnaround_table()
    rome, share = rome_context()
    rows = late_no_link_rome_rows(components)
    ids = components.MVT_ID_mvt.to_numpy()[rows]
    later = later_local_date_of(rome, ids)
    inputs = next_day_inputs(components.pP.to_numpy()[rows], components.pS.to_numpy()[rows], components.msched.to_numpy()[rows] / 3600, later, share.reindex(ids).to_numpy(), turnaround.turn.reindex(ids).to_numpy())
    regime = components.regime.to_numpy()[rows]
    classes = np.select([regime == 'P', regime == 'S'], [0, 1], 2)
    value, msched = components.v.to_numpy()[rows], components.msched.to_numpy()[rows]
    day_stake = np.where(later | components.next_day.to_numpy()[rows], 86400 + value - np.minimum(msched, 86400), 0)
    stake = np.clip(np.maximum(np.abs(msched - value), day_stake), STAKE_FLOOR_SECONDS, REGIME_MODEL_STAKE_CAP_SECONDS) ** 2
    return LogisticRegression(**REGIME_MODEL_SETTINGS).fit(inputs, classes, sample_weight=stake / stake.mean())


def opdi_outage_days() -> set[str]:
    """
    This function finds the 2026 days when OPDI's flight list is likely broken/missing - days with fewer than half the flights of their month's median day
    """
    flight_lists = sorted(OPDI_FOLDER.glob('flight_list_2026*.parquet'))
    first_seen = pd.concat([pd.read_parquet(path, columns=['first_seen']) for path in flight_lists]).first_seen
    counts = pd.to_datetime(first_seen, utc=True).dt.strftime('%Y-%m-%d').value_counts()
    days = pd.date_range(counts.index.min()[:7] + '-01', counts.index.max(), freq='D').strftime('%Y-%m-%d')
    counts = counts.reindex(days, fill_value=0)
    month_median = counts.groupby(counts.index.str[:7]).transform('median')
    return set(counts.index[counts < OUTAGE_SHARE * month_median])


def ranking_turnaround(parts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    This function finds each flight's turnaround: from OPDI when it is known, otherwise from the ADS-B record of the plane's previous landing, shifted by the gap measured in 2025
    """
    turnaround, offsets = airframe_turnaround.load_turnaround_table()
    table = turnaround.reindex(parts.MVT_ID_mvt)
    turn = np.where(table.turn.notna(), table.turn, -table.ads_prev_fast_rel + offsets['turn'])
    takeoff_day = pd.to_datetime(parts.takeoff, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    on_outage_day = pd.Series(takeoff_day).isin(opdi_outage_days()).to_numpy()
    return turn, on_outage_day & ~np.isfinite(turn)


def regime_model_inputs(part: pd.DataFrame, turn: np.ndarray, outage_without_turnaround: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function builds the 16 clues for the 2026 flights
    """
    rome, share = rome_context()
    ids = part.MVT_ID_mvt.to_numpy()
    later = later_local_date_of(rome, ids)
    inputs = next_day_inputs(part.pP.to_numpy(), part.pS.to_numpy(), part.msched.to_numpy() / 3600, later, share.reindex(ids).to_numpy(), turn)
    known = np.isfinite(turn)
    if outage_without_turnaround.any():
        inputs[outage_without_turnaround, TURNAROUND_COLUMNS] = inputs[known, TURNAROUND_COLUMNS].mean(0)
        inputs[outage_without_turnaround, MISSING_TURNAROUND_COLUMN] = 0.0
    return inputs, later


def regime_model_values(parts: pd.DataFrame, model: LogisticRegression, strength: float) -> pd.Series:
    """
    This function uses the P / S / N model on 2026 Rome no-link flights more than 30 minutes late: it moves the P and S chances 'strength' of the way toward the model's and rebuilds the guess
    """
    rows = (parts.airport.to_numpy() == 'LIRF') & (parts.msched.to_numpy() > MINIMUM_DELAY_SECONDS)
    part = parts[rows]
    turn, outage_without_turnaround = ranking_turnaround(part)
    ids = part.MVT_ID_mvt.to_numpy()
    inputs, later = regime_model_inputs(part, turn, outage_without_turnaround)
    probability = np.zeros((len(part), 3))           # P, S, N; a class the model never saw keeps probability 0
    probability[:, model.classes_] = model.predict_proba(inputs)
    p_copy = part.pP.to_numpy() + strength * (probability[:, 0] - part.pP.to_numpy())
    p_previous = part.pS.to_numpy() + strength * (probability[:, 1] - part.pS.to_numpy())
    msched, value = part.msched.to_numpy(), part.v.to_numpy()
    mixture = p_copy * msched + p_previous * (86400 + value) + (1 - p_copy - p_previous) * value
    later_utc_day = part.day_difference.to_numpy() >= 1               # the late-day cap and the msched cap act on these rows only
    cap = np.where(later_utc_day, np.maximum(msched, 86400 + value), np.inf)
    lane = np.minimum(part.model_weight.to_numpy() * mixture + (1 - part.model_weight.to_numpy()) * part.prior.to_numpy(), cap)
    same_date_late = later_utc_day & ~later & (msched >= MSCHED_CAP_HOURS[0] * 3600) & (msched <= MSCHED_CAP_HOURS[1] * 3600)
    lane = np.where(same_date_late, np.minimum(lane, msched), lane)
    plain = np.abs(part.submitted_before_corrections.to_numpy() - np.minimum(part.lane_pred.to_numpy(), cap)) < PLAIN_TOLERANCE_SECONDS
    return pd.Series(lane[plain], index=ids[plain])
