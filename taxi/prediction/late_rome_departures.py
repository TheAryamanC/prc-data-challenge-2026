"""
This file looks at only a handful of flights: 2026 departures from Rome Fiumicino (LIRF) with no NM link (the organizers could not match them to their flight plan) that took off at least 3 hours after their timetable time
These few flights matter a lot, because a guess that is wrong by a whole day makes a huge error
Such a flight's answer can be one of three things: P (the timetable time was copied, so the answer is take-off minus timetable time, called msched), S (the real time was written with the timetable's Rome date, so the answer is a whole day, 86,400 seconds, plus the taxi time) or N (a normal, correct record)

This is how it works:
-measure clues from the 2025 Rome flights:
    1. how late a flight could be and still be recorded correctly (about 12.45 h), and where the first yesterday's-date record beyond that lies (about 14.05 h)
    2. how often each kind of record shows up in each group of flights
    3. each flight's (or airline's) habit of having its time written down
-for flights 8 hours or more late, work out the chances of P, S and N from these clues and move the guess part of the way toward the expected answer
-for flights still on the same Rome date that are 3 to about 12.45 hours late, never let the guess go above msched (no such 2025 flight had a bigger answer)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.models.regime_mixture import flight_and_operator_keys
from taxi.main import competition_columns, departure_frame

LATE_ROME_TRANSFER_FACTOR = 0.5    # from the 2025 out-of-fold checks (0.12, 0.27, 0.5, 1.0 tried): 0.5 helped in all three (cross-fitted, Jan-Jun -> Jul-Dec, the reverse), 1.0 lost in the reverse
MINIMUM_DELAY_HOURS = 3.0          # the msched cap: same-date rows from here on never had a label above msched (2-3 h: 3 of 341 did)
OVERRIDE_DELAY_HOURS = 8.0         # the overrides: from here on no-link rows carry no correct block
MINIMUM_FLIGHT_DEPARTURES = 5      # a flight with fewer 2025 Rome departures is judged by its operator
RARE_OPERATOR_DEPARTURES = 30      # operators below this many 2025 Rome departures give the share for unknown operators
PREVIOUS_DATE_OFFSET = 86400.0     # seconds a previous-date (S) record adds to the taxi time: one day
JEFFREYS_PRIOR_EVENTS = 0.5        # pseudo-events of the Jeffreys prior: 0 of 0 gives 0.5, and no share is exactly 0 or 1 (far too sure for tiny groups)
JEFFREYS_PRIOR_ROWS = 1            # pseudo-rows of the same prior (half an event and half a non-event)
MINIMUM_OPERATOR_DEPARTURES = 1    # an operator with any 2025 Rome departure left (after leaving the row out) uses its own share


def jeffreys_share(events: float, rows: float) -> float:
    """
    This function guesses a share carefully from few examples
    """
    return float((events + JEFFREYS_PRIOR_EVENTS) / (rows + JEFFREYS_PRIOR_ROWS))


def rome_local_dates(utc_times: pd.Series) -> pd.Series:
    """
    This function turns UTC times into the date on a calendar in Rome
    """
    return utc_times.dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()


def rome_departures() -> pd.DataFrame:
    """
    This function builds the main table: every Rome departure of 2025 and of the ranking file, one row each
    For each flight it notes whether it has an NM link, how it was recorded (P, S or N; empty for 2026), how late it took off, whether the Rome date changed, and its flight number and airline
    """
    frame = departure_frame()
    rome = frame[frame.is_rome].reset_index(drop=True)
    # MVT_TIME_UTC_mvt = take-off time, SCHED_TIME_UTC_mvt = scheduled time (UTC)
    times = competition_columns(['SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt'], rome.MVT_ID_mvt)
    local_days_after_schedule = (rome_local_dates(times['MVT_TIME_UTC_mvt']) - rome_local_dates(times['SCHED_TIME_UTC_mvt'])).dt.days
    later_local_date = (local_days_after_schedule >= 1).to_numpy()
    flights, operators = flight_and_operator_keys(rome)
    regime = np.select([rome.is_schedule_copy.to_numpy(), rome.is_previous_date.to_numpy()], ['P', 'S'], 'N').astype(object)
    regime[rome.TAXITIME_SEC_mvt.isna().to_numpy()] = None          # no label (ranking rows): no regime
    return pd.DataFrame({'MVT_ID_mvt': rome.MVT_ID_mvt.to_numpy(), 
                         'year': rome.year.to_numpy(), 
                         'no_link': rome.no_nm_link.to_numpy(),
                         'regime': regime, 
                         'msched': rome.seconds_after_schedule.to_numpy(),
                         'delay_hours': rome.seconds_after_schedule.to_numpy() / 3600, 
                         'later_local_date': later_local_date,
                         'flight': flights, 
                         'operator': operators})


class RecordingHabit:
    """
    This is a class that counts, for each flight number and each airline, how often the real leaving time was written down in 2025 (S or N) instead of copied from the timetable (P)
    """

    def __init__(self, rome: pd.DataFrame) -> None:
        """
        This function fills the counts from the 2025 Rome flights that have a flight number, per flight and per airline
        """
        past = rome[(rome.year == 2025) & (rome.flight != '')].assign(recorded=lambda frame: frame.regime != 'P')
        self.flights = past.groupby('flight').recorded.agg(['sum', 'size'])
        self.operators = past.groupby('operator').recorded.agg(['sum', 'size'])
        rare = self.operators[self.operators['size'] < RARE_OPERATOR_DEPARTURES]
        self.unknown = jeffreys_share(rare['sum'].sum(), rare['size'].sum())

    def share(self, flight: str, operator: str, leave_out: bool | None = None) -> float:
        """
        This function gives the recording share of one flight
        """
        for table, key, minimum_rows in ((self.flights, flight, MINIMUM_FLIGHT_DEPARTURES), (self.operators, operator, MINIMUM_OPERATOR_DEPARTURES)):
            if key in table.index:
                recorded, rows = table.loc[key, 'sum'], table.loc[key, 'size']
                if leave_out is not None:
                    recorded, rows = recorded - int(leave_out), rows - 1
                if rows >= minimum_rows:
                    return jeffreys_share(recorded, rows)
        return self.unknown


def regime_share(rows: pd.DataFrame, regime: str) -> float:
    """
    This function gives the Jeffreys share of rows in a group that were recorded one way (P, S or N)
    """
    return jeffreys_share(int((rows.regime == regime).sum()), len(rows))


def mechanism_evidence(rome: pd.DataFrame, habit: RecordingHabit) -> dict[str, float]:
    """
    This function measures the seven clue numbers the corrections need from 2025
    """
    past = rome[rome.year == 2025]
    latest_correct = float(past.delay_hours[~past.no_link & (past.regime == 'N')].max())
    previous_date_delays = past.delay_hours[past.regime == 'S']
    first_previous_date = float(previous_date_delays[previous_date_delays > latest_correct].min())
    no_link = past[past.no_link]
    inside = no_link[no_link.later_local_date & (no_link.delay_hours <= latest_correct)]
    zone = no_link[no_link.later_local_date & (no_link.delay_hours > latest_correct) & (no_link.delay_hours < first_previous_date)]
    beyond = no_link[no_link.later_local_date & (no_link.delay_hours >= first_previous_date)]
    # each row's own record is left out of its habit (leave-one-out), so the calibration is not fitted on itself
    expected_previous_date = sum(habit.share(row.flight, row.operator, leave_out=row.regime != 'P') for row in beyond.itertuples())
    overridden_before_window = no_link[(no_link.delay_hours >= OVERRIDE_DELAY_HOURS) & (no_link.delay_hours < first_previous_date)]
    return {'latest_correct_hours': latest_correct,
            'first_previous_date_hours': first_previous_date,
            'ordinary_share': regime_share(overridden_before_window, 'N'),
            'ordinary_share_beyond': regime_share(past[past.delay_hours > latest_correct], 'N'),
            'previous_date_share_inside': regime_share(inside, 'S'),
            'previous_date_share_zone': regime_share(zone, 'S'),
            'recorded_calibration': float((beyond.regime == 'S').sum() / expected_previous_date)}


def regime_probabilities(delay_hours: float, later_local_date: bool, evidence: dict[str, float], recorded_share: float) -> dict[str, float]:
    """
    This function gives the chances of P, S and N for one very late no-link flight, from the group it falls in: same Rome date (no S possible), a later Rome date up to the latest correct record, the unresolved zone up to the first S record, or beyond that
    """
    if not later_local_date:
        ordinary, previous_date = evidence['ordinary_share'], 0.0
    elif delay_hours <= evidence['latest_correct_hours']:
        ordinary, previous_date = evidence['ordinary_share'], evidence['previous_date_share_inside']
    elif delay_hours < evidence['first_previous_date_hours']:
        ordinary, previous_date = evidence['ordinary_share'], evidence['previous_date_share_zone']
    else:
        ordinary = evidence['ordinary_share_beyond']
        previous_date = (1 - ordinary) * min(1.0, evidence['recorded_calibration'] * recorded_share)
    return {'P': 1 - ordinary - previous_date, 'S': previous_date, 'N': ordinary}


def correct_late_rome_departures(predictions: pd.Series, lane_value: pd.Series, transfer_factor: float = LATE_ROME_TRANSFER_FACTOR) -> tuple[pd.Series, pd.DataFrame]:
    """
    This is the main function: it takes the 2026 Rome no-link flights at least 3 hours late and, for those 8 hours or more late, moves the guess the transfer factor's share of the way toward the expected answer from the three chances
    """
    rome = rome_departures()
    habit = RecordingHabit(rome)
    evidence = mechanism_evidence(rome, habit)
    examined = rome[(rome.year == 2026) & rome.no_link & (rome.delay_hours >= MINIMUM_DELAY_HOURS) & rome.MVT_ID_mvt.isin(predictions.index)]
    corrected = predictions.copy()
    changes = []
    for row in examined.itertuples():
        before = float(predictions[row.MVT_ID_mvt])
        after = before
        value = float(lane_value[row.MVT_ID_mvt])
        probability = {'P': np.nan, 'S': np.nan, 'N': np.nan}
        mean = np.nan
        if row.delay_hours >= OVERRIDE_DELAY_HOURS:          # the override: toward the conditional mean of the three labels
            probability = regime_probabilities(row.delay_hours, bool(row.later_local_date), evidence, habit.share(row.flight, row.operator))
            mean = probability['P'] * row.msched + probability['S'] * (PREVIOUS_DATE_OFFSET + value) + probability['N'] * value
            after = before + transfer_factor * (mean - before)
        if not row.later_local_date and row.delay_hours <= evidence['latest_correct_hours']:
            after = min(after, row.msched)                   # the msched cap: same local date, inside the window
        if after != before:
            changes.append({'MVT_ID_mvt': row.MVT_ID_mvt, 
                            'flight': row.flight, 
                            'delay_hours': row.delay_hours,
                            'later_local_date': bool(row.later_local_date), 
                            'pP': probability['P'], 
                            'pS': probability['S'],
                            'pN': probability['N'], 
                            'mean': mean, 
                            'before': before, 
                            'after': after})
            corrected[row.MVT_ID_mvt] = after
    return corrected, pd.DataFrame(changes)
