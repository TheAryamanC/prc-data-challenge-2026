"""
A plane does not taxi alone - how long it takes to drive from its stand to the runway depends on what all the other planes are doing
This file writes down those clues - things like how busy the runway is, whether planes jump the queue, etc

This is how it works:
-look at the queue of take-offs and landings around each take-off
-look at the typical taxi times of groups such as the stand or the runway, learned only from other months
-look for who took off out of turn compared with their off-block time or timetable time
-look for what the take-offs just before and after were
-get the latest airport weather report and the wind along the runway
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taxi.main import DOWNLOADS, FEATURE_MATRIX, competition_files, departure_frame

CHALLENGE_COLUMNS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'EOBT_1_flt', 'BLOCK_TIME_UTC_mvt']
# takeoff_sequence columns left out of airport_state
AIRPORT_STATE_DROPPED_COLUMNS = ['sq_rwy_since_arr', 'sq_rwy_to_arr', 'sq_rwy_since_any']
# metar_weather columns of airport_state_extended
AIRPORT_STATE_EXTENDED_WEATHER_COLUMNS = ['wx_sn', 'wx_fz', 'wx_icing', 'wx_icing_3h', 'wx_sn_3h', 'wx_icing_6h', 'wx_sn_6h', 'wx_icing_day', 'wx_sn_day', 'wx_temp', 'wx_lvp', 'wx_lvp_2h', 'wx_vis', 'wx_ceil', 'wx_gust', 'wx_cross', 'wx_head', 'wx_ts_90m']
BLOCK_NAMES = ('takeoff_sequence', 'unimpeded_times', 'ordering', 'neighbours_adsb', 'neighbours', 'neighbours_wide', 'metar_weather')
STATE_TABLE_NAMES = ('airport_state', 'airport_state_extended')

# an NM off-block or scheduled time counts as the start of the taxi when 0 - 6 h before take-off
SIX_HOURS = 21600

# takeoff_sequence
MOVEMENT_COUNT_PAST_WINDOWS_SECONDS = (300, 600, 900, 1800)    # 5, 10, 15 and 30 min: from "right now" to about the length of a long taxi
MOVEMENT_COUNT_NEXT_WINDOWS_SECONDS = (300, 600, 900)          # 5, 10 and 15 min after: the planes queued behind this one
TAKEOFF_GAP_PREVIOUS_RANKS = (1, 2, 3, 5)                      # gaps to the 1st, 2nd, 3rd and 5th take-off before: a short look back down the queue
TAKEOFF_GAP_NEXT_MAX_RANK = 3                                  # gaps to the next 1st, 2nd and 3rd take-off
DECAY_TIME_CONSTANTS_SECONDS = (300, 900)                      # fading counts with a 5 and a 15 min memory, the same scale as the count windows
BUSY_CHAIN_GAP_RUNWAY_SECONDS = 150                            # a "rush hour" ends at a gap over 2.5 min: a little more than the 1-2 min between back-to-back take-offs on one runway
BUSY_CHAIN_GAP_AIRPORT_SECONDS = 90                            # shorter for the whole airport, where several runways together send planes off more often
PRESSURE_WINDOW_SECONDS = 900                                  # 15 min, the middle one of the count windows
PRESSURE_CAPACITY_QUANTILE = 0.99                              # "the most it usually handles" = the 99th percentile of its own 15-min counts, so one odd burst does not set it
PRESSURE_CAPACITY_FLOOR = 1.0                                  # at least 1 take-off, so a very quiet runway never divides by 0
QUEUE_BEFORE_START_WINDOWS_SECONDS = (600, 1200)               # 10 and 20 min before the plane left its stand: the queue it joins
MIN_TAXI_FOR_RATE_SECONDS = 60                                 # no rate for taxis under 1 min, where dividing by a tiny time gives silly rates
TAKEOFF_RATE_UNIT_SECONDS = 600                                # the rate is written as take-offs per 10 min, the same scale as the count windows
RATIO_PSEUDO_COUNT = 1                                         # add-one smoothing: keeps the ratios finite when a count is 0

# unimpeded_times
RESIDUAL_CLIP_SECONDS = 6000                                   # +-100 min: keeps a few hours-long outliers from dragging a group's mean
LABEL_CLIP_FLOOR_SECONDS = -12                                 # the lowest 2025 label is -12 s, the same floor as the rest of the pipeline
LABEL_CLIP_CAP_SECONDS = 7200                                  # 2 h: the few much longer 2025 labels would otherwise swamp a group's mean
SHRINKAGE_PRIOR_WEIGHT = 20.0                                  # the parent group's mean counts like 20 extra flights
UNIMPEDED_QUANTILE = 0.2                                       # a fast but not freak taxi stands for "no queue"
UNIMPEDED_MIN_ROWS = 10                                        # a 20th percentile of fewer than 10 flights is too shaky - the runway value is used instead

# ordering
OVERTAKING_OFFSET_STOP = 61                                    # look at up to 60 take-offs on each side (range() stops before 61), with the 1 h window below
OVERTAKING_WINDOW_SECONDS = 3600                               # 1 h on each side: picked when the block was built, not tuned
RANK_DIFFERENCE_WINDOW_SECONDS = 1800                          # 30 min: a tighter look at the swaps right around the plane
DISORDER_WINDOW_SECONDS = 1800                                 # 30 min on each side, the same span as the rank difference

# neighbours_adsb, neighbours, neighbours_wide
WINDOW_MAX_OFFSET_STOP = 65                                    # the window maximum checks at most 64 neighbours on each side (range() stops before 65): a speed limit, a very busy 45-min airport window can miss its farthest rows
CLOCK_EVIDENCE_CLIP_FLOOR_SECONDS = -3600                      # clock differences are kept between -1 h ...
CLOCK_EVIDENCE_CLIP_CAP_SECONDS = 21600                        # ... and 6 h, the same 6 h limit as SIX_HOURS, so a broken timestamp cannot swamp a mean
DWELL_CLIP_CAP_SECONDS = 7200                                  # ADS-B stops longer than 2 h are cut there, so one parked aircraft cannot swamp a mean
NM_TAXI_SHARE_THRESHOLD_SECONDS = 1500.0                       # share of neighbours with an NM taxi over 25 min (a long taxi)
ADSB_TAXI_SHARE_THRESHOLD_SECONDS = 1800.0                     # share with an ADS-B taxi over 30 min
DWELL_SHARE_THRESHOLD_SECONDS = 600.0                          # share that stood still for over 10 min on the way out
EXCESS_SHARE_THRESHOLD_SECONDS = 600.0                         # share whose ADS-B taxi was over 10 min longer than the NM one
DELAY_SHARE_THRESHOLD_SECONDS = 3600.0                         # share taking off over 1 h after the timetable
PUSH_DELAY_SHARE_THRESHOLD_SECONDS = 900.0                     # share leaving the stand over 15 min after the filed EOBT, the usual line for a "delayed" flight in punctuality statistics
TAKEOFF_AFTER_EOBT_SHARE_THRESHOLD_SECONDS = 2400.0            # share taking off over 40 min after the filed EOBT
TREND_SIDE_WINDOW_SECONDS = 900                                # 15 min before against 15 min after, the middle neighbour window

# metar_weather
CEILING_CAP_FEET = 12000                                       # cloud this high does not matter on the ground - "no ceiling" is set to this round cap
RVR_CAP_METRES = 2000                                          # METARs report runway visual range only up to 2,000 m (higher is written P2000, ICAO Annex 3)
VISIBILITY_CAP_METRES = 10000                                  # METARs report visibility only up to 10 km (written 9999)
# low-visibility procedures: about the ICAO CAT I approach limits (200 ft decision height, 550 m RVR, here rounded to 600 m), below which airports switch to low-visibility procedures
LVP_VISIBILITY_METRES = 600                                    # visibility under 600 m
LVP_CEILING_FEET = 200                                         # or cloud ceiling under 200 ft
LVP_RVR_METRES = 600                                           # or runway visual range under 600 m
IMC_VISIBILITY_METRES = 5000                                   # below 5 km ground visibility VFR flights may not use a controlled European airport (SERA.5005)
IMC_CEILING_FEET = 1000                                        # 1,000 ft: the US line between IFR and marginal VFR cloud ceilings
# four weather classes loosely shaped like the US flight categories (LIFR / IFR / MVFR / VFR), but with lower, metric limits picked when the block was built
LIFR_VISIBILITY_METRES = 600                                   # the same limits as the low-visibility flag above
LIFR_CEILING_FEET = 200                                        # the same 200 ft as the low-visibility flag
IFR_VISIBILITY_METRES = 1500                                   # 1,500 m: the lowest ground visibility for special VFR aeroplane flights in Europe (SERA.5010)
IFR_CEILING_FEET = 500                                         # 500 ft: the US line between LIFR and IFR ceilings
MVFR_VISIBILITY_METRES = 5000                                  # the same 5 km as the IMC flag
MVFR_CEILING_FEET = 1000                                       # the same 1,000 ft as the IMC flag
WIND_SHIFT_MIN_SPEED_KNOTS = 4                                 # below about 3 kt a METAR may give the wind as VRB (variable), so its direction means little
ICING_MAX_TEMPERATURE_CELSIUS = 3                              # a few degrees above freezing, so wet snow or rain on cold-soaked wings counts (the temperature is also an input)
ICING_SNOW_WINDOWS_HOURS = (3, 6)                              # icing and snow over the last 3 and 6 h (plus the whole day): how long de-icing demand has been building
TAIL_WIND_FLAG_KNOTS = 5                                       # ICAO noise-abatement runway rules allow at most 5 kt tail wind, gusts included
CROSS_WIND_FLAG_KNOTS = 15                                     # ... and at most 15 kt cross wind (same rule), so above these the preferred runway may have to change
METAR_MAX_AGE_SECONDS = 3 * 3600                               # these airports issue a METAR every 30 or 60 min, so a report over 3 h old means the archive has a gap


def epoch_seconds(times: pd.Series) -> np.ndarray:
    """
    This function turns clock times into seconds since 1970 for easy comparison
    """
    return times.dt.as_unit('ns').astype('int64').to_numpy() / 1e9


def all_challenge_rows() -> pd.DataFrame:
    """
    This function reads every competition file and stacks everything into one big table
    """
    return pd.concat([pd.read_parquet(path, columns=CHALLENGE_COLUMNS) for path in competition_files()], ignore_index=True)


def count_between(sorted_times: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    This function counts how many times in a sorted list fall inside each window
    """
    return np.searchsorted(sorted_times, upper, side='left') - np.searchsorted(sorted_times, lower, side='left')


def exponentially_decayed_count(sorted_times: np.ndarray, query_times: np.ndarray, time_constant: float) -> np.ndarray:
    """
    This function counts the events before each moment, but the older an event is, the less it counts
    """
    decayed_counts = np.zeros(len(query_times))
    next_event = 0
    running_sum = 0.0            # the decayed count at last_event_time, just after that event
    last_event_time = 0.0
    for query in np.argsort(query_times):
        while next_event < len(sorted_times) and sorted_times[next_event] < query_times[query]:
            running_sum = running_sum * np.exp(-(sorted_times[next_event] - last_event_time) / time_constant) + 1.0
            last_event_time = sorted_times[next_event]
            next_event += 1
        if next_event:
            decayed_counts[query] = running_sum * np.exp(-(query_times[query] - last_event_time) / time_constant)
        else:
            decayed_counts[query] = 0.0
    return decayed_counts


def key_groups(key: pd.Series) -> Iterator[tuple[object, np.ndarray]]:
    """
    This function splits the rows into groups by a key such as the runway and hands them out one group at a time (rows with a missing key are left out)
    """
    for key_value, rows in key.groupby(key).groups.items():
        yield key_value, np.asarray(rows)


# takeoff_sequence
@dataclass
class SequenceGroup:
    """
    This data structure holds everything the queue helpers need about one runway or one airport
    """
    level: str                     # 'rwy' or 'apt': the prefix of the column names
    rows: np.ndarray               # row positions of the departures
    own_takeoffs: np.ndarray       # their take-off times, in row order
    sorted_takeoffs: np.ndarray    # the same, ascending
    position: np.ndarray           # how many of the key's take-offs come strictly before each row's own
    landings: np.ndarray           # the key's landing times, ascending (empty when there are none)


def sorted_times_by_key(times: np.ndarray, keys: np.ndarray) -> dict[object, np.ndarray]:
    """
    This function groups times by a key and sorts each group, e.g. a separate list of landing times for every runway
    """
    return pd.Series(times).groupby(keys).apply(lambda key_times: np.sort(key_times.to_numpy())).to_dict()


def nm_off_block_within_six_hours(takeoff_times: np.ndarray, nm_off_block_times: np.ndarray) -> np.ndarray:
    """
    This function keeps the Network Manager's off-block time (when the plane left its stand) only when it is 0 to 6 hours before take-off
    """
    nm_lead = takeoff_times - nm_off_block_times
    return np.where(np.isfinite(nm_off_block_times) & (nm_lead >= 0) & (nm_lead <= SIX_HOURS), nm_off_block_times, np.nan)


def taxi_start_times(takeoff_times: np.ndarray, nm_off_block_times: np.ndarray, scheduled_times: np.ndarray) -> np.ndarray:
    """
    This function makes the best guess of when each plane started to drive
    """
    nm_taxi_start = nm_off_block_within_six_hours(takeoff_times, nm_off_block_times)
    scheduled_lead = takeoff_times - scheduled_times
    scheduled_start = np.where((scheduled_lead >= 0) & (scheduled_lead <= SIX_HOURS), scheduled_times, np.nan)
    return np.where(np.isfinite(nm_taxi_start), nm_taxi_start, scheduled_start)


def add_movement_counts(column: Callable[[str], np.ndarray], group: SequenceGroup) -> None:
    """
    This function counts the take-offs and landings on the same runway (or airport) in the 5, 10, 15 and 30 minutes before each take-off and in the 5, 10 and 15 minutes after
    """
    level, rows, own = group.level, group.rows, group.own_takeoffs
    for window in MOVEMENT_COUNT_PAST_WINDOWS_SECONDS:
        column(f'sq_{level}_dep_past_{window}')[rows] = count_between(group.sorted_takeoffs, own - window, own)
        column(f'sq_{level}_arr_past_{window}')[rows] = count_between(group.landings, own - window, own)
    for window in MOVEMENT_COUNT_NEXT_WINDOWS_SECONDS:
        # take-offs after: [take-off + 1e-6 s, take-off + window), so take-offs at the same time are left out
        column(f'sq_{level}_dep_next_{window}')[rows] = count_between(group.sorted_takeoffs, own + 1e-6, own + window)
        column(f'sq_{level}_arr_next_{window}')[rows] = count_between(group.landings, own, own + window)


def add_takeoff_gaps(column: Callable[[str], np.ndarray], group: SequenceGroup) -> None:
    """
    This function measures how long ago the 1st, 2nd, 3rd and 5th earlier take-offs on the same runway (or airport) were, and how long until the next 1st, 2nd and 3rd
    """
    level, rows, own, sorted_takeoffs = group.level, group.rows, group.own_takeoffs, group.sorted_takeoffs
    after_own = np.searchsorted(sorted_takeoffs, own, side='right')        # first take-off strictly after the row's own
    for rank in TAKEOFF_GAP_PREVIOUS_RANKS:
        previous_position = group.position - rank
        has_previous = previous_position >= 0
        previous_takeoff = sorted_takeoffs[np.maximum(previous_position, 0)]
        column(f'sq_{level}_gap_prev{rank}')[rows] = np.where(has_previous, own - previous_takeoff, np.nan)
        if rank <= TAKEOFF_GAP_NEXT_MAX_RANK:
            next_position = after_own + rank - 1
            has_next = next_position < len(sorted_takeoffs)
            next_takeoff = sorted_takeoffs[np.minimum(next_position, len(sorted_takeoffs) - 1)]
            column(f'sq_{level}_gap_next{rank}')[rows] = np.where(has_next, next_takeoff - own, np.nan)


def add_landing_gaps(column: Callable[[str], np.ndarray], group: SequenceGroup) -> None:
    """
    This function measures the time since the last landing and until the next landing on the same runway (or airport)
    """
    if not len(group.landings):
        return
    level, rows, own, landings = group.level, group.rows, group.own_takeoffs, group.landings
    previous_landing = np.searchsorted(landings, own, side='left') - 1
    next_landing = np.searchsorted(landings, own, side='right')
    previous_landing_time = landings[np.maximum(previous_landing, 0)]
    next_landing_time = landings[np.minimum(next_landing, len(landings) - 1)]
    column(f'sq_{level}_since_arr')[rows] = np.where(previous_landing >= 0, own - previous_landing_time, np.nan)
    column(f'sq_{level}_to_arr')[rows] = np.where(next_landing < len(landings), next_landing_time - own, np.nan)


def add_decayed_counts(column: Callable[[str], np.ndarray], group: SequenceGroup) -> None:
    """
    This function measures how busy the recent past was with fading counts of the earlier take-offs and landings
    """
    for time_constant in DECAY_TIME_CONSTANTS_SECONDS:
        decayed_takeoffs = exponentially_decayed_count(group.sorted_takeoffs, group.own_takeoffs, time_constant)
        column(f'sq_{group.level}_dep_ewm_{time_constant}')[group.rows] = decayed_takeoffs
        decayed_landings = exponentially_decayed_count(group.landings, group.own_takeoffs, time_constant)
        column(f'sq_{group.level}_arr_ewm_{time_constant}')[group.rows] = decayed_landings


def add_busy_period(column: Callable[[str], np.ndarray], group: SequenceGroup) -> None:
    """
    This function finds "rush hours"
    """
    sorted_takeoffs = group.sorted_takeoffs
    busy_gap_limit = BUSY_CHAIN_GAP_RUNWAY_SECONDS if group.level == 'rwy' else BUSY_CHAIN_GAP_AIRPORT_SECONDS
    starts_new_run = np.diff(sorted_takeoffs, prepend=-np.inf) > busy_gap_limit
    run_start = np.maximum.accumulate(np.where(starts_new_run, np.arange(len(sorted_takeoffs)), 0))   # first take-off of each one's run
    own_position = np.minimum(group.position, len(sorted_takeoffs) - 1)
    column(f'sq_{group.level}_busy_n')[group.rows] = own_position - run_start[own_position]
    column(f'sq_{group.level}_busy_dur')[group.rows] = sorted_takeoffs[own_position] - sorted_takeoffs[run_start[own_position]]


def add_pressure(column: Callable[[str], np.ndarray], group: SequenceGroup) -> None:
    """
    This function compares the take-offs of the last 15 minutes with the most the runway (or airport) usually handles
    """
    count_15_minutes = count_between(group.sorted_takeoffs, group.own_takeoffs - PRESSURE_WINDOW_SECONDS, group.own_takeoffs)
    capacity = max(np.quantile(count_15_minutes, PRESSURE_CAPACITY_QUANTILE), PRESSURE_CAPACITY_FLOOR)
    column(f'sq_{group.level}_pressure')[group.rows] = count_15_minutes / capacity
    column(f'sq_{group.level}_cap15')[group.rows] = capacity


def add_counts_during_taxi(column: Callable[[str], np.ndarray], group: SequenceGroup, taxi_start: np.ndarray) -> None:
    """
    This function looks at what happened while the plane was driving to the runway e.g. the queue before and after
    """
    level, rows, own = group.level, group.rows, group.own_takeoffs
    has_start = np.isfinite(taxi_start)
    start_or_takeoff = np.where(has_start, taxi_start, own)
    column(f'sq_{level}_dep_during')[rows] = np.where(has_start, count_between(group.sorted_takeoffs, start_or_takeoff, own), np.nan)
    column(f'sq_{level}_arr_during')[rows] = np.where(has_start, count_between(group.landings, start_or_takeoff, own), np.nan)
    for window in QUEUE_BEFORE_START_WINDOWS_SECONDS:
        queue_ahead = count_between(group.sorted_takeoffs, start_or_takeoff - window, start_or_takeoff)
        column(f'sq_{level}_dep_before_start_{window}')[rows] = np.where(has_start, queue_ahead, np.nan)
    takeoffs_during = count_between(group.sorted_takeoffs, start_or_takeoff, own)
    taxi_seconds = own - start_or_takeoff
    takeoffs_per_10_minutes = takeoffs_during / np.maximum(taxi_seconds, MIN_TAXI_FOR_RATE_SECONDS) * TAKEOFF_RATE_UNIT_SECONDS
    column(f'sq_{level}_dep_rate_during')[rows] = np.where(has_start & (taxi_seconds > MIN_TAXI_FOR_RATE_SECONDS), takeoffs_per_10_minutes, np.nan)


def build_takeoff_sequence_block(all_rows: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    """
    This function builds the runway queue
    """
    block = pd.DataFrame(index=departures.index)
    takeoff_times = epoch_seconds(departures.MVT_TIME_UTC_mvt)
    nm_off_block_times = epoch_seconds(departures.AOBT_3_flt)          # AOBT_3: the NM actual off-block time
    scheduled_times = epoch_seconds(departures.SCHED_TIME_UTC_mvt)
    arrivals = all_rows[all_rows.PHASE_mvt == 'ARR']
    landing_times = epoch_seconds(arrivals.MVT_TIME_UTC_mvt)
    arrival_runway_keys = (arrivals.ADES_mvt.astype(str) + ':' + arrivals.RUNWAY_mvt.astype(str)).to_numpy()
    arrival_airport_keys = arrivals.ADES_mvt.astype(str).to_numpy()
    sorted_landings_by_key = {'rwy': sorted_times_by_key(landing_times, arrival_runway_keys),
                              'apt': sorted_times_by_key(landing_times, arrival_airport_keys)}
    level_keys = {'rwy': (departures.ADEP_mvt.astype(str) + ':' + departures.RUNWAY_mvt.astype(str)),
                  'apt': departures.ADEP_mvt.astype(str)}
    taxi_start = taxi_start_times(takeoff_times, nm_off_block_times, scheduled_times)
    columns: dict[str, np.ndarray] = {}

    def column(name: str) -> np.ndarray:
        """
        This function hands back the column with this name, creating it full of blanks the first time it is asked for
        """
        return columns.setdefault(name, np.full(len(departures), np.nan))

    for level, key in level_keys.items():
        for key_value, rows in key_groups(key):
            own_takeoffs = takeoff_times[rows]
            sorted_takeoffs = np.sort(own_takeoffs)
            group = SequenceGroup(level=level, rows=rows, own_takeoffs=own_takeoffs, sorted_takeoffs=sorted_takeoffs, position=np.searchsorted(sorted_takeoffs, own_takeoffs, side='left'), landings=sorted_landings_by_key[level].get(key_value, np.zeros(0)))
            add_movement_counts(column, group)
            add_takeoff_gaps(column, group)
            add_landing_gaps(column, group)
            add_decayed_counts(column, group)
            add_busy_period(column, group)
            add_pressure(column, group)
            add_counts_during_taxi(column, group, taxi_start[rows])
    for name, values in columns.items():
        block[name] = values.astype('float32')
    block['sq_rwy_dep_arr_ratio_900'] = (block.sq_rwy_dep_past_900 + RATIO_PSEUDO_COUNT) / (block.sq_rwy_arr_past_900 + RATIO_PSEUDO_COUNT)
    block['sq_rwy_share_of_apt_900'] = (block.sq_rwy_dep_past_900 + RATIO_PSEUDO_COUNT) / (block.sq_apt_dep_past_900 + RATIO_PSEUDO_COUNT)
    block['sq_rwy_since_any'] = np.fmin(block.sq_rwy_gap_prev1, block.sq_rwy_since_arr)
    return block


# unimpeded_times
def shrunk_mean(value_sum: np.ndarray, count: np.ndarray, parent_mean: np.ndarray, prior_weight: float) -> np.ndarray:
    """
    This function takes an average but pulls it toward the parent group's average when there are only a few values, so one lucky value does not fool the model
    """
    return (value_sum + prior_weight * parent_mean) / (count + prior_weight)


def unimpeded_key_hierarchy(departures: pd.DataFrame) -> dict[str, tuple[pd.Series, str | None]]:
    """
    This function builds the family tree of ways to group departures
    """
    hour = departures.MVT_TIME_UTC_mvt.dt.hour.astype(str)
    day_of_week = departures.MVT_TIME_UTC_mvt.dt.dayofweek.astype(str)
    airport = departures.ADEP_mvt.astype(str)
    runway_number = departures.RUNWAY_mvt.astype(str)
    runway = airport + ':' + runway_number
    stand = airport + ':' + departures.STAND_mvt.astype(str)
    flight = departures.FLIGHT_mvt.fillna('???')
    operator = airport + ':' + flight.str[:3]
    destination = departures.ADES_mvt.astype(str)
    return {'apt': (airport, None),
            'rwy': (runway, 'apt'),
            'stand': (stand, 'apt'),
            'stand_rwy': (stand + ':' + runway_number, 'stand'),
            'stand_rwy_h': (stand + ':' + runway_number + ':' + hour, 'stand_rwy'),
            'rwy_h': (runway + ':' + hour, 'rwy'),
            'rwy_dow_h': (runway + ':' + day_of_week + ':' + hour, 'rwy_h'),
            'op': (operator, 'apt'),
            'op_h': (operator + ':' + hour, 'op'),
            'flight': (airport + ':' + flight, 'op'),
            'flight_dow': (airport + ':' + flight + ':' + day_of_week, 'flight'),
            'dest': (airport + ':' + destination, 'apt'),
            'op_dest': (operator + ':' + destination, 'op'),
            'type_rwy': (runway + ':' + departures.AIRCRAFT_TYPE_mvt.astype(str), 'rwy'),
            'op_stand': (operator + ':' + departures.STAND_mvt.astype(str), 'stand')}


def out_of_month_totals(key_codes: np.ndarray, held_out_month: np.ndarray, target_values: np.ndarray, counted_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function adds up and counts the known answers of each row's group, leaving out the row's own month
    """
    labelled = pd.DataFrame({'key_code': key_codes[counted_rows],
                             'month': held_out_month[counted_rows],
                             'value': target_values[counted_rows]})
    totals = labelled.groupby('key_code')['value'].agg(['sum', 'count'])
    per_month = labelled.groupby(['key_code', 'month'])['value'].agg(['sum', 'count'])
    sum_all = totals['sum'].reindex(key_codes).to_numpy()
    count_all = totals['count'].reindex(key_codes).to_numpy()
    key_and_month = pd.MultiIndex.from_arrays([key_codes, held_out_month])
    sum_own_month = per_month['sum'].reindex(key_and_month).to_numpy()
    count_own_month = per_month['count'].reindex(key_and_month).to_numpy()
    # all months minus the row's own month (NaN = no labelled rows = 0)
    sum_out_of_month = np.nan_to_num(sum_all) - np.nan_to_num(sum_own_month)
    count_out_of_month = np.nan_to_num(count_all) - np.nan_to_num(count_own_month)
    return sum_out_of_month, count_out_of_month


def out_of_month_percentile_20(key_codes: np.ndarray, held_out_month: np.ndarray, clipped_taxi_time: np.ndarray, is_labelled: np.ndarray) -> np.ndarray:
    """
    As above but the 20% percentile
    """
    labelled = pd.DataFrame({'key_code': key_codes[is_labelled], 'month': held_out_month[is_labelled], 'value': clipped_taxi_time})
    percentile_20 = np.full(len(key_codes), np.nan)
    for month in range(0, 13):
        other_months = labelled[labelled['month'] != month]
        key_quantile = other_months.groupby('key_code')['value'].quantile(UNIMPEDED_QUANTILE)
        key_count = other_months.groupby('key_code')['value'].size()
        key_quantile = key_quantile[key_count >= UNIMPEDED_MIN_ROWS]
        rows_of_month = held_out_month == month
        percentile_20[rows_of_month] = key_quantile.reindex(key_codes[rows_of_month]).to_numpy()
    return percentile_20


def build_unimpeded_times_block(frame: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    """
    This function builds the typical taxi times of every group in the family tree, learned only from other months 
    """
    taxi_time = frame.TAXITIME_SEC_mvt.to_numpy()
    is_normal_regime = (~frame.is_schedule_copy & ~frame.is_previous_date).to_numpy()
    is_labelled = np.isfinite(taxi_time) & (frame.year == 2025).to_numpy() & is_normal_regime
    residual = np.clip(taxi_time - frame.baseline.to_numpy(), -RESIDUAL_CLIP_SECONDS, RESIDUAL_CLIP_SECONDS)          # linked rows only (no-link rows use y below)
    key_hierarchy = unimpeded_key_hierarchy(departures)
    held_out_month = np.where(frame.year.to_numpy() == 2025, frame.month.to_numpy(), 0)   # ranking rows: month 0 -> all of 2025
    block = pd.DataFrame(index=departures.index)
    shrunk_means: dict[str, dict[str, np.ndarray]] = {}
    targets = {'r': (residual, is_labelled & ~frame.no_nm_link.to_numpy()), 'y': (np.clip(taxi_time, LABEL_CLIP_FLOOR_SECONDS, LABEL_CLIP_CAP_SECONDS), is_labelled)}
    for key_name, (key_values, parent_name) in key_hierarchy.items():
        key_codes = pd.factorize(key_values)[0]
        shrunk_means[key_name] = {}
        for target_name, (target_values, counted_rows) in targets.items():
            sum_out_of_month, count_out_of_month = out_of_month_totals(key_codes, held_out_month, target_values, counted_rows)
            if parent_name is None:
                mean = sum_out_of_month / np.maximum(count_out_of_month, 1)
            else:
                mean = shrunk_mean(sum_out_of_month, count_out_of_month, shrunk_means[parent_name][target_name], SHRINKAGE_PRIOR_WEIGHT)
            shrunk_means[key_name][target_name] = mean
            block[f'te_{key_name}_{target_name}'] = mean.astype('float32')
        block[f'te_{key_name}_n'] = np.log1p(count_out_of_month).astype('float32')     # count of the last target (y)
    # unimpeded reference: 20th percentile of y per stand x runway out of month (keys with >= 10 rows), fallback runway
    for key_name in ('stand_rwy', 'rwy'):
        key_codes = pd.factorize(key_hierarchy[key_name][0])[0]
        percentile_20 = out_of_month_percentile_20(key_codes, held_out_month, np.clip(taxi_time[is_labelled], LABEL_CLIP_FLOOR_SECONDS, LABEL_CLIP_CAP_SECONDS), is_labelled)
        block[f'te_{key_name}_p20'] = percentile_20.astype('float32')
    block['te_unimpeded'] = block.te_stand_rwy_p20.fillna(block.te_rwy_p20)
    return block


# ordering
def overtaking_counts(sorted_takeoffs: np.ndarray, clock_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function compares the order the planes took off in with the order of a clock (the off-block time or the timetable), looking at up to 60 take-offs on each side within 1 hour
    It counts how many planes got ahead of this one, how many this one got ahead of, and the difference of the two within 30 minutes
    """
    group_size = len(sorted_takeoffs)
    positions = np.arange(group_size)
    overtaken = np.zeros(group_size)
    overtook = np.zeros(group_size)
    rank_difference = np.zeros(group_size)
    for offset in range(1, OVERTAKING_OFFSET_STOP):
        neighbour = positions - offset
        in_range = neighbour >= 0
        neighbour = np.maximum(neighbour, 0)
        near = in_range & (sorted_takeoffs - sorted_takeoffs[neighbour] <= OVERTAKING_WINDOW_SECONDS)
        overtaken += near & (clock_times[neighbour] > clock_times)        # took off before me although it started / was scheduled later
        rank_difference += near & (sorted_takeoffs - sorted_takeoffs[neighbour] <= RANK_DIFFERENCE_WINDOW_SECONDS) & (clock_times[neighbour] > clock_times)
        neighbour = positions + offset
        in_range = neighbour < group_size
        neighbour = np.minimum(neighbour, group_size - 1)
        near = in_range & (sorted_takeoffs[neighbour] - sorted_takeoffs <= OVERTAKING_WINDOW_SECONDS)
        overtook += near & (clock_times[neighbour] < clock_times)          # took off after me although it started / was scheduled earlier
        rank_difference -= near & (sorted_takeoffs[neighbour] - sorted_takeoffs <= RANK_DIFFERENCE_WINDOW_SECONDS) & (clock_times[neighbour] < clock_times)
    return overtaken, overtook, rank_difference


def neighbour_disorder(sorted_takeoffs: np.ndarray, inversions: np.ndarray) -> np.ndarray:
    """
    This function works out the average number of order swaps of the other take-offs within 30 minutes on each side
    """
    cumulative_inversions = np.r_[0, np.cumsum(inversions)]
    window_start = np.searchsorted(sorted_takeoffs, sorted_takeoffs - DISORDER_WINDOW_SECONDS)
    window_end = np.searchsorted(sorted_takeoffs, sorted_takeoffs + DISORDER_WINDOW_SECONDS, side='right')
    inversions_around = cumulative_inversions[window_end] - cumulative_inversions[window_start] - inversions
    return inversions_around / np.maximum(window_end - window_start - 1, 1)


def build_ordering_block(departures: pd.DataFrame) -> pd.DataFrame:
    """
    This function finds how many planes each plane was overtaken by and overtook
    """
    takeoff_times = epoch_seconds(departures.MVT_TIME_UTC_mvt)
    nm_off_block_times = epoch_seconds(departures.AOBT_3_flt)
    scheduled_times = epoch_seconds(departures.SCHED_TIME_UTC_mvt)
    nm_off_block_times = nm_off_block_within_six_hours(takeoff_times, nm_off_block_times)
    level_keys = {'rwy': departures.ADEP_mvt.astype(str) + ':' + departures.RUNWAY_mvt.astype(str),
                  'apt': departures.ADEP_mvt.astype(str)}
    columns: dict[str, np.ndarray] = {}
    for level, key in level_keys.items():
        for clock_name in ('off', 'sch'):
            for direction in ('overtaken', 'overtook'):
                columns[f'or_{level}_{clock_name}_{direction}'] = np.full(len(departures), np.nan)
        columns[f'or_{level}_rank_diff'] = np.full(len(departures), np.nan)
        for _, rows in key_groups(key):
            by_takeoff = np.argsort(takeoff_times[rows], kind='stable')
            rows_by_takeoff = rows[by_takeoff]
            sorted_takeoffs = takeoff_times[rows_by_takeoff]
            for clock_name, clock_times in (('off', nm_off_block_times[rows_by_takeoff]), ('sch', scheduled_times[rows_by_takeoff])):
                overtaken, overtook, rank_difference = overtaking_counts(sorted_takeoffs, clock_times)
                missing_clock = ~np.isfinite(clock_times)
                columns[f'or_{level}_{clock_name}_overtaken'][rows_by_takeoff] = np.where(missing_clock, np.nan, overtaken)
                columns[f'or_{level}_{clock_name}_overtook'][rows_by_takeoff] = np.where(missing_clock, np.nan, overtook)
                if clock_name == 'sch':
                    columns[f'or_{level}_rank_diff'][rows_by_takeoff] = np.where(missing_clock, np.nan, rank_difference)
            # disorder among the neighbours (+-30 min, the row excluded)
            schedule_overtaken = np.nan_to_num(columns[f'or_{level}_sch_overtaken'][rows_by_takeoff])
            schedule_overtook = np.nan_to_num(columns[f'or_{level}_sch_overtook'][rows_by_takeoff])
            inversions = schedule_overtaken + schedule_overtook
            disorder = columns.setdefault(f'or_{level}_disorder', np.full(len(departures), np.nan))
            disorder[rows_by_takeoff] = neighbour_disorder(sorted_takeoffs, inversions)
    return pd.DataFrame({name: values.astype('float32') for name, values in columns.items()})


# neighbours_adsb, neighbours, neighbours_wide
NEIGHBOUR_COLUMN_PREFIX = {'neighbours_adsb': 'nb', 'neighbours': 'n2', 'neighbours_wide': 'nw'}
# +-15 and +-45 min (+-5 min added in the no-ADS-B block) for the state around the take-off and +-2 h and +-12 h for the state of the day
NEIGHBOUR_WINDOWS_SECONDS = {'neighbours_adsb': (900, 2700), 'neighbours': (300, 900, 2700), 'neighbours_wide': (7200, 43200)}


def leave_one_out_window_max(times: np.ndarray, values: np.ndarray, has_value: np.ndarray, window: float) -> np.ndarray:
    """
    This function finds the largest value among the other rows within the time window of each row
    """
    row_count = len(times)
    window_max = np.full(row_count, -np.inf)
    values_or_minus_infinity = np.where(has_value, values, -np.inf)
    for offset in range(1, WINDOW_MAX_OFFSET_STOP):
        if offset >= row_count:
            break                                      # no row has a neighbour this far away
        has_earlier = slice(offset, None)              # rows i >= offset, whose neighbour i - offset exists
        has_later = slice(None, row_count - offset)    # rows i < row_count - offset, whose neighbour i + offset exists
        # neighbour offset positions earlier: row i (in has_earlier) against row i - offset (the same positions as has_later)
        in_window = np.abs(times[has_later] - times[has_earlier]) <= window
        larger = np.maximum(window_max[has_earlier], values_or_minus_infinity[has_later])
        window_max[has_earlier] = np.where(in_window, larger, window_max[has_earlier])
        # neighbour offset positions later: row i (in has_later) against row i + offset (the same positions as has_earlier)
        in_window = np.abs(times[has_earlier] - times[has_later]) <= window
        larger = np.maximum(window_max[has_later], values_or_minus_infinity[has_earlier])
        window_max[has_later] = np.where(in_window, larger, window_max[has_later])
    return np.where(np.isfinite(window_max), window_max, np.nan)


def neighbour_window_statistics(times: np.ndarray, values: np.ndarray, window: float, threshold: float | None, side: str = 'both', need_max: bool = True) -> dict[str, np.ndarray]:
    """
    This function summarizes the values of the other planes in the same group within a time window of each plane
    """
    has_value = np.isfinite(values)
    values_or_zero = np.where(has_value, values, 0.0)
    cumulative_sum = np.r_[0, np.cumsum(values_or_zero)]
    cumulative_count = np.r_[0, np.cumsum(has_value)]
    window_start = np.searchsorted(times, times - (window if side != 'after' else -1e-3), side='left')
    window_end = np.searchsorted(times, times + (window if side != 'before' else -1e-3), side='right')
    own = has_value if side == 'both' else np.zeros(len(times), bool)       # one-sided windows exclude the row (and ties) by construction
    window_end = np.maximum(window_end, window_start)
    count = cumulative_count[window_end] - cumulative_count[window_start] - own
    value_sum = cumulative_sum[window_end] - cumulative_sum[window_start] - np.where(own, values_or_zero, 0.0)
    statistics = {'mean': np.where(count > 0, value_sum / np.maximum(count, 1), np.nan), 'n': count.astype(float)}
    if threshold is not None:
        above_threshold = has_value & (values > threshold)
        cumulative_above = np.r_[0, np.cumsum(above_threshold)]
        above_count = cumulative_above[window_end] - cumulative_above[window_start] - (above_threshold & own)
        statistics['share'] = np.where(count > 0, above_count / np.maximum(count, 1), np.nan)
    if side != 'both' or not need_max:
        return statistics
    statistics['max'] = leave_one_out_window_max(times, values, has_value, window)
    return statistics


def schedule_delay_evidence(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    This function works out the delay
    """
    return np.clip(frame.seconds_after_schedule.to_numpy(dtype=float), CLOCK_EVIDENCE_CLIP_FLOOR_SECONDS, CLOCK_EVIDENCE_CLIP_CAP_SECONDS), DELAY_SHARE_THRESHOLD_SECONDS


def neighbour_evidence(block_name: str, frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
    """
    This function picks which clues the neighbor blocks look at
    """
    nm_taxi = frame.baseline.to_numpy(dtype=float)
    nm_taxi = np.where(frame.no_nm_link.to_numpy() | (nm_taxi < 0) | (nm_taxi > SIX_HOURS), np.nan, nm_taxi)
    if block_name == 'neighbours_adsb':
        adsb_columns = ['MVT_ID_mvt', 'ads_near_stand_next_rel', 'ads_far_leave30_rel', 'ads_dw_far_dur']
        adsb = pd.read_parquet(FEATURE_MATRIX / 'adsb_pass1.parquet', columns=adsb_columns)
        adsb = adsb.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(frame.MVT_ID_mvt)
        adsb_taxi = -adsb.ads_near_stand_next_rel.to_numpy(dtype=float)                    # seconds from leaving the stand to take-off
        adsb_taxi = np.where(np.isfinite(adsb_taxi), adsb_taxi, -adsb.ads_far_leave30_rel.to_numpy(dtype=float))
        adsb_taxi = np.where((adsb_taxi < 0) | (adsb_taxi > SIX_HOURS), np.nan, adsb_taxi)
        return {'nmb': (nm_taxi, NM_TAXI_SHARE_THRESHOLD_SECONDS),
                'ads': (adsb_taxi, ADSB_TAXI_SHARE_THRESHOLD_SECONDS),
                'dwell': (np.clip(adsb.ads_dw_far_dur.to_numpy(dtype=float), 0, DWELL_CLIP_CAP_SECONDS), DWELL_SHARE_THRESHOLD_SECONDS),
                'excess': (adsb_taxi - nm_taxi, EXCESS_SHARE_THRESHOLD_SECONDS),
                'delay': schedule_delay_evidence(frame)}
    # no ADS-B evidence (robust to the coverage change in 2026)
    clock_columns = ['EOBT_1_flt__minus__AOBT_3_flt', 'MVT_TIME_UTC_mvt__minus__EOBT_1_flt']      # first clock minus second, s
    base_clocks = pd.read_parquet(FEATURE_MATRIX / 'base_features.parquet', columns=clock_columns)
    push_delay = -base_clocks['EOBT_1_flt__minus__AOBT_3_flt'].to_numpy(dtype=float)      # NM off-block minus filed EOBT (push delay)
    takeoff_after_eobt = base_clocks['MVT_TIME_UTC_mvt__minus__EOBT_1_flt'].to_numpy(dtype=float)
    return {'nmb': (nm_taxi, NM_TAXI_SHARE_THRESHOLD_SECONDS),
            'delay': schedule_delay_evidence(frame),
            'late': (np.clip(push_delay, CLOCK_EVIDENCE_CLIP_FLOOR_SECONDS, CLOCK_EVIDENCE_CLIP_CAP_SECONDS), PUSH_DELAY_SHARE_THRESHOLD_SECONDS),
            'tke': (np.clip(takeoff_after_eobt, CLOCK_EVIDENCE_CLIP_FLOOR_SECONDS, CLOCK_EVIDENCE_CLIP_CAP_SECONDS), TAKEOFF_AFTER_EOBT_SHARE_THRESHOLD_SECONDS)}


@dataclass
class KeyOrder:
    """
    This data structure holds the rows sorted by group and then by take-off time
    """
    order: np.ndarray              # row positions in sorted order
    sorted_times: np.ndarray       # take-off times (s) in sorted order
    group_bounds: np.ndarray       # group g = sorted positions group_bounds[g] .. group_bounds[g + 1] - 1

    def groups(self) -> Iterator[tuple[int, int]]:
        return zip(self.group_bounds[:-1], self.group_bounds[1:])


def key_order(key: np.ndarray, takeoff_seconds: np.ndarray) -> KeyOrder:
    """
    This function sorts the rows by group and then by take-off time
    """
    order = np.lexsort((takeoff_seconds, pd.factorize(key)[0]))
    sorted_keys = key[order]
    group_bounds = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])
    return KeyOrder(order=order, sorted_times=takeoff_seconds[order], group_bounds=group_bounds)


def window_statistic_columns(key: KeyOrder, values: np.ndarray, window: float, threshold: float | None, need_max: bool) -> dict[str, np.ndarray]:
    """
    This function runs neighbour_window_statistics on every group and puts the answers back in the original row order
    """
    row_count = len(values)
    statistic_names = ('mean', 'max', 'n', 'share') if threshold is not None else ('mean', 'max', 'n')
    statistic_columns = {statistic: np.full(row_count, np.nan) for statistic in statistic_names}
    sorted_values = values[key.order]
    for group_start, group_end in key.groups():
        group_times = key.sorted_times[group_start:group_end]
        group_values = sorted_values[group_start:group_end]
        group_statistics = neighbour_window_statistics(group_times, group_values, window, threshold, need_max=need_max)
        for statistic, statistic_values in group_statistics.items():
            statistic_columns[statistic][key.order[group_start:group_end]] = statistic_values
    return statistic_columns


def side_mean_columns(key: KeyOrder, values: np.ndarray) -> dict[str, np.ndarray]:
    """
    This function takes the average value of the take-offs in the same group in the 15 minutes before each plane and, separately, in the 15 minutes after
    """
    sorted_values = values[key.order]
    side_means = {side: np.full(len(values), np.nan) for side in ('before', 'after')}
    for group_start, group_end in key.groups():
        group_times = key.sorted_times[group_start:group_end]
        group_values = sorted_values[group_start:group_end]
        for side in side_means:
            group_means = neighbour_window_statistics(group_times, group_values, TREND_SIDE_WINDOW_SECONDS, None, side)['mean']
            side_means[side][key.order[group_start:group_end]] = group_means
    return side_means


def build_neighbour_block(block_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function builds one neighbor block per runway and per airport, for each clue and each time window
    """
    # whole epoch seconds: departure_metadata stores the take-off time in microseconds
    takeoff_seconds = (frame.MVT_TIME_UTC_mvt.astype('int64') // 10 ** 6).to_numpy().astype(float)
    base_runway = pd.read_parquet(FEATURE_MATRIX / 'base_features.parquet', columns=['RUNWAY_mvt'])
    evidence = neighbour_evidence(block_name, frame)
    prefix = NEIGHBOUR_COLUMN_PREFIX[block_name]
    count_kept_for = ('ads',) if block_name == 'neighbours_adsb' else ('nmb',)
    block = pd.DataFrame(index=frame.index)
    level_keys = {'rwy': (frame.ADEP_mvt.astype(str) + ':' + base_runway.RUNWAY_mvt.astype(str)).to_numpy(),
                  'apt': frame.ADEP_mvt.astype(str).to_numpy()}
    for level, key_values in level_keys.items():
        key = key_order(key_values, takeoff_seconds)
        for evidence_name, (values, threshold) in evidence.items():
            for window in NEIGHBOUR_WINDOWS_SECONDS[block_name]:
                statistic_columns = window_statistic_columns(key, values, window, threshold, need_max=block_name != 'neighbours_wide')
                for statistic, statistic_values in statistic_columns.items():
                    # the count of neighbours with evidence is kept once per source
                    skip_count = statistic == 'n' and evidence_name not in count_kept_for
                    if skip_count or (statistic == 'max' and block_name == 'neighbours_wide'):
                        continue
                    block[f'{prefix}_{level}_{evidence_name}_{statistic}_{window}'] = statistic_values.astype('float32')
        if block_name == 'neighbours':   # trend: evidence of the take-offs just before against those just after
            for evidence_name in ('nmb', 'tke'):
                side_means = side_mean_columns(key, evidence[evidence_name][0])
                block[f'{prefix}_{level}_{evidence_name}_before_900'] = side_means['before'].astype('float32')
                block[f'{prefix}_{level}_{evidence_name}_after_900'] = side_means['after'].astype('float32')
                block[f'{prefix}_{level}_{evidence_name}_trend_900'] = (side_means['after'] - side_means['before']).astype('float32')
        print(level, 'done', flush=True)
    relative_evidence = {'neighbours_adsb': ('nmb', 'ads'), 'neighbours': ('nmb', 'tke', 'late'), 'neighbours_wide': ()}[block_name]
    for evidence_name in relative_evidence:   # the row's own evidence relative to its neighbours
        own_values = evidence[evidence_name][0]
        block[f'{prefix}_rwy_{evidence_name}_rel_900'] = (own_values - block[f'{prefix}_rwy_{evidence_name}_mean_900']).astype('float32')
    return block


# metar_weather
METAR_DIRECTORY = DOWNLOADS / 'metar'
# present-weather flags: column name -> METAR code searched in the weather codes
WEATHER_CODE_FLAGS = {'fg': 'FG', 'br': 'BR', 'sn': 'SN', 'ra': 'RA', 'dz': 'DZ', 'ts': 'TS', 'fz': 'FZ', 'sh': 'SH'}


def read_metar_csv(station: str) -> pd.DataFrame:
    """
    This function reads one airport's downloaded weather reports
    """
    report_file = METAR_DIRECTORY / f'{station}.csv'
    raw = pd.read_csv(report_file, na_values=['M'], low_memory=False)
    raw['t'] = pd.to_datetime(raw.valid, utc=True)
    return raw.sort_values('t').drop_duplicates('t').reset_index(drop=True)


def metar_number(raw: pd.DataFrame, column: str) -> pd.Series:
    """
    This function turns a weather column into numbers
    """
    return pd.to_numeric(raw[column].replace('T', 0.0), errors='coerce')


def ceiling_feet(raw: pd.DataFrame) -> pd.Series:
    """
    This function finds the cloud ceiling, the height of the lowest layer of cloud that covers much of the sky - 12,000 feet when there is none, and never more than that
    """
    layers = [np.where(raw[f'skyc{layer}'].astype(str).str.strip().isin(['BKN', 'OVC', 'VV']), metar_number(raw, f'skyl{layer}'), np.nan) for layer in range(1, 5)]      # the archive has up to 4 cloud layers, skyc1-4 / skyl1-4
    return pd.Series(np.nanmin(np.stack(layers, 1), axis=1)).fillna(CEILING_CAP_FEET).clip(upper=CEILING_CAP_FEET)


def lowest_runway_visual_range(raw: pd.DataFrame) -> pd.Series:
    """
    This function finds how far a pilot can see along the runway (the lowest runway visual range in the report) - 2,000 meters when there is none, and never more than that
    """
    runway_visual_ranges = raw.metar.astype(str).str.findall(r' R\d\d[LRC]?/[PM]?(\d{4})')
    lowest = runway_visual_ranges.map(lambda ranges: min(int(value) for value in ranges) if ranges else np.nan)
    return lowest.astype(float).fillna(RVR_CAP_METRES).clip(upper=RVR_CAP_METRES)


def current_weather_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    This function turns each coded weather report into a tidy row of numbers and yes/no flags for things like visibility, cloud ceiling, etc.
    """
    reports = pd.DataFrame({'t': raw.t})
    reports['vis'] = (metar_number(raw, 'vsby') * 1609.34).clip(upper=VISIBILITY_CAP_METRES)            # statute miles -> m
    reports['ceil'] = ceiling_feet(raw)
    reports['rvr'] = lowest_runway_visual_range(raw)
    reports['lvp'] = ((reports.vis < LVP_VISIBILITY_METRES) | (reports.ceil < LVP_CEILING_FEET) | (reports.rvr < LVP_RVR_METRES)).astype(float)
    reports['imc'] = ((reports.vis < IMC_VISIBILITY_METRES) | (reports.ceil < IMC_CEILING_FEET)).astype(float)
    weather_codes = raw.wxcodes.fillna('').astype(str)
    for flag, code in WEATHER_CODE_FLAGS.items():
        reports[flag] = weather_codes.str.contains(code).astype(float)
    reports['heavy'] = weather_codes.str.contains(r'\+').astype(float)
    reports['temp'] = (metar_number(raw, 'tmpf') - 32) / 1.8
    reports['spread'] = (metar_number(raw, 'tmpf') - metar_number(raw, 'dwpf')) / 1.8
    reports['spd'] = metar_number(raw, 'sknt')
    reports['gust'] = metar_number(raw, 'gust').fillna(reports.spd)
    reports['dir'] = metar_number(raw, 'drct')
    reports['vrb'] = raw.metar.astype(str).str.contains(r' VRB\d\d| \d{3}V\d{3} ').astype(float)
    reports['qnh'] = metar_number(raw, 'alti') * 33.8639                                 # inches of mercury -> hPa
    return reports


def flight_category_codes(reports: pd.DataFrame) -> np.ndarray:
    """
    This function sorts each report into one of four weather classes by visibility and cloud ceiling, from 0 (good weather, VFR) to 3 (the worst, LIFR)
    """
    conditions = [(reports.vis < LIFR_VISIBILITY_METRES) | (reports.ceil < LIFR_CEILING_FEET),
                  (reports.vis < IFR_VISIBILITY_METRES) | (reports.ceil < IFR_CEILING_FEET),
                  (reports.vis < MVFR_VISIBILITY_METRES) | (reports.ceil < MVFR_CEILING_FEET)]
    return np.select(conditions, [3, 2, 1], 0).astype(float)


def wind_shift_degrees(reports: pd.DataFrame) -> pd.Series:
    """
    This function measures how far the wind direction has turned (0 to 180 degrees) compared with the average wind of the previous 2 hours
    """
    direction_radians = np.deg2rad(reports.dir)
    wind_u = (reports.spd * np.sin(direction_radians))
    wind_v = (reports.spd * np.cos(direction_radians))
    mean_u = wind_u.rolling('2h', closed='left').mean()
    mean_v = wind_v.rolling('2h', closed='left').mean()
    shift = np.rad2deg(np.arctan2(wind_u, wind_v) - np.arctan2(mean_u, mean_v))
    return np.abs((shift + 180) % 360 - 180).where((reports.spd >= WIND_SHIFT_MIN_SPEED_KNOTS), 0.0)


def add_weather_history_columns(reports: pd.DataFrame) -> None:
    """
    This function gives the weather table a memory for things such as the pressure change over 3 hours, icing and snow over 3 hours etc.
    """
    reports['qnh_d3h'] = reports.qnh - reports.qnh.rolling('3h', closed='left').mean()
    # icing: at most 3 deg C with snow, rain, drizzle or freezing precipitation, or at most 0 deg C with fog / mist
    precipitation = reports.sn + reports.ra + reports.dz + reports.fz > 0
    freezing_fog = (reports.temp <= 0) & (reports.fg + reports.br > 0)
    icing = ((reports.temp <= ICING_MAX_TEMPERATURE_CELSIUS) & (precipitation | freezing_fog)).astype(float)
    reports['icing'] = icing
    for hours in ICING_SNOW_WINDOWS_HOURS:
        reports[f'icing_{hours}h'] = icing.rolling(f'{hours}h').mean()
        reports[f'sn_{hours}h'] = reports.sn.rolling(f'{hours}h').mean()
    reports['icing_day'] = icing.groupby(reports.index.floor('D')).transform('mean')
    reports['sn_day'] = reports.sn.groupby(reports.index.floor('D')).transform('mean')
    reports['lvp_2h'] = reports.lvp.rolling('2h').mean()
    reports['ts_90m'] = reports.ts.rolling('90min').max()
    reports['gust_90m'] = reports.gust.rolling('90min').max()
    flight_category = flight_category_codes(reports)
    reports['fcat'] = flight_category
    flight_category_series = pd.Series(flight_category, index=reports.index)
    reports['fcat_change_2h'] = flight_category_series - flight_category_series.rolling('2h', closed='left').mean()
    reports['wind_shift_2h'] = wind_shift_degrees(reports)


def metar_reports(station: str) -> pd.DataFrame:
    """
    This function runs the whole weather pipeline above
    """
    raw = read_metar_csv(station)
    reports = current_weather_columns(raw)
    reports = reports.set_index('t')
    add_weather_history_columns(reports)
    return reports.reset_index()


def add_runway_wind_columns(part: pd.DataFrame, runway_heading: np.ndarray) -> None:
    """
    This function splits the wind into parts along and across the runway
    """
    wind_angle = np.deg2rad(part.dir.to_numpy() - runway_heading)
    part['head'] = part.spd.to_numpy() * np.cos(wind_angle)
    part['tail'] = np.maximum(-part['head'], 0)
    part['cross'] = np.abs(part.spd.to_numpy() * np.sin(wind_angle))
    part['gust_cross'] = np.abs(part.gust.to_numpy() * np.sin(wind_angle))
    part['gust_factor'] = part.gust - part.spd
    part['tail_over_5'] = (part['tail'] > TAIL_WIND_FLAG_KNOTS).astype(float)
    part['cross_over_15'] = (part['cross'] > CROSS_WIND_FLAG_KNOTS).astype(float)


def build_metar_weather_block(frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function finds the latest weather report at or before each takeoff and the wind along its runway
    """
    base_runway = pd.read_parquet(FEATURE_MATRIX / 'base_features.parquet', columns=['RUNWAY_mvt'])
    takeoff_times = frame.MVT_TIME_UTC_mvt.dt.as_unit('ns')
    block: pd.DataFrame | None = None
    for airport, rows in frame.groupby('ADEP_mvt').groups.items():
        reports = metar_reports(airport)
        rows = np.asarray(rows)
        report_times = reports.t.dt.as_unit('ns')
        row_takeoff_times = takeoff_times.iloc[rows]
        report_nanoseconds = report_times.astype('int64').to_numpy()
        report_position = np.searchsorted(report_nanoseconds, row_takeoff_times.astype('int64').to_numpy(), side='right') - 1
        latest_report_time = report_times.to_numpy()[np.maximum(report_position, 0)]
        report_age = (row_takeoff_times.to_numpy() - latest_report_time) / np.timedelta64(1, 's')
        part = reports.drop(columns='t').iloc[np.maximum(report_position, 0)].reset_index(drop=True)
        no_recent_report = (report_position < 0) | (report_age > METAR_MAX_AGE_SECONDS)
        part[no_recent_report] = np.nan
        part['report_age'] = np.where(report_position >= 0, report_age, np.nan)
        runway_number = base_runway.RUNWAY_mvt.iloc[rows].astype(str).str.extract(r'(\d+)', expand=False)      # '09L' -> '09'
        runway_heading = pd.to_numeric(runway_number, errors='coerce').to_numpy() * 10.0      # a runway's number is its magnetic heading in tens of degrees: 09 -> 090
        add_runway_wind_columns(part, runway_heading)
        part.index = rows
        block = part if block is None else pd.concat([block, part])
        print(airport, len(reports), 'reports; rows without a report within 3 h: %.4f' % float(np.mean(no_recent_report)), flush=True)
    block = block.sort_index().drop(columns=['dir']).add_prefix('wx_').astype('float32')
    assert len(block) == len(frame)
    return block


# These next functions build data for the tree models
def build_airport_state(block_directory: Path) -> pd.DataFrame:
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['MVT_ID_mvt'])
    neighbours = pd.read_parquet(block_directory / 'block_neighbours.parquet')
    takeoff_sequence = pd.read_parquet(block_directory / 'block_takeoff_sequence.parquet').drop(columns=AIRPORT_STATE_DROPPED_COLUMNS)
    return pd.concat([metadata, neighbours, takeoff_sequence], axis=1)


def build_airport_state_extended(block_directory: Path) -> pd.DataFrame:
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['MVT_ID_mvt'])
    neighbours_wide = pd.read_parquet(block_directory / 'block_neighbours_wide.parquet')
    ordering = pd.read_parquet(block_directory / 'block_ordering.parquet')
    weather = pd.read_parquet(block_directory / 'block_metar_weather.parquet', columns=AIRPORT_STATE_EXTENDED_WEATHER_COLUMNS)
    return pd.concat([metadata, neighbours_wide, ordering, weather], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build traffic and weather feature blocks.')
    parser.add_argument('blocks', nargs='+', choices=list(BLOCK_NAMES + STATE_TABLE_NAMES))
    parser.add_argument('--out-dir', default=str(FEATURE_MATRIX))
    args = parser.parse_args()
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = departure_frame()
    all_rows = all_challenge_rows()
    departures = all_rows[all_rows.PHASE_mvt == 'DEP'].reset_index(drop=True)
    assert departures.MVT_ID_mvt.equals(frame.MVT_ID_mvt), 'row order differs from departure_metadata.parquet'
    builders: dict[str, Callable[[], pd.DataFrame]] = {
        'takeoff_sequence': lambda: build_takeoff_sequence_block(all_rows, departures),
        'unimpeded_times': lambda: build_unimpeded_times_block(frame, departures),
        'ordering': lambda: build_ordering_block(departures),
        'neighbours_adsb': lambda: build_neighbour_block('neighbours_adsb', frame),
        'neighbours': lambda: build_neighbour_block('neighbours', frame),
        'neighbours_wide': lambda: build_neighbour_block('neighbours_wide', frame),
        'metar_weather': lambda: build_metar_weather_block(frame)}
    for name in args.blocks:
        if name in STATE_TABLE_NAMES:
            state_table = (build_airport_state if name == 'airport_state' else build_airport_state_extended)(output_directory)
            state_table.to_parquet(output_directory / f'{name}.parquet', index=False)
            print(name, 'written', flush=True)
            continue
        block = builders[name]()
        assert len(block) == len(frame)
        block.to_parquet(output_directory / f'block_{name}.parquet', index=False)
        print(name, block.shape, 'missing share max %.3f' % float(block.isna().mean().max()), flush=True)


if __name__ == '__main__':
    main()
