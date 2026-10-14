"""
Every airplane carries a transponder. Every few seconds, it sends some data indicating its position, speed, plane ID etc. This is all recorded in the asdb.lol archive. The problem is that this data doesn't say which flight it belongs to, so this file is a kind of matchmaker.

This is how it works:
-for all flights in challenge files:
    -first try matching flights by radio name or by finding a take-off/landing 90 seconds before/after the recorded time
    -then try expanding the take-off/landing window to 2 minutes
    -then try matching flights by airplane make and model
-for all matched flights:
    -calculate out a flight's drive to the runway and answer several questions such as:
        1. when did the plane first appear?
        2. where did it stand still?
        3. when did it leave its stand?
        4. how far did it still have to drive one minute, five minutes, an hour before take-off?
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# AIRPORT_REFERENCE_POINTS: (latitude, longitude) of each airport, the origin of the local x / y metres
from taxi.main import ADSB_DAY_FILES, AIRPORT_REFERENCE_POINTS, DOWNLOADS, FEATURE_MATRIX, competition_files

POINTS_DIRECTORY = DOWNLOADS / 'adsblol/points'                  # points_<day>.parquet: reduced adsb.lol points of one UTC day
PASS1_DAY_FILES = ADSB_DAY_FILES / 'pass1'                       # pass1_<day>.parquet
PASS2_DAY_FILES = ADSB_DAY_FILES / 'pass2'                       # pass2_<day>.parquet
PASS3_DAY_FILES = ADSB_DAY_FILES / 'pass3'                       # pass3_<day>.parquet
STAND_MAP_FILE = ADSB_DAY_FILES / 'stand_map.parquet'            # airport, STAND_mvt, x, y, n, spread (read by pass 1 / 2 / 3)
ADSB_ONLY_STAND_MAP_FILE = ADSB_DAY_FILES / 'stand_map_adsb_only.parquet'   # the stand map from parked arrivals alone (stand-map)

SLOW_SPEED_KNOTS = 50.0     # below this ground speed an aircraft is on the surface (taxiing or parked)
MOVEMENT_COLUMNS = [
    'MVT_ID_mvt',
    'PHASE_mvt',
    'ADEP_mvt',
    'ADES_mvt',
    'MVT_TIME_UTC_mvt',
    'BLOCK_TIME_UTC_mvt',
    'AOBT_3_flt',
    'SCHED_TIME_UTC_mvt',
    'EOBT_1_flt',
    'CALLSIGN_flt',
    'FLIGHT_mvt',
    'STAND_mvt'
]
# Airports whose X-Plane stand names agree with the ADS-B stand map
XPLANE_STAND_AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LIRF']
# Seconds before the movement time at which the departure's surface state is sampled (ads_togo_<s>, ads_gs_<s>, ...).
TIME_GRID_SECONDS = (60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 720, 840, 960, 1080, 1200, 1350, 1500, 1800, 2100, 2400, 3000, 3600)

# Matching windows (seconds) around the recorded movement time (take-off or landing time of the challenge files)
CALLSIGN_WINDOW_SECONDS = 900          # pass 1: a callsign match's take-off / landing lies closer than this
TIME_WINDOW_SECONDS = 90               # pass 1 without a callsign: the only take-off / landing in [-90 s, +90 s)
LATER_PASS_WINDOW_SECONDS = 120        # pass 2 / 3: free take-offs in [-120 s, +120 s)
MIN_RUNNER_UP_MARGIN_SECONDS = 20      # pass 2 / 3: a pair is skipped when the next candidate is less than this much farther
POINT_COLUMNS = ['icao', 't', 'lat', 'lon', 'ground', 'gs', 'callsign', 'airport', 'src']   # read from the points files
ESTIMATED_AIRPLANE_SIZE = 40           # metres: about the length of a typical airliner (an A320 or a 737-800 is 38-40 m long), so a point this close is still at the stand
ESTIMATED_RUNWAY_WAIT_TIME = 300       # 5 minutes: from my own flying experience, a plane is very rarely idle on the runway for longer

# Process-wide inputs, filled by run_pass_over_days before the worker pool starts (the forked workers inherit them).
STAND_POSITIONS: dict[tuple[str, str], tuple[float, float]] = {}   # (airport, stand) -> local (x, y) of the stand map
AIRCRAFT_TYPE_BY_MOVEMENT: dict[object, str] = {}                  # movement id -> aircraft type designator (pass 3 only)
# Output folder of each pass; the CLI's --out-dir replaces one entry (e.g. the pass-1 run with the ADS-B-only stand map).
PASS_OUTPUT_DIRECTORIES: dict[str, Path] = {'pass1': PASS1_DAY_FILES, 'pass2': PASS2_DAY_FILES, 'pass3': PASS3_DAY_FILES}
_MOVEMENTS_CACHE: pd.DataFrame | None = None

POINTS_BEFORE_DAY_SECONDS = 6 * 3600           # 6 h of the day before, so flights near midnight are not cut off with room to spare just in case there is a REALLY long delay
POINTS_AFTER_DAY_START_SECONDS = 30 * 3600     # the whole day plus the same 6 h margin into the next day

# Departure surface sequence and its features
SURFACE_LOOKBACK_SECONDS = 4 * 3600            # a round limit far longer than a normal taxi-out, so a plane parked for many hours does not bring its whole stay at the stand
STATIONARY_SPEED_KNOTS = 3                     # below this ground speed a point counts as stationary
DWELL_MAX_STEP_METRES = 15                     # a still step moves less than this: well under one plane length, so small position noise does not break a stop
DWELL_GAP_STEP_SECONDS = 30                    # a step longer than this counts as a gap in the data, so the speed reported before it is not trusted
MIN_DWELL_SECONDS = 60                         # one minute - shorter pauses are left out so only real stops count
FAR_DWELL_LEAVE_METRES = (10, 30, 80)          # three hand-picked steps from 'just started to roll' (10 m) to 'clearly gone' (80 m, about two plane lengths)
RECEPTION_GAP_SECONDS = (120, 300, 600, 1200)  # silences of 2, 5, 10 and 20 minutes
GRID_BEFORE_FIRST_POINT_SECONDS = 1            # 1 s of slack, so a snapshot moment just before the first point is still taken
MIN_PATH_FOR_SHARE_METRES = 1.0                # only there to stop a division by zero when the whole path is 0 m long

# Arrivals and the stand map
PARKED_RECEPTION_GAP_SECONDS = 4 * 3600        # the same round 4 h as SURFACE_LOOKBACK_SECONDS - a longer silence ends the parked part of the track
PARKED_AFTER_IN_BLOCK_SECONDS = 120            # 2 minutes of slack after the recorded in-block time, which is rounded to the minute
STAND_MAP_MAX_BLOCK_OFFSET_SECONDS = 300       # the parked position must lie within 5 minutes of the in-block time, so the plane has really reached its stand
STAND_MAP_MAX_PARKED_SPEED_KNOTS = 2           # a parked plane reports about 0 kt, and 2 kt leaves a little room for speed noise
MIN_STAND_OBSERVATIONS = 2                     # at least two arrivals, so a single bad match can never place a stand on its own
MAX_STAND_SPREAD_METRES = 40                   # about one airliner length, as above

# Passes 2 and 3
SOLE_CANDIDATE_MARGIN_SECONDS = 999.0          # stands for 'no runner-up': far above MIN_RUNNER_UP_MARGIN_SECONDS (20 s), so a sole candidate always passes
TYPE_LOOKUP_DAY_OFFSETS = (-1, 0, 1)           # the day before, the day itself and the day after: the same three points files that load_points reads

# Worker processes of the matching passes (main)
PASS1_WORKER_PROCESSES = 6                     # to keep memory within 32GB limit
LATER_PASS_WORKER_PROCESSES = 3                # to keep memory within 32GB limit


"""
This data structure just describes a flight (e.g. did it arrive or depart from an airport, and so on)
"""
class MovementRow(Protocol):
    MVT_ID_mvt: float
    PHASE_mvt: str                    # 'DEP' (departure) or 'ARR' (arrival)
    airport: str                      # the movement's own airport
    STAND_mvt: str | float            # stand (parking position) name; NaN when unknown
    movement_time: float              # take-off / landing time, epoch seconds
    arrival_block_time: float
    nm_off_block_time: float
    scheduled_time: float
    estimated_off_block_time: float
    callsign: str


TrackIndex = dict[tuple[str, str], 'AircraftTrack']                                # (icao, airport) -> track
TransitionEntry = tuple[float, str, int]                                           # (transition time, icao, point index of the track)
Match = tuple[float, str, int, int]                                                # (|transition time - movement time|, icao, point index, ads_cs_match flag)
FreeTransitions = dict[str, tuple[np.ndarray, list[TransitionEntry]]]              # airport -> (times, entries), both in time order
CandidatePair = tuple[float, int, str, int, float]                                 # (time difference, row position, airport, candidate index, runner-up margin)
TransitionLookup = dict[str, dict[str, tuple[np.ndarray, list[str], list[int]]]]   # phase -> airport -> (times, icaos, point indices)


def movements() -> pd.DataFrame:
    """
    This function reads all flights in the competition data and adds some information to it, such as the airport it arrived at/left from
    It only needs to do this once - all this data is cached so that future calls are ~instant (or at the very least much faster)
    """
    global _MOVEMENTS_CACHE
    if _MOVEMENTS_CACHE is None:
        movement_rows = pd.concat([pd.read_parquet(path, columns=MOVEMENT_COLUMNS) for path in competition_files()], ignore_index=True)
        movement_rows['airport'] = np.where(movement_rows.PHASE_mvt.eq('DEP'), movement_rows.ADEP_mvt, movement_rows.ADES_mvt)
        # the times are stored with microsecond resolution
        movement_rows['movement_time'] = movement_rows.MVT_TIME_UTC_mvt.astype('int64') / 1e6
        # arrival block time only - departure blocks are never used because that is what we need to predict
        # an arrival without a block time gets ~0 so no parked position qualifies for it
        arrival_block_seconds = movement_rows.BLOCK_TIME_UTC_mvt.astype('int64') / 1e6
        movement_rows['arrival_block_time'] = np.where(movement_rows.PHASE_mvt.eq('ARR'), arrival_block_seconds, np.nan)
        movement_rows['nm_off_block_time'] = movement_rows.AOBT_3_flt.astype('int64') / 1e6
        movement_rows.loc[movement_rows.AOBT_3_flt.isna(), 'nm_off_block_time'] = np.nan
        for source_column, time_column in (('SCHED_TIME_UTC_mvt', 'scheduled_time'), ('EOBT_1_flt', 'estimated_off_block_time')):
            movement_rows[time_column] = movement_rows[source_column].astype('int64') / 1e6
            movement_rows.loc[movement_rows[source_column].isna(), time_column] = np.nan
        movement_rows['day'] = movement_rows.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
        movement_rows['callsign'] = movement_rows.CALLSIGN_flt.fillna('').str.strip()
        source_columns = ['BLOCK_TIME_UTC_mvt', 'CALLSIGN_flt', 'AOBT_3_flt', 'SCHED_TIME_UTC_mvt', 'EOBT_1_flt']
        _MOVEMENTS_CACHE = movement_rows.drop(columns=source_columns)
    return _MOVEMENTS_CACHE


def point_file(day: pd.Timestamp) -> Path:
    """
    This function takes in a day and returns the name of the file which corresponds to where the asdb data is for that day
    """
    return POINTS_DIRECTORY / f'points_{day.strftime("%Y-%m-%d")}.parquet'


def load_points(day: str) -> pd.DataFrame | None:
    """
    This function loads all the asdb data for that day (includes data taken 6 hours before and 30 hours after) - so technically it reads 3 files
    The reason for this is some flights leave near midnight, and some data is cut off, which is detrimental to the model
    """
    day_start = pd.Timestamp(day)
    parts: list[pd.DataFrame] = []
    for day_offset in (-1, 0, 1):
        path = point_file(day_start + pd.Timedelta(days=day_offset))
        if path.exists():
            parts.append(pd.read_parquet(path, columns=POINT_COLUMNS))
    if not parts:
        return None
    points = pd.concat(parts, ignore_index=True)
    # The -6/+30 hour window is applied after the concatenation
    window_start = day_start.timestamp() - POINTS_BEFORE_DAY_SECONDS
    window_end = day_start.timestamp() + POINTS_AFTER_DAY_START_SECONDS
    points = points[(points.t >= window_start) & (points.t < window_end)]
    # The address becomes a category before de-duplicating and sorting
    points['icao'] = points.icao.astype(object).astype('category')
    points = one_point_per_address_and_time(points)
    points['callsign'] = points.callsign.fillna('').str.strip().astype(object).astype('category')
    points['mlat'] = points.src.eq('mlat').fillna(False).to_numpy(dtype=bool)
    points['airport'] = points.airport.astype(object).astype('category')
    return points.drop(columns='src')


def one_point_per_address_and_time(points: pd.DataFrame) -> pd.DataFrame:
    """
    This function makes sure that each plane only has one dot per moment, because a plane can only be in one place at a time
    First it sorts the points by plane and then by time
    """
    if points.t.dtype != np.float64: # if not sortable by numpy
        return points.drop_duplicates(['icao', 't']).sort_values(['icao', 't'], kind='stable').reset_index(drop=True)
    address_codes = points.icao.cat.codes.to_numpy().astype(np.int64)
    address_codes = np.where(address_codes < 0, len(points.icao.cat.categories), address_codes)
    times = points.t.to_numpy()
    order = np.lexsort((times, address_codes))
    sorted_codes = address_codes[order]
    sorted_times = times[order]
    same_time = (sorted_times[1:] == sorted_times[:-1]) | (np.isnan(sorted_times[1:]) & np.isnan(sorted_times[:-1]))
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = (sorted_codes[1:] != sorted_codes[:-1]) | ~same_time
    return points.iloc[order[is_first]].reset_index(drop=True)


def local_xy_metres(latitude: np.ndarray | float, longitude: np.ndarray | float, reference_latitude: float, reference_longitude: float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    This function turns latitude/longitude positions into north/east distances from the center of the airport - flat earth projections
    """
    # the usual flat-earth factors: about 111,320 m per degree of longitude at the equator (shrinking with cos(latitude)) and about 110,540 m per degree of latitude
    metres_east = (longitude - reference_longitude) * 111320.0 * np.cos(np.deg2rad(reference_latitude))
    metres_north = (latitude - reference_latitude) * 110540.0
    return metres_east, metres_north


class AircraftTrack:
    """
    This is a class that takes in an airplanes data at an airport and figures out things like when an airplane started the take-off sequence (speed increase)
    """

    def __init__(self, track_points: pd.DataFrame) -> None:
        self.fill(track_points.t.to_numpy(), track_points.lat.to_numpy(), track_points.lon.to_numpy(), track_points.gs.to_numpy(dtype=float), track_points.ground.to_numpy(), track_points.mlat.to_numpy())

    @classmethod
    def from_arrays(cls, times: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, reported_speeds: np.ndarray, on_ground: np.ndarray, is_mlat: np.ndarray) -> AircraftTrack:
        track = cls.__new__(cls)
        track.fill(times, latitudes, longitudes, reported_speeds, on_ground, is_mlat)
        return track

    def fill(self, times: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, reported_speeds: np.ndarray, on_ground: np.ndarray, is_mlat: np.ndarray) -> None:
        """
        This is the actual function that figures out airplane speed and builds the track
        """
        self.times = times
        self.latitudes = latitudes
        self.longitudes = longitudes
        # a point on the ground without a speed report counts as stationary
        self.ground_speeds = np.where(np.isnan(reported_speeds) & on_ground, 0.0, reported_speeds)
        self.is_mlat = is_mlat
        self.is_slow = (self.ground_speeds < SLOW_SPEED_KNOTS) | (on_ground & np.isnan(reported_speeds))
        is_fast = ~self.is_slow
        previous_is_slow = np.r_[False, self.is_slow[:-1]]
        # take-off roll starts (slow -> fast), landing ends (fast -> slow)
        self.takeoff_roll_starts = np.flatnonzero(is_fast & previous_is_slow)
        self.landing_ends = np.flatnonzero(self.is_slow & np.r_[False, is_fast[:-1]])


# departure track features
@dataclass
class SurfaceSequence:
    """
    This data structure stores information about a departure
    """
    times: np.ndarray                    # epoch s
    speeds: np.ndarray                   # ground speed, knots
    is_mlat: np.ndarray                  # multilateration position
    x: np.ndarray
    y: np.ndarray
    step_lengths: np.ndarray             # metres between consecutive points (one value fewer than points)
    step_durations: np.ndarray           # seconds between consecutive points
    path_to_go: np.ndarray               # metres of path from each point to the take-off-roll start (0 at the last point)
    distance_to_stand: np.ndarray | None # metres from each point to the stand; None when the stand position is unknown


def last_point_before_surface_sequence(track: AircraftTrack, roll_start: int, movement_time: float) -> int:
    """
    This function figures out where the take-off run starts
    """
    before_sequence = roll_start - 1
    while before_sequence >= 0 and track.is_slow[before_sequence] and track.times[before_sequence] > movement_time - SURFACE_LOOKBACK_SECONDS:
        before_sequence -= 1
    return before_sequence


def surface_sequence(track: AircraftTrack, point_indices: np.ndarray, stand_xy: tuple[float, float] | None, reference_latitude: float, reference_longitude: float) -> SurfaceSequence:
    """
    Builds the data structure containing information about a departure
    """
    times = track.times[point_indices]
    x, y = local_xy_metres(track.latitudes[point_indices], track.longitudes[point_indices], reference_latitude, reference_longitude)
    step_lengths = np.hypot(np.diff(x), np.diff(y))
    path_to_go = np.r_[np.cumsum(step_lengths[::-1])[::-1], 0.0]
    distance_to_stand = np.hypot(x - stand_xy[0], y - stand_xy[1]) if stand_xy is not None else None
    return SurfaceSequence(times, track.ground_speeds[point_indices], track.is_mlat[point_indices], x, y, step_lengths, np.diff(times), path_to_go, distance_to_stand)


def sequence_summary_features(sequence: SurfaceSequence, movement_time: float) -> dict[str, float]:
    """
    This function returns details about the airplane drive such as when the take-off run started
    """
    return {
        'ads_first_rel': sequence.times[0] - movement_time,
        'ads_first_gs': sequence.speeds[0],
        'ads_first_x': sequence.x[0],
        'ads_first_y': sequence.y[0],
        'ads_tko_x': sequence.x[-1],
        'ads_tko_y': sequence.y[-1],
        'ads_first_mlat': float(sequence.is_mlat[0]),
        'ads_mlat_frac': float(np.mean(sequence.is_mlat)),
        'ads_path': sequence.path_to_go[0],
        'ads_maxgap': sequence.step_durations.max(),
    }


def runs_of_true(flags: np.ndarray) -> list[tuple[int, int]]:
    """
    This function figures out at which points the airplane stops and starts
    """
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def dwells(sequence: SurfaceSequence) -> list[tuple[int, int]]:
    """
    This function finds times when airplanes stand still for at least 60 seconds
    """
    is_still_step = (sequence.step_lengths < DWELL_MAX_STEP_METRES) & ((sequence.speeds[:-1] < STATIONARY_SPEED_KNOTS) | (sequence.step_durations > DWELL_GAP_STEP_SECONDS))
    # a run of still steps first .. end - 1 covers the points first .. end; `end` is the point where the dwell ends
    return [(first, end) for first, end in runs_of_true(is_still_step) if sequence.times[end] - sequence.times[first] >= MIN_DWELL_SECONDS]


def dwell_features(sequence: SurfaceSequence, dwell_runs: list[tuple[int, int]], movement_time: float) -> dict[str, float]:
    """
    Turns these standstill times into usable information, which can help figure out when the airplane is idle or when it is waiting on the runway
    I have chosen a time of 300 seconds just based on my own experience flying, where 99% of time time, the most time we are idle on the runway if 5 minutes before moving (even if that move is just a few meters)
    """
    times = sequence.times
    path_to_go = sequence.path_to_go
    features: dict[str, float] = {'ads_ndwell': len(dwell_runs)}
    if not dwell_runs:
        return features
    selected_dwells = {
        'last': dwell_runs[-1],
        'far': max(dwell_runs, key=lambda dwell: (path_to_go[dwell[1]], times[dwell[1]])),
        'long': max(dwell_runs, key=lambda dwell: times[dwell[1]] - times[dwell[0]])
    }
    dwells_of_300_seconds = [dwell for dwell in dwell_runs if times[dwell[1]] - times[dwell[0]] >= ESTIMATED_RUNWAY_WAIT_TIME]
    if dwells_of_300_seconds:
        selected_dwells['last300'] = dwells_of_300_seconds[-1]
    for name, (dwell_start, dwell_end) in selected_dwells.items():
        features[f'ads_dw_{name}_end_rel'] = times[dwell_end] - movement_time
        features[f'ads_dw_{name}_dur'] = times[dwell_end] - times[dwell_start]
        features[f'ads_dw_{name}_togo'] = path_to_go[dwell_end]
        features[f'ads_dw_{name}_gs_after'] = sequence.speeds[dwell_end + 1] if dwell_end + 1 < len(sequence.speeds) else np.nan
        features[f'ads_dw_{name}_dt_after'] = times[dwell_end + 1] - times[dwell_end] if dwell_end + 1 < len(times) else np.nan
        if sequence.distance_to_stand is not None:
            features[f'ads_dw_{name}_dstand'] = sequence.distance_to_stand[dwell_end]
    _, far_dwell_end = selected_dwells['far']
    distance_from_dwell_end = np.hypot(sequence.x - sequence.x[far_dwell_end], sequence.y - sequence.y[far_dwell_end])
    is_after_dwell = np.arange(len(distance_from_dwell_end)) > far_dwell_end
    for leave_distance in FAR_DWELL_LEAVE_METRES:
        points_away = np.flatnonzero((distance_from_dwell_end > leave_distance) & is_after_dwell)
        if len(points_away):
            features[f'ads_far_leave{leave_distance}_rel'] = times[points_away[0]] - movement_time
            features[f'ads_far_leave{leave_distance}_prev_rel'] = times[points_away[0] - 1] - movement_time
    return features


def reception_gap_features(sequence: SurfaceSequence, movement_time: float) -> dict[str, float]:
    """
    Sometimes there are radio silences, when no antenna can hear the plane's signals
    This function figures out the length of the silence, the speed and distance moved during this time, and other features
    """
    features: dict[str, float] = {}
    for gap_seconds in RECEPTION_GAP_SECONDS:
        long_gap_steps = np.flatnonzero(sequence.step_durations >= gap_seconds)
        if not len(long_gap_steps):
            continue
        last_gap_step = long_gap_steps[-1]
        first_point_after_gap = last_gap_step + 1
        features[f'ads_gap{gap_seconds}_end_rel'] = sequence.times[first_point_after_gap] - movement_time
        features[f'ads_gap{gap_seconds}_len'] = sequence.step_durations[last_gap_step]
        features[f'ads_gap{gap_seconds}_gs'] = sequence.speeds[first_point_after_gap]
        features[f'ads_gap{gap_seconds}_disp'] = sequence.step_lengths[last_gap_step]
        features[f'ads_gap{gap_seconds}_togo'] = sequence.path_to_go[first_point_after_gap]
        features[f'ads_gap{gap_seconds}_n'] = len(long_gap_steps)
        if sequence.distance_to_stand is not None:
            features[f'ads_gap{gap_seconds}_dstand'] = sequence.distance_to_stand[first_point_after_gap]
    return features


def stand_features(sequence: SurfaceSequence, distance_to_stand: np.ndarray, stand_xy: tuple[float, float], movement_time: float) -> dict[str, float]:
    """
    This function finds out details about the stand, so that the start time of the taxi time can be accurately calculated (e.g. if it first starts moving from this stand after 13 hours, the taxi time started at this time)
    I have chosen a distance of 40 meters just arbitrarily to indicate when taxi-time started, but that is just a best-estimate based on average airplane size/length and not any real evidence
    """
    features: dict[str, float] = {
        'ads_first_dstand': distance_to_stand[0],
        'ads_min_dstand': distance_to_stand.min(),
        'ads_stand_to_tko': np.hypot(sequence.x[-1] - stand_xy[0], sequence.y[-1] - stand_xy[1]),
    }
    points_near_stand = np.flatnonzero(distance_to_stand < ESTIMATED_AIRPLANE_SIZE)
    if len(points_near_stand):
        last_point_near_stand = points_near_stand[-1]
        features['ads_near_stand_last_rel'] = sequence.times[last_point_near_stand] - movement_time
        features['ads_near_stand_gs'] = sequence.speeds[last_point_near_stand]
        first_point_after_stand = last_point_near_stand + 1
        if first_point_after_stand < len(sequence.times):
            features['ads_near_stand_next_rel'] = sequence.times[first_point_after_stand] - movement_time
            features['ads_near_stand_next_dstand'] = distance_to_stand[first_point_after_stand]
    return features

# These next set of functions are essentially performing some kind of interpolation to help figure out where the airplane is on the taxi journey
def bracketing_step(times: np.ndarray, query_time: float) -> tuple[int, float]:
    upper = np.searchsorted(times, query_time)
    upper = min(max(upper, 1), len(times) - 1)
    weight = (query_time - times[upper - 1]) / max(times[upper] - times[upper - 1], 1e-6) # minimum capped at 1/1000000th of a second
    weight = min(max(weight, 0.0), 1.0)
    return upper, weight


def interpolate(values: np.ndarray, upper: int, weight: float) -> float:
    return values[upper - 1] + weight * (values[upper] - values[upper - 1])


def nearer_point(times: np.ndarray, upper: int, query_time: float) -> int:
    return upper - 1 if query_time - times[upper - 1] < times[upper] - query_time else upper


def time_grid_features(sequence: SurfaceSequence, movement_time: float) -> dict[str, float]:
    """
    This function takes snapshots of the plane to figure out where in the taxi journey it is at
    I have chosen snapshots every 1 minute at the beginning, and this between-snapshot time gradually increases
    """
    features: dict[str, float] = {}
    for seconds_before in TIME_GRID_SECONDS:
        query_time = movement_time - seconds_before
        if query_time < sequence.times[0] - GRID_BEFORE_FIRST_POINT_SECONDS:
            continue
        upper, weight = bracketing_step(sequence.times, query_time)
        features[f'ads_togo_{seconds_before}'] = interpolate(sequence.path_to_go, upper, weight)
        features[f'ads_gap_{seconds_before}'] = sequence.times[upper] - sequence.times[upper - 1]
        features[f'ads_gs_{seconds_before}'] = sequence.speeds[nearer_point(sequence.times, upper, query_time)]
        if sequence.distance_to_stand is not None:
            features[f'ads_dstand_{seconds_before}'] = interpolate(sequence.distance_to_stand, upper, weight)
    return features


def reference_clock_features(sequence: SurfaceSequence, clocks: dict[str, float], leave_time: float, movement_time: float) -> dict[str, float]:
    """
    This function compares the scheduled time, the estimated off-block time, the reported off-block time with the physical pings the airplane gave off, recorded by the antennas in asdb.lol
    """
    times = sequence.times
    features: dict[str, float] = {}
    for name, clock_time in clocks.items():
        if not np.isfinite(clock_time):
            continue
        features[f'ads_first_minus_{name}'] = times[0] - clock_time
        features[f'ads_leave_minus_{name}'] = (movement_time + leave_time - clock_time) if np.isfinite(leave_time) else np.nan
        if times[0] <= clock_time <= times[-1]:
            upper, weight = bracketing_step(times, clock_time)
            path_still_to_go = interpolate(sequence.path_to_go, upper, weight)
            features[f'ads_togo_{name}'] = path_still_to_go
            features[f'ads_gs_{name}'] = sequence.speeds[nearer_point(times, upper, clock_time)]
            features[f'ads_gap_{name}'] = times[upper] - times[upper - 1]
            features[f'ads_frac_{name}'] = path_still_to_go / max(sequence.path_to_go[0], MIN_PATH_FOR_SHARE_METRES)
            if sequence.distance_to_stand is not None:
                features[f'ads_dstand_{name}'] = interpolate(sequence.distance_to_stand, upper, weight)
    return features


def departure_track_features(track: AircraftTrack, roll_start: int, movement_time: float, stand_xy: tuple[float, float] | None, reference_latitude: float, reference_longitude: float, nm_off_block_time: float = np.nan, reference_clocks: dict[str, float] | None = None) -> dict[str, float]:
    """
    This is the main function that turns one departure's track into its features
    It looks back from the start of the take-off run to the plane's previous flight (or at most 4 hours if that was longer)
    """
    times = track.times
    features: dict[str, float] = {'ads_tko_rel': times[roll_start] - movement_time}
    before_sequence = last_point_before_surface_sequence(track, roll_start, movement_time)
    point_indices = np.arange(before_sequence + 1, roll_start + 1)          # surface sequence incl. the first fast point
    previous_flight_seen = before_sequence >= 0 and not track.is_slow[before_sequence]
    features['ads_prev_fast_rel'] = (times[before_sequence] - movement_time) if previous_flight_seen else np.nan
    features['ads_nsurf'] = len(point_indices) - 1
    if len(point_indices) < 2:
        return features
    sequence = surface_sequence(track, point_indices, stand_xy, reference_latitude, reference_longitude)
    features.update(sequence_summary_features(sequence, movement_time))
    features.update(dwell_features(sequence, dwells(sequence), movement_time))
    features.update(reception_gap_features(sequence, movement_time))
    # stops are moving points followed by stationary ones
    is_moving = sequence.speeds >= STATIONARY_SPEED_KNOTS
    features['ads_nstops'] = int(np.sum(is_moving[:-1] & ~is_moving[1:]))
    if stand_xy is not None and sequence.distance_to_stand is not None:      # set together; checking both narrows the types
        features.update(stand_features(sequence, sequence.distance_to_stand, stand_xy, movement_time))
    features.update(time_grid_features(sequence, movement_time))
    # reference clocks in this order: those given (schedule, estimated off-block), then the NM actual off-block time
    clocks = dict(reference_clocks or {})
    clocks['aobt'] = nm_off_block_time
    # time of leaving the stand: the first point after the last one near the stand, else 30 m away from the farthest dwell
    leave_time = features.get('ads_near_stand_next_rel', features.get('ads_far_leave30_rel', np.nan))
    features.update(reference_clock_features(sequence, clocks, leave_time, movement_time))
    return features


def arrival_track_features(track: AircraftTrack, landing_end: int, movement_time: float, arrival_block_time: float, reference_latitude: float, reference_longitude: float) -> dict[str, float]:
    """
    Same as above, but for arrivals
    """
    times = track.times
    features: dict[str, float] = {'ads_ldg_rel': times[landing_end] - movement_time}
    last_slow = landing_end
    while last_slow + 1 < len(times) and track.is_slow[last_slow + 1] and times[last_slow + 1] - times[last_slow] < PARKED_RECEPTION_GAP_SECONDS:
        last_slow += 1
    slow_sequence = np.arange(landing_end, last_slow + 1)
    if np.isfinite(arrival_block_time):
        # position at the latest point up to in-block + 120 s
        points_before_block = slow_sequence[times[slow_sequence] <= arrival_block_time + PARKED_AFTER_IN_BLOCK_SECONDS]
        if len(points_before_block):
            parked_point = points_before_block[-1]
            x, y = local_xy_metres(track.latitudes[parked_point], track.longitudes[parked_point], reference_latitude, reference_longitude)
            features['ads_park_x'] = x
            features['ads_park_y'] = y
            features['ads_park_dt'] = times[parked_point] - arrival_block_time
            features['ads_park_gs'] = track.ground_speeds[parked_point]
    return features


def load_stand_map() -> dict[tuple[str, str], tuple[float, float]]:
    """
    This function reads the stand map into a quick lookup of where each stand is (in meters from the airport center) - it is empty if the stand map has not been built yet
    """
    if STAND_MAP_FILE.exists():
        stand_map = pd.read_parquet(STAND_MAP_FILE)
        return {(airport, stand): (x, y) for airport, stand, x, y in stand_map[['airport', 'STAND_mvt', 'x', 'y']].itertuples(index=False)}
    return {}


def departure_clock_features(track: AircraftTrack, roll_start: int, movement: MovementRow) -> dict[str, float]:
    """
    This function gets everything ready for one departure (the airport center, its stand position and its schedule and off-block times)
    """
    reference_latitude, reference_longitude = AIRPORT_REFERENCE_POINTS[movement.airport]
    stand_xy = STAND_POSITIONS.get((movement.airport, movement.STAND_mvt))
    clocks = {'sched': movement.scheduled_time, 'eobt': movement.estimated_off_block_time}
    return departure_track_features(track, roll_start, movement.movement_time, stand_xy, reference_latitude, reference_longitude, movement.nm_off_block_time, clocks)


# pass 1
def keep_closest_claim(matches: dict[int, Match], movement_rows: list[MovementRow]) -> dict[int, Match]:
    """
    This function makes sure that one take-off or landing pattern in the data goes to only one flight
    If several flights claimed the same one, only the flight whose recorded time is closest keeps it
    """
    winners: dict[tuple[str, str, int], int] = {}
    for row_position, match in matches.items():
        transition = (movement_rows[row_position].airport, match[1], match[2])
        holder = winners.get(transition)
        if holder is None or (match[0], movement_rows[row_position].MVT_ID_mvt) < (matches[holder][0], movement_rows[holder].MVT_ID_mvt):
            winners[transition] = row_position
    return {row_position: matches[row_position] for row_position in winners.values()}


def tracks_by_aircraft(points: pd.DataFrame) -> TrackIndex:
    """
    This function splits the day's dots into one track per plane and airport
    """
    groups = points.groupby(['icao', 'airport'], sort=False, observed=True)
    if not (isinstance(points.icao.dtype, pd.CategoricalDtype) and isinstance(points.airport.dtype, pd.CategoricalDtype)):
        # keys that are not categories
        return {(icao, airport): AircraftTrack(track_points) for (icao, airport), track_points in groups}
    # ngroup numbers the groups in order of first appearance
    group_numbers = groups.ngroup().to_numpy()
    rows_in_groups = np.flatnonzero(~np.isnan(group_numbers)) if group_numbers.dtype.kind == 'f' else np.arange(len(group_numbers))
    if len(rows_in_groups) == 0:
        return {}
    numbers = group_numbers[rows_in_groups].astype(np.int64)
    rows_by_group = rows_in_groups[np.argsort(numbers, kind='stable')]
    group_ends = np.cumsum(np.bincount(numbers))[:-1]
    columns = (points.t.to_numpy(), points.lat.to_numpy(), points.lon.to_numpy(), points.gs.to_numpy(dtype=float), points.ground.to_numpy(), points.mlat.to_numpy())
    addresses = points.icao.cat.categories
    address_codes = points.icao.cat.codes.to_numpy()
    airports = points.airport.cat.categories
    airport_codes = points.airport.cat.codes.to_numpy()
    tracks: TrackIndex = {}
    for group_rows in np.split(rows_by_group, group_ends):
        first_row = group_rows[0]
        key = (addresses[address_codes[first_row]], airports[airport_codes[first_row]])
        tracks[key] = AircraftTrack.from_arrays(*(column[group_rows] for column in columns))
    return tracks


def addresses_by_callsign(points: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """
    This function builds a phone book of which planes used which callsign (radio name) at each airport
    It is needed because the challenge files know a flight's callsign but not its plane ID
    """
    identified = points.loc[points.callsign != '', ['airport', 'callsign', 'icao']]
    # one row per (airport, callsign, address), at its first appearance; much faster than one group per callsign
    identified = identified[identified.airport.notna() & identified.callsign.notna()].drop_duplicates()
    addresses: dict[tuple[str, str], list[str]] = {}
    airports = identified.airport.astype(object)
    callsigns = identified.callsign.astype(object)
    icaos = identified.icao.astype(object)
    for airport, callsign, icao in zip(airports, callsigns, icaos):
        addresses.setdefault((airport, callsign), []).append(icao)
    return addresses


def transition_lookup(tracks: TrackIndex) -> TransitionLookup:
    """
    This function lists every take-off run start and every landing run end of the day per airport, sorted by time, to quickly find the ones that happened around a given time
    """
    entries_by_phase: dict[str, dict[str, list[TransitionEntry]]] = {'DEP': {}, 'ARR': {}}
    for (icao, airport), track in tracks.items():
        for phase, transition_indices in (('DEP', track.takeoff_roll_starts), ('ARR', track.landing_ends)):
            for point_index in transition_indices:
                entries_by_phase[phase].setdefault(airport, []).append((track.times[point_index], icao, point_index))
    lookup: TransitionLookup = {}
    for phase, entries_by_airport in entries_by_phase.items():
        lookup[phase] = {}
        for airport, entries in entries_by_airport.items():
            entries.sort()
            transition_times = np.array([entry[0] for entry in entries])
            lookup[phase][airport] = (transition_times, [entry[1] for entry in entries], [entry[2] for entry in entries])
    return lookup


def callsign_matches(movement_rows: list[MovementRow], tracks: TrackIndex, addresses: dict[tuple[str, str], list[str]]) -> dict[int, Match]:
    """
    This function matches flights by callsign (radio name)
    For each flight, it looks at the planes that used its callsign at its airport and picks the take-off or landing closest to the recorded time, if it is less than 15 minutes away
    """
    matches: dict[int, Match] = {}
    for row_position, movement in enumerate(movement_rows):
        best_match = None
        candidate_addresses = addresses.get((movement.airport, movement.callsign), []) if movement.callsign else []
        for icao in candidate_addresses:
            track = tracks.get((icao, movement.airport))
            if track is None:
                continue
            transition_indices = track.takeoff_roll_starts if movement.PHASE_mvt == 'DEP' else track.landing_ends
            if len(transition_indices) == 0:
                continue
            time_differences = np.abs(track.times[transition_indices] - movement.movement_time)
            nearest = time_differences.argmin()
            is_closer = best_match is None or time_differences[nearest] < best_match[0]
            if time_differences[nearest] < CALLSIGN_WINDOW_SECONDS and is_closer:
                best_match = (time_differences[nearest], icao, transition_indices[nearest], 1)
        if best_match is not None:
            matches[row_position] = best_match
    return matches


def time_window_matches(movement_rows: list[MovementRow], lookup: TransitionLookup, callsign_candidates: dict[int, Match], claimed: set[tuple[str, str, int]]) -> dict[int, Match]:
    """
    This function matches flights by looking at a +-90 seconds window for a take-off or landing
    """
    candidates: dict[int, Match] = {}
    candidate_transitions: dict[int, tuple[str, str, int]] = {}      # row position -> (airport, icao, point index)
    for row_position, movement in enumerate(movement_rows):
        if row_position in callsign_candidates or movement.airport not in lookup[movement.PHASE_mvt]:
            continue
        transition_times, transition_addresses, transition_points = lookup[movement.PHASE_mvt][movement.airport]
        window_start = np.searchsorted(transition_times, movement.movement_time - TIME_WINDOW_SECONDS)
        window_end = np.searchsorted(transition_times, movement.movement_time + TIME_WINDOW_SECONDS)
        if window_end - window_start != 1:
            continue
        transition = (movement.airport, transition_addresses[window_start], transition_points[window_start])
        if transition not in claimed:
            time_difference = abs(transition_times[window_start] - movement.movement_time)
            candidates[row_position] = (time_difference, transition_addresses[window_start], transition_points[window_start], 0)
            candidate_transitions[row_position] = transition
    claim_counts = Counter(candidate_transitions.values())
    return {position: match for position, match in candidates.items() if claim_counts[candidate_transitions[position]] == 1}


def matched_feature_rows(accepted: dict[int, Match], movement_rows: list[MovementRow], tracks: TrackIndex) -> list[dict[str, float]]:
    """
    This function works out the features of every matched flight (departure or arrival) and adds the flight id, the plane ID, how it was matched and how far apart the two times were
    """
    matched_rows: list[dict[str, float]] = []
    for row_position in sorted(accepted):
        movement = movement_rows[row_position]
        time_difference, icao, transition_index, match_flag = accepted[row_position]
        airport = movement.airport
        reference_latitude, reference_longitude = AIRPORT_REFERENCE_POINTS[airport]
        track = tracks[(icao, airport)]
        if movement.PHASE_mvt == 'DEP':
            features = departure_clock_features(track, transition_index, movement)
        else:
            features = arrival_track_features(track, transition_index, movement.movement_time, movement.arrival_block_time, reference_latitude, reference_longitude)
        features.update(MVT_ID_mvt=movement.MVT_ID_mvt, ads_icao=icao, ads_cs_match=match_flag, ads_match_dt=time_difference)
        matched_rows.append(features)
    return matched_rows


def match_first_pass_day(day: str) -> tuple[str, int]:
    """
    This function:
        1. loads the dots
        2. builds the tracks
        3. matches flights by callsign first and then by time
        4. saves the features
    """
    points = load_points(day)
    if points is None:
        return day, 0
    day_movements = movements()
    day_movements = day_movements[day_movements.day == day]
    if day_movements.empty: # used if cases like midnight flights
        return day, 0
    tracks = tracks_by_aircraft(points)
    addresses = addresses_by_callsign(points)
    lookup = transition_lookup(tracks)
    movement_rows = list(day_movements.itertuples(index=False))
    callsign_candidates = callsign_matches(movement_rows, tracks, addresses)
    accepted = keep_closest_claim(callsign_candidates, movement_rows)
    claimed = {(movement_rows[row_position].airport, match[1], match[2]) for row_position, match in accepted.items()}
    accepted.update(time_window_matches(movement_rows, lookup, callsign_candidates, claimed))
    day_table = pd.DataFrame(matched_feature_rows(accepted, movement_rows, tracks))
    day_table.to_parquet(PASS_OUTPUT_DIRECTORIES['pass1'] / f'pass1_{day}.parquet', index=False)
    return day, len(day_table)


# stand map and tables
def build_stand_map() -> None:
    """
    This function builds the stand map from landing planes
        -for each stand
            -take the median of the positions where arrivals parked
    Only stands seen more than once, where the parked planes agree within 40 meters, are kept
    """
    parked = pd.concat([pd.read_parquet(path) for path in sorted(PASS1_DAY_FILES.glob('pass1_*.parquet'))], ignore_index=True)
    parked = parked[parked.get('ads_park_x').notna()] if 'ads_park_x' in parked else parked.iloc[:0]
    arrival_columns = ['MVT_ID_mvt', 'PHASE_mvt', 'ADES_mvt', 'STAND_mvt']
    arrivals = pd.concat([pd.read_parquet(path, columns=arrival_columns) for path in competition_files()])
    arrivals = arrivals[arrivals.PHASE_mvt.eq('ARR')]
    parked = parked.merge(arrivals, on='MVT_ID_mvt')
    parked = parked[(parked.ads_park_dt.abs() < STAND_MAP_MAX_BLOCK_OFFSET_SECONDS) & (parked.ads_park_gs < STAND_MAP_MAX_PARKED_SPEED_KNOTS)]
    by_stand = parked.groupby(['ADES_mvt', 'STAND_mvt'])

    def median_distance_from_median(stand: pd.DataFrame) -> float:
        """
        This function measures how spread out one stand's parked positions are
        """
        return np.median(np.hypot(stand.ads_park_x - stand.ads_park_x.median(), stand.ads_park_y - stand.ads_park_y.median()))

    stand_map = pd.DataFrame({'x': by_stand.ads_park_x.median(),
                              'y': by_stand.ads_park_y.median(),
                              'n': by_stand.size(),
                              'spread': by_stand.apply(median_distance_from_median)}).reset_index()
    stand_map = stand_map.rename(columns={'ADES_mvt': 'airport'})
    stand_map = stand_map[(stand_map.n >= MIN_STAND_OBSERVATIONS) & (stand_map.spread < MAX_STAND_SPREAD_METRES)]
    stand_map.to_parquet(ADSB_ONLY_STAND_MAP_FILE, index=False)
    print(len(stand_map), 'stands')
    print(stand_map.groupby('airport').agg(n=('n', 'size'), obs=('n', 'sum'), spread=('spread', 'median')))


def merge_xplane_stands() -> None:
    """
    This function fills the gaps in the stand map with stands from X-Plane (a flight simulator's airport maps) at the seven airports where its stand names agree with the ADS-B stand map
    Only stands that the ADS-B map does not have yet are added, with a count of 0 so they can be told apart
    """
    adsb_stands = pd.read_parquet(ADSB_ONLY_STAND_MAP_FILE)
    xplane_stands = pd.read_parquet(FEATURE_MATRIX / 'xplane_stand_positions.parquet')
    xplane_stands = xplane_stands[xplane_stands.airport.isin(XPLANE_STAND_AIRPORTS)]
    new_stands = xplane_stands.merge(adsb_stands[['airport', 'STAND_mvt']], on=['airport', 'STAND_mvt'], how='left', indicator=True)
    new_stands = new_stands[new_stands._merge == 'left_only'].drop(columns='_merge').assign(n=0, spread=float('nan'))
    merged = pd.concat([adsb_stands, new_stands[adsb_stands.columns]], ignore_index=True)
    merged.to_parquet(STAND_MAP_FILE, index=False)
    print('stands', len(adsb_stands), '->', len(merged), new_stands.groupby('airport').size().to_dict())


def build_pass1_table(output_path: str | Path) -> None:
    """
    This function combines all the pass 1 day files into one big table of departures
    """
    day_files = sorted(PASS1_DAY_FILES.glob('pass1_*.parquet'))
    table = pd.concat([pd.read_parquet(path) for path in day_files], ignore_index=True)
    table = table.drop(columns=[column for column in table if column.startswith('ads_park') or column == 'ads_ldg_rel'])
    table = table[table.ads_tko_rel.notna()] if 'ads_tko_rel' in table else table
    table.to_parquet(output_path, index=False)
    print(len(day_files), 'days', table.shape)


# passes 2 and 3
def unused_takeoff_transitions(points: pd.DataFrame, earlier_matches: pd.DataFrame) -> tuple[TrackIndex, FreeTransitions]:
    """
    This function builds the day's tracks and lists, per airport, for every take-off that no earlier pass has used yet
    """
    earlier_takeoff_times = earlier_matches.movement_time + earlier_matches.ads_tko_rel
    used_transitions = {(airport, icao, int(round(takeoff_time))) for airport, icao, takeoff_time in zip(earlier_matches.airport, earlier_matches.ads_icao, earlier_takeoff_times)}
    tracks = tracks_by_aircraft(points)
    free_transitions: dict[str, list[TransitionEntry]] = {}
    for (icao, airport), track in tracks.items():
        for point_index in track.takeoff_roll_starts:
            if (airport, icao, int(round(track.times[point_index]))) not in used_transitions:
                free_transitions.setdefault(airport, []).append((float(track.times[point_index]), icao, int(point_index)))
    candidates: FreeTransitions = {}
    for airport, entries in free_transitions.items():
        entries_in_time_order = sorted(entries)
        candidates[airport] = (np.array([entry[0] for entry in entries_in_time_order]), entries_in_time_order)
    return tracks, candidates


def aircraft_types_by_address(day: str) -> dict[str, str]:
    """
    This function reads the airplane model that each plane broadcasts and keeps the first one seen
    """
    parts: list[pd.DataFrame] = []
    for day_offset in TYPE_LOOKUP_DAY_OFFSETS:
        path = point_file(pd.Timestamp(day) + pd.Timedelta(days=day_offset))
        if path.exists():
            parts.append(pd.read_parquet(path, columns=['icao', 'typ']).dropna())
    if not parts:
        return {}
    first_type_seen = pd.concat(parts).drop_duplicates('icao')
    return first_type_seen.set_index('icao').typ.astype(str).to_dict()


def closest_candidate_pair(row_position: int, airport: str, movement_time: float, transition_times: np.ndarray, candidate_indices: list[int]) -> CandidatePair:
    """
    This function finds the take-off closest in time to one departure, and how much farther away the second-closest one is
    """
    time_differences = np.abs(transition_times[candidate_indices] - movement_time)
    by_difference = np.argsort(time_differences)
    closest = by_difference[0]
    runner_up_margin = float(time_differences[by_difference[1]] - time_differences[closest]) if len(by_difference) > 1 else SOLE_CANDIDATE_MARGIN_SECONDS
    return float(time_differences[closest]), row_position, airport, candidate_indices[int(closest)], runner_up_margin


def assign_closest_first(pairs: list[CandidatePair], departure_rows: list[MovementRow], candidates: FreeTransitions, tracks: TrackIndex, match_flag: int) -> list[dict[str, float]]:
    """
    This function hands out take-offs to departures one to one, closest pair first
    A pair is skipped if its take-off is already taken or if the second-closest take-off was less than 20 seconds farther away (too close to call) - each departure only ever tries its own closest take-off
    """
    pairs.sort()
    taken_transitions: set[tuple[str, int]] = set()
    matched_rows: list[dict[str, float]] = []
    for time_difference, row_position, airport, candidate_index, runner_up_margin in pairs:
        if (airport, candidate_index) in taken_transitions or runner_up_margin < MIN_RUNNER_UP_MARGIN_SECONDS:
            continue
        taken_transitions.add((airport, candidate_index))
        departure = departure_rows[row_position]
        _, icao, roll_start = candidates[airport][1][candidate_index]
        features = departure_clock_features(tracks[(icao, airport)], roll_start, departure)
        features.update(MVT_ID_mvt=departure.MVT_ID_mvt, ads_icao=icao, ads_cs_match=match_flag, ads_match_dt=time_difference)
        matched_rows.append(features)
    return matched_rows


def day_departures(day: str) -> pd.DataFrame:
    """
    This function picks out the departures of one day
    """
    all_movements = movements()
    return all_movements[(all_movements.day == day) & (all_movements.PHASE_mvt == 'DEP')]


def window_indices(transition_times: np.ndarray, movement_time: float) -> range:
    """
    This function finds the free take-offs from 2 minutes before to just under 2 minutes after a given time
    """
    window_start = np.searchsorted(transition_times, movement_time - LATER_PASS_WINDOW_SECONDS)
    window_end = np.searchsorted(transition_times, movement_time + LATER_PASS_WINDOW_SECONDS)
    return range(window_start, window_end)


def match_second_pass_day(day: str) -> tuple[str, int, int]:
    """
    This function runs pass 2 for one day which looks at departures that pass 1 did not match get the closest take-off that pass 1 did not use, within +-2 minutes
    """
    points = load_points(day)
    if points is None:
        return day, 0, 0
    departures = day_departures(day)
    if departures.empty:
        return day, 0, 0
    pass1_file = PASS1_DAY_FILES / f'pass1_{day}.parquet'
    match_columns = ['MVT_ID_mvt', 'ads_icao', 'ads_tko_rel']
    earlier_matches = pd.read_parquet(pass1_file, columns=match_columns) if pass1_file.exists() else pd.DataFrame(columns=match_columns)
    # the day's departures matched in pass 1, with their movement time and airport
    earlier_matches = earlier_matches.merge(departures[['MVT_ID_mvt', 'movement_time', 'airport']], on='MVT_ID_mvt')
    unmatched = departures[~departures.MVT_ID_mvt.isin(set(earlier_matches.MVT_ID_mvt))]
    tracks, candidates = unused_takeoff_transitions(points, earlier_matches)
    departure_rows = list(unmatched.itertuples(index=False))
    pairs: list[CandidatePair] = []
    for row_position, departure in enumerate(departure_rows):
        if departure.airport not in candidates:
            continue
        transition_times, _ = candidates[departure.airport]
        window = window_indices(transition_times, departure.movement_time)
        if not window:
            continue
        pairs.append(closest_candidate_pair(row_position, departure.airport, departure.movement_time, transition_times, list(window)))
    # 2 is the ads_cs_match code of a pass-2 match (pass 1 writes 1 for a callsign match and 0 for a time match)
    matched_rows = assign_closest_first(pairs, departure_rows, candidates, tracks, 2)
    pd.DataFrame(matched_rows).to_parquet(PASS_OUTPUT_DIRECTORIES['pass2'] / f'pass2_{day}.parquet', index=False)
    return day, len(unmatched), len(matched_rows)


def pass1_and_pass2_matches(day: str, departures: pd.DataFrame) -> pd.DataFrame:
    """
    This function collects the day's pass 1 and pass 2 matches, so that pass 3 knows which departures are done and which take-offs are already taken
    """
    earlier_parts: list[pd.DataFrame] = []
    for earlier_file in (PASS1_DAY_FILES / f'pass1_{day}.parquet', PASS2_DAY_FILES / f'pass2_{day}.parquet'):
        if earlier_file.exists():
            earlier_day = pd.read_parquet(earlier_file)
            if len(earlier_day):
                earlier_parts.append(earlier_day[['MVT_ID_mvt', 'ads_icao', 'ads_tko_rel']])
    if not earlier_parts:
        return pd.DataFrame(columns=['MVT_ID_mvt', 'ads_icao', 'ads_tko_rel', 'movement_time', 'airport'])
    return pd.concat(earlier_parts, ignore_index=True).merge(departures[['MVT_ID_mvt', 'movement_time', 'airport']], on='MVT_ID_mvt')


def match_third_pass_day(day: str) -> tuple[str, int, int]:
    """
    This function runs pass 3 for one day by only looking at take-offs of planes of the same model as the flight
    """
    points = load_points(day)
    if points is None:
        return day, 0, 0
    departures = day_departures(day)
    if departures.empty:
        return day, 0, 0
    earlier_matches = pass1_and_pass2_matches(day, departures)
    unmatched = departures[~departures.MVT_ID_mvt.isin(set(earlier_matches.MVT_ID_mvt))]
    tracks, candidates = unused_takeoff_transitions(points, earlier_matches)
    departure_rows = list(unmatched.itertuples(index=False))
    type_by_address = aircraft_types_by_address(day)
    # one-to-one, closest first: free transitions within +-120 s whose aircraft type equals the movement's, next one >= 20 s farther
    pairs: list[CandidatePair] = []
    for row_position, departure in enumerate(departure_rows):
        movement_type = AIRCRAFT_TYPE_BY_MOVEMENT.get(departure.MVT_ID_mvt, '')
        if departure.airport not in candidates or not movement_type:
            continue
        transition_times, transition_entries = candidates[departure.airport]
        window = window_indices(transition_times, departure.movement_time)
        same_type = [index for index in window if type_by_address.get(transition_entries[index][1], '') == movement_type]
        if not same_type:
            continue
        pairs.append(closest_candidate_pair(row_position, departure.airport, departure.movement_time, transition_times, same_type))
    # 4 is the ads_cs_match code of a pass-3 match
    matched_rows = assign_closest_first(pairs, departure_rows, candidates, tracks, 4)
    pd.DataFrame(matched_rows).to_parquet(PASS_OUTPUT_DIRECTORIES['pass3'] / f'pass3_{day}.parquet', index=False)
    return day, len(unmatched), len(matched_rows)


def extend_table_with_later_pass(base_table: str, pass_folder: str, output_table: str) -> None:
    """
    This function adds all the matched departures
    """
    base = pd.read_parquet(FEATURE_MATRIX / f'{base_table}.parquet')
    day_files = sorted((ADSB_DAY_FILES / pass_folder).glob(f'{pass_folder}_*.parquet'))
    later_pass = pd.concat([pd.read_parquet(path) for path in day_files], ignore_index=True)
    later_pass = later_pass[~later_pass.MVT_ID_mvt.isin(set(base.MVT_ID_mvt))].drop_duplicates('MVT_ID_mvt')[base.columns]
    later_pass = later_pass.astype(base.dtypes.to_dict())
    extended = pd.concat([base, later_pass], ignore_index=True)
    assert not extended.MVT_ID_mvt.duplicated().any()
    extended.to_parquet(FEATURE_MATRIX / f'{output_table}.parquet', index=False)
    match_flags = extended.ads_cs_match.value_counts().to_dict()
    print(f'{base_table}: {len(base)} rows + {len(later_pass)} from {pass_folder} -> {output_table}: {len(extended)} rows; '
          f'match flags {match_flags}')


def run_pass_over_days(pass_function: Callable[[str], tuple[str, int] | tuple[str, int, int]], days: list[str], worker_count: int) -> None:
    """
    This function runs the pipeline over many days at once using several worker processes (for speed/parallelism - it would be too inefficient otherwise)
    """
    STAND_POSITIONS.update(load_stand_map())
    movements()
    if pass_function is match_third_pass_day:
        types = pd.concat([pd.read_parquet(path, columns=['MVT_ID_mvt', 'AIRCRAFT_TYPE_mvt']) for path in competition_files()]).dropna()
        AIRCRAFT_TYPE_BY_MOVEMENT.update(dict(zip(types.MVT_ID_mvt, types.AIRCRAFT_TYPE_mvt.astype(str).str.strip())))
    with Pool(min(len(days), worker_count)) as pool:
        for result in pool.imap_unordered(pass_function, days):
            print(*result, flush=True)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build ADS-B ground tracks.')
    commands = parser.add_subparsers(dest='command', required=True)
    for pass_name in ('pass1', 'pass2', 'pass3'):
        pass_parser = commands.add_parser(pass_name)
        pass_parser.add_argument('days', nargs='*')
        pass_parser.add_argument('--out-dir', default='')
    commands.add_parser('stand-map')
    commands.add_parser('merge-xplane')
    table_parser = commands.add_parser('table')
    table_parser.add_argument('--out', default=str(FEATURE_MATRIX / 'adsb_pass1.parquet'))
    extend_parser = commands.add_parser('extend')
    extend_parser.add_argument('base_table')
    extend_parser.add_argument('pass_folder')
    extend_parser.add_argument('output_table')
    return parser


def main() -> None:
    """
    This function runs the steps in the pipeline as specified
    """
    args = argument_parser().parse_args()
    if args.command in ('pass1', 'pass2', 'pass3'):
        days = args.days or sorted(path.stem.removeprefix('points_') for path in POINTS_DIRECTORY.glob('points_*.parquet'))
        if args.out_dir:
            PASS_OUTPUT_DIRECTORIES[args.command] = Path(args.out_dir)
        PASS_OUTPUT_DIRECTORIES[args.command].mkdir(parents=True, exist_ok=True)
        pass_functions = {'pass1': match_first_pass_day, 'pass2': match_second_pass_day, 'pass3': match_third_pass_day}
        run_pass_over_days(pass_functions[args.command], days, PASS1_WORKER_PROCESSES if args.command == 'pass1' else LATER_PASS_WORKER_PROCESSES)
    elif args.command == 'stand-map':
        build_stand_map()
    elif args.command == 'merge-xplane':
        merge_xplane_stands()
    elif args.command == 'table':
        build_pass1_table(args.out)
    else:
        extend_table_with_later_pass(args.base_table, args.pass_folder, args.output_table)


if __name__ == '__main__':
    main()
