"""
This file runs the whole project, from downloading the data to writing the submission file with one guessed taxi-out time per departure
Taxi-out time is how many seconds a plane takes from leaving its parking spot (the stand) to taking off from the runway - the challenge gives the real times for 2025 to learn from and hides them for January, February, June and July 2026, which have to be guessed

This is how it works:
-stage 1, data (about a day):
    -download every source into taxi/data/downloads (a file that is already there is not downloaded again):
        1. competition - the organizers' challenge files
        2. adsblol - ADS-B plane positions from adsb.lol, keeping only the points on the ground or below 4,000 feet near the 10 airports
        3. opdi - OPDI, the open flight lists and flight events of EUROCONTROL and OpenSky
        4. eurocontrol - EUROCONTROL's monthly taxi-out and taxi-in figures per airport
        5. weather - hourly weather at each airport from Open-Meteo
        6. metar - METAR, the routine weather reports of each airport
        7. xplane - each airport's map with its stands, from the X-Plane Scenery Gateway
    -build every feature table (the tables of clues the models learn from)
-stage 2, models (about two days on one 8 GB graphics card):
    -train and tune every model, skipping any step that is already finished and up to date, so a stopped run can carry on later
    -write the submission file taxi/prediction/submission.parquet and a copy of its January and July rows

You will need:
    -a python 3.12 environment
    -a GPU with 8GB VRAM minimum
    -32 GB RAM minimum
    -about 70GB of free space on a HDD/SSD

Licenses: adsb.lol ODbL 1.0, Open-Meteo CC BY 4.0, METAR public domain, X-Plane GPL v2+, EUROCONTROL's monthly figures for non-commercial use only and not modified (downloaded, never shared on), and OPDI has no open license but E. Spinielli (an organiser) confirmed on the challenge's Discord that it can be used
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from minio import Minio

PYTHON = sys.executable


# ============================================================================================== 1. folders and files
PACKAGE_DIR = Path(__file__).resolve().parent                        # taxi/: the code, the data and the results
PROJECT_ROOT = PACKAGE_DIR.parent                                    # the repository root: every step runs from here

DATA = PACKAGE_DIR / 'data'                                          # created by the first run, before anything is downloaded
DOWNLOADS = DATA / 'downloads'                                       # every download, never modified
COMPETITION_DIR = DOWNLOADS / 'competition'                          # the organisers' training, ranking and template files
RANKING_FILE = COMPETITION_DIR / 'final_ranking.parquet'             # the rows to predict (final ranking: Jan, Feb, Jun, Jul 2026)
SUBMITTING_FILE = COMPETITION_DIR / 'final_submitting.parquet'       # the organisers' template: the departures to predict, in order
INITIAL_SUBMITTING_FILE = COMPETITION_DIR / 'submitting.parquet'     # the first template (January and July 2026 only)
FEATURES = DATA / 'features'                                         # every feature table built from the downloads
ADSB_DAY_FILES = FEATURES / 'adsblol'                                # per-day ADS-B matches: pass1/, pass2/, pass3/, stand maps
FEATURE_MATRIX = FEATURES / 'matrix'                                 # row-aligned feature tables (one row per departure) and network inputs
PRIOR_TABLES = FEATURES / 'prior'                                    # the feature tables of the prior model

MODEL_WEIGHTS = PACKAGE_DIR / 'models/weights'                       # every saved model: <run>/, the prior's steps in prior/<step>/
RUN_PREDICTIONS = PACKAGE_DIR / 'models/predictions'                 # predictions of every run: <run>/pred.parquet + metrics.json
TUNED_CLASSIFIERS = PACKAGE_DIR / 'models/tune'                      # Optuna studies and ranking predictions of the tuned classifiers
PREDICTION_DIR = PACKAGE_DIR / 'prediction'                          # the submission builder and its result
SUBMISSION_FILE = PREDICTION_DIR / 'submission.parquet'              # the final file (MVT_ID_mvt, TAXITIME_SEC_mvt)
INITIAL_MONTHS_FILE = PREDICTION_DIR / 'submission_initial_months.parquet'   # its January / July rows, first template's order

# ============================================================================================== 2. shared loaders
# The four cross-validation folds of 2025: fold wk<k> holds out the departures whose ISO week number modulo 4 is k.
WEEK_FOLDS = ('wk0', 'wk1', 'wk2', 'wk3')

# The organisers linked a departure to its NM flight plan only when |block time - LOBT| <= 3606 s. The LOBT label window
# widens that limit by 10 s on each side.
_LOBT_MATCH_LIMIT_SECONDS = 3606.0
_LOBT_WINDOW_TOLERANCE_SECONDS = 10.0    # slack: one 2026 arrival in the challenge files lies 3615 s from its LOBT, just past the 3606 s limit

# ---- recording regimes and week folds
SCHEDULE_COPY_TOLERANCE_SECONDS = 6      # the published block times carry an independent +-6 s jitter (README), so a copy of the schedule can be up to 6 s off
PREVIOUS_DATE_MIN_LABEL_SECONDS = 80000  # an S label is one day (86,400 s) plus the real taxi time
PREVIOUS_DATE_MAX_LABEL_SECONDS = 95000  # see above
WEEK_FOLD_COUNT = 4                      # each fold holds out every fourth ISO week, so it has weeks of every season (held-out whole months had hidden the Feb 2025 Istanbul snow)
# ---- downloads
DOWNLOAD_ATTEMPTS = 5                    # a few retries ride out short network or server hiccups
LAST_DOWNLOAD_ATTEMPT = 4                # DOWNLOAD_ATTEMPTS - 1: the number of the last try when counting from 0
RETRY_WAIT_STEP_SECONDS = 10             # waits 10, 20, 30 ... s between tries, giving a busy server time to recover
DOWNLOAD_TIMEOUT_SECONDS = 600           # give up when a server sends nothing for 10 minutes (urllib's timeout is per network read, not per file)
DOWNLOAD_CHUNK_BYTES = 1 << 22           # 4 MiB per copy step: fast streaming with little memory
GITHUB_API_TIMEOUT_SECONDS = 60          # the API replies are small JSON documents, so one minute is plenty
RATE_LIMIT_MARGIN_SECONDS = 5            # wait 5 s past GitHub's X-RateLimit-Reset time, in case the two clocks differ a little
LOW_ALTITUDE_FEET = 4000                 # a round ceiling well above every airport (the highest, Madrid, is about 2,000 ft up): take-offs and landings are kept, traffic high overhead is dropped
LONGITUDE_BOX_FACTOR = 1.3               # the box reaches 1.3 times as far east-west as north-south (about 14 km against 11 km)
ARCHIVE_BUFFER_BYTES = 1 << 22           # the split archive is read 4 MiB at a time: fast streaming with little memory
TRACE_FILES_PER_TASK = 64                # trace files go to a worker 64 at a time, so the pool spends little time handing out tasks
ADSB_WORKERS = 8                         # cut from 16 to 8 in September 2026 to keep memory safe on the 32 GB laptop
OPDI_EVENT_FILE_ATTEMPTS = 3             # fewer tries than DOWNLOAD_ATTEMPTS: a failed event file is reported and skipped (the 2-12 October 2025 file does not exist upstream)
METAR_MIN_LINES = 1000                   # a sanity floor far below a full reply (half-hourly reports over about 590 days make tens of thousands of lines): catches error pages and cut-off replies
# ---- model steps: the numbers on the command lines of stage 2 (kept as text)
IMPORTANCE_MODEL_SEED = '42'             # any fixed seed, only so that runs repeat (the first of REGIME_SEEDS)
REGIME_SEEDS = (42, 7)                   # any fixed seeds, only so that runs repeat; the same two seeds as the two XGBoost value fits
SECOND_VALUE_MODEL_SEED = '7'            # any fixed seed, only so that runs repeat (the second of REGIME_SEEDS)
STAKE_WEIGHT_POWER = '1'                 # a row counts in proportion to its stake (power 1 = plain linear weighting)
ROME_LINKED_TUNING_TRIALS = '25'         # a time budget: the Rome classifier trains on about 160,000 rows, so each trial is slow
NO_LINK_TUNING_TRIALS = '60'             # a time budget: the no-link classifier is small (about a minute per trial, an hour for 60)
TWO_PART_TOP_FEATURES = '200'            # the 200 strongest inputs of the first XGBoost model, to keep the tree models small in memory
TWO_PART_LOW_THRESHOLD_SECONDS = '600'   # half of the two-part model's main 1,200 s long-taxi threshold: an extra tree variant, run on the 2025 week folds
TWO_PART_HIGH_THRESHOLD_SECONDS = '2400' # double the main 1,200 s threshold: the second extra tree variant, run on the 2025 week folds
COVERAGE_OUTAGE_BLEND = '0.5' # chosen on 2025 data: best of 0 / 0.25 / 0.5 / 0.75 / 1 on 2025 airport-days with an ADS-B outage, and better than no blend in all 4 folds of Rome June-September 2025 with ADS-B blanked
NEXT_DAY_REGIME_MODEL_STRENGTH = '0.5' # chosen on 2025 data only, by a selection rule written down before the comparison was run
LATE_ROME_FACTOR = '0.5' # hand-chosen because it had the best cross-validation score


def competition_files() -> list[Path]:
    return sorted(COMPETITION_DIR.glob('training_*.parquet')) + [RANKING_FILE]


def competition_columns(columns: list[str], movement_ids: np.ndarray | pd.Series) -> pd.DataFrame:
    all_rows = pd.concat([pd.read_parquet(path, columns=['MVT_ID_mvt'] + columns) for path in competition_files()])
    first_row_per_movement = all_rows.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
    return first_row_per_movement.reindex(movement_ids)


def recording_regimes(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function works out how each departure's off-block time (when the plane left its stand) was written down: P when it is a copy of the scheduled time (within 6 seconds), S when it is not P and the taxi time is between 80,000 and 95,000 seconds (recorded with the previous day's date, so about a day too long), and N for every other labeled row
    """
    challenge = competition_columns(['SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'FLIGHT_mvt'], metadata.MVT_ID_mvt)
    scheduled_time = challenge.SCHED_TIME_UTC_mvt
    takeoff_time = challenge.MVT_TIME_UTC_mvt
    seconds_after_schedule = (takeoff_time - scheduled_time).dt.total_seconds().to_numpy()
    day_difference = (takeoff_time.dt.normalize() - scheduled_time.dt.normalize()).dt.days.to_numpy()
    block_minus_schedule = (challenge.BLOCK_TIME_UTC_mvt - scheduled_time).dt.total_seconds().to_numpy()
    label = metadata.TAXITIME_SEC_mvt.to_numpy()
    is_schedule_copy = np.abs(block_minus_schedule) <= SCHEDULE_COPY_TOLERANCE_SECONDS
    is_previous_date = ~is_schedule_copy & (label > PREVIOUS_DATE_MIN_LABEL_SECONDS) & (label < PREVIOUS_DATE_MAX_LABEL_SECONDS)
    operator_code = challenge.FLIGHT_mvt.fillna('???').str[:3].to_numpy()
    return seconds_after_schedule, day_difference, is_schedule_copy, is_previous_date, operator_code


def departure_frame() -> pd.DataFrame:
    """
    This function builds the main departure table almost every model uses
    """
    frame = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet')
    seconds_after_schedule, day_difference, is_schedule_copy, is_previous_date, operator_code = recording_regimes(frame)
    frame['seconds_after_schedule'] = seconds_after_schedule
    frame['day_difference'] = day_difference
    frame['is_schedule_copy'] = is_schedule_copy
    frame['is_previous_date'] = is_previous_date
    frame['operator_code'] = operator_code
    frame['no_nm_link'] = frame.AOBT_3_flt.isna()
    frame['is_rome'] = frame.ADEP_mvt == 'LIRF'
    return frame


def held_out_week_fold_rows(frame: pd.DataFrame, fold: str) -> np.ndarray:
    """
    This function marks the rows a week fold holds out for practice tests
    A model can then learn from the other three folds and be tested on this one
    """
    assert fold in WEEK_FOLDS, fold
    takeoff_time = pd.to_datetime(frame.MVT_TIME_UTC_mvt)
    iso_week = takeoff_time.dt.isocalendar().week.to_numpy().astype(int)
    fold_number = int(fold[2:])
    return (frame.year.to_numpy() == 2025) & (iso_week % WEEK_FOLD_COUNT == fold_number)


# Filled by the first call of lobt_label_window: the window's lower bound (s) per movement id over every challenge file.
_LOBT_WINDOW_LOWER_BOUND: list[pd.Series] = []


def lobt_label_window(movement_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function gives the lowest and highest possible taxi time of each movement from the off-block time in its flight plan (the LOBT)
    """
    window_half_width = _LOBT_MATCH_LIMIT_SECONDS + _LOBT_WINDOW_TOLERANCE_SECONDS
    if not _LOBT_WINDOW_LOWER_BOUND:
        columns = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt']
        all_rows = pd.concat([pd.read_parquet(path, columns=columns) for path in competition_files()])
        all_rows = all_rows.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
        takeoff_minus_lobt = (all_rows.MVT_TIME_UTC_mvt - pd.to_datetime(all_rows.LOBT_flt, utc=True)).dt.total_seconds()
        _LOBT_WINDOW_LOWER_BOUND.append(takeoff_minus_lobt - window_half_width)
    lower = _LOBT_WINDOW_LOWER_BOUND[0].reindex(movement_ids).to_numpy()
    return lower, lower + 2 * window_half_width


def departures_with_later_pass_track(pass_folder: str) -> set[float]:
    """
    This function collects the 2026 departures that were matched to their ADS-B track
    """
    movement_ids: set[float] = set()
    for day_file in ADSB_DAY_FILES.glob(f'{pass_folder}/{pass_folder}_2026-*.parquet'):
        day = pd.read_parquet(day_file)
        if len(day):
            movement_ids |= set(day.MVT_ID_mvt)
    return movement_ids


def rmse(errors: np.ndarray) -> float:
    if not len(errors):
        return float('nan')
    return float(np.sqrt(np.mean(np.square(errors))))


# ============================================================================================== 3. downloads
COMPETITION_BUCKET = 'prc-2026-datasets'                     # the organisers' bucket with the challenge files
ADSB_FOLDER = DOWNLOADS / 'adsblol'
POINTS_FOLDER = ADSB_FOLDER / 'points'                       # points_<day>.parquet, one per UTC day
OPDI_FOLDER = DOWNLOADS / 'opdi'
OPDI_URL = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002'
OPDI_FIRST_EVENT_WINDOW = pd.Timestamp('2024-12-26')         # the 10-day event files start on this day and follow each other
EUROCONTROL_FOLDER = DOWNLOADS / 'eurocontrol'
EUROCONTROL_CSV_URL = 'https://www.eurocontrol.int/performance/data/download/csv'
# monthly per-airport aggregates (flights, valid flights, reference and additional minutes) for the years of the challenge files
EUROCONTROL_AGGREGATES = [f'{kind}_additional_time_{year}.csv' for kind in ('taxi_out', 'taxi_in') for year in (2025, 2026)]
# Open-Meteo request coordinates (latitude, longitude) per airport; the API snaps them to the ERA5 grid
WEATHER_COORDINATES = {
    'EDDF': (50.033333, 8.570556),
    'EDDM': (48.353802, 11.7861),
    'EGLL': (51.470748, -0.459909),
    'EHAM': (52.308601, 4.76389),
    'LEBL': (41.2971, 2.07846),
    'LEMD': (40.493407, -3.572249),
    'LFPG': (49.00896, 2.554117),
    'LIRF': (41.804532, 12.251998),
    'LSZH': (47.458056, 8.548056),
    'LTFM': (41.274874, 28.732136)
}
# hourly ERA5 weather from 2025-01-01 to 2026-07-31, UTC
WEATHER_URL = ('https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}'
               '&start_date=2025-01-01&end_date=2026-07-31'
               '&hourly=temperature_2m%2Crelative_humidity_2m%2Cdew_point_2m%2Cprecipitation%2Csnowfall%2Csnow_depth%2Cweather_code'
               '%2Cpressure_msl%2Ccloud_cover%2Cwind_speed_10m%2Cwind_direction_10m%2Cwind_gusts_10m'
               '&timezone=UTC&models=era5')
# every report of the station from 2024-12-25 to 2026-08-05, UTC, as comma-separated text
METAR_URL = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={station}&data=all'
             '&year1=2024&month1=12&day1=25&year2=2026&month2=8&day2=5&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no'
             '&missing=M&trace=T&direct=no&report_type=3&report_type=4')
# Scenery Gateway packs used per airport (the layouts change when new packs are approved, so they are pinned)
XPLANE_SCENERY_IDS = {
    'EDDF': 103368,
    'EDDM': 111124,
    'EGLL': 111341,
    'EHAM': 111943,
    'LEBL': 102255,
    'LEMD': 99851,
    'LFPG': 111022,
    'LIRF': 110482,
    'LSZH': 106102,
    'LTFM': 105504
}
# Aerodrome reference points (latitude, longitude): a reduced adsb.lol point lies within about 11 km of one of them. They are also
# the origin of the local x / y metre coordinates of taxi.features.adsb_tracks and taxi.features.matrix_tables.
AIRPORT_REFERENCE_POINTS = {
    'EDDF': (50.033333, 8.570556),
    'EDDM': (48.353783, 11.786086),
    'EGLL': (51.4706, -0.461941),
    'EHAM': (52.308601, 4.76389),
    'LEBL': (41.2971, 2.07846),
    'LEMD': (40.471926, -3.56264),
    'LFPG': (49.012798, 2.55),
    'LIRF': (41.804532, 12.251998),
    'LSZH': (47.464699, 8.54917),
    'LTFM': (41.262222, 28.727778)
}
AIRPORTS = list(AIRPORT_REFERENCE_POINTS)
AIRPORT_LATITUDES = np.array([AIRPORT_REFERENCE_POINTS[a][0] for a in AIRPORTS])            # latitude of each airport, in AIRPORTS order
AIRPORT_LONGITUDES = np.array([AIRPORT_REFERENCE_POINTS[a][1] for a in AIRPORTS])           # longitude of each airport, in AIRPORTS order
LONGITUDE_SCALE = np.cos(np.deg2rad(AIRPORT_LATITUDES))     # degrees of longitude -> degrees of latitude at each airport
RADIUS_DEGREES = 0.10                                       # about 11 km of latitude (longitude: x1.3)
# The 2026 repository also carries a prod build of 2025-12-19; that day was reduced from the staging build of the 2025 repository.
EXCLUDED_RELEASES = ('globe_history_2026/releases/download/v2025.12.19-planes-readsb-prod-0/',)
# Tag endings tried in order: 28 May - 10 June 2025 are only published as ...-prod-0tmp / ...-staging-0tmp
RELEASE_TAG_SUFFIXES = ('0', '0tmp')


# ---------------------------------------------------------------------------------------------- HTTP helpers
def fetch(url: str, attempts: int = DOWNLOAD_ATTEMPTS, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - any network error is retried
            if attempt == attempts - 1:
                raise
            print(f'  retry {attempt + 1} after {error}', flush=True)
            time.sleep(RETRY_WAIT_STEP_SECONDS * (attempt + 1))
    raise AssertionError('unreachable')     # reached only when attempts < 1


def fetch_to_file(url: str, target: Path, attempts: int = DOWNLOAD_ATTEMPTS) -> None:
    partial = target.with_name(target.name + '.part')
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(partial, 'wb') as output:
                shutil.copyfileobj(response, output, DOWNLOAD_CHUNK_BYTES)
            partial.rename(target)
            return
        except Exception as error:  # noqa: BLE001 - any network error is retried
            if attempt == attempts - 1:
                partial.unlink(missing_ok=True)
                raise
            print(f'  retry {attempt + 1} after {error}', flush=True)
            time.sleep(RETRY_WAIT_STEP_SECONDS * (attempt + 1))


# ---------------------------------------------------------------------------------------------- competition files
def bucket_client() -> Minio:
    alias = json.loads((Path.home() / '.mc/config.json').read_text())['aliases']['opensky']
    endpoint = urlparse(alias['url'])
    return Minio(endpoint.netloc, access_key=alias['accessKey'], secret_key=alias['secretKey'], secure=endpoint.scheme == 'https')


def download_competition_files() -> None:
    client = bucket_client()
    COMPETITION_DIR.mkdir(parents=True, exist_ok=True)
    for item in client.list_objects(COMPETITION_BUCKET, recursive=True):
        target = COMPETITION_DIR / item.object_name
        if not target.exists():
            client.fget_object(COMPETITION_BUCKET, item.object_name, str(target))
            print('competition', item.object_name, flush=True)


def challenge_days() -> tuple[list[str], list[str]]:
    days: set[str] = set()
    ranking_days: set[str] = set()
    for path in competition_files():
        file_days = set(pd.read_parquet(path, columns=['MVT_TIME_UTC_mvt']).MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d'))
        days |= file_days
        if path == RANKING_FILE:
            ranking_days |= file_days
    months = sorted({day[:7].replace('-', '') for day in days})
    days |= {(pd.Timestamp(day) + pd.Timedelta(days=step)).strftime('%Y-%m-%d') for day in ranking_days for step in (-1, 1)}
    return sorted(days), months


# ---------------------------------------------------------------------------------------------- adsb.lol
def release_assets(year: int, tag: str) -> list[str] | None:
    """
    This function asks GitHub for the download links of one adsb.lol release (one day's data) by its tag, and returns None if it does not exist
    GitHub allows only 60 requests per hour without a token, so when the limit is used up it sleeps until it resets and tries again - other errors are retried after 10, 20, ... seconds, up to 5 tries in all before it gives up with an error
    """
    url = f'https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}'
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=GITHUB_API_TIMEOUT_SECONDS) as response:
                return sorted(asset['browser_download_url'] for asset in json.loads(response.read())['assets'])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.headers.get('X-RateLimit-Remaining') == '0':     # 60 requests per hour without a token: wait for the next hour
                time.sleep(max(0.0, float(error.headers['X-RateLimit-Reset']) - time.time()) + RATE_LIMIT_MARGIN_SECONDS)
                continue
            if attempt == LAST_DOWNLOAD_ATTEMPT:
                raise
        except OSError:                                               # any other network error is retried
            if attempt == LAST_DOWNLOAD_ATTEMPT:
                raise
        print(f'  retry {attempt + 1} of {url}', flush=True)
        time.sleep(RETRY_WAIT_STEP_SECONDS * (attempt + 1))
    raise RuntimeError(f'GitHub API rate limit not lifted for {url}')


def day_release_urls(day: str) -> list[str] | None:
    """
    This function finds the download links of one day's adsb.lol data
    It tries the prod build before the staging one, and the usual '-0' tag before '-0tmp' (the only tag of 28 May - 10 June 2025)
    """
    for build in ('prod', 'staging'):
        for suffix in RELEASE_TAG_SUFFIXES:
            tag = f"v{day.replace('-', '.')}-planes-readsb-{build}-{suffix}"
            for year in (2025, 2026):
                if f'globe_history_{year}/releases/download/{tag}/' in EXCLUDED_RELEASES:
                    continue
                urls = release_assets(year, tag)
                if urls:
                    return urls
    return None


@dataclass
class TraceColumns:
    """
    This data structure holds the points of one plane's trace
    """
    times: np.ndarray                      # epoch seconds (UTC)
    latitudes: np.ndarray                  # degrees
    longitudes: np.ndarray                 # degrees
    altitudes: np.ndarray                  # feet; NaN on the ground and when not reported
    on_ground: np.ndarray                  # bool: the aircraft reported being on the ground
    ground_speeds: np.ndarray              # knots; NaN when not reported
    tracks: np.ndarray                     # direction of motion over the ground, degrees; NaN when not reported
    flags: np.ndarray                      # readsb's flag bits of the point (int16; 0 when absent)
    callsigns: list[str | None]            # the last callsign broadcast up to the point; None before the first one
    position_sources: list[str | None]     # how the position was obtained (e.g. 'adsb_icao', 'mlat'); None when absent


def trace_columns(trace: list[list[Any]], day_start: float) -> TraceColumns:
    """
    This function sorts a raw adsb.lol trace (a list of little lists, one per point) into the TraceColumns data structure
    It turns each point's seconds into a real time, marks 'ground' points as on the ground
    """
    point_count = len(trace)
    times = np.empty(point_count)
    latitudes = np.empty(point_count)
    longitudes = np.empty(point_count)
    altitudes = np.full(point_count, np.nan)
    on_ground = np.zeros(point_count, bool)
    ground_speeds = np.full(point_count, np.nan)
    tracks = np.full(point_count, np.nan)
    flags = np.zeros(point_count, np.int16)
    callsigns: list[str | None] = [None] * point_count
    current_callsign = None
    position_sources: list[str | None] = [None] * point_count
    for i, point in enumerate(trace):
        times[i] = day_start + point[0]
        latitudes[i] = point[1]
        longitudes[i] = point[2]
        if point[3] == 'ground':
            on_ground[i] = True
        elif point[3] is not None:
            altitudes[i] = point[3]
        if point[4] is not None:
            ground_speeds[i] = point[4]
        if point[5] is not None:
            tracks[i] = point[5]
        flags[i] = point[6] or 0
        if point[8] is not None and point[8].get('flight'):
            current_callsign = point[8]['flight'].strip()
        callsigns[i] = current_callsign
        position_sources[i] = point[9] if len(point) > 9 else None
    return TraceColumns(times, latitudes, longitudes, altitudes, on_ground, ground_speeds, tracks, flags, callsigns, position_sources)


def airport_boxes(latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function checks which points lie in the box around one of the 10 airports and which airport that is
    The airports are hundreds of kilometers apart, so a point is in at most one box
    """
    latitude_distance = np.abs(latitudes[:, None] - AIRPORT_LATITUDES[None, :])
    longitude_distance = np.abs(longitudes[:, None] - AIRPORT_LONGITUDES[None, :]) * LONGITUDE_SCALE[None, :]
    in_box = (latitude_distance < RADIUS_DEGREES) & (longitude_distance < RADIUS_DEGREES * LONGITUDE_BOX_FACTOR)
    return in_box.any(1), in_box.argmax(1)


def points_near_airports(trace_file: str) -> pd.DataFrame | None:
    """
    This function opens one plane's zipped trace file and keeps only the points that are low (on the ground or below 4,000 feet) and near one of the 10 airports - this shrinks the huge worldwide archive to what the project needs
    """
    try:
        with gzip.open(trace_file) as compressed:
            trace_json = json.loads(compressed.read())
    except Exception:  # noqa: BLE001 - an unreadable trace is skipped
        return None
    trace = trace_json.get('trace') or []
    if not trace:
        return None
    columns = trace_columns(trace, trace_json['timestamp'])
    is_low = columns.on_ground | (columns.altitudes < LOW_ALTITUDE_FEET)
    if not is_low.any():
        return None
    is_near, airport_numbers = airport_boxes(columns.latitudes, columns.longitudes)
    kept = is_near & is_low
    if not kept.any():
        return None
    # the first messages may precede the identification: they take the trace's first callsign
    filled_callsigns = pd.Series(columns.callsigns).bfill().ffill().to_numpy()
    rows = np.flatnonzero(kept)
    return pd.DataFrame(dict(icao=trace_json['icao'],
                             t=columns.times[rows],
                             lat=columns.latitudes[rows].astype('f8'),
                             lon=columns.longitudes[rows].astype('f8'),
                             ground=columns.on_ground[rows],
                             alt=columns.altitudes[rows].astype('f4'),
                             gs=columns.ground_speeds[rows].astype('f4'),
                             track=columns.tracks[rows].astype('f4'),
                             flags=columns.flags[rows],
                             callsign=filled_callsigns[rows],
                             airport=np.array(AIRPORTS)[airport_numbers[rows]],
                             src=np.array(columns.position_sources, dtype=object)[rows],
                             typ=trace_json.get('t'),
                             reg=trace_json.get('r')))


class ConcatenatedParts(io.RawIOBase):
    """
    This is a class that reads a day's archive, which comes split into several part files (.tar.aa, .tar.ab, ...)
    """

    def __init__(self, part_files: list[Path]) -> None:
        self.part_files = list(part_files)
        self.current = open(self.part_files.pop(0), 'rb')

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:
        while True:
            count = self.current.readinto(buffer)
            if count or not self.part_files:
                return count or 0
            self.current.close()
            self.current = open(self.part_files.pop(0), 'rb')


def download_release_parts(urls: list[str], work_folder: Path) -> list[Path]:
    """
    This function downloads every part of a day's archive
    """
    part_files: list[Path] = []
    for number, url in enumerate(urls):
        part_file = work_folder / f'part{number}'
        part_files.append(part_file)
        fetch_to_file(url, part_file)
    return part_files


def extract_full_traces(part_files: list[Path], folder: Path) -> None:
    """
    This function reads the split archive as one stream and unpacks only the trace_full_* files (each one plane's whole path for the day), with a safety filter so nothing is written outside the folder
    """
    with io.BufferedReader(ConcatenatedParts(part_files), ARCHIVE_BUFFER_BYTES) as stream, tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            if member.isfile() and os.path.basename(member.name).startswith('trace_full_'):
                archive.extract(member, folder, filter='data')


def points_of_traces(trace_files: list[str], workers: int) -> list[pd.DataFrame]:
    with Pool(workers) as pool:
        return [points for points in pool.imap_unordered(points_near_airports, trace_files, chunksize=TRACE_FILES_PER_TASK) if points is not None]


def download_adsb_day(day: str, workers: int) -> str:
    points_file = POINTS_FOLDER / f'points_{day}.parquet'
    if points_file.exists():
        return 'exists'
    urls = day_release_urls(day)
    if urls is None:
        return 'NO_RELEASE'
    work_folder = ADSB_FOLDER / 'raw' / day
    shutil.rmtree(work_folder, ignore_errors=True)
    (work_folder / 'traces').mkdir(parents=True)          # makes the work folder (traces/ itself stays empty)
    try:
        part_files = download_release_parts(urls, work_folder)
        extract_full_traces(part_files, work_folder / 'extracted')
        for part in part_files:
            part.unlink()
        trace_files = sorted(glob.glob(str(work_folder / 'extracted' / '**' / 'trace_full_*.json'), recursive=True))
        parts = points_of_traces(trace_files, workers)
        POINTS_FOLDER.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(points_file, index=False)
        return f'{len(trace_files)} traces from {urls[0].rsplit("/", 2)[1]}'
    except Exception as error:  # noqa: BLE001 - the day is reported and skipped, the other days continue
        points_file.unlink(missing_ok=True)
        return f'FAILED ({error})'
    finally:
        shutil.rmtree(work_folder, ignore_errors=True)


def download_adsb(workers: int = ADSB_WORKERS) -> None:
    for day in challenge_days()[0]:
        result = download_adsb_day(day, workers)
        if result != 'exists':
            print('adsb.lol', day, result, flush=True)
    shutil.rmtree(ADSB_FOLDER / 'raw', ignore_errors=True)     # the per-day work folders; every day's points file holds what is kept


# ---------------------------------------------------------------------------------------------- OPDI and EUROCONTROL
def opdi_event_windows(months: list[str]) -> list[str]:
    # month by month (parse, then add a month): a bad entry raises its error before the later entries are read
    month_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for month in months:
        month_start = pd.Timestamp(f'{month[:4]}-{month[4:]}-01')
        month_ranges.append((month_start, month_start + pd.offsets.MonthBegin(1)))
    windows: list[str] = []
    start = OPDI_FIRST_EVENT_WINDOW
    while start < month_ranges[-1][1]:
        end = start + pd.Timedelta(days=10)     # OPDI publishes its flight events in 10-day files
        if any(start < month_end and end > month_start for month_start, month_end in month_ranges):
            windows.append(f'flight_events_{start:%Y%m%d}_{end:%Y%m%d}.parquet')
        start = end
    return windows


def download_opdi() -> None:
    OPDI_FOLDER.mkdir(parents=True, exist_ok=True)
    months = challenge_days()[1]
    for month in months:
        target = OPDI_FOLDER / f'flight_list_{month}.parquet'
        if not target.exists():
            fetch_to_file(f'{OPDI_URL}/flight_list/{target.name}', target)
            print('opdi', target.name, flush=True)
    for name in opdi_event_windows(months):
        target = OPDI_FOLDER / name
        if not target.exists():
            try:
                fetch_to_file(f'{OPDI_URL}/flight_events/{name}', target, attempts=OPDI_EVENT_FILE_ATTEMPTS)
                print('opdi', name, flush=True)
            except Exception as error:  # noqa: BLE001 - one 10-day file (2-12 October 2025) does not exist upstream
                print('opdi', name, 'not available:', error, flush=True)


def download_eurocontrol() -> None:
    EUROCONTROL_FOLDER.mkdir(parents=True, exist_ok=True)
    for name in EUROCONTROL_AGGREGATES:
        target = EUROCONTROL_FOLDER / name
        if not target.exists():
            fetch_to_file(f'{EUROCONTROL_CSV_URL}/{name}', target)
            if target.read_text()[:4] != 'YEAR':
                target.unlink()
                raise RuntimeError(f'EUROCONTROL {name}: not a CSV reply')
            print('eurocontrol', name, flush=True)


# ---------------------------------------------------------------------------------------------- weather and airport layouts
def download_weather() -> None:
    folder = DOWNLOADS / 'weather'
    folder.mkdir(parents=True, exist_ok=True)
    for airport, (latitude, longitude) in WEATHER_COORDINATES.items():
        target = folder / f'weather_{airport}.json'
        if not target.exists():
            target.write_bytes(fetch(WEATHER_URL.format(latitude=latitude, longitude=longitude)))
            print('weather', airport, flush=True)


def download_metar() -> None:
    folder = DOWNLOADS / 'metar'
    folder.mkdir(parents=True, exist_ok=True)
    for airport in AIRPORTS:
        target = folder / f'{airport}.csv'
        if not target.exists():
            data = fetch(METAR_URL.format(station=airport))
            if data.count(b'\n') <= METAR_MIN_LINES:
                raise RuntimeError(f'METAR {airport}: incomplete reply')
            target.write_bytes(data)
            print('metar', airport, flush=True)


def download_xplane() -> None:
    folder = DOWNLOADS / 'xplane'
    folder.mkdir(parents=True, exist_ok=True)
    for airport, scenery_id in XPLANE_SCENERY_IDS.items():
        url = f'https://gateway.x-plane.com/apiv1/scenery/{scenery_id}'
        target = folder / f'{airport}.dat'
        if not target.exists():
            reply = json.loads(fetch(url))
            archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(reply['scenery']['masterZipBlob'])))
            target.write_bytes(archive.read(f'{airport}.dat'))
            print('xplane', airport, flush=True)


SOURCES = {
    'competition': download_competition_files,
    'adsblol': download_adsb,
    'opdi': download_opdi,
    'eurocontrol': download_eurocontrol,
    'weather': download_weather,
    'metar': download_metar,
    'xplane': download_xplane
}


def download_sources(names: list[str]) -> None:
    for source in names or list(SOURCES):
        SOURCES[source]()


# ============================================================================================== 4. pipeline steps
# ---------------------------------------------------------------- stage 1: data (every step runs, from --from-step on)
# (step name, [module run with python -m, its arguments ...]), in running order
DATA_STEPS: list[tuple[str, list[str]]] = [
    ('downloads', ['taxi.main', '--download-only']),
    ('prior_tables', ['taxi.features.prior_tables', 'base', 'weather', 'clock', 'opdi', 'operational', 'ground-extract', 'ground', 'no-link', 'linked-arrivals']),
    ('matrix_tables', ['taxi.features.matrix_tables', 'matrix', 'xplane', 'arrival-proxy']),
    ('adsb_pass1_for_stand_map', ['taxi.features.adsb_tracks', 'pass1']),
    ('adsb_stand_map', ['taxi.features.adsb_tracks', 'stand-map']),
    ('adsb_merge_xplane_stands', ['taxi.features.adsb_tracks', 'merge-xplane']),
    ('adsb_pass1', ['taxi.features.adsb_tracks', 'pass1']),
    ('adsb_pass2', ['taxi.features.adsb_tracks', 'pass2']),
    ('adsb_pass3', ['taxi.features.adsb_tracks', 'pass3']),
    ('adsb_tables', ['taxi.features.adsb_tracks', 'table']),
    ('adsb_table_pass2', ['taxi.features.adsb_tracks', 'extend', 'adsb_pass1', 'pass2', 'adsb_pass2']),
    ('adsb_table_pass3', ['taxi.features.adsb_tracks', 'extend', 'adsb_pass2', 'pass3', 'adsb_pass3']),
    ('feature_blocks', ['taxi.features.feature_blocks', 'takeoff_sequence', 'unimpeded_times', 'ordering', 'neighbours_adsb', 'neighbours', 'neighbours_wide', 'metar_weather', 'airport_state', 'airport_state_extended']),
    ('airframe_turnaround', ['taxi.features.airframe_turnaround']),
]


# ---------------------------------------------------------------- stage 2: models (a step whose outputs exist is skipped)
@dataclass
class Step:
    """
    This data structure describes one model step
    """
    name: str
    module_and_arguments: list[str]
    outputs: list[Path]
    newer_than: list[Path] = field(default_factory=list)

    def is_done(self) -> bool:
        if not all(output.exists() for output in self.outputs):
            return False
        newest_input = max((path.stat().st_mtime for path in self.newer_than if path.exists()), default=0.0)
        return all(output.stat().st_mtime > newest_input for output in self.outputs)


def run_prediction_file(run_name: str) -> Path:
    return RUN_PREDICTIONS / run_name / 'pred.parquet'


REGIME_MODULE = 'taxi.models.regime_mixture'
TUNED_MODULE = 'taxi.models.tuned_classifiers'
TREE_MODULE = 'taxi.models.tree_value_models'
OUT_OF_FOLD_MODULE = 'taxi.models.out_of_fold'


def importance_model_step() -> Step:
    """
    This function makes the step for the first model, an XGBoost taxi-time model with seed 42 - its main job is to rank which inputs matter most, so the networks and tree models know which ones to use
    """
    arguments = [REGIME_MODULE, 'train', '--name', 'xgboost_value_seed42', '--seed', IMPORTANCE_MODEL_SEED, '--iteration-set', 'xgboost_value']
    return Step('XGBoost value fit seed 42', arguments, [run_prediction_file('xgboost_value_seed42')])


def network_steps() -> list[Step]:
    """
    This function makes the neural network steps: first three steps that build ready-made input files (caches) for the three ADS-B tables, rebuilt when the first XGBoost model or the table changed after them, then one step for every job in NETWORK_JOBS
    """
    from taxi.models.network_jobs import NETWORK_JOBS     # imported here: taxi.models imports this module
    importance_model = MODEL_WEIGHTS / 'xgboost_value_seed42/value.ubj'
    steps = []
    for pass_number in (1, 2, 3):
        adsb_table = f'adsb_pass{pass_number}'
        cache = f'network_inputs_pass{pass_number}'
        arguments = ['taxi.features.network_inputs', '--adsb-table', adsb_table, '--out', cache]
        newer_than = [importance_model, FEATURE_MATRIX / f'{adsb_table}.parquet']
        steps.append(Step(f'network inputs {cache}', arguments, [FEATURE_MATRIX / cache / 'columns.json'], newer_than=newer_than))
    for job in NETWORK_JOBS:
        fold_outputs = [run_prediction_file(f'{job.name}_{fold}') for fold in job.folds.split(',')]
        steps.append(Step(f'network {job.name} ({job.folds})', ['taxi.models.neural_network', *job.arguments()], fold_outputs))
    return steps


def regime_steps() -> list[Step]:
    """
    This function makes the steps of the regime models, which guess both how the off-block time was recorded (P, S or N) and the ordinary taxi time
    It also adds a second XGBoost taxi-time model, the stake-weighted classifiers (which pay extra attention to rows where a wrong guess costs a lot) and the Rome next-day prior (for Rome flights without a flight plan link that take off on a later day than scheduled)
    """
    steps = []
    for seed in REGIME_SEEDS:
        arguments = [REGIME_MODULE, 'train', '--name', f'regime_seed{seed}', '--seed', str(seed), '--iteration-set', 'regime', '--value-variants', 'two_part,catboost', '--average-value-variants']
        steps.append(Step(f'regime fit seed {seed}', arguments, [run_prediction_file(f'regime_seed{seed}')]))
    steps += [
        Step('XGBoost value fit seed 7',
             [REGIME_MODULE, 'train', '--name', 'xgboost_value_seed7', '--seed', SECOND_VALUE_MODEL_SEED, '--iteration-set', 'xgboost_value'],
             [run_prediction_file('xgboost_value_seed7')]),
        Step('stake-weighted classifiers',
             [REGIME_MODULE, 'stake-classifiers', '--runs', 'regime_seed42,regime_seed7', '--stake-weight-power', STAKE_WEIGHT_POWER],
             [run_prediction_file('regime_seed42_stake_classifiers'), run_prediction_file('regime_seed7_stake_classifiers')]),
        Step('Rome next-day prior', [REGIME_MODULE, 'nextday', 'regime_seed42_stake_classifiers', 'regime_seed7_stake_classifiers'],
             [run_prediction_file('regime_seed42_stake_classifiers_nextday'), run_prediction_file('regime_seed7_stake_classifiers_nextday')]),
    ]
    return steps


def tuned_classifier_steps() -> list[Step]:
    """
    This function makes three steps: two that tune the 'was the time copied from the schedule?' classifiers with Optuna
    """
    return [
        Step('tuning: Rome NM-linked rows', 
             [TUNED_MODULE, 'tune', '--model', 'rome_linked', '--trials', ROME_LINKED_TUNING_TRIALS],
             [TUNED_CLASSIFIERS / 'study_rome_linked.json']),
        Step('tuning: rows without an NM link', 
             [TUNED_MODULE, 'tune', '--model', 'no_link', '--trials', NO_LINK_TUNING_TRIALS],
             [TUNED_CLASSIFIERS / 'study_no_link.json']),
        Step('tuned classifiers', 
             [TUNED_MODULE, 'final', '--models', 'rome_linked,no_link'],
             [TUNED_CLASSIFIERS / 'rome_linked_classifier.parquet', TUNED_CLASSIFIERS / 'no_link_classifier.parquet']),
    ]


def tree_value_steps() -> list[Step]:
    """
    This function makes the 25 tree model steps: five taxi-time models (four two-part models, which handle unusually long taxis separately, and one CatBoost model), each trained on all of 2025 and on each of the four week folds
    They use the inputs ranked by the first XGBoost model and leave out the calendar columns, and all five models of one fold run before the next fold
    """
    two_part = [TREE_MODULE, 'two-part', '--top-features', TWO_PART_TOP_FEATURES, '--importance-from', 'xgboost_value_seed42', '--drop-columns', 'month,day,dayofyear']
    catboost = [TREE_MODULE, 'catboost', '--importance-from', 'xgboost_value_seed42', '--drop-columns', 'month,day,dayofyear', '--extra-tables', 'airport_state']
    extended = ['--extra-tables', 'airport_state,airport_state_extended', '--adsb-table', 'adsb_pass3']
    tree_runs = [
        ('two_part_residual_pass1_airport_state', two_part + ['--extra-tables', 'airport_state']),
        ('catboost_value_pass1_airport_state', catboost),
        ('two_part_residual_pass3_airport_state_extended', two_part + extended),
        ('two_part_residual_pass3_airport_state_extended_threshold600', two_part + extended + ['--threshold', TWO_PART_LOW_THRESHOLD_SECONDS]),
        ('two_part_residual_pass3_airport_state_extended_threshold2400', two_part + extended + ['--threshold', TWO_PART_HIGH_THRESHOLD_SECONDS]),
    ]
    steps = []
    for fold in ('full', 'wk0', 'wk1', 'wk2', 'wk3'):
        for base_name, arguments in tree_runs:
            run_name = f'{base_name}_{fold}'
            steps.append(Step(f'tree {run_name}', arguments + ['--fold', fold, '--name', run_name], [run_prediction_file(run_name)]))
    return steps


def push_delay_classifier_step() -> Step:
    """
    This function makes the step that trains the Rome classifier for flights with a flight plan link again
    """
    return Step('Rome classifier with push-delay inputs', [TUNED_MODULE, 'final', '--models', 'rome_linked', '--suffix', '_with_push_delay', '--push-delay-network', 'network_pass3_clip15000_full'], [TUNED_CLASSIFIERS / 'rome_linked_classifier_with_push_delay.parquet'])


def out_of_fold_steps() -> list[Step]:
    """
    This function makes the five steps of the honest 2025 rehearsal in out_of_fold.py (tuned classifiers, second-stage value, global classifier, prior and the combined components), used to fit the last corrections
    """
    from taxi.models.out_of_fold import MONTH_PAIRS, OUT_OF_FOLD
    return [
        Step('out-of-fold: tuned classifiers', [OUT_OF_FOLD_MODULE, 'tuned'], [OUT_OF_FOLD / 'tuned_rome_linked.parquet', OUT_OF_FOLD / 'tuned_no_link.parquet']),
        Step('out-of-fold: second-stage value', [OUT_OF_FOLD_MODULE, 'value'], [OUT_OF_FOLD / 'value.parquet']),
        Step('out-of-fold: global classifier', [OUT_OF_FOLD_MODULE, 'global'], [OUT_OF_FOLD / 'global.parquet']),
        Step('out-of-fold: prior', [OUT_OF_FOLD_MODULE, 'prior'], [OUT_OF_FOLD / f'prior_{a}_{b}.parquet' for a, b in MONTH_PAIRS]),
        Step('out-of-fold: components', [OUT_OF_FOLD_MODULE, 'components'], [OUT_OF_FOLD / 'components.parquet']),
    ]


def submission_step() -> Step:
    arguments = ['taxi.prediction.build_submission', '--coverage-outage-blend', COVERAGE_OUTAGE_BLEND, '--aggregate-calibration', '--support-caps', '--next-day-regime-model', NEXT_DAY_REGIME_MODEL_STRENGTH, '--late-rome-corrections', '--late-rome-factor', LATE_ROME_FACTOR]
    return Step('submission', arguments, [SUBMISSION_FILE])


def model_steps() -> list[Step]:
    return [importance_model_step(), *network_steps(), *regime_steps(), *tuned_classifier_steps(), *tree_value_steps(), push_delay_classifier_step(), *out_of_fold_steps(), submission_step()]


def delete_working_files() -> None:
    for folder in (RUN_PREDICTIONS, MODEL_WEIGHTS, TUNED_CLASSIFIERS):
        shutil.rmtree(folder, ignore_errors=True)
    print('deleted the working files under taxi/models; the result is', SUBMISSION_FILE, flush=True)


def create_data_folders() -> None:
    for folder in (DOWNLOADS, FEATURES):
        folder.mkdir(parents=True, exist_ok=True)


def run_data_stage(first_step: str) -> None:
    create_data_folders()        # before anything is downloaded
    step_names = [name for name, _ in DATA_STEPS]
    first_index = step_names.index(first_step)
    for number, (name, module_and_arguments) in enumerate(DATA_STEPS[first_index:], first_index + 1):
        print(f'\n##### data step {number}/{len(DATA_STEPS)}: {name}', flush=True)
        subprocess.run([PYTHON, '-u', '-m', *module_and_arguments], cwd=PROJECT_ROOT, check=True)


def run_model_stage(keep_outputs: bool) -> None:
    if not (FEATURE_MATRIX / 'airport_state_extended.parquet').exists():
        raise SystemExit('the feature tables are missing: run python taxi/main.py --data-only first')
    steps = model_steps()
    for number, step in enumerate(steps, 1):
        if step.is_done():
            continue
        print(f'\n##### model step {number}/{len(steps)}: {step.name}', flush=True)
        subprocess.run([PYTHON, '-u', '-m', *step.module_and_arguments], cwd=PROJECT_ROOT, check=True)
    if not keep_outputs:
        delete_working_files()


# ============================================================================================== 5. command line
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full pipeline.')
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--data-only', action='store_true')
    stage.add_argument('--models-only', action='store_true')
    stage.add_argument('--download-only', nargs='*', choices=list(SOURCES), metavar='SOURCE')
    parser.add_argument('--from-step', default=DATA_STEPS[0][0], choices=[name for name, _ in DATA_STEPS])
    # outputs (runs, weights, tuned classifiers) are kept by default; --no-keep-outputs deletes them after the submission is written
    parser.add_argument('--keep-outputs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--list', action='store_true')
    return parser.parse_args()


def print_step_list() -> None:
    for name, _ in DATA_STEPS:
        print(f'data    {name}')
    for step in model_steps():
        print(f"{'done' if step.is_done() else '    '}    {step.name}")


def main() -> None:
    arguments = parse_arguments()
    if arguments.download_only is not None:
        download_sources(arguments.download_only)
        return
    if arguments.list:
        print_step_list()
        return
    if not arguments.models_only:
        run_data_stage(arguments.from_step)
    if not arguments.data_only:
        run_model_stage(arguments.keep_outputs)
    print('\nDone!\n', flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(PROJECT_ROOT))
    from taxi.main import main as run_pipeline
    run_pipeline()
