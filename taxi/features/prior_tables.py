"""
Every plane that leaves one of the ten airports in the challenge gets one row in a big table, and this file fills those rows with clues
The thing to predict is the taxi-out time: the seconds between the plane leaving its stand (off-block) and lifting off the runway
The clues come from the airport's own record (columns ending in _mvt), the flight plan record of EUROCONTROL's Network Manager (NM, columns ending in _flt), OPDI (EUROCONTROL's open flight lists and ground events, made from ADS-B position messages) and hourly weather
The NM record has planned and reported times such as LOBT (the last planned off-block time) and AOBT (the reported actual off-block time) - departures with no NM record at all are called no-link rows and are the hardest ones
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from taxi.main import COMPETITION_DIR, DOWNLOADS, FEATURES, PRIOR_TABLES, competition_files

TARGET_COLUMN = 'TAXITIME_SEC_mvt'          # the label: take-off minus off-block time, in seconds (NaN on ranking rows)
MOVEMENT_TIME_COLUMN = 'MVT_TIME_UTC_mvt'   # take-off time of a departure, landing time of an arrival
BLOCK_TIME_COLUMN = 'BLOCK_TIME_UTC_mvt'    # off-block time of a departure (never used: it gives the label), in-block of an arrival

# Columns kept as categorical features (missing values become the level '<MISSING>').
CATEGORICAL_COLUMNS = ['ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'FLIGHT_RULE_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'ADES_FILED_flt', 'MARKET_SEGMENT_flt', 'FLIGHT_RULE_flt', 'FLIGHT_TYPE_flt', 'AIRCRAFT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt']

# Clock fields of a departure row
CLOCK_COLUMNS = [MOVEMENT_TIME_COLUMN, 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt']

# OPDI surface event types kept by ground-extract
GROUND_EVENT_TYPES = ['exit-parking_position', 'entry-parking_position', 'exit-apron', 'entry-taxiway', 'entry-runway', 'entry-deicing_pad', 'exit-deicing_pad']

# The ten departure airports of the challenge (ICAO codes)
STUDY_AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LTFM', 'LSZH']

# ground-extract output
GROUND_EVENT_EXTRACTS = FEATURES / 'ground_events_exact'

FINGERPRINT_CHUNK_BYTES = 2 ** 20 # memory cap

ARRIVAL_ERROR_CAP_SECONDS = 1800                  # landing minus NM arrival estimate is clipped to +-30 min, so a few odd flights cannot swamp the nearby mean
TAXI_IN_CAP_SECONDS = 3600                        # taxi-in times are clipped at 1 hour, so rare extreme values cannot swamp the averages
NEARBY_ARRIVAL_HALF_WINDOW_SECONDS = 3600         # landings from 1 h before to 1 h after the take-off are averaged
PRIOR_FALLBACK_BASELINE_SECONDS = 1000.0          # about 17 minutes: a round, taxi-out-sized starting guess for non-Rome rows without an AOBT

# weather table
PRECIPITATION_SHORT_WINDOW_HOURS = 3              # rain / snow summed over the past 3 h (and 12 h below): a short and a half-day window, hand-picked round values
PRECIPITATION_LONG_WINDOW_HOURS = 12              # half a day
TEMPERATURE_CHANGE_HOURS = 3                      # the same 3 h span as the short precipitation window
ICING_TEMPERATURE_CELSIUS = 3                     # a few degrees above freezing, because frost and ice can form on aircraft while the air 2 m up is still slightly above 0 C
ICING_HUMIDITY_PERCENT = 90                       # nearly saturated air, where frost and fog form easily
BASELINE_RATIO_OFFSET_SECONDS = 60                # added so the ratio never divides by zero
PLAUSIBLE_BASELINE_MAX_SECONDS = 21600            # 6 h: a take-off minus AOBT above this cannot be a real taxi-out, so it is not trusted
PROXY_BASELINE_MAX_SECONDS = 7200                 # 2 h: a tighter cap for the neighbours' mean, so one strange row cannot swamp it
STAND_ARRIVAL_WINDOW_DAYS = 7                     # one week, so every weekday of the timetable counts once

# clock table
LINKED_ARRIVAL_CLOCK_TOLERANCE = '18h'            # the same flight's arrival row lands at most 18 h after the take-off: generous, as flights between these airports take a few hours
PREVIOUS_STAND_ARRIVAL_TOLERANCE = '48h'          # the last arrival at the same stand counts when its in-block is at most 2 days old

# opdi and ground tables
OPDI_MATCH_TOLERANCE = '90min'                    # an OPDI flight matches when it was first seen at most 90 min from the take-off

# operational table
PRESSURE_PSEUDO_COUNT = 1                         # 1 added on top and bottom, so the ratio never divides by zero (and is 1 when both counts are 0)
RUNWAY_ACTIVE_SHARE = 0.05                        # a runway counts as in use above 5 % of the movements around the take-off, which ignores a stray movement or two
PREVIOUS_STAND_DEPARTURE_MAX_AGE_SECONDS = 86400  # one day
LATE_STAND_DEPARTURE_DELAY_SECONDS = 900          # 15 minutes: the usual punctuality limit in aviation (a flight more than 15 min late counts as delayed)
RECENT_STAND_DEPARTURE_AGE_SECONDS = 3600         # within the last hour

# no-link table
NO_LINK_OPDI_TOLERANCE = '3h'                     # a wider window than OPDI_MATCH_TOLERANCE for the rows without an NM link


def seconds_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds()


def epoch_seconds(times: pd.Series) -> np.ndarray:
    """
    This function turns times into plain numbers of seconds since 1 January 1970
    """
    return times.dt.as_unit('ns').astype('int64').to_numpy() / 1e9


def read_all_challenge_rows() -> pd.DataFrame:
    training_files = competition_files()[:-1]
    ranking_file = competition_files()[-1]
    # Two concatenations (training files, then training + ranking) as in the original build
    training_rows = pd.concat([pd.read_parquet(path) for path in training_files], ignore_index=True)
    return pd.concat([training_rows, pd.read_parquet(ranking_file)], ignore_index=True)


def competition_file_fingerprints() -> dict[str, dict[str, int | str]]:
    """
    This function takes a fingerprint of every challenge file to verify the input files did not change
    """
    fingerprints: dict[str, dict[str, int | str]] = {}
    for path in sorted(COMPETITION_DIR.glob('*.parquet')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(FINGERPRINT_CHUNK_BYTES), b''):      # 1 MiB at a time
                digest.update(chunk)
        fingerprints[path.name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    return fingerprints


def load_prior_tables(include_opdi_features: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the two files of the base table
    """
    features = pd.read_parquet(PRIOR_TABLES / 'row_features.parquet')
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    if include_opdi_features:
        opdi_features = pd.read_parquet(PRIOR_TABLES / 'opdi_features.parquet')
        assert len(opdi_features) == len(features)
        features = pd.concat([features, opdi_features], axis=1)
    return features, metadata


# base
MOVEMENT_COUNT_WINDOWS = [300, 900, 3600]       # seconds before / after the take-off in which other movements are counted


def departure_row_features(departures: pd.DataFrame) -> pd.DataFrame:
    """
    This function builds the main clues of the base table using the helpers below
    """
    features = pd.DataFrame(index=departures.index)
    _add_categorical_columns(features, departures)
    _add_combined_keys(features)
    _add_stand_and_operator_parts(features)
    _add_calendar_and_clock_parts(features, departures)
    _add_clock_differences(features, departures)
    _add_record_mismatch_flags(features, departures)
    return features


def _add_categorical_columns(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLUMNS:
        features[column] = departures[column].fillna('<MISSING>').astype(str)


def _add_combined_keys(features: pd.DataFrame) -> None:
    combined_keys = {'airport_runway': ['ADEP_mvt', 'RUNWAY_mvt'], 
                     'airport_stand': ['ADEP_mvt', 'STAND_mvt'], 
                     'airport_runway_stand': ['ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt'], 
                     'route': ['ADEP_mvt', 'ADES_mvt']}
    for key_name, key_columns in combined_keys.items():
        value = features[key_columns[0]]
        for column in key_columns[1:]:
            value = value + ':' + features[column]
        features[key_name] = value


def _add_stand_and_operator_parts(features: pd.DataFrame) -> None:
    features['stand_prefix'] = features.STAND_mvt.str.extract(r'^([A-Za-z]+)', expand=False).fillna('numeric')
    features['stand_number'] = pd.to_numeric(features.STAND_mvt.str.extract(r'(\d+)', expand=False), errors='coerce')
    features['airline_prefix'] = features.FLIGHT_mvt.str.extract(r'^([A-Z0-9]{2})', expand=False).fillna('<MISSING>')
    features['callsign_prefix'] = features.CALLSIGN_flt.str.extract(r'^([A-Z]{2,3})', expand=False).fillna('<MISSING>')


def _add_calendar_and_clock_parts(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    takeoff_time = departures[MOVEMENT_TIME_COLUMN]
    features['month'] = takeoff_time.dt.month
    features['day'] = takeoff_time.dt.day
    features['dayofweek'] = takeoff_time.dt.dayofweek
    features['dayofyear'] = takeoff_time.dt.dayofyear
    for column in CLOCK_COLUMNS:
        clock = departures[column]
        features[column + '_hour'] = clock.dt.hour + clock.dt.minute / 60      # fractional hour of the day
        # second / minute: a clock rounded to the minute (second 0) hints at a manual or estimated time
        features[column + '_second'] = clock.dt.second
        features[column + '_minute'] = clock.dt.minute
        features[column + '_missing'] = clock.isna().astype('int8')


def _add_clock_differences(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    for first_column, second_column in itertools.combinations(CLOCK_COLUMNS, 2):
        features[first_column + '__minus__' + second_column] = seconds_between(departures[first_column], departures[second_column])
    features['aobt_baseline'] = seconds_between(departures[MOVEMENT_TIME_COLUMN], departures.AOBT_3_flt)
    features['scheduled_baseline'] = seconds_between(departures[MOVEMENT_TIME_COLUMN], departures.SCHED_TIME_UTC_mvt)


def _add_record_mismatch_flags(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    for movement_column, flight_column in [('ADEP_mvt', 'ADEP_flt'), ('ADES_mvt', 'ADES_flt'), ('AIRCRAFT_TYPE_mvt', 'AIRCRAFT_TYPE_flt')]:
        features[movement_column + '_mismatch'] = features[movement_column].ne(features[flight_column]).astype('int8')
    features['flight_id_missing'] = departures.FLIGHT_ID_mvt.isna().astype('int8')


def add_movement_context(features: pd.DataFrame, departures: pd.DataFrame, all_rows: pd.DataFrame) -> pd.DataFrame:
    """
    This function counts how busy the airport was around each take-off
        -calculate the number of departures and arrivals in the 5, 15 and 60 minutes before and after it
        -for these arrivals, average how far their landing time is from the NM arrival time and their taxi-in time
    """
    airports = departures.ADEP_mvt.fillna('<MISSING>')
    departure_times = epoch_seconds(departures[MOVEMENT_TIME_COLUMN])
    for phase in ['DEP', 'ARR']:
        phase_rows = all_rows.loc[all_rows.PHASE_mvt.eq(phase)].copy()
        # a departure's airport is its origin (ADEP), an arrival's its destination (ADES)
        airport_column = 'ADEP_mvt' if phase == 'DEP' else 'ADES_mvt'
        for window in MOVEMENT_COUNT_WINDOWS:
            features[f'{phase}_count_past_{window}'] = np.zeros(len(departures), dtype='float32')
            features[f'{phase}_count_future_{window}'] = np.zeros(len(departures), dtype='float32')
        if phase == 'ARR':
            phase_rows['arrival_error'] = seconds_between(phase_rows[MOVEMENT_TIME_COLUMN], phase_rows.ARVT_3_flt).clip(-ARRIVAL_ERROR_CAP_SECONDS, ARRIVAL_ERROR_CAP_SECONDS)
            phase_rows['arrival_taxi'] = seconds_between(phase_rows[BLOCK_TIME_COLUMN], phase_rows[MOVEMENT_TIME_COLUMN]).clip(0, TAXI_IN_CAP_SECONDS)
            for column in ['arrival_error', 'arrival_taxi']:
                features[column + '_nearby_mean'] = np.nan
        # row labels of `departures` (a RangeIndex) are also positions, so they index the numpy arrays directly
        for airport, row_labels in airports.groupby(airports).groups.items():
            airport_rows = phase_rows.loc[phase_rows[airport_column].eq(airport)].sort_values(MOVEMENT_TIME_COLUMN)
            sorted_movement_times = epoch_seconds(airport_rows[MOVEMENT_TIME_COLUMN])
            query_times = departure_times[row_labels]
            _count_movements_around_takeoff(features, phase, row_labels, sorted_movement_times, query_times)
            if phase == 'ARR':
                _add_nearby_arrival_means(features, row_labels, airport_rows, sorted_movement_times, query_times)
    return features


def _count_movements_around_takeoff(features: pd.DataFrame, phase: str, row_labels: pd.Index, sorted_movement_times: np.ndarray, query_times: np.ndarray) -> None:
    """
    This function does the counting for one airport
    """
    # left: movements strictly before the take-off & right: movements up to and including it
    before_takeoff = np.searchsorted(sorted_movement_times, query_times, side='left')
    up_to_takeoff = np.searchsorted(sorted_movement_times, query_times, side='right')
    for window in MOVEMENT_COUNT_WINDOWS:
        features.loc[row_labels, f'{phase}_count_past_{window}'] = (
            before_takeoff - np.searchsorted(sorted_movement_times, query_times - window, side='left'))
        features.loc[row_labels, f'{phase}_count_future_{window}'] = (
            np.searchsorted(sorted_movement_times, query_times + window, side='right') - up_to_takeoff)


def _add_nearby_arrival_means(features: pd.DataFrame, row_labels: pd.Index, airport_rows: pd.DataFrame, sorted_movement_times: np.ndarray, query_times: np.ndarray) -> None:
    """
    This function does the averaging above
    """
    # the window holds the landings at or after take-off - 1 h and before take-off + 1 h
    window_start = np.searchsorted(sorted_movement_times, query_times - NEARBY_ARRIVAL_HALF_WINDOW_SECONDS)
    window_end = np.searchsorted(sorted_movement_times, query_times + NEARBY_ARRIVAL_HALF_WINDOW_SECONDS)
    for column in ['arrival_error', 'arrival_taxi']:
        values = airport_rows[column].to_numpy()
        # prefix sums of the finite values and their counts: window sum = sums[end] - sums[start]
        prefix_sums = np.r_[0, np.nancumsum(values)]
        prefix_counts = np.r_[0, np.cumsum(np.isfinite(values))]
        count_in_window = prefix_counts[window_end] - prefix_counts[window_start]
        features.loc[row_labels, column + '_nearby_mean'] = (
            (prefix_sums[window_end] - prefix_sums[window_start]) / np.maximum(count_in_window, 1))


def prior_baseline(features: pd.DataFrame) -> np.ndarray:
    """
    This function makes the prior model's starting guess for each departure: take-off minus AOBT when the AOBT is known
    Without an AOBT, Rome uses take-off minus the timetable time
    """
    aobt_baseline = features.aobt_baseline.to_numpy(dtype=float)
    scheduled_baseline = features.scheduled_baseline.to_numpy(dtype=float)
    fallback = np.where(features.ADEP_mvt.eq('LIRF'), np.maximum(scheduled_baseline, 0), PRIOR_FALLBACK_BASELINE_SECONDS)
    return np.where(np.isfinite(aobt_baseline), aobt_baseline, fallback)


def build_base_tables(output_dir: Path) -> None:
    fingerprints_before = competition_file_fingerprints()
    all_rows = read_all_challenge_rows()
    departures = all_rows.loc[all_rows.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    features = add_movement_context(departure_row_features(departures), departures, all_rows)
    for column in features:
        if pd.api.types.is_string_dtype(features[column].dtype) or features[column].dtype == object:
            features[column] = features[column].astype('category')
        else:
            features[column] = features[column].astype('float32')
    features.to_parquet(output_dir / 'row_features.parquet', index=False)
    metadata = departures[['MVT_ID_mvt', TARGET_COLUMN, MOVEMENT_TIME_COLUMN, 'ADEP_mvt', 'AOBT_3_flt']].copy()
    # computed from the float32 / category features above, so the baseline has float32 precision (kept as built)
    metadata['baseline'] = prior_baseline(features)
    metadata['month'] = departures[MOVEMENT_TIME_COLUMN].dt.month
    metadata['year'] = departures[MOVEMENT_TIME_COLUMN].dt.year
    metadata.to_parquet(output_dir / 'metadata.parquet', index=False)
    assert fingerprints_before == competition_file_fingerprints(), 'Input files changed'
    print('base', features.shape, flush=True)


# weather + other additional features
TRAFFIC_WINDOWS = [300, 900, 1800]      # seconds either side of the take-off for the traffic context of the weather table


def build_weather_table(output_dir: Path) -> None:
    """
    This function builds weather features using the helpers below
    """
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    all_rows = read_all_challenge_rows()
    departures = all_rows.loc[all_rows.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    assert departures.MVT_ID_mvt.equals(metadata.MVT_ID_mvt)
    features = pd.DataFrame(index=departures.index)
    hourly_table = _hourly_weather(sorted(departures.ADEP_mvt.unique()))
    weather_at_takeoff = _weather_at_takeoff(departures, hourly_table)
    features = pd.concat([features, weather_at_takeoff.astype('float32')], axis=1)
    _add_wind_and_icing(features, departures)
    _add_clock_precision(features, departures)
    takeoff_minus_aobt = seconds_between(departures[MOVEMENT_TIME_COLUMN], departures.AOBT_3_flt)
    _add_offblock_delay_ratios(features, departures, takeoff_minus_aobt)
    takeoff_epoch = epoch_seconds(departures[MOVEMENT_TIME_COLUMN])
    _add_traffic_and_queues(features, departures, takeoff_minus_aobt, takeoff_epoch)
    _add_stand_arrival_history(features, departures, all_rows, takeoff_epoch)
    assert len(features) == len(metadata)
    features.astype('float32').to_parquet(output_dir / 'weather_features.parquet', index=False)
    print('weather', features.shape, flush=True)


def _hourly_weather(airports: list[str]) -> pd.DataFrame:
    hourly_frames = []
    for airport in airports:
        weather_path = DOWNLOADS / 'weather' / f'weather_{airport}.json'
        weather_json = json.loads(weather_path.read_text())
        hourly = pd.DataFrame(weather_json['hourly'])
        hourly['time'] = pd.to_datetime(hourly.time, utc=True)
        hourly['airport'] = airport
        hourly = hourly.sort_values('time')
        for column in ['precipitation', 'snowfall']:
            hourly[column + '_past_3h'] = hourly[column].rolling(PRECIPITATION_SHORT_WINDOW_HOURS, min_periods=1).sum()
            hourly[column + '_past_12h'] = hourly[column].rolling(PRECIPITATION_LONG_WINDOW_HOURS, min_periods=1).sum()
        hourly['temperature_change_3h'] = hourly.temperature_2m.diff(TEMPERATURE_CHANGE_HOURS)
        hourly_frames.append(hourly)
    return pd.concat(hourly_frames).set_index(['airport', 'time'])


def _weather_at_takeoff(departures: pd.DataFrame, hourly_table: pd.DataFrame) -> pd.DataFrame:
    takeoff_hour_key = pd.MultiIndex.from_arrays([departures.ADEP_mvt, departures[MOVEMENT_TIME_COLUMN].dt.floor('h')])
    weather_at_takeoff = hourly_table.reindex(takeoff_hour_key).reset_index(drop=True).add_prefix('weather_')
    assert weather_at_takeoff.weather_temperature_2m.notna().all(), 'Incomplete weather coverage'
    return weather_at_takeoff


def _add_wind_and_icing(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    # runway designator number x 10 = magnetic heading in degrees (close enough to true heading for head / crosswind)
    runway_heading = pd.to_numeric(departures.RUNWAY_mvt.str.extract(r'(\d+)', expand=False), errors='coerce') * 10
    wind_angle = np.deg2rad(features.weather_wind_direction_10m - runway_heading)
    features['weather_headwind'] = features.weather_wind_speed_10m * np.cos(wind_angle)
    features['weather_crosswind'] = (features.weather_wind_speed_10m * np.sin(wind_angle)).abs()
    features['weather_dewpoint_spread'] = features.weather_temperature_2m - features.weather_dew_point_2m
    # icing conditions (de-icing delays): below 3 degrees C and humid or precipitating
    is_cold = features.weather_temperature_2m < ICING_TEMPERATURE_CELSIUS
    is_humid_or_wet = (features.weather_relative_humidity_2m > ICING_HUMIDITY_PERCENT) | (features.weather_precipitation > 0)
    features['weather_cold_wet'] = (is_cold & is_humid_or_wet).astype(float)


def _add_clock_precision(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    takeoff_time = departures[MOVEMENT_TIME_COLUMN]
    for column in ['AOBT_3_flt', 'LOBT_flt', 'EOBT_1_flt', 'ARVT_3_flt']:
        takeoff_minus_clock = seconds_between(takeoff_time, departures[column])
        seconds_past_minute = takeoff_minus_clock % 60
        features[column + '_modulo_minute'] = np.minimum(seconds_past_minute, 60 - seconds_past_minute)
        features[column + '_rounded_minute'] = departures[column].dt.second.eq(0).astype(float)
        features[column + '_day_difference'] = (takeoff_time.dt.floor('D') - departures[column].dt.floor('D')).dt.total_seconds() / 86400


def _add_offblock_delay_ratios(features: pd.DataFrame, departures: pd.DataFrame, takeoff_minus_aobt: pd.Series) -> None:
    for column in ['LOBT_flt', 'EOBT_1_flt', 'SCHED_TIME_UTC_mvt']:
        takeoff_minus_clock = seconds_between(departures[MOVEMENT_TIME_COLUMN], departures[column])
        features[column + '_baseline_ratio'] = takeoff_minus_aobt / (takeoff_minus_clock.abs() + BASELINE_RATIO_OFFSET_SECONDS)
        features[column + '_delay_signed_log'] = np.sign(takeoff_minus_clock) * np.log1p(takeoff_minus_clock.abs())


def _add_traffic_and_queues(features: pd.DataFrame, departures: pd.DataFrame, takeoff_minus_aobt: pd.Series, takeoff_epoch: np.ndarray) -> None:
    aobt_epoch = epoch_seconds(departures.AOBT_3_flt)       # NaT -> about -9.2e9 (no AOBT) - such rows are masked below
    has_plausible_baseline = takeoff_minus_aobt.between(0, PLAUSIBLE_BASELINE_MAX_SECONDS).to_numpy()
    baselines = takeoff_minus_aobt.to_numpy()
    for group_name, group_keys in [('airport', departures.ADEP_mvt), ('runway', departures.ADEP_mvt + ':' + departures.RUNWAY_mvt)]:
        for window in TRAFFIC_WINDOWS:
            features[f'{group_name}_takeoffs_near_{window}'] = 0.
            features[f'{group_name}_proxy_mean_{window}'] = np.nan
        for query_name in ['offblock', 'takeoff']:
            features[f'{group_name}_queue_at_{query_name}'] = np.nan
        for _, group_rows in group_keys.groupby(group_keys).groups.items():
            group_rows = np.asarray(group_rows)       # row labels = positions (RangeIndex)
            _add_takeoffs_and_proxy_means(features, group_name, group_rows, baselines, takeoff_epoch)
            _add_queue_lengths(features, group_name, group_rows, has_plausible_baseline, aobt_epoch, takeoff_epoch)


def _add_takeoffs_and_proxy_means(features: pd.DataFrame, group_name: str, group_rows: np.ndarray, baselines: np.ndarray, takeoff_epoch: np.ndarray) -> None:
    rows_by_takeoff = group_rows[np.argsort(takeoff_epoch[group_rows])]
    sorted_takeoffs = takeoff_epoch[rows_by_takeoff]
    sorted_baselines = baselines[rows_by_takeoff]
    # only baselines of 0 - 7200 s count towards the proxy mean
    sorted_baselines = np.where((sorted_baselines >= 0) & (sorted_baselines <= PROXY_BASELINE_MAX_SECONDS), sorted_baselines, np.nan)
    baseline_prefix_sums = np.r_[0, np.nancumsum(sorted_baselines)]
    baseline_prefix_counts = np.r_[0, np.cumsum(np.isfinite(sorted_baselines))]
    for window in TRAFFIC_WINDOWS:
        window_start = np.searchsorted(sorted_takeoffs, takeoff_epoch[group_rows] - window)
        window_end = np.searchsorted(sorted_takeoffs, takeoff_epoch[group_rows] + window, side='right')
        features.loc[group_rows, f'{group_name}_takeoffs_near_{window}'] = window_end - window_start - 1   # minus itself
        # mean baseline of the other departures in the window: remove the departure's own baseline when it counted
        own_baseline = baselines[group_rows]
        own_baseline_counted = (own_baseline >= 0) & (own_baseline <= PROXY_BASELINE_MAX_SECONDS)
        other_count = baseline_prefix_counts[window_end] - baseline_prefix_counts[window_start] - own_baseline_counted
        other_sum = baseline_prefix_sums[window_end] - baseline_prefix_sums[window_start] - np.where(own_baseline_counted, own_baseline, 0)
        other_mean = other_sum / np.maximum(other_count, 1)
        features.loc[group_rows, f'{group_name}_proxy_mean_{window}'] = np.where(other_count > 0, other_mean, np.nan)


def _add_queue_lengths(features: pd.DataFrame, group_name: str, group_rows: np.ndarray, has_plausible_baseline: np.ndarray, aobt_epoch: np.ndarray, takeoff_epoch: np.ndarray) -> None:
    plausible_rows = group_rows[has_plausible_baseline[group_rows]]
    sorted_offblocks = np.sort(aobt_epoch[plausible_rows])
    sorted_plausible_takeoffs = np.sort(takeoff_epoch[plausible_rows])
    for query_name, query_times in [('offblock', aobt_epoch[group_rows]), ('takeoff', takeoff_epoch[group_rows])]:
        already_off_block = np.searchsorted(sorted_offblocks, query_times, side='right')
        already_airborne = np.searchsorted(sorted_plausible_takeoffs, query_times, side='right')
        queue_length = already_off_block - already_airborne
        if query_name == 'offblock':
            # at its own off-block the departure counts itself (minus 1)
            queue_length = np.where(has_plausible_baseline[group_rows], queue_length - 1, np.nan)
        features.loc[group_rows, f'{group_name}_queue_at_{query_name}'] = queue_length


def _add_stand_arrival_history(features: pd.DataFrame, departures: pd.DataFrame, all_rows: pd.DataFrame, takeoff_epoch: np.ndarray) -> None:
    arrivals = all_rows.loc[all_rows.PHASE_mvt.eq('ARR')].copy()
    arrivals['stand_key'] = arrivals.ADES_mvt.fillna('') + ':' + arrivals.STAND_mvt.fillna('')
    arrivals['taxi'] = seconds_between(arrivals[BLOCK_TIME_COLUMN], arrivals[MOVEMENT_TIME_COLUMN]).clip(0, TAXI_IN_CAP_SECONDS)
    departure_stand_keys = departures.ADEP_mvt.fillna('') + ':' + departures.STAND_mvt.fillna('')
    arrival_positions_by_stand = arrivals.groupby('stand_key').indices
    features['stand_arrival_taxi_7d'] = np.nan
    features['stand_arrival_count_7d'] = 0.
    seven_days = STAND_ARRIVAL_WINDOW_DAYS * 86400
    for stand_key, row_labels in departure_stand_keys.groupby(departure_stand_keys).groups.items():
        if stand_key not in arrival_positions_by_stand:
            continue
        stand_arrivals = arrivals.iloc[arrival_positions_by_stand[stand_key]].sort_values(MOVEMENT_TIME_COLUMN)
        landing_times = epoch_seconds(stand_arrivals[MOVEMENT_TIME_COLUMN])
        taxi_in_times = stand_arrivals.taxi.to_numpy()
        window_start = np.searchsorted(landing_times, takeoff_epoch[row_labels] - seven_days)
        window_end = np.searchsorted(landing_times, takeoff_epoch[row_labels] + seven_days)
        prefix_sums = np.r_[0, np.nancumsum(taxi_in_times)]
        prefix_counts = np.r_[0, np.cumsum(np.isfinite(taxi_in_times))]
        count_in_window = prefix_counts[window_end] - prefix_counts[window_start]
        features.loc[row_labels, 'stand_arrival_count_7d'] = count_in_window
        features.loc[row_labels, 'stand_arrival_taxi_7d'] = np.where(
            count_in_window > 0, (prefix_sums[window_end] - prefix_sums[window_start]) / np.maximum(count_in_window, 1), np.nan)


# clock
def canonical_flight_number(flight_numbers: pd.Series) -> pd.Series:
    """
    This function cleans flight numbers so the same flight is spelled the same way in every record
    """
    flight_numbers = flight_numbers.fillna('').str.strip()
    has_doubled_code = flight_numbers.str[1:3].eq(flight_numbers.str[3:5])
    return flight_numbers.where(~has_doubled_code, flight_numbers.str[:3] + flight_numbers.str[5:])


CLOCK_TABLE_COLUMNS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', MOVEMENT_TIME_COLUMN, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', BLOCK_TIME_COLUMN]


def build_clock_table(output_dir: Path) -> None:
    """
    This function builds clock features using the helpers below
    """
    departures, arrivals = _clock_table_rows()
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    assert departures.MVT_ID_mvt.equals(metadata.MVT_ID_mvt)
    features = pd.DataFrame(index=departures.index)
    for column in ['AOBT_3_flt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'SCHED_TIME_UTC_mvt']:
        features['clock_' + column] = seconds_between(departures[MOVEMENT_TIME_COLUMN], departures[column])
    _add_linked_arrival_clocks(features, departures, arrivals)
    _add_previous_stand_arrival(features, departures, arrivals)
    _add_previous_same_flight(features, departures)
    features.astype('float32').to_parquet(output_dir / 'clock_features.parquet', index=False)

    print('clock', features.shape, flush=True)


def _clock_table_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = pd.concat([pd.read_parquet(path, columns=CLOCK_TABLE_COLUMNS) for path in competition_files()], ignore_index=True)
    all_rows.loc[all_rows.PHASE_mvt.eq('DEP'), BLOCK_TIME_COLUMN] = pd.NaT      # every departure block masked first
    departures = all_rows.loc[all_rows.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    arrivals = all_rows.loc[all_rows.PHASE_mvt.eq('ARR')].copy()
    departures['row'] = np.arange(len(departures))      # restores the departure order after the as-of joins
    return departures, arrivals


def _add_linked_arrival_clocks(features: pd.DataFrame, departures: pd.DataFrame, arrivals: pd.DataFrame) -> None:
    for frame in [departures, arrivals]:
        frame['key'] = frame.ADEP_mvt.fillna('') + ':' + frame.ADES_mvt.fillna('') + ':' + canonical_flight_number(frame.FLIGHT_mvt)
    arrival_clock_columns = ['key', 'AOBT_3_flt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', MOVEMENT_TIME_COLUMN]
    arrival_clocks = arrivals.loc[arrivals.AOBT_3_flt.notna(), arrival_clock_columns].copy()
    arrival_clocks = arrival_clocks.rename(columns={column: 'arrival_' + column for column in arrival_clocks if column != 'key'})
    departures_by_time = departures[['row', 'key', MOVEMENT_TIME_COLUMN]].sort_values(MOVEMENT_TIME_COLUMN)
    arrivals_by_time = arrival_clocks.sort_values('arrival_' + MOVEMENT_TIME_COLUMN)
    joined = pd.merge_asof(departures_by_time, arrivals_by_time, left_on=MOVEMENT_TIME_COLUMN, right_on='arrival_' + MOVEMENT_TIME_COLUMN, by='key', direction='forward', tolerance=pd.Timedelta(LINKED_ARRIVAL_CLOCK_TOLERANCE)).sort_values('row')
    for column in ['AOBT_3_flt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt']:
        features['clock_arrival_' + column] = seconds_between(joined[MOVEMENT_TIME_COLUMN], joined['arrival_' + column]).to_numpy()
    features['clock_linked_airborne'] = seconds_between(joined['arrival_' + MOVEMENT_TIME_COLUMN], joined[MOVEMENT_TIME_COLUMN]).to_numpy()


def _add_previous_stand_arrival(features: pd.DataFrame, departures: pd.DataFrame, arrivals: pd.DataFrame) -> None:
    departures['stand_key'] = departures.ADEP_mvt.fillna('') + ':' + departures.STAND_mvt.fillna('')
    arrivals['stand_key'] = arrivals.ADES_mvt.fillna('') + ':' + arrivals.STAND_mvt.fillna('')
    has_block_and_stand = arrivals[BLOCK_TIME_COLUMN].notna() & arrivals.STAND_mvt.notna()
    stand_arrivals = arrivals.loc[has_block_and_stand, ['stand_key', BLOCK_TIME_COLUMN, MOVEMENT_TIME_COLUMN]].copy()
    stand_arrivals = stand_arrivals.rename(columns={MOVEMENT_TIME_COLUMN: 'arrival_movement'}).sort_values(BLOCK_TIME_COLUMN)
    departures_by_time = departures[['row', 'stand_key', MOVEMENT_TIME_COLUMN]].sort_values(MOVEMENT_TIME_COLUMN)
    joined = pd.merge_asof(departures_by_time, stand_arrivals, left_on=MOVEMENT_TIME_COLUMN, right_on=BLOCK_TIME_COLUMN, by='stand_key', direction='backward', tolerance=pd.Timedelta(PREVIOUS_STAND_ARRIVAL_TOLERANCE)).sort_values('row')
    features['clock_previous_stand_arrival_block'] = seconds_between(joined[MOVEMENT_TIME_COLUMN], joined[BLOCK_TIME_COLUMN]).to_numpy()
    features['clock_previous_stand_arrival_movement'] = seconds_between(joined[MOVEMENT_TIME_COLUMN], joined.arrival_movement).to_numpy()


def _add_previous_same_flight(features: pd.DataFrame, departures: pd.DataFrame) -> None:
    departures['flight_key'] = departures.ADEP_mvt.fillna('') + ':' + canonical_flight_number(departures.FLIGHT_mvt)
    departures_by_time = departures.sort_values(MOVEMENT_TIME_COLUMN)
    for column in [MOVEMENT_TIME_COLUMN, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt']:
        previous_value = departures_by_time.groupby('flight_key')[column].shift(1)
        takeoff_minus_previous = seconds_between(departures_by_time[MOVEMENT_TIME_COLUMN], previous_value)
        features['clock_previous_flight_' + column] = takeoff_minus_previous.reindex(departures.index)


# opdi
def build_opdi_table(output_dir: Path) -> None:
    """
    This function builds opdi features using the helpers below
    """
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    matched_parts = []
    for path in competition_files():
        file_departures = pd.read_parquet(path)
        file_departures = file_departures.loc[file_departures.PHASE_mvt.eq('DEP')].copy()
        for month, month_departures in file_departures.groupby(file_departures[MOVEMENT_TIME_COLUMN].dt.strftime('%Y%m')):
            flight_list_path = DOWNLOADS / 'opdi' / f'flight_list_{month}.parquet'
            if not flight_list_path.exists():
                raise FileNotFoundError(flight_list_path)
            matched_parts.append(_match_month_to_opdi(month_departures, pd.read_parquet(flight_list_path)))
    opdi_features = pd.concat(matched_parts).set_index('MVT_ID_mvt').reindex(metadata.MVT_ID_mvt).reset_index(drop=True)
    for column in opdi_features.select_dtypes(include=['str', 'string', 'object']):
        opdi_features[column] = opdi_features[column].astype('category')
    opdi_features.to_parquet(output_dir / 'opdi_features.parquet', index=False)
    print('opdi', opdi_features.shape, flush=True)


def _match_month_to_opdi(month_departures: pd.DataFrame, flight_list: pd.DataFrame) -> pd.DataFrame:
    # match key: departure airport (adep, else the probable adep_p) and callsign
    flight_list['key'] = flight_list.adep.fillna(flight_list.adep_p).fillna('') + ':' + flight_list.flt_id.fillna('').str.strip()
    month_departures = month_departures.copy()
    month_departures['key'] = month_departures.ADEP_mvt + ':' + month_departures.CALLSIGN_flt.fillna('').str.strip()
    for column in ['first_seen', 'last_seen']:
        flight_list[column] = pd.to_datetime(flight_list[column], utc=True)
    joined = pd.merge_asof(month_departures.sort_values(MOVEMENT_TIME_COLUMN), flight_list.sort_values('first_seen'), left_on=MOVEMENT_TIME_COLUMN, right_on='first_seen', by='key', tolerance=pd.Timedelta(OPDI_MATCH_TOLERANCE), direction='nearest')
    matched = joined[['MVT_ID_mvt']].copy()
    for seen_column in ['first_seen', 'last_seen']:
        for reference_column in [MOVEMENT_TIME_COLUMN, 'AOBT_3_flt', 'ARVT_3_flt', 'LOBT_flt']:
            seen_minus_reference = seconds_between(joined[seen_column], joined[reference_column])
            matched[f'opdi_{seen_column}_minus_{reference_column}'] = seen_minus_reference.astype('float32')
    matched['opdi_seen_duration'] = seconds_between(joined.last_seen, joined.first_seen).astype('float32')
    matched['opdi_destination_match'] = (joined.ades == joined.ADES_mvt).astype('float32')
    matched['opdi_type_match'] = (joined.typecode == joined.AIRCRAFT_TYPE_mvt).astype('float32')
    for column in ['icao24', 'typecode', 'icao_operator']:      # icao24 = the aircraft's transponder address
        matched['opdi_' + column] = joined[column].fillna('<MISSING>')
    return matched


# operational
OPERATIONAL_TABLE_COLUMNS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', MOVEMENT_TIME_COLUMN, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt']
SCHEDULE_WINDOWS = [1800, 3600]     # seconds ahead in which scheduled movements are counted
RUNWAY_USE_HALF_WINDOW = 1800       # runway use is measured over the movements within +-30 min of the take-off


def build_operational_table(output_dir: Path) -> None:
    """
    This function builds operational features
    """
    all_rows = pd.concat([pd.read_parquet(path, columns=OPERATIONAL_TABLE_COLUMNS) for path in competition_files()], ignore_index=True)
    departures = all_rows.loc[all_rows.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet', columns=['MVT_ID_mvt'])
    assert departures.MVT_ID_mvt.equals(metadata.MVT_ID_mvt)
    features = pd.DataFrame(index=departures.index)
    takeoff_epoch = epoch_seconds(departures[MOVEMENT_TIME_COLUMN])
    scheduled_epoch = epoch_seconds(departures.SCHED_TIME_UTC_mvt)     # a missing schedule becomes about -9.2e9 (counts 0)
    for phase in ['DEP', 'ARR']:
        phase_rows = all_rows.loc[all_rows.PHASE_mvt.eq(phase)]
        # a departure's airport is its origin (ADEP), an arrival's its destination (ADES)
        airport_column = 'ADEP_mvt' if phase == 'DEP' else 'ADES_mvt'
        for airport, row_labels in departures.groupby('ADEP_mvt').groups.items():
            airport_rows = phase_rows.loc[phase_rows[airport_column].eq(airport)]
            _add_scheduled_movement_counts(features, phase, row_labels, airport_rows, scheduled_epoch, takeoff_epoch)
            _add_runway_use(features, phase, row_labels, airport_rows, takeoff_epoch)
    # pressure: scheduled departures per scheduled arrival in the next window (+1 smoothing)
    for window in SCHEDULE_WINDOWS:
        for query_name in ['scheduled', 'actual']:
            departures_next = features[f'ops_DEP_{query_name}_next_{window}']
            arrivals_next = features[f'ops_ARR_{query_name}_next_{window}']
            features[f'ops_pressure_{query_name}_{window}'] = (departures_next + PRESSURE_PSEUDO_COUNT) / (arrivals_next + PRESSURE_PSEUDO_COUNT)
    _add_previous_stand_departure(features, departures, takeoff_epoch, scheduled_epoch)
    features.astype('float32').to_parquet(output_dir / 'operational_features.parquet', index=False)
    print('operational', features.shape, flush=True)


def _add_scheduled_movement_counts(features: pd.DataFrame, phase: str, row_labels: pd.Index, airport_rows: pd.DataFrame, scheduled_epoch: np.ndarray, takeoff_epoch: np.ndarray) -> None:
    sorted_schedule_times = np.sort(epoch_seconds(airport_rows.SCHED_TIME_UTC_mvt.dropna()))
    for window in SCHEDULE_WINDOWS:
        for query_name, query_times in [('scheduled', scheduled_epoch[row_labels]), ('actual', takeoff_epoch[row_labels])]:
            window_start = np.searchsorted(sorted_schedule_times, query_times)
            window_end = np.searchsorted(sorted_schedule_times, query_times + window)
            features.loc[row_labels, f'ops_{phase}_{query_name}_next_{window}'] = window_end - window_start


def _add_runway_use(features: pd.DataFrame, phase: str, row_labels: pd.Index, airport_rows: pd.DataFrame, takeoff_epoch: np.ndarray) -> None:
    movement_times = epoch_seconds(airport_rows[MOVEMENT_TIME_COLUMN])
    sorted_movement_times = np.sort(movement_times)
    runways = airport_rows.RUNWAY_mvt.fillna('unknown')
    window_start_times = takeoff_epoch[row_labels] - RUNWAY_USE_HALF_WINDOW
    window_end_times = takeoff_epoch[row_labels] + RUNWAY_USE_HALF_WINDOW
    movements_before_window_end = np.searchsorted(sorted_movement_times, window_end_times)
    movements_in_window = movements_before_window_end - np.searchsorted(sorted_movement_times, window_start_times)
    runway_shares = []
    for runway in sorted(runways.unique()):
        sorted_runway_times = np.sort(movement_times[runways.eq(runway)])
        runway_movements = np.searchsorted(sorted_runway_times, window_end_times) - np.searchsorted(sorted_runway_times, window_start_times)
        runway_share = runway_movements / np.maximum(movements_in_window, 1)
        runway_shares.append(runway_share)
        features.loc[row_labels, f'ops_{phase}_runway_share_{runway}'] = runway_share
    share_matrix = np.array(runway_shares)        # one row per runway, one column per departure
    features.loc[row_labels, f'ops_{phase}_runway_entropy'] = -(share_matrix * np.log(np.maximum(share_matrix, 1e-8))).sum(axis=0)
    features.loc[row_labels, f'ops_{phase}_runways_active'] = (share_matrix > RUNWAY_ACTIVE_SHARE).sum(axis=0)


def _add_previous_stand_departure(features: pd.DataFrame, departures: pd.DataFrame, takeoff_epoch: np.ndarray, scheduled_epoch: np.ndarray) -> None:
    departures['key'] = departures.ADEP_mvt + ':' + departures.STAND_mvt.fillna('<MISSING>')
    aobt_epoch = epoch_seconds(departures.AOBT_3_flt)
    for stand_key, row_labels in departures.groupby('key').groups.items():
        if stand_key.endswith('<MISSING>'):
            continue
        row_labels = np.asarray(row_labels)      # row labels = positions (RangeIndex)
        rows_by_takeoff = row_labels[np.argsort(takeoff_epoch[row_labels])]
        # position (in take-off order) of the last take-off strictly before each departure's NM off-block
        previous_position = np.searchsorted(takeoff_epoch[rows_by_takeoff], aobt_epoch[row_labels], side='left') - 1
        found = (previous_position >= 0) & departures.AOBT_3_flt.iloc[row_labels].notna().to_numpy()
        previous_departure = rows_by_takeoff[np.maximum(previous_position, 0)]
        age = aobt_epoch[row_labels] - takeoff_epoch[previous_departure]
        found &= (age >= 0) & (age <= PREVIOUS_STAND_DEPARTURE_MAX_AGE_SECONDS) & (previous_departure != row_labels)
        features.loc[row_labels, 'ops_previous_stand_departure_age'] = np.where(found, age, np.nan)
        previous_delay = takeoff_epoch[previous_departure] - scheduled_epoch[previous_departure]
        features.loc[row_labels, 'ops_previous_stand_departure_delay'] = np.where(found, previous_delay, np.nan)
        features.loc[row_labels, 'ops_stand_recent_late_proxy'] = np.where(found, (previous_delay > LATE_STAND_DEPARTURE_DELAY_SECONDS) & (age < RECENT_STAND_DEPARTURE_AGE_SECONDS), np.nan)


# ground events
def opdi_event_files() -> list[Path]:
    return sorted((DOWNLOADS / 'opdi').glob('flight_events_*.parquet'))


def as_unsigned_flight_ids(flight_ids: pd.Series) -> pd.Series:
    """
    This function makes OPDI flight ids match across files
    """
    if not pd.api.types.is_integer_dtype(flight_ids.dtype):
        raise TypeError(f'Flight IDs must be integers, got {flight_ids.dtype}')
    is_missing = flight_ids.isna().to_numpy()
    unsigned_values = flight_ids.fillna(0).to_numpy().astype('uint64')
    result = pd.Series(pd.array(unsigned_values, dtype='UInt64'), index=flight_ids.index)
    result.loc[is_missing] = pd.NA
    return result


def extract_ground_events() -> None:
    """
    This function makes small copies of the huge OPDI event files that keep only the 7 kinds of ground events at the ten airports
    """
    GROUND_EVENT_EXTRACTS.mkdir(parents=True, exist_ok=True)
    for event_file in opdi_event_files():
        extract_file = GROUND_EVENT_EXTRACTS / event_file.name
        if extract_file.exists() or not event_file.exists():
            continue
        try:
            pq.ParquetFile(event_file)
        except Exception:
            continue  # a concurrent download may not yet have a footer
        events = _study_airport_ground_events(event_file)
        events.to_parquet(extract_file, index=False)
        print(event_file.name, len(events), flush=True)


def _study_airport_ground_events(event_file: Path) -> pd.DataFrame:
    events = pd.read_parquet(event_file, columns=['flight_id', 'type', 'event_time', 'info'], filters=[('type', 'in', GROUND_EVENT_TYPES)], dtype_backend='pyarrow')
    events = events.loc[events.flight_id.notna()].copy()
    events['flight_id'] = as_unsigned_flight_ids(events.flight_id)
    # cheap text filter on the JSON 'info' field before parsing it
    study_airport_pattern = '"osm_airport":"(?:' + '|'.join(STUDY_AIRPORTS) + ')"'
    events = events.loc[events['info'].fillna('').str.contains(study_airport_pattern, regex=True)].copy()
    info = pd.DataFrame([json.loads(text) for text in events['info']], index=events.index)
    events['airport'] = info.get('osm_airport', '')
    events['callsign'] = info.get('osn_flight_id', pd.Series('', index=events.index)).str.strip()
    events['duration'] = pd.to_numeric(info.get('opdi_time_in_use_s', np.nan), errors='coerce')
    events['reference'] = info.get('osm_ref', '')
    return events.drop(columns='info')


GROUND_TABLE_COLUMNS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', MOVEMENT_TIME_COLUMN]
GROUND_EVENT_WINDOW = (-120, 14400)     # events kept from 2 min after to 4 h before the take-off (s before take-off)


def build_ground_event_table(output_dir: Path) -> None:
    """
    This function builds features that OPDI saw each plane doing on the ground before take-off
    """
    event_files = opdi_event_files()
    missing_extracts = [event_file.name for event_file in event_files if not (GROUND_EVENT_EXTRACTS / event_file.name).exists()]
    if missing_extracts:
        raise FileNotFoundError(f'Missing filtered event files: {missing_extracts}')
    events = _read_ground_event_extracts(event_files)
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    summaries = []
    for path in competition_files():
        file_departures = _ground_table_departures(path)
        for month, month_departures in file_departures.groupby(file_departures[MOVEMENT_TIME_COLUMN].dt.strftime('%Y%m')):
            flight_list = _ground_table_flight_list(month)
            joined = pd.merge_asof(month_departures.sort_values(MOVEMENT_TIME_COLUMN), flight_list.sort_values('first_seen'), left_on=MOVEMENT_TIME_COLUMN, right_on='first_seen', by='key', direction='nearest', tolerance=pd.Timedelta(OPDI_MATCH_TOLERANCE))
            summaries.append(_summarise_ground_events(joined, events))
    ground_features = pd.concat(summaries)
    assert ground_features.index.is_unique
    ground_features = ground_features.reindex(metadata.MVT_ID_mvt).reset_index(drop=True)
    ground_features['ground_available'] = ground_features.notna().any(axis=1).astype(float)
    ground_features.astype('float32').to_parquet(output_dir / 'ground_event_features.parquet', index=False)
    print('ground', ground_features.shape, flush=True)


def _read_ground_event_extracts(event_files: list[Path]) -> pd.DataFrame:
    event_parts = []
    for event_file in event_files:
        event_part = pd.read_parquet(GROUND_EVENT_EXTRACTS / event_file.name, dtype_backend='pyarrow')
        event_part['flight_id'] = as_unsigned_flight_ids(event_part.flight_id)
        event_parts.append(event_part)
    events = pd.concat(event_parts, ignore_index=True).drop_duplicates()     # the event files overlap in time
    events['event_time'] = pd.to_datetime(events.event_time, utc=True)
    events['flight_id'] = events.flight_id.astype('UInt64')
    return events


def _ground_table_departures(path: Path) -> pd.DataFrame:
    file_departures = pd.read_parquet(path, columns=GROUND_TABLE_COLUMNS)
    file_departures = file_departures.loc[file_departures.PHASE_mvt.eq('DEP')].copy()
    file_departures[MOVEMENT_TIME_COLUMN] = pd.to_datetime(file_departures[MOVEMENT_TIME_COLUMN], utc=True).dt.as_unit('ns')
    flight_numbers = file_departures.FLIGHT_mvt.fillna('')
    has_doubled_code = flight_numbers.str[1:3].eq(flight_numbers.str[3:5])
    flight_numbers = flight_numbers.where(~has_doubled_code, flight_numbers.str[:3] + flight_numbers.str[5:])
    file_departures['key'] = file_departures.ADEP_mvt + ':' + file_departures.CALLSIGN_flt.fillna(flight_numbers).fillna('').str.strip()
    return file_departures


def _ground_table_flight_list(month: str) -> pd.DataFrame:
    flight_list = pd.read_parquet(DOWNLOADS / 'opdi' / f'flight_list_{month}.parquet', columns=['id', 'adep', 'flt_id', 'first_seen'], dtype_backend='pyarrow')
    flight_list['id'] = as_unsigned_flight_ids(flight_list.id)
    flight_list['key'] = flight_list.adep.fillna('') + ':' + flight_list.flt_id.fillna('').str.strip()
    flight_list['key'] = flight_list['key'].astype(str)
    flight_list['first_seen'] = pd.to_datetime(flight_list.first_seen, utc=True).astype('datetime64[ns, UTC]')
    return flight_list


def _summarise_ground_events(joined: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    relevant_events = events.loc[events.flight_id.isin(joined.id.dropna())]
    departure_columns = joined[['MVT_ID_mvt', 'id', 'ADEP_mvt', MOVEMENT_TIME_COLUMN]]
    matched_events = departure_columns.merge(relevant_events, left_on='id', right_on='flight_id', how='inner')
    matched_events['delta'] = seconds_between(matched_events[MOVEMENT_TIME_COLUMN], matched_events.event_time)   # time before take-off
    at_departure_airport = matched_events.ADEP_mvt.eq(matched_events.airport)
    in_window = matched_events.delta.between(*GROUND_EVENT_WINDOW)
    matched_events = matched_events.loc[at_departure_airport & in_window].copy()
    matched_events['start_delta'] = matched_events.delta + matched_events.duration
    # 'earliest' = largest time before take-off; the column names ground_<type>_<statistic> are fixed
    summary = matched_events.groupby(['MVT_ID_mvt', 'type']).agg(
        earliest=('delta', 'max'), latest=('delta', 'min'), count=('delta', 'size'), duration=('duration', 'max'),
        earliest_start=('start_delta', 'max')).unstack('type')
    summary.columns = ['ground_' + event_type + '_' + statistic for statistic, event_type in summary.columns]
    return summary


# no-link rows
NO_LINK_ROW_COLUMNS = ['MVT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', MOVEMENT_TIME_COLUMN, 'SCHED_TIME_UTC_mvt']
NO_LINK_CANDIDATE_COLUMNS = ['flt_id', 'adep', 'ades', 'first_seen', 'last_seen', 'typecode', 'icao_operator']


def build_no_link_table(output_dir: Path) -> None:
    features, metadata = load_prior_tables(include_opdi_features=True)
    is_no_link = metadata.AOBT_3_flt.isna()
    features = features.loc[is_no_link].reset_index(drop=True)
    weather_features = pd.read_parquet(PRIOR_TABLES / 'weather_features.parquet').loc[is_no_link].reset_index(drop=True)
    features = pd.concat([features, weather_features], axis=1)
    no_link_rows = _no_link_rows(metadata.loc[is_no_link, 'MVT_ID_mvt'])
    flight_numbers = no_link_rows.FLIGHT_mvt.fillna('')
    _add_flight_number_shape(features, flight_numbers)
    # repeated two-letter code removed (see canonical_flight_number)
    has_doubled_code = flight_numbers.str[1:3].eq(flight_numbers.str[3:5])
    no_link_rows['canonical'] = np.where(has_doubled_code, flight_numbers.str[:3] + flight_numbers.str[5:], flight_numbers)
    no_link_rows['key'] = no_link_rows.ADEP_mvt + ':' + no_link_rows.canonical
    joined = _recover_opdi_flights(no_link_rows)
    for column in ['first_seen', 'last_seen']:
        features['recovered_' + column + '_delta'] = seconds_between(joined[MOVEMENT_TIME_COLUMN], joined[column]).to_numpy()
    features['recovered_destination_match'] = joined.ADES_mvt.eq(joined.ades).astype(float)
    for column in ['typecode', 'icao_operator']:
        features['recovered_' + column] = joined[column].fillna('<MISSING>')
    for column in features:
        if isinstance(features[column].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(features[column]):
            features[column] = features[column].astype(str).fillna('<MISSING>').astype('category')
        else:
            features[column] = features[column].astype('float32')
    features.to_parquet(output_dir / 'no_link_features.parquet', index=False)
    print('no-link', features.shape, 'recovered OPDI for', int(joined.first_seen.notna().sum()), 'of', len(no_link_rows), flush=True)


def _no_link_rows(no_link_ids: pd.Series) -> pd.DataFrame:
    file_parts = []
    for path in competition_files():
        file_rows = pd.read_parquet(path, columns=NO_LINK_ROW_COLUMNS)
        file_parts.append(file_rows[file_rows.MVT_ID_mvt.isin(no_link_ids)])
    return pd.concat(file_parts).set_index('MVT_ID_mvt').reindex(no_link_ids).reset_index()


def _add_flight_number_shape(features: pd.DataFrame, flight_numbers: pd.Series) -> None:
    features['raw_prefix3'] = flight_numbers.str[:3].fillna('')
    features['raw_length'] = flight_numbers.str.len().astype(float)
    features['raw_repeated_prefix'] = flight_numbers.str[1:3].eq(flight_numbers.str[3:5]).astype(float)
    features['raw_digits'] = flight_numbers.str.count(r'\d').astype(float)
    features['raw_letters'] = flight_numbers.str.count(r'[A-Z]').astype(float)


def _recover_opdi_flights(no_link_rows: pd.DataFrame) -> pd.DataFrame:
    candidate_parts = []
    for flight_list_path in sorted((DOWNLOADS / 'opdi').glob('flight_list_*.parquet')):
        flight_list = pd.read_parquet(flight_list_path, columns=NO_LINK_CANDIDATE_COLUMNS)
        flight_list['key'] = flight_list.adep.fillna('') + ':' + flight_list.flt_id.fillna('').str.strip()
        candidate_parts.append(flight_list[flight_list.key.isin(no_link_rows.key)])
    candidates = pd.concat(candidate_parts, ignore_index=True)
    for column in ['first_seen', 'last_seen']:
        candidates[column] = pd.to_datetime(candidates[column], utc=True)
    joined = pd.merge_asof(no_link_rows.sort_values(MOVEMENT_TIME_COLUMN), candidates.sort_values('first_seen'), left_on=MOVEMENT_TIME_COLUMN, right_on='first_seen', by='key', direction='nearest', tolerance=pd.Timedelta(NO_LINK_OPDI_TOLERANCE))
    return joined.set_index('MVT_ID_mvt').reindex(no_link_rows.MVT_ID_mvt).reset_index()


# linked arrivals
LINKED_ARRIVAL_KEY = ['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt']     # same flight number, origin and destination
LINKED_ARRIVAL_MAX_SECONDS = 6 * 3600                            # the arrival's NM off-block lies at most 6 h before the take-off


def link_departures_to_arrivals(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """
    This function tries to find the same flight's arrival record at its destination
    """
    departures = departures.assign(day=departures[MOVEMENT_TIME_COLUMN].dt.floor('D'))
    arrivals = arrivals.rename(columns={MOVEMENT_TIME_COLUMN: 'linked_landing', 'AOBT_3_flt': 'linked_aobt'})
    departure_keys = departures[['MVT_ID_mvt', MOVEMENT_TIME_COLUMN, 'day', *LINKED_ARRIVAL_KEY]]
    candidates_by_day = []
    for days_before in (0, 1):         # the off-block is on the take-off day or the day before: joining by day bounds the join
        arrivals_by_day = arrivals.assign(day=arrivals.linked_aobt.dt.floor('D') + pd.Timedelta(days=days_before))
        candidates_by_day.append(departure_keys.merge(arrivals_by_day, on=[*LINKED_ARRIVAL_KEY, 'day']))
    candidates = pd.concat(candidates_by_day, ignore_index=True)
    candidates['linked_base'] = (candidates[MOVEMENT_TIME_COLUMN] - candidates.linked_aobt).dt.total_seconds()
    candidates = candidates[(candidates.linked_base >= 0) & (candidates.linked_base <= LINKED_ARRIVAL_MAX_SECONDS)]
    closest = candidates.sort_values(['MVT_ID_mvt', 'linked_base', 'linked_landing']).drop_duplicates('MVT_ID_mvt')
    return closest.set_index('MVT_ID_mvt')[['linked_base', 'linked_landing', 'linked_aobt']]


def build_linked_arrivals_table(output_dir: Path) -> None:
    columns = ['MVT_ID_mvt', 'PHASE_mvt', *LINKED_ARRIVAL_KEY, MOVEMENT_TIME_COLUMN, 'AOBT_3_flt']
    movements = pd.concat([pd.read_parquet(path, columns=columns) for path in competition_files()]).drop_duplicates('MVT_ID_mvt')
    departures = movements[movements.PHASE_mvt.eq('DEP')]
    has_nm_offblock = movements.AOBT_3_flt.notna()
    arrivals = movements[movements.PHASE_mvt.eq('ARR') & has_nm_offblock][[*LINKED_ARRIVAL_KEY, MOVEMENT_TIME_COLUMN, 'AOBT_3_flt']]
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet', columns=['MVT_ID_mvt'])
    table = link_departures_to_arrivals(departures, arrivals).reindex(metadata.MVT_ID_mvt).reset_index()
    table.to_parquet(output_dir / 'linked_arrivals.parquet', index=False)
    print('linked arrivals', table.shape, 'linked departures', int(table.linked_base.notna().sum()), flush=True)


TABLE_BUILDERS: dict[str, Callable[[Path], None]] = {
    'base': build_base_tables, 
    'weather': build_weather_table, 
    'clock': build_clock_table, 
    'opdi': build_opdi_table,
    'operational': build_operational_table, 
    'ground': build_ground_event_table, 
    'no-link': build_no_link_table,
    'linked-arrivals': build_linked_arrivals_table
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the per-departure feature tables.')
    table_names = ['base', 'weather', 'clock', 'opdi', 'operational', 'ground-extract', 'ground', 'no-link', 'linked-arrivals']
    parser.add_argument('tables', nargs='+', choices=table_names)
    parser.add_argument('--out-dir', default=str(PRIOR_TABLES))
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for table in args.tables:
        if table == 'ground-extract':
            extract_ground_events()
        else:
            TABLE_BUILDERS[table](output_dir)


if __name__ == '__main__':
    main()
