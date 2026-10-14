"""
The challenge files never say which actual airplane flew a departure
This file works out how long that airplane had been on the ground since it last landed at the airport (Rome, LIRF, by default) - this rest time is called the turnaround
It uses OPDI, EUROCONTROL's open flight lists made from the planes' ADS-B position messages

This is how it works:
-for each departure of the airport:
    -find its OPDI flight (same callsign within a -10/+40 minute window)
    -take that flight's transponder address
    -find the latest OPDI flight of the same airplane into the airport before that - the time between the two is the turnaround
    -as a backup, look up when the plane last landed in the project's own ADS-B tracks
-on the 2025 departures that have both, measure the typical difference between the OPDI and the ADS-B measures
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from taxi.main import FEATURE_MATRIX, OPDI_FOLDER, competition_files

OUTPUT_FILE = FEATURE_MATRIX / 'airframe_turnaround.parquet'
OFFSETS_FILE = FEATURE_MATRIX / 'airframe_turnaround_offsets.json'
# an OPDI flight matches a departure when it was first seen from 10 min before to 40 min after the take-off
MATCH_BEFORE = pd.Timedelta('10min')
MATCH_AFTER = pd.Timedelta('40min')
ADSB_TABLES = ('adsb_pass3', 'adsb_pass2', 'adsb_pass1')      # first table with a track wins
DEPARTURE_COLUMNS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']


# These functions cleans callsigns so that one flight is spelled the same way in every list
def normalised_callsigns(callsigns: pd.Series) -> pd.Series:
    text = callsigns.astype(object).fillna('').map(_upper_case_stripped)
    return text.map(_without_doubled_operator_letter)


def _upper_case_stripped(value: object) -> str:
    return str(value).strip().upper()


def _without_doubled_operator_letter(value: str) -> str:
    has_doubled_letter = len(value) > 4 and value[:4].isalpha() and value[2] == value[3]
    return value[:3] + value[4:] if has_doubled_letter else value


def opdi_flights(airport: str) -> pd.DataFrame:
    """
    This function reads every monthly OPDI flight list and keeps the flights that leave from or arrive at the airport, with their first and last seen times and a cleaned callsign
    """
    flight_list_columns = ['icao24', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen']
    parts = [pd.read_parquet(path, columns=flight_list_columns) for path in sorted(OPDI_FOLDER.glob('flight_list_*.parquet'))]
    flights = pd.concat(parts)
    flights = flights[(flights.adep == airport) | (flights.ades == airport)].copy()
    flights['first_seen'] = pd.to_datetime(flights.first_seen, utc=True)
    flights['last_seen'] = pd.to_datetime(flights.last_seen, utc=True)
    flights['callsign'] = normalised_callsigns(flights.flt_id)
    return flights


def match_turnaround(departures: pd.DataFrame, flights: pd.DataFrame, airport: str) -> pd.DataFrame:
    """
    This function looks at each departure and it finds the OPDI flight out of the airport with the same callsign within that window specified above and calculates the turnaround time
    """
    outbound = flights[flights.adep == airport][['callsign', 'first_seen', 'icao24']].sort_values('first_seen')
    inbound = flights[flights.ades == airport][['icao24', 'last_seen']].rename(columns={'last_seen': 'inbound_last_seen'})
    inbound = inbound.sort_values('inbound_last_seen')
    rows = departures.assign(takeoff=departures.MVT_TIME_UTC_mvt).sort_values('takeoff')
    matched = pd.merge_asof(rows, outbound, left_on='takeoff', right_on='first_seen', by='callsign', direction='nearest', tolerance=MATCH_AFTER)
    matched = matched[matched.first_seen.notna() & (matched.first_seen >= matched.takeoff - MATCH_BEFORE)]
    matched = pd.merge_asof(matched.sort_values('first_seen'), inbound, left_on='first_seen', right_on='inbound_last_seen', by='icao24', direction='backward', allow_exact_matches=False)
    matched['turn'] = (matched.first_seen - matched.inbound_last_seen).dt.total_seconds()
    matched['inbound_after_sched'] = (matched.inbound_last_seen - matched.SCHED_TIME_UTC_mvt).dt.total_seconds()
    matched_columns = matched[['MVT_ID_mvt', 'icao24', 'turn', 'inbound_after_sched']]
    result = departures[['MVT_ID_mvt']].merge(matched_columns, on='MVT_ID_mvt', how='left')
    return result


def adsb_previous_landing(movement_ids: pd.Series) -> pd.Series:
    """
    This function is the backup from the ADS-B tracks
    """
    tables = [pd.read_parquet(FEATURE_MATRIX / f'{table}.parquet', columns=['MVT_ID_mvt', 'ads_prev_fast_rel']) for table in ADSB_TABLES]
    first_table_wins = pd.concat(tables).drop_duplicates('MVT_ID_mvt')
    return first_table_wins.set_index('MVT_ID_mvt').ads_prev_fast_rel.reindex(movement_ids)


def build(airport: str = 'LIRF') -> pd.DataFrame:
    """
    This function runs everything for one airport (Rome by default as it is the hardest to match)
    """
    departures = _airport_departures(airport)
    table = match_turnaround(departures, opdi_flights(airport), airport)
    table['ads_prev_fast_rel'] = adsb_previous_landing(table.MVT_ID_mvt).to_numpy()
    seconds_after_schedule = (departures.MVT_TIME_UTC_mvt - departures.SCHED_TIME_UTC_mvt).dt.total_seconds()
    table['seconds_after_schedule'] = seconds_after_schedule.to_numpy()
    offsets = _measurement_offsets(table, departures)
    FEATURE_MATRIX.mkdir(parents=True, exist_ok=True)
    table.drop(columns='seconds_after_schedule').to_parquet(OUTPUT_FILE, index=False)
    OFFSETS_FILE.write_text(json.dumps(offsets, indent=1))
    opdi_count = int(table.turn.notna().sum())
    adsb_count = int(table.ads_prev_fast_rel.notna().sum())
    print(f'{airport} departures {len(table)}: OPDI turnaround {opdi_count}, ADS-B previous landing {adsb_count}; offsets (OPDI - ADS-B) {offsets}', flush=True)
    return table


def _airport_departures(airport: str) -> pd.DataFrame:
    """
    This function reads the airport's departures from the challenge files and gives each a cleaned callsign - the flight number is used when there is no callsign, e.g. for departures with no NM link
    """
    file_parts = [pd.read_parquet(path, columns=DEPARTURE_COLUMNS) for path in competition_files()]
    movements = pd.concat(file_parts).drop_duplicates('MVT_ID_mvt')
    departures = movements[(movements.PHASE_mvt == 'DEP') & (movements.ADEP_mvt == airport)].copy()
    callsign_or_flight_number = departures.CALLSIGN_flt.where(departures.CALLSIGN_flt.notna(), departures.FLIGHT_mvt)
    departures['callsign'] = normalised_callsigns(callsign_or_flight_number)
    return departures


def _measurement_offsets(table: pd.DataFrame, departures: pd.DataFrame) -> dict[str, float]:
    """
    OPDI's first and last seen times are not exactly the same moments as the take-off and landing in the ADS-B tracks
    This function measures the median difference between the two on the 2025 departures that have both, so that an ADS-B measure can later be turned into an OPDI-like one
    """
    in_2025 = departures.TAXITIME_SEC_mvt.notna().to_numpy()
    both = in_2025 & table.turn.notna().to_numpy() & table.ads_prev_fast_rel.notna().to_numpy()
    adsb_after_schedule = table.seconds_after_schedule + table.ads_prev_fast_rel
    return {
        'turn': float(np.median((table.turn + table.ads_prev_fast_rel)[both])),
        'inbound_after_sched': float(np.median((table.inbound_after_sched - adsb_after_schedule)[both])),
        'rows': int(both.sum())
    }


def load_turnaround_table() -> tuple[pd.DataFrame, dict[str, float]]:
    return pd.read_parquet(OUTPUT_FILE).set_index('MVT_ID_mvt'), json.loads(OFFSETS_FILE.read_text())


if __name__ == '__main__':
    build()
