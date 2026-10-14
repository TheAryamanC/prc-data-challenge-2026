"""
The models need one big table where every row is one departing flight

This is how it works:
-combine the seven tables of the older model
-look at how arriving flights of the same airline, airport and stand had their parking (block) times written down
-read the free X-Plane airport maps (stands, taxiways and runway ends) and find each stand on the map, then measure how far each stand is from each runway end, both in a straight line and by driving along the taxiways
"""
from __future__ import annotations

import argparse
import heapq
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taxi.main import AIRPORT_REFERENCE_POINTS, DOWNLOADS, FEATURE_MATRIX, PRIOR_TABLES, competition_files

# High-cardinality identifiers of the prior's tables that are not model features.
IDENTIFIER_COLUMNS = ['FLIGHT_mvt', 'CALLSIGN_flt', 'opdi_icao24', 'linked_landing', 'linked_aobt']
# The prior's tables in the order of their columns in base_features.parquet (the order is part of every model's inputs).
PRIOR_TABLE_NAMES = ['row_features', 'weather_features', 'clock_features', 'opdi_features', 'operational_features', 'ground_event_features', 'linked_arrivals']

ARRIVAL_PROXY_COLUMNS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']
SCHEDULE_COPY_SECONDS = 6          # an arrival whose block time is within 6 s of its schedule is a schedule copy (regime P)

# Row codes of an X-Plane apt.dat airport file (one record per line, the code first)
RUNWAY_ROW = '100'                 # a land runway: width, surface, ... then both ends (designator, latitude, longitude, ...)
TAXI_NODE_ROW = '1201'             # a node of the taxi-route network: latitude, longitude, usage, node id, name
TAXI_EDGE_ROW = '1202'             # an edge of the taxi-route network: node id, node id, direction, type (runway / taxiway_X)
STAND_ROW = '1300'                 # a start-up location (stand / gate): latitude, longitude, heading, type, aircraft types, name

# Arrival proxies
PREVIOUS_ARRIVAL_MAX_AGE = '6h'   # an arrival at the same stand up to 6 h before counts as this aircraft's inbound
OPERATOR_WINDOW_DAYS = 7           # one week centred on each day, so every weekday of the timetable counts once


def aligned_prior_table(table_name: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    This function combines one of the old model's tables and lines its rows up with the master list of departures
    """
    table = pd.read_parquet(PRIOR_TABLES / f'{table_name}.parquet')
    assert len(table) == len(metadata), table_name
    if 'MVT_ID_mvt' in table:
        table = table.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(metadata.MVT_ID_mvt).reset_index(drop=True)
    return table.drop(columns=[column for column in IDENTIFIER_COLUMNS if column in table])


def convert_to_model_types(base_features: pd.DataFrame) -> None:
    """
    This function gives every column the type the models expect
    """
    for column in base_features:
        if base_features[column].dtype == object or pd.api.types.is_string_dtype(base_features[column].dtype):
            base_features[column] = base_features[column].astype('category')
        elif base_features[column].dtype != 'category':
            base_features[column] = base_features[column].astype('float32')


def build_base_features(output_directory: Path) -> None:
    """
    This function combines all seven old model tables
    """
    metadata = pd.read_parquet(PRIOR_TABLES / 'metadata.parquet')
    parts = [aligned_prior_table(table_name, metadata) for table_name in PRIOR_TABLE_NAMES]
    base_features = pd.concat(parts, axis=1)
    base_features = base_features.loc[:, ~base_features.columns.duplicated()]
    convert_to_model_types(base_features)
    base_features.to_parquet(output_directory / 'base_features.parquet', index=False)
    metadata.to_parquet(output_directory / 'departure_metadata.parquet', index=False)
    print('base_features', base_features.shape, base_features.dtypes.value_counts().to_dict())


# arrival proxies
def arrival_rows(all_movements: pd.DataFrame) -> pd.DataFrame:
    """
    This function picks out the arriving flights and marks each one as a schedule copy when its recorded block time is within 6 seconds of the scheduled time
    """
    arrivals = all_movements[all_movements.PHASE_mvt.eq('ARR')].copy()
    arrivals['airport'] = arrivals.ADES_mvt
    block_minus_schedule = (arrivals.BLOCK_TIME_UTC_mvt - arrivals.SCHED_TIME_UTC_mvt).dt.total_seconds()
    arrivals['is_copy'] = (block_minus_schedule.abs() <= SCHEDULE_COPY_SECONDS).astype(float)
    arrivals['delay'] = block_minus_schedule
    arrivals['day'] = arrivals.MVT_TIME_UTC_mvt.dt.floor('D')
    return arrivals


def departure_rows(all_movements: pd.DataFrame) -> pd.DataFrame:
    """
    This function picks out the departing flights and adds their departure airport and the UTC day of the take-off
    """
    departures = all_movements[all_movements.PHASE_mvt.eq('DEP')].copy()
    departures['airport'] = departures.ADEP_mvt
    departures['day'] = departures.MVT_TIME_UTC_mvt.dt.floor('D')
    return departures


def previous_arrival_at_stand(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """
    This function finds, for each departure, the last arrival that parked at the same airport and stand at most 6 hours before the departure's scheduled time - often it is the same plane that is about to leave
    """
    departure_keys = departures[['MVT_ID_mvt', 'airport', 'STAND_mvt', 'SCHED_TIME_UTC_mvt']].dropna()
    departure_keys = departure_keys.rename(columns={'SCHED_TIME_UTC_mvt': 'time'})
    arrival_keys = arrivals[['airport', 'STAND_mvt', 'BLOCK_TIME_UTC_mvt', 'is_copy', 'delay']].dropna()
    arrival_keys = arrival_keys.rename(columns={'BLOCK_TIME_UTC_mvt': 'time'})
    departure_keys['time'] = departure_keys.time.dt.as_unit('ns')
    arrival_keys['time'] = arrival_keys.time.dt.as_unit('ns')
    arrival_keys['arrival_block'] = arrival_keys.time
    joined = pd.merge_asof(departure_keys.sort_values('time'), arrival_keys.sort_values('time'), on='time', by=['airport', 'STAND_mvt'], direction='backward', tolerance=pd.Timedelta(PREVIOUS_ARRIVAL_MAX_AGE))
    joined['prev_arr_age'] = (joined.time - joined.arrival_block).dt.total_seconds()
    return joined[['MVT_ID_mvt', 'is_copy', 'delay', 'prev_arr_age']].rename(columns={'is_copy': 'prev_arr_P', 'delay': 'prev_arr_delay'})


def operator_week_copy_counts(operator_day: pd.DataFrame) -> pd.DataFrame:
    """
    This function counts, for every airport and airline, how many arrivals there were and how many of them were schedule copies over the 7 days centered on each day
    """
    operator_week_parts: list[pd.DataFrame] = []
    for (airport, operator), operator_days in operator_day.groupby(['airport', 'operator']):
        daily = operator_days.set_index('day')[['sum', 'count']].asfreq('D', fill_value=0)
        rolling_sums = daily.rolling(OPERATOR_WINDOW_DAYS, center=True, min_periods=1).sum()
        rolling_sums['airport'] = airport
        rolling_sums['operator'] = operator
        operator_week_parts.append(rolling_sums.reset_index())
    return pd.concat(operator_week_parts).rename(columns={'sum': 'wsum', 'count': 'wcount'})


def arrival_copy_rates(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """
    This function works out, for each departure, how often nearby arrivals were schedule copies
    """
    operator_day = arrivals.groupby(['airport', 'operator', 'day']).is_copy.agg(['sum', 'count']).reset_index()
    rates = departures[['MVT_ID_mvt', 'airport', 'operator', 'day']].merge(operator_day, on=['airport', 'operator', 'day'], how='left')
    rates['op_day_arrP'] = rates['sum'] / rates['count']
    rates['op_day_n'] = rates['count'].fillna(0)
    operator_week = operator_week_copy_counts(operator_day)
    rates = rates.merge(operator_week, on=['airport', 'operator', 'day'], how='left')
    rates['op_wk_arrP'] = rates.wsum / rates.wcount
    rates['op_wk_n'] = rates.wcount.fillna(0)
    airport_day = arrivals.groupby(['airport', 'day']).is_copy.mean().rename('ap_day_arrP').reset_index()
    return rates.merge(airport_day, on=['airport', 'day'], how='left')


def build_arrival_proxy_features(output_directory: Path) -> None:
    """
    The idea of this function is that if an airline's arrivals were written down as schedule copies, its departures probably were too
    """
    all_movements = pd.concat([pd.read_parquet(path, columns=ARRIVAL_PROXY_COLUMNS) for path in competition_files()], ignore_index=True)
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['MVT_ID_mvt'])
    all_movements['operator'] = all_movements.FLIGHT_mvt.fillna('???').str[:3]
    arrivals = arrival_rows(all_movements)
    departures = departure_rows(all_movements)
    previous_arrival = previous_arrival_at_stand(departures, arrivals)
    rates = arrival_copy_rates(departures, arrivals)
    rate_columns = ['MVT_ID_mvt', 'op_day_arrP', 'op_day_n', 'op_wk_arrP', 'op_wk_n', 'ap_day_arrP']
    features = rates[rate_columns].merge(previous_arrival, on='MVT_ID_mvt', how='left')
    features = features.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(metadata.MVT_ID_mvt).reset_index()
    features.to_parquet(output_directory / 'arrival_proxy_features.parquet', index=False)
    print('arrival_proxy_features', features.shape)


# X-Plane airport geometry
def airport_local_xy(latitude: np.ndarray | list[float] | float, longitude: np.ndarray | list[float] | float, airport: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This function turns latitude/longitude positions into meters east and north of the airport's reference point - flat earth approximation
    """
    reference_latitude, reference_longitude = AIRPORT_REFERENCE_POINTS[airport]
    # the usual flat-earth factors: about 111,320 m per degree of longitude at the equator (shrinking with cos(latitude)) and about 110,540 m per degree of latitude
    east = (np.asarray(longitude) - reference_longitude) * 111320.0 * np.cos(np.deg2rad(reference_latitude))
    north = (np.asarray(latitude) - reference_latitude) * 110540.0
    return east, north


def stand_name_keys(name: str) -> list[str]:
    """
    This function makes two tidy spellings of a stand name
    """
    key = name.upper()
    key = re.sub(r'-?\([A-Z]\)$', '', key)                                        # trailing '(A)' / '-(A)'
    key = re.sub(r'^(STANDS?|GATE|RAMP|PARKING|POSITION|POS|APRON)\s+', '', key)
    key = key.replace(' ', '').replace('-', '')
    key_without_zeros = re.sub(r'(?<![0-9])0+(?=[0-9])', '', key)                 # drop leading zeros: H01 -> H1
    return [key, key_without_zeros]


XPlaneAirport = tuple[list[tuple[float, float, str]], dict[int, tuple[float, float]], list[tuple[int, int]], dict[str, tuple[float, float]]]


def parse_xplane_airport(airport: str) -> XPlaneAirport:
    """
    This function reads one airport's X-Plane map file and pulls out the named stands, the taxiway points, the taxiway roads between them (leaving out roads that are part of a runway) and both ends of every runway
    """
    stands: list[tuple[float, float, str]] = []
    nodes: dict[int, tuple[float, float]] = {}
    edges: list[tuple[int, int]] = []
    runway_ends: dict[str, tuple[float, float]] = {}
    with open(DOWNLOADS / f'xplane/{airport}.dat', errors='ignore') as apt_file:
        for line in apt_file:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == STAND_ROW and len(fields) >= 7:
                stands.append((float(fields[1]), float(fields[2]), ' '.join(fields[6:])))
            elif fields[0] == TAXI_NODE_ROW:
                nodes[int(fields[4])] = (float(fields[1]), float(fields[2]))
            elif fields[0] == TAXI_EDGE_ROW:
                if len(fields) >= 5 and fields[4] == 'runway':
                    continue
                edges.append((int(fields[1]), int(fields[2])))
            elif fields[0] == RUNWAY_ROW:
                # a runway row holds both ends: designator at field 8 / 17, followed by the threshold latitude and longitude
                for designator_field in (8, 17):
                    runway_ends[fields[designator_field]] = (float(fields[designator_field + 1]), float(fields[designator_field + 2]))
    return stands, nodes, edges, runway_ends


def shortest_path_lengths(adjacency: dict[int, list[tuple[int, float]]], source: int) -> dict[int, float]:
    """
    This function finds the shortest driving distance from one point to every point it can reach on the taxiway network using Dijkstra's algorithm
    """
    distances = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, 1e18):
            continue # shorter path found
        for neighbour, edge_length in adjacency[node]:
            new_distance = distance + edge_length
            if new_distance < distances.get(neighbour, 1e18):
                distances[neighbour] = new_distance
                heapq.heappush(queue, (new_distance, neighbour))
    return distances


def find_stand_position(stand_code: object, stand_lookup: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """
    This function tries to find one challenge stand code on the X-Plane map, and returns its position (or None when it is not found)
    """
    keys = stand_name_keys(str(stand_code))
    for key in keys:
        if key in stand_lookup:
            return stand_lookup[key]
    without_letter = re.sub(r'[A-Z]$', '', keys[1])
    if without_letter in stand_lookup:
        return stand_lookup[without_letter]
    for key in (keys[1] + 'L', keys[1] + 'R', keys[1] + 'A'):
        if key in stand_lookup:
            return stand_lookup[key]
    return None


def match_stand_positions(stands: list[tuple[float, float, str]], stand_codes: np.ndarray) -> dict[object, tuple[float, float]]:
    """
    This function looks up every stand code of an airport on the X-Plane map and keeps the ones it finds, with their latitude/longitude
    """
    stand_lookup: dict[str, tuple[float, float]] = {}
    for latitude, longitude, stand_name in stands:
        for key in stand_name_keys(stand_name):
            stand_lookup.setdefault(key, (latitude, longitude))
    stand_positions: dict[object, tuple[float, float]] = {}
    for stand_code in stand_codes:
        position = find_stand_position(stand_code, stand_lookup)
        if position is not None:
            stand_positions[stand_code] = position
    return stand_positions


def largest_connected_component(node_ids: list[int], adjacency: dict[int, list[tuple[int, float]]]) -> set[int]:
    """
    This function finds the biggest piece of the taxiway network in which every point can reach every other point
    """
    visited: set[int] = set()
    largest_component: set[int] = set()
    for start_node in node_ids:
        if start_node in visited or not adjacency[start_node]:
            continue
        stack = [start_node]
        component = {start_node}
        while stack:
            node = stack.pop()
            for neighbour, _ in adjacency[node]:
                if neighbour not in component:
                    component.add(neighbour)
                    stack.append(neighbour)
        visited |= component
        if len(component) > len(largest_component):
            largest_component = component
    return largest_component


@dataclass
class TaxiGraph:
    """
    This data structure holds one airport's taxiway network
    """
    node_ids: list[int]
    node_xy: np.ndarray                                  # (nodes, 2): east, north of each node (m)
    node_position: dict[int, np.ndarray]                 # node id -> its row of node_xy
    adjacency: dict[int, list[tuple[int, float]]]        # node id -> (neighbour id, edge length in m), both directions
    largest_component: set[int]
    snap_node_ids: list[int]                             # the nodes of the largest component, the only ones positions snap to
    snap_node_xy: np.ndarray

    def nearest_node(self, position: tuple[float, float] | np.ndarray) -> int:
        """
        This function finds the point of the biggest connected piece of the taxiway network that is closest to a given position, in a straight line
        """
        distances = np.hypot(self.snap_node_xy[:, 0] - position[0], self.snap_node_xy[:, 1] - position[1])
        return self.snap_node_ids[int(np.argmin(distances))]


def build_taxi_graph(airport: str, nodes: dict[int, tuple[float, float]], edges: list[tuple[int, int]]) -> TaxiGraph:
    """
    This function turns the X-Plane taxiway points and roads into a TaxiGraph
    """
    node_ids = list(nodes)
    latitudes = [nodes[node][0] for node in node_ids]
    longitudes = [nodes[node][1] for node in node_ids]
    node_xy = np.c_[airport_local_xy(latitudes, longitudes, airport)]
    node_position = {node: node_xy[row] for row, node in enumerate(node_ids)}
    adjacency: dict[int, list[tuple[int, float]]] = {node: [] for node in node_ids}
    for node_a, node_b in edges:
        if node_a in node_position and node_b in node_position:
            edge_length = float(np.hypot(*(node_position[node_a] - node_position[node_b])))
            adjacency[node_a].append((node_b, edge_length))
            adjacency[node_b].append((node_a, edge_length))
    # snap only to nodes of the largest connected taxiway component
    largest_component = largest_connected_component(node_ids, adjacency)
    in_largest = np.array([node in largest_component for node in node_ids])
    snap_node_ids = [node for node, keep in zip(node_ids, in_largest) if keep]
    return TaxiGraph(node_ids, node_xy, node_position, adjacency, largest_component, snap_node_ids, node_xy[in_largest])


def airport_stand_geometry(airport: str, all_movements: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    This function matches the airport's stands to the X-Plane map, builds the taxiway network and measures the distance from every matched stand to every runway end
    For each stand and runway end it records the straight-line distance and the driving distance along the taxiways (including the short hop from the stand to its nearest taxiway point, and blank when the stand cannot be reached)
    """
    stands, nodes, edges, runway_ends = parse_xplane_airport(airport)
    stand_codes = all_movements[all_movements.airport == airport].STAND_mvt.dropna().unique()
    stand_positions = match_stand_positions(stands, stand_codes)
    graph = build_taxi_graph(airport, nodes, edges)
    threshold_xy = {runway: np.array(airport_local_xy(latitude, longitude, airport)) for runway, (latitude, longitude) in runway_ends.items()}
    path_lengths_from_threshold = {runway: shortest_path_lengths(graph.adjacency, graph.nearest_node(position)) for runway, position in threshold_xy.items()}
    stand_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for stand_code, (latitude, longitude) in stand_positions.items():
        stand_x, stand_y = airport_local_xy(latitude, longitude, airport)
        stand_rows.append(dict(airport=airport, STAND_mvt=stand_code, x=float(stand_x), y=float(stand_y)))
        node = graph.nearest_node((stand_x, stand_y))
        snap_distance = float(np.hypot(*(graph.node_position[node] - np.array([stand_x, stand_y]))))
        for runway, position in threshold_xy.items():
            feature_rows.append(dict(airport=airport,
                                     STAND_mvt=stand_code,
                                     RUNWAY_mvt=runway,
                                     xp_stand_x=float(stand_x),
                                     xp_stand_y=float(stand_y),
                                     xp_line_dist=float(np.hypot(*(position - np.array([stand_x, stand_y])))),
                                     xp_taxi_path=path_lengths_from_threshold[runway].get(node, np.nan) + snap_distance,
                                     xp_snap=snap_distance))
    is_departure_here = (all_movements.airport == airport) & all_movements.PHASE_mvt.eq('DEP')
    matched_share = all_movements[is_departure_here].STAND_mvt.isin(list(stand_positions)).mean()
    print(airport, 'graph', len(graph.largest_component), 'of', len(graph.node_ids), 'stands matched rows %.3f' % matched_share)
    return stand_rows, feature_rows


def departure_geometry(all_movements: pd.DataFrame, stand_table: pd.DataFrame, stand_runway_features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    This function gives every departure the map measurements of its own stand and runway
    """
    departures = all_movements[all_movements.PHASE_mvt.eq('DEP')][['MVT_ID_mvt', 'airport', 'STAND_mvt', 'RUNWAY_mvt']]
    geometry = departures.merge(stand_runway_features, on=['airport', 'STAND_mvt', 'RUNWAY_mvt'], how='left')
    geometry = geometry.merge(stand_table.rename(columns={'x': 'xp_sx', 'y': 'xp_sy'}), on=['airport', 'STAND_mvt'], how='left')
    geometry['xp_stand_x'] = geometry.xp_stand_x.fillna(geometry.xp_sx)
    geometry['xp_stand_y'] = geometry.xp_stand_y.fillna(geometry.xp_sy)
    geometry = geometry.drop(columns=['airport', 'STAND_mvt', 'RUNWAY_mvt', 'xp_sx', 'xp_sy'])
    return geometry.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(metadata.MVT_ID_mvt).reset_index()


def build_xplane_geometry(output_directory: Path) -> None:
    """
    This function runs the map work for all ten airports
    """
    metadata = pd.read_parquet(FEATURE_MATRIX / 'departure_metadata.parquet', columns=['MVT_ID_mvt'])
    columns = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt']
    all_movements = pd.concat([pd.read_parquet(path, columns=columns) for path in competition_files()])
    # the movement's own airport: the departure airport (ADEP) of a departure, the destination (ADES) of an arrival
    all_movements['airport'] = np.where(all_movements.PHASE_mvt.eq('DEP'), all_movements.ADEP_mvt, all_movements.ADES_mvt)
    stand_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for airport in AIRPORT_REFERENCE_POINTS:
        airport_stand_rows, airport_feature_rows = airport_stand_geometry(airport, all_movements)
        stand_rows += airport_stand_rows
        feature_rows += airport_feature_rows
    stand_table = pd.DataFrame(stand_rows)
    stand_table.to_parquet(output_directory / 'xplane_stand_positions.parquet', index=False)
    geometry = departure_geometry(all_movements, stand_table, pd.DataFrame(feature_rows), metadata)
    geometry.to_parquet(output_directory / 'xplane_geometry_features.parquet', index=False)
    print('xplane_geometry_features', geometry.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the feature matrix.')
    parser.add_argument('tables', nargs='+', choices=['matrix', 'arrival-proxy', 'xplane'])
    parser.add_argument('--out-dir', default=str(FEATURE_MATRIX))
    args = parser.parse_args()
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    builders: dict[str, Callable[[Path], None]] = {'matrix': build_base_features, 
                                                   'arrival-proxy': build_arrival_proxy_features,
                                                   'xplane': build_xplane_geometry}
    for table in args.tables:
        builders[table](output_directory)


if __name__ == '__main__':
    main()
