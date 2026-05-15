from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from py123d.api import MapAPI
from py123d.datatypes import BaseMapSurfaceObject, EgoStateSE2, Lane, LaneGroup, MapLayer
from py123d.geometry import OccupancyMap2D
from shapely.geometry import Point

from nav123d.pdm.observation.pdm_occupancy_map import PDMDrivableMap
from nav123d.pdm.utils.graph_search.dijkstra import Dijkstra
from nav123d.pdm.utils.pdm_geometry_utils import normalize_angle, parallel_discrete_path
from nav123d.pdm.utils.pdm_path import PDMPath
from nav123d.pdm.utils.route_utils import route_lane_group_correction


def build_route_dicts(
    map_api: MapAPI,
    route_roadblock_ids: List[int],
) -> Tuple[Dict[int, LaneGroup], Dict[int, Lane]]:
    """
    Builds roadblock and lane dictionaries of the target route from the map-api.
    :param map_api: map interface
    :param route_roadblock_ids: ID's of on-route roadblocks
    :return: tuple of (route_roadblock_dict, route_lane_dict)
    """
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

    route_lane_group_dict: Dict[int, LaneGroup] = {}
    route_lane_dict: Dict[int, Lane] = {}

    for id_ in route_roadblock_ids:
        _lane_group = map_api.get_map_object_in_layer(id_, MapLayer.LANE_GROUP)

        if isinstance(_lane_group, LaneGroup):
            route_lane_group_dict[int(_lane_group.object_id)] = _lane_group

            for lane in _lane_group.lanes:
                route_lane_dict[int(lane.object_id)] = lane

    return route_lane_group_dict, route_lane_dict


def correct_route_lane_groups(
    ego_state_se2: EgoStateSE2,
    map_api: MapAPI,
    route_lane_group_dict: Dict[int, LaneGroup],
) -> Tuple[Dict[int, LaneGroup], Dict[int, Lane]]:
    """
    Corrects the roadblock route and rebuilds lane-graph dictionaries.
    :param ego_state: state of the ego vehicle
    :param map_api: map interface
    :param route_roadblock_dict: current roadblock dict (used for correction context)
    :return: tuple of (corrected route_roadblock_dict, corrected route_lane_dict)
    """
    corrected_lane_groups, corrected_ids = route_lane_group_correction(
        ego_pose_se2=ego_state_se2.rear_axle_se2,
        map_api=map_api,
        route_lane_group_dict=route_lane_group_dict,
    )
    return build_route_dicts(map_api, corrected_ids)


def build_drivable_area_occupancy_map(
    map_api: MapAPI,
    ego_state_se2: EgoStateSE2,
    map_radius: float = 50.0,
    layers: List[MapLayer] = [
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.CARPARK,
    ],
) -> OccupancyMap2D:
    query_dict = map_api.get_map_objects_in_radius(
        point=ego_state_se2.center_2d,
        radius=map_radius,
        layers=layers,  # type: ignore
    )
    drivable_objects_dict: Dict[str, geom.Polygon] = {}
    for map_layer in query_dict.keys():
        for map_object in query_dict[map_layer]:
            assert isinstance(map_object, BaseMapSurfaceObject), f"Expected Surface, got {type(map_object)}"
            drivable_objects_dict[f"{map_layer.serialize()}_{map_object.object_id}"] = map_object.shapely_polygon

    return OccupancyMap2D.from_dict(drivable_objects_dict)  # type: ignore


def _get_intersecting_lanes(
    ego_state: EgoState,
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    drivable_area_map: PDMDrivableMap,
) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
    """
    Returns on-route lanes and heading errors where ego-vehicle intersects.
    :param ego_state: state of ego-vehicle
    :param route_lane_dict: on-route lane dictionary
    :param drivable_area_map: drivable area occupancy map
    :return: tuple of lists with lane objects and heading errors [rad].
    """
    ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
    ego_rear_axle_point: Point = Point(*ego_position_array)
    ego_heading: float = ego_state.rear_axle.heading

    intersecting_lanes = drivable_area_map.intersects(ego_rear_axle_point)

    on_route_lanes, on_route_heading_errors = [], []
    for lane_id in intersecting_lanes:
        if lane_id in route_lane_dict.keys():
            lane_object = route_lane_dict[lane_id]
            lane_discrete_path: List[StateSE2] = lane_object.baseline_path.discrete_path
            lane_state_se2_array = np.array([state.array for state in lane_discrete_path], dtype=np.float64)
            lane_distances = (ego_position_array[None, ...] - lane_state_se2_array) ** 2
            lane_distances = lane_distances.sum(axis=-1) ** 0.5

            heading_error = lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
            heading_error = np.abs(normalize_angle(heading_error))

            on_route_lanes.append(lane_object)
            on_route_heading_errors.append(heading_error)

    return on_route_lanes, on_route_heading_errors


def _get_starting_lane(
    ego_state: EgoState,
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    drivable_area_map: PDMDrivableMap,
) -> LaneGraphEdgeMapObject:
    """
    Returns the most suitable starting lane, in ego's vicinity.
    :param ego_state: state of ego-vehicle
    :param route_lane_dict: on-route lane dictionary
    :param drivable_area_map: drivable area occupancy map
    :return: lane object (on-route)
    """
    on_route_lanes, heading_error = _get_intersecting_lanes(ego_state, route_lane_dict, drivable_area_map)

    if on_route_lanes:
        # 1. Option: find lanes from lane occupancy-map; select lane with lowest heading error
        return on_route_lanes[np.argmin(np.abs(heading_error))]

    # 2. Option: find any intersecting or close lane on-route
    starting_lane: LaneGraphEdgeMapObject = None
    closest_distance = np.inf
    for edge in route_lane_dict.values():
        if edge.contains_point(ego_state.center):
            return edge

        distance = edge.polygon.distance(ego_state.car_footprint.geometry)
        if distance < closest_distance:
            starting_lane = edge
            closest_distance = distance

    return starting_lane


def _get_discrete_centerline(
    current_lane: LaneGraphEdgeMapObject,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    search_depth: int = 30,
) -> List[StateSE2]:
    """
    Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
    :param current_lane: lane object of starting lane.
    :param route_roadblock_dict: on-route roadblock dictionary
    :param route_lane_dict: on-route lane dictionary
    :param search_depth: depth of search (for runtime), defaults to 30
    :return: list of discrete states on centerline (x,y,θ)
    """
    roadblocks = list(route_roadblock_dict.values())
    roadblock_ids = list(route_roadblock_dict.keys())

    start_idx = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
    roadblock_window = roadblocks[start_idx : start_idx + search_depth]

    graph_search = Dijkstra(current_lane, list(route_lane_dict.keys()))
    route_plan, _ = graph_search.search(roadblock_window[-1])

    centerline_discrete_path: List[StateSE2] = []
    for lane in route_plan:
        centerline_discrete_path.extend(lane.baseline_path.discrete_path)

    return centerline_discrete_path


def _get_proposal_paths(
    current_lane: LaneGraphEdgeMapObject,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    lateral_offsets: Optional[List[float]],
) -> List[PDMPath]:
    """
    Builds proposal paths: centerline at index 0, plus optional lateral offsets.
    :param current_lane: current or starting lane of path-planning
    :param route_roadblock_dict: on-route roadblock dictionary
    :param route_lane_dict: on-route lane dictionary
    :param lateral_offsets: optional centerline offsets for proposals
    :return: list of paths (index 0 is centerline)
    """
    centerline_discrete_path = _get_discrete_centerline(current_lane, route_roadblock_dict, route_lane_dict)
    centerline = PDMPath(centerline_discrete_path)

    output_paths: List[PDMPath] = [centerline]

    if lateral_offsets is not None:
        for lateral_offset in lateral_offsets:
            offset_discrete_path = parallel_discrete_path(discrete_path=centerline_discrete_path, offset=lateral_offset)
            output_paths.append(PDMPath(offset_discrete_path))

    return output_paths
