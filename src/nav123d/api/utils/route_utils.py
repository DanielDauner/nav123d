from typing import List, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
from py123d.api import MapAPI, SceneAPI
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.datatypes import LaneGroup, MapLayer
from py123d.geometry import Point2D, PolylineSE2

from nav123d.datatypes.driving_command import DrivingCommand

_CANDIDATE_FALLBACK_RADII: List[float] = [5.0, 20.0, 50.0]
_USE_NUPLAN_DEFAULT_ROUTE: bool = True
_TARGET_ROUTE_LOOKAHEAD_TIME_S: float = 60.0

# Driving-command thresholds (yaw-free, route-geometry based).
_COMMAND_HEADING_THRESHOLD_RAD: float = float(np.deg2rad(20.0))
_COMMAND_TANGENT_LOOKAHEAD_M: float = 30.0
_COMMAND_INTERSECTION_HORIZON_M: float = 50.0
_COMMAND_INTERSECTION_PAD_M: float = 5.0


def get_driving_command_heuristic_from_api(scene_api: SceneAPI) -> DrivingCommand:
    """High-level discrete driving command derived from the route geometry ahead.

    Yaw-free and ground-truth-free for the agent: only uses the current ego (x, y), the map,
    and :func:`get_route_lane_group_ids_from_api`. Strategy (option 4):
        1. If the route enters an intersection within ``_COMMAND_INTERSECTION_HORIZON_M`` of the
           current arc-length, compute the heading change across that intersection.
        2. Otherwise, compute the tangent delta over a fixed lookahead window.
    """
    from nav123d.api.base_agent_api import AgentAPI

    if isinstance(scene_api, AgentAPI):
        scene_api_ = _get_scene_api_from_agent_api(scene_api)
    else:
        scene_api_ = scene_api

    initial_ego_state_se3 = scene_api_.get_ego_state_se3_at_iteration(0)
    map_api = scene_api_.get_map_api()
    if initial_ego_state_se3 is None or map_api is None:
        return DrivingCommand.UNKNOWN

    route_lane_group_ids = get_route_lane_group_ids_from_api(scene_api_)
    if not route_lane_group_ids:
        return DrivingCommand.UNKNOWN

    ego_point_2d = initial_ego_state_se3.center_2d
    route_lane_groups, route_polyline_se2, route_arc_boundaries = _stitch_route_polyline(
        route_lane_group_ids, map_api, ego_point_2d
    )
    if route_polyline_se2 is None or route_polyline_se2.length < 1.0:
        return DrivingCommand.UNKNOWN

    s0 = float(route_polyline_se2.project(ego_point_2d))

    delta_yaw_rad = _next_intersection_heading_delta(route_lane_groups, route_arc_boundaries, route_polyline_se2, s0)
    if delta_yaw_rad is None:
        delta_yaw_rad = _tangent_heading_delta(route_polyline_se2, s0, _COMMAND_TANGENT_LOOKAHEAD_M)
    if delta_yaw_rad is None:
        return DrivingCommand.UNKNOWN

    if delta_yaw_rad > _COMMAND_HEADING_THRESHOLD_RAD:
        return DrivingCommand.LEFT
    if delta_yaw_rad < -_COMMAND_HEADING_THRESHOLD_RAD:
        return DrivingCommand.RIGHT
    return DrivingCommand.STRAIGHT


def _stitch_route_polyline(
    route_lane_group_ids: List[int],
    map_api: MapAPI,
    anchor_point_2d: Point2D,
) -> tuple:
    """Concatenate per-lane-group centerlines into one :class:`PolylineSE2` along the route.

    For each lane group, picks the lane whose centerline start is closest to the previous
    segment's end (or to ``anchor_point_2d`` for the first lane group). Duplicate join points
    are dropped so arc-length is monotonic across segments.

    :return: ``(lane_groups, polyline_se2, arc_boundaries)`` where ``arc_boundaries[i]`` is the
        arc-length of the start of lane group ``i`` on the stitched polyline (length is
        ``len(lane_groups) + 1``). Returns ``([], None, [])`` if no usable centerline was found.
    """
    arrays: List[npt.NDArray[np.float64]] = []
    lane_groups: List[LaneGroup] = []
    arc_boundaries: List[float] = [0.0]
    prev_end = np.array([anchor_point_2d.x, anchor_point_2d.y], dtype=np.float64)
    cumulative_length = 0.0

    for lg_id in route_lane_group_ids:
        lane_group = map_api.get_map_object_in_layer(lg_id, MapLayer.LANE_GROUP)
        if not isinstance(lane_group, LaneGroup) or not lane_group.lanes:
            continue

        best_lane = min(
            lane_group.lanes,
            key=lambda lane: float(np.linalg.norm(lane.centerline_2d.array[0] - prev_end)),
        )
        centerline_array = best_lane.centerline_2d.array
        if centerline_array.shape[0] < 2:
            continue

        if arrays:
            centerline_array = centerline_array[1:]  # drop join duplicate
        arrays.append(centerline_array)
        lane_groups.append(lane_group)
        cumulative_length += float(best_lane.centerline_2d.length)
        arc_boundaries.append(cumulative_length)
        prev_end = best_lane.centerline_2d.array[-1]

    if not arrays:
        return [], None, []

    stitched_xy = np.vstack(arrays)
    if stitched_xy.shape[0] < 2:
        return [], None, []
    return lane_groups, PolylineSE2.from_array(stitched_xy), arc_boundaries


def _tangent_heading_delta(route_polyline_se2: PolylineSE2, s_start: float, lookahead_m: float) -> Optional[float]:
    """Wrapped (start, start+lookahead) yaw delta along the route polyline."""
    s_end = min(s_start + lookahead_m, route_polyline_se2.length)
    if s_end - s_start < 1.0:
        return None
    pose_start = route_polyline_se2.interpolate(float(s_start))
    pose_end = route_polyline_se2.interpolate(float(s_end))
    return _wrap_to_pi(pose_end.yaw - pose_start.yaw)  # type: ignore[union-attr]


def _next_intersection_heading_delta(
    route_lane_groups: List[LaneGroup], route_arc_boundaries: List[float], route_polyline_se2: PolylineSE2, s0: float
) -> Optional[float]:
    """Heading change across the first intersection encountered along the route within horizon.

    Returns ``None`` if no intersection-bearing lane group sits in
    ``[s0, s0 + _COMMAND_INTERSECTION_HORIZON_M]`` on the route, or if the route ends inside
    the intersection (no post-intersection lane group to compare against).
    """
    entry_idx: Optional[int] = None
    for i, lane_group in enumerate(route_lane_groups):
        lg_start_s = route_arc_boundaries[i]
        if lg_start_s < s0:
            continue
        if lg_start_s > s0 + _COMMAND_INTERSECTION_HORIZON_M:
            break
        if lane_group.intersection_id is not None:
            entry_idx = i
            break
    if entry_idx is None:
        return None

    intersection_id = route_lane_groups[entry_idx].intersection_id
    exit_idx = entry_idx + 1
    while exit_idx < len(route_lane_groups) and route_lane_groups[exit_idx].intersection_id == intersection_id:
        exit_idx += 1
    if exit_idx >= len(route_lane_groups):
        return None  # route ends inside the intersection

    entry_s = max(route_arc_boundaries[entry_idx] - _COMMAND_INTERSECTION_PAD_M, 0.0)
    exit_s = min(route_arc_boundaries[exit_idx] + _COMMAND_INTERSECTION_PAD_M, route_polyline_se2.length)
    if exit_s - entry_s < 1.0:
        return None

    pose_entry = route_polyline_se2.interpolate(float(entry_s))
    pose_exit = route_polyline_se2.interpolate(float(exit_s))
    return _wrap_to_pi(pose_exit.yaw - pose_entry.yaw)  # type: ignore[union-attr]


def _wrap_to_pi(angle_rad: float) -> float:
    return float(((angle_rad + np.pi) % (2.0 * np.pi)) - np.pi)


def get_route_lane_group_ids_from_api(scene_api: SceneAPI) -> List[int]:
    """Returns the on-route lane group ids for the scene.

    Uses nuPlan's logged route roadblocks when available, otherwise infers the route via
    :func:`_infer_route_lane_group_ids`.

    :param scene_api: scene interface providing map and ego state access
    :return: ordered on-route lane group ids (empty if none can be determined)
    """
    from nav123d.api.base_agent_api import AgentAPI

    if isinstance(scene_api, AgentAPI):
        scene_api_ = _get_scene_api_from_agent_api(scene_api)
    else:
        scene_api_ = scene_api

    route_lane_group_ids: List[int] = []
    if scene_api_.get_map_metadata() is not None:
        dataset_name = scene_api_.get_log_metadata().dataset
        if "nuplan" in dataset_name and _USE_NUPLAN_DEFAULT_ROUTE:
            if "scenario" in scene_api_.get_all_custom_modality_metadatas().keys():
                modality = scene_api_.get_custom_modality_at_iteration(0, "scenario")
                assert modality is not None
                route_lane_group_ids = [int(id_) for id_ in modality.data["route_roadblock_ids"]]

        if len(route_lane_group_ids) == 0:
            route_lane_group_ids = _infer_route_lane_group_ids(scene_api_)
    return route_lane_group_ids


def _infer_route_lane_group_ids(scene_api: SceneAPI) -> List[int]:
    """Infers the route lane group ids via shortest-path search on the lane-group graph.

    Looks up the ego position ``_TARGET_ROUTE_LOOKAHEAD_TIME_S`` ahead (via an oracle), then finds
    the shortest lane-group path from the current position's candidates to the look-ahead candidates.

    :param scene_api: Arrow-backed scene interface
    :return: lane group ids along the inferred route (empty if no path is found)
    """

    initial_ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state modality not found at iteration 0."

    target_ts_us = initial_ego_state_se3.timestamp.time_us + int(_TARGET_ROUTE_LOOKAHEAD_TIME_S * 1e6)
    end_ego_state_se3 = scene_api.get_ego_state_se3_at_timestamp(target_ts_us, criteria="nearest")
    map_api = scene_api.get_map_api()
    assert end_ego_state_se3 is not None, (
        f"Ego state modality not found at iteration {scene_api.number_of_iterations - 1}."
    )
    assert map_api is not None, "Map API not found in Agent API."

    # Query nearest lane groups to initial and end ego states, several candidates for start and end lane groups.
    start_candidates = _query_lane_group_candidates(map_api, initial_ego_state_se3.center_2d)
    end_candidates = _query_lane_group_candidates(map_api, end_ego_state_se3.center_2d)
    if not start_candidates or not end_candidates:
        return []

    # Use lane group digraph to find optional lane group sequence from start to end candidates.
    lane_group_digraph = map_api.get_layer_graph(layer="lane_group")

    best_path: List[int] = []
    for start_id in start_candidates:
        if start_id not in lane_group_digraph:
            continue
        for end_id in end_candidates:
            if end_id not in lane_group_digraph:
                continue
            try:
                path = nx.shortest_path(lane_group_digraph, source=start_id, target=end_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not best_path or len(path) < len(best_path):
                best_path = [int(node) for node in path]

    return best_path


def _query_lane_group_candidates(map_api: MapAPI, point_2d: Point2D) -> List[int]:
    """Return candidate lane-group IDs at ``point_2d``.

    Containment first (cheapest correct query when ego is on a lane group), then a
    radius fallback for ego poses slightly off any polygon (e.g. parked at a curb).
    """
    contained = map_api.query_object_ids(
        geometry=point_2d.shapely_point,
        layers=[MapLayer.LANE_GROUP],
        predicate="intersects",
    )
    candidate_ids = contained.get(MapLayer.LANE_GROUP, [])
    if candidate_ids:
        return [int(id_) for id_ in candidate_ids]  # type: ignore[union-attr]

    for radius in _CANDIDATE_FALLBACK_RADII:
        nearby = map_api.get_map_objects_in_radius(
            point=point_2d,
            radius=radius,
            layers=[MapLayer.LANE_GROUP],
        )
        nearby_objects = nearby.get(MapLayer.LANE_GROUP, [])
        if nearby_objects:
            return [int(obj.object_id) for obj in nearby_objects]

    return []


def _get_scene_api_from_agent_api(agent_api) -> SceneAPI:
    """Extracts the scene API from an agent API, if possible."""
    # Local import to avoid the route_utils → arrow_agent_api → base_agent_api → route_utils cycle.
    from nav123d.api.arrow_agent_api import ArrowAgentSceneAPI

    # NOTE: This is a hacky way of getting access ground-truth ego position at look ahead time.
    # This implementation shouldn't be the end-results but is currently practical for inferring route info.
    assert isinstance(agent_api, (ArrowSceneAPI, ArrowAgentSceneAPI)), "Expected scene_api to be of Arrow-backed types."
    return ArrowSceneAPI(agent_api._log_dir, agent_api._scene_metadata)
