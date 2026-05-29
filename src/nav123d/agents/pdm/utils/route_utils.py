from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.geometry as geom
from py123d.api import MapAPI
from py123d.datatypes import Intersection, LaneGroup, MapLayer
from py123d.geometry import OccupancyMap2D, PoseSE2, PoseSE2Index
from py123d.geometry.utils.rotation_utils import normalize_angle

from nav123d.agents.pdm.utils.graph_search.bfs_roadblock import BreadthFirstSearchLaneGroup


def get_current_lane_group_candidates(
    ego_pose_se2: PoseSE2,
    map_api: MapAPI,
    route_lane_group_dict: Dict[int, LaneGroup],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[LaneGroup, List[LaneGroup]]:
    """Finds the ego's current lane group and the set of nearby candidate lane groups.

    Prefers on-route candidates, falling back to the closest off-route candidate, and finally
    to any close lane group.

    :param ego_pose_se2: pose of the ego vehicle
    :param map_api: map interface
    :param route_lane_group_dict: on-route lane group dictionary
    :param heading_error_thresh: [rad] max heading error for a lane to count as a candidate
    :param displacement_error_thresh: [m] max displacement error for a lane to count as a candidate
    :return: tuple of (best current lane group, list of candidate lane groups)
    """
    lane_group_candidates = []

    lane_group_dict = map_api.get_map_objects_in_radius(
        point=ego_pose_se2.point_2d, radius=1.0, layers=[MapLayer.LANE_GROUP]
    )
    lane_group_candidates: List[LaneGroup] = lane_group_dict[MapLayer.LANE_GROUP]  # type: ignore

    if len(lane_group_candidates) == 0:
        # TODO: Use query nearest once implemented in py123d.
        lane_group_dict = map_api.get_map_objects_in_radius(
            point=ego_pose_se2.point_2d, radius=100.0, layers=[MapLayer.LANE_GROUP]
        )
        lane_group_candidates: List[LaneGroup] = lane_group_dict[MapLayer.LANE_GROUP]  # type: ignore

    on_route_candidates: List[LaneGroup] = []
    on_route_candidate_displacement_errors: List[float] = []
    candidates: List[LaneGroup] = []
    candidate_displacement_errors: List[float] = []

    lane_group_displacement_errors: List[float] = []
    lane_group_heading_errors: List[float] = []

    for lane_group_idx, lane_group in enumerate(lane_group_candidates):
        assert isinstance(lane_group, LaneGroup), f"Expected LaneGroup, got {type(lane_group)}"
        lane_displacement_error, lane_heading_error = np.inf, np.inf

        for lane in lane_group.lanes:
            lane_discrete_poses_se2 = lane.centerline.polyline_se2.array

            lane_state_distances = np.linalg.norm(
                lane_discrete_poses_se2[..., PoseSE2Index.XY] - ego_pose_se2.point_2d.array[None, ...], axis=-1
            )
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(lane_discrete_poses_se2[argmin, PoseSE2Index.YAW] - ego_pose_se2.yaw)
            )
            displacement_error = lane_state_distances[argmin]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if heading_error < heading_error_thresh and displacement_error < displacement_error_thresh:
                if lane_group.object_id in route_lane_group_dict.keys():
                    on_route_candidates.append(lane_group)
                    on_route_candidate_displacement_errors.append(displacement_error)
                else:
                    candidates.append(lane_group)
                    candidate_displacement_errors.append(displacement_error)

        lane_group_displacement_errors.append(lane_displacement_error)
        lane_group_heading_errors.append(float(lane_heading_error))

    if on_route_candidates:  # prefer on-route lane_groups
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)],
            on_route_candidates,
        )
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close lane_group
    return (
        lane_group_candidates[np.argmin(lane_group_displacement_errors)],
        lane_group_candidates,
    )


def _lane_group_entry_exit_yaw(lane_group: LaneGroup) -> Tuple[float, float]:
    """Returns a representative ``(entry_yaw, exit_yaw)`` [rad] for a lane group.

    Uses the lane group's first lane centerline, whose start/end heading approximate the heading of
    the (near-parallel) lanes in the group at the group's entry and exit.

    :param lane_group: lane group with at least one lane
    :return: tuple of (entry yaw, exit yaw) in radians
    """
    poses_se2 = lane_group.lanes[0].centerline.polyline_se2.array
    return float(poses_se2[0, PoseSE2Index.YAW]), float(poses_se2[-1, PoseSE2Index.YAW])


def _roll_out_forward_lane_groups(
    start_lane_group: LaneGroup,
    search_depth_forward: int = 30,
) -> Tuple[List[LaneGroup], List[int]]:
    """Greedily rolls out a 'follow-the-road' route by following the straightest successor.

    At each step the successor minimizing the heading change between the current lane group's exit
    tangent and the candidate's entry tangent is chosen. Stops at ``search_depth_forward`` steps,
    when there is no successor, or when a lane group repeats (loop guard).

    :param start_lane_group: lane group to start the rollout from
    :param search_depth_forward: max number of successor lane groups to append, defaults to 30
    :return: tuple of (route lane groups, route lane group ids), starting with ``start_lane_group``
    """
    route_lane_groups: List[LaneGroup] = [start_lane_group]
    route_lane_group_ids: List[int] = [int(start_lane_group.object_id)]
    visited = {int(start_lane_group.object_id)}

    current = start_lane_group
    for _ in range(search_depth_forward):
        successors = [successor for successor in current.successors if successor.lanes]
        if not successors:
            break

        _, current_exit_yaw = _lane_group_entry_exit_yaw(current)
        next_group = min(
            successors,
            key=lambda successor: abs(normalize_angle(_lane_group_entry_exit_yaw(successor)[0] - current_exit_yaw)),
        )
        next_id = int(next_group.object_id)
        if next_id in visited:  # loop guard
            break

        route_lane_groups.append(next_group)
        route_lane_group_ids.append(next_id)
        visited.add(next_id)
        current = next_group

    return route_lane_groups, route_lane_group_ids


def infer_follow_the_road_route(
    ego_pose_se2: PoseSE2,
    map_api: MapAPI,
    search_depth_forward: int = 30,
) -> List[int]:
    """Synthesizes a 'follow-the-road' route from the ego's current lane group.

    Fallback used when no on-route lane groups are available (e.g. there is no logged route and the
    oracle-based inference found no path). Finds the ego's current lane group purely from the map
    (no oracle / ground-truth) and greedily follows the straightest successor along the lane-group
    graph via :func:`_roll_out_forward_lane_groups`.

    :param ego_pose_se2: pose of the ego vehicle
    :param map_api: map interface
    :param search_depth_forward: max number of successor lane groups to roll out, defaults to 30
    :return: ordered lane group ids starting at ego (empty if ego is not on/near any lane group)
    """
    # Guard get_current_lane_group_candidates, which calls np.argmin over the candidate list and
    # would raise on an empty list when no lane group exists near ego (e.g. ego off-map).
    nearby = map_api.get_map_objects_in_radius(point=ego_pose_se2.point_2d, radius=100.0, layers=[MapLayer.LANE_GROUP])
    if not nearby[MapLayer.LANE_GROUP]:
        return []

    starting_group, _ = get_current_lane_group_candidates(
        ego_pose_se2=ego_pose_se2,
        map_api=map_api,
        route_lane_group_dict={},
    )
    _, route_lane_group_ids = _roll_out_forward_lane_groups(starting_group, search_depth_forward)
    return route_lane_group_ids


def route_lane_group_correction(
    ego_pose_se2: PoseSE2,
    map_api: MapAPI,
    route_lane_group_dict: Dict[int, LaneGroup],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> Tuple[List[LaneGroup], List[int]]:
    """Corrects and repairs the on-route lane groups for the current ego pose.

    Handles three cases: an off-route start (backward then forward graph search), disconnected
    consecutive lane groups (search for connecting links), and route loops.

    :param ego_pose_se2: pose of the ego vehicle
    :param map_api: map interface
    :param route_lane_group_dict: on-route lane group dictionary
    :param search_depth_backward: max BFS depth for the backward search, defaults to 15
    :param search_depth_forward: max BFS depth for the forward search, defaults to 30
    :return: tuple of (corrected lane groups, corrected lane group ids)
    """
    # TODO: Refactor code for readability

    starting_group, starting_group_candidates = get_current_lane_group_candidates(
        ego_pose_se2=ego_pose_se2,
        map_api=map_api,
        route_lane_group_dict=route_lane_group_dict,
    )
    starting_block_ids = [lane_group.object_id for lane_group in starting_group_candidates]

    route_lane_groups = list(route_lane_group_dict.values())
    route_lane_group_ids = list(route_lane_group_dict.keys())

    # When no route is provided, synthesize a follow-the-road route from the current lane group.
    # Avoids indexing the empty route below and keeps the agent driving down the road.
    if not route_lane_group_ids:
        return _roll_out_forward_lane_groups(starting_group, search_depth_forward)

    # Fix 1: when agent starts off-route
    if starting_group.object_id not in route_lane_group_ids:
        # Backward search if current lane_group not in route
        graph_search = BreadthFirstSearchLaneGroup(route_lane_group_ids[0], map_api, forward_search=False)
        path, path_id, path_found = graph_search.search(starting_block_ids, max_depth=search_depth_backward)  # type: ignore

        if path_found:
            route_lane_groups[:0] = path[:-1]
            route_lane_group_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route lane_group
            graph_search = BreadthFirstSearchLaneGroup(int(starting_group.object_id), map_api, forward_search=True)
            path, path_id, path_found = graph_search.search(route_lane_group_ids[:3], max_depth=search_depth_forward)

            if path_found:
                end_lane_group_idx = np.argmax(np.array(route_lane_group_ids) == path_id[-1])

                route_lane_groups = route_lane_groups[end_lane_group_idx + 1 :]
                route_lane_group_ids = route_lane_group_ids[end_lane_group_idx + 1 :]

                route_lane_groups[:0] = path
                route_lane_group_ids[:0] = path_id

    # Fix 2: check if lane_groups are linked, search for links if not
    lane_groups_to_append = {}
    for i in range(len(route_lane_groups) - 1):
        next_incoming_block_ids = [_lane_group.object_id for _lane_group in route_lane_groups[i + 1].predecessors]
        is_incoming = route_lane_group_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchLaneGroup(route_lane_group_ids[i], map_api, forward_search=True)
        path, path_id, path_found = graph_search.search(route_lane_group_ids[i + 1], max_depth=search_depth_forward)  # type: ignore

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            lane_groups_to_append[i] = (path, path_id)

    # append missing intermediate lane_groups
    offset = 1
    for i, (path, path_id) in lane_groups_to_append.items():
        route_lane_groups[i + offset : i + offset] = path
        route_lane_group_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    route_lane_groups, route_lane_group_ids = remove_route_loops(route_lane_groups, route_lane_group_ids)

    return route_lane_groups, route_lane_group_ids


def remove_route_loops(
    route_lane_groups: List[LaneGroup],
    route_lane_group_ids: List[int],
) -> Tuple[List[LaneGroup], List[int]]:
    """Removes the end of the route where a lane group intersects an earlier one (forming a loop).

    :param route_lane_groups: input route lane_groups
    :param route_lane_group_ids: input route lane_groups ids
    :return: tuple of ids and lane_groups of route without loops
    """

    lane_group_intersection_dict: Dict[int, geom.Polygon] = {}

    loop_idx: Optional[int] = None

    for idx, lane_group in enumerate(route_lane_groups):
        # loops only occur at intersection, thus searching for lane_group-connectors.
        intersection: Optional[Intersection] = lane_group.intersection
        if intersection is not None:
            if len(lane_group_intersection_dict) == 0:
                for intersection_lane_group in intersection.lane_groups:
                    lane_group_intersection_dict[int(intersection_lane_group.object_id)] = (
                        intersection_lane_group.shapely_polygon
                    )
                continue
            # else:
            occupancy_map = OccupancyMap2D.from_dict(lane_group_intersection_dict)  # type: ignore
            intersecting_ids = occupancy_map.intersects(lane_group.shapely_polygon)
            for intersecting_id in intersecting_ids:
                intersecting_polygon = lane_group_intersection_dict[int(intersecting_id)]
                area = intersecting_polygon.intersection(lane_group.shapely_polygon).area
                if area > 1.0:
                    loop_idx = idx
                    break

    if loop_idx:
        route_lane_groups = route_lane_groups[:loop_idx]
        route_lane_group_ids = route_lane_group_ids[:loop_idx]

    return route_lane_groups, route_lane_group_ids
