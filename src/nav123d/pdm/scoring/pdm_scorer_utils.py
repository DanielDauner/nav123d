from enum import IntEnum

import numpy as np
import numpy.typing as npt
from py123d.datatypes import BoxDetectionSE2
from py123d.geometry import PoseSE2
from shapely import LineString, Polygon

# from nuplan.planning.simulation.observation.idm.utils import is_agent_behind, is_track_stopped
from nav123d.pdm.utils.pdm_constants import DYNAMIC_OBJECT_LABELS
from nav123d.pdm.utils.pdm_enums import StateIndex


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


def get_collision_type(
    state: npt.NDArray[np.float64],
    ego_polygon: Polygon,
    box_detection_se2: BoxDetectionSE2,
    box_detection_polygon: Polygon,
    stopped_speed_threshold: float = 5e-02,
) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param box_detection_se2: Box detection state.
    :param box_detection_polygon: Polygon representing the box detection.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """

    ego_speed = np.hypot(
        state[StateIndex.VELOCITY_X],
        state[StateIndex.VELOCITY_Y],
    )

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    center_point = box_detection_polygon.centroid
    tracked_object_center = PoseSE2(center_point.x, center_point.y, box_detection_se2.center_se2.yaw)

    ego_rear_axle_pose: PoseSE2 = PoseSE2.from_array(state[StateIndex.STATE_SE2])

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(box_detection_se2):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString([
        ego_polygon.exterior.coords[0],
        ego_polygon.exterior.coords[3],
    ]).intersects(box_detection_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def is_track_stopped(box_detection_se2: BoxDetectionSE2, stopped_speed_threshold: float = 5e-02) -> bool:
    """
    Evaluates if a tracked object is stopped
    :param box_detection_se2: Box detection state
    :param stopped_speed_threshold: Threshold for 0 speed due to noise
    :return: True if track is stopped else False.
    """
    is_stopped: bool = True
    if box_detection_se2.attributes.default_label in DYNAMIC_OBJECT_LABELS:
        assert box_detection_se2.velocity_2d is not None, (
            "Velocity information is required for dynamic objects to determine if they are stopped."
        )
        is_stopped = bool(box_detection_se2.velocity_2d.magnitude <= stopped_speed_threshold)
    return is_stopped


def is_agent_behind(ego_pose_se2: PoseSE2, agent_pose_se2: PoseSE2, angle_tolerance: float = 150) -> bool:
    """
    Determines if an agent is behind of the ego
    """
    return bool(get_agent_relative_angle(ego_pose_se2, agent_pose_se2) > np.deg2rad(angle_tolerance))


def is_agent_ahead(ego_pose_se2: PoseSE2, agent_pose_se2: PoseSE2, angle_tolerance: float = 30) -> bool:
    """
    Determines if an agent is ahead of the ego
    :param ego_pose_se2: ego's pose
    :param agent_pose_se2: agent's pose
    :param angle_tolerance: tolerance to consider if agent is ahead, where zero is the heading of the ego [deg]
    :return: true if agent is ahead, false otherwise.
    """
    return bool(get_agent_relative_angle(ego_pose_se2, agent_pose_se2) < np.deg2rad(angle_tolerance))


def get_agent_relative_angle(ego_pose_se2: PoseSE2, agent_pose_se2: PoseSE2) -> float:
    """
    Get the the relative angle of an agent position to the ego
    :param ego_pose_se2: pose of ego
    :param agent_pose_se2: pose of an agent
    :return: relative angle in radians.
    """
    agent_vector: npt.NDArray[np.float32] = np.array([
        agent_pose_se2.x - ego_pose_se2.x,
        agent_pose_se2.y - ego_pose_se2.y,
    ])
    ego_vector: npt.NDArray[np.float32] = np.array([np.cos(ego_pose_se2.yaw), np.sin(ego_pose_se2.yaw)])
    dot_product = np.dot(ego_vector, agent_vector / np.linalg.norm(agent_vector))
    return float(np.arccos(dot_product))
