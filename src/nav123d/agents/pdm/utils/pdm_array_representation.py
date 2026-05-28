# TODO: Move & rename this file for common usage (not specific for PDM)
from typing import List

import numpy as np
import numpy.typing as npt
import shapely
from py123d.datatypes import EgoStateSE2, EgoStateSE3Metadata, Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE2, DynamicStateSE2Index
from py123d.geometry import PoseSE2, PoseSE2Index

from nav123d.agents.pdm.utils.pdm_enums import (
    BBCoordsIndex,
    PointIndex,
    StateIndex,
)
from nav123d.agents.pdm.utils.pdm_geometry_utils import (
    get_acceleration_shifted,
    get_velocity_shifted,
    se2_array_translate_longitudinally,
    translate_lon_and_lat,
)


def array_to_states_se2(array: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    """
    Converts array representation to PoseSE2 over the last dim.
    :param array: array filled with (x,y,θ) on last dim
    :return: object array of PoseSE2, with shape array.shape[:-1]
    """
    assert array.shape[-1] == len(PoseSE2Index), "Last dimension of input array must be of size 3 (x,y,yaw)"
    leading_shape = array.shape[:-1]
    flat = array.reshape(-1, len(PoseSE2Index))
    poses = np.empty(flat.shape[0], dtype=object)
    for i in range(flat.shape[0]):
        poses[i] = PoseSE2.from_array(flat[i])
    return poses.reshape(leading_shape)


def states_se2_to_array(states_se2: List[PoseSE2]) -> npt.NDArray[np.float64]:
    """
    Converts list of PoseSE2 to array representation.
    :param states_se2: list of PoseSE2 objects
    :return: array of shape (N, 3) with (x,y,yaw) on last dim
    """
    return np.stack([pose.array for pose in states_se2])


def ego_state_to_state_array(ego_state: EgoStateSE2) -> npt.NDArray[np.float64]:
    """
    Converts an EgoStateSE2 into an array representation (drops timestamp and metadata).
    The returned array follows StateIndex layout; STEERING_RATE and ANGULAR_ACCELERATION
    are not represented in py123d's EgoStateSE2 and stay zero.
    :param ego_state: EgoStateSE2 instance
    :return: array filled with ego state values (from the rear axle)
    """
    state_array = np.zeros(len(StateIndex), dtype=np.float64)

    state_array[StateIndex.STATE_SE2] = ego_state.rear_axle_se2.array

    if ego_state.dynamic_state_se2 is not None:
        state_array[StateIndex.VELOCITY_2D] = ego_state.dynamic_state_se2.velocity_2d.array
        state_array[StateIndex.ACCELERATION_2D] = ego_state.dynamic_state_se2.acceleration_2d.array
        state_array[StateIndex.ANGULAR_VELOCITY] = ego_state.dynamic_state_se2.angular_velocity

    if ego_state.tire_steering_angle is not None:
        state_array[StateIndex.STEERING_ANGLE] = ego_state.tire_steering_angle

    return state_array


def ego_state_to_center_state_array(ego_state: EgoStateSE2) -> npt.NDArray[np.float64]:
    """
    Converts an EgoStateSE2 into an array representation referenced from the vehicle center.
    Velocity/acceleration are body-frame on DynamicStateSE2 and therefore identical to the
    rear-axle representation; only the SE2 pose differs.
    :param ego_state: EgoStateSE2 instance
    :return: array filled with ego state values (from the center)
    """
    state_array = ego_state_to_state_array(ego_state)
    state_array[StateIndex.STATE_SE2] = ego_state.center_se2.array
    return state_array


def ego_states_to_state_array(ego_states: List[EgoStateSE2]) -> npt.NDArray[np.float64]:
    """
    Converts a list of EgoStateSE2 into an array representation (rear-axle reference).
    :param ego_states: list of EgoStateSE2 instances
    :return: 2D array of shape (N, len(StateIndex))
    """
    return np.array([ego_state_to_state_array(ego_state) for ego_state in ego_states], dtype=np.float64)


def ego_states_to_center_state_array(ego_states: List[EgoStateSE2]) -> npt.NDArray[np.float64]:
    """
    Converts a list of EgoStateSE2 into an array representation (center reference).
    :param ego_states: list of EgoStateSE2 instances
    :return: 2D array of shape (N, len(StateIndex))
    """
    return np.array([ego_state_to_center_state_array(ego_state) for ego_state in ego_states], dtype=np.float64)


def state_array_to_ego_state(
    state_array: npt.NDArray[np.float64],
    timestamp: Timestamp,
    metadata: EgoStateSE3Metadata,
) -> EgoStateSE2:
    """
    Converts array representation of an ego state back to an EgoStateSE2.
    Note: STEERING_RATE and ANGULAR_ACCELERATION cannot be carried into EgoStateSE2
    and are silently dropped (py123d's DynamicStateSE2 has no field for them).
    :param state_array: array representation of an ego state
    :param timestamp: timestamp of state
    :param metadata: vehicle metadata
    :return: EgoStateSE2 instance built from the rear-axle pose
    """
    dyn_array = np.zeros(len(DynamicStateSE2Index), dtype=np.float64)
    dyn_array[DynamicStateSE2Index.VELOCITY_2D] = state_array[StateIndex.VELOCITY_2D]
    dyn_array[DynamicStateSE2Index.ACCELERATION_2D] = state_array[StateIndex.ACCELERATION_2D]
    dyn_array[DynamicStateSE2Index.ANGULAR_VELOCITY] = state_array[StateIndex.ANGULAR_VELOCITY]

    return EgoStateSE2.from_rear_axle(
        rear_axle_se2=PoseSE2.from_array(state_array[StateIndex.STATE_SE2]),
        metadata=metadata,
        timestamp=timestamp,
        dynamic_state_se2=DynamicStateSE2.from_array(dyn_array, copy=False),
        tire_steering_angle=float(state_array[StateIndex.STEERING_ANGLE]),
    )


def state_array_to_ego_states(
    state_array: npt.NDArray[np.float64],
    timestamps: List[Timestamp],
    metadata: EgoStateSE3Metadata,
) -> List[EgoStateSE2]:
    """
    Converts array representation of ego states back to a list of EgoStateSE2.
    :param state_array: array representation of ego states
    :param timestamps: list of timestamps, one per state
    :param metadata: vehicle metadata
    :return: list of EgoStateSE2 instances
    """
    ego_states: List[EgoStateSE2] = []
    for i, timestamp in enumerate(timestamps):
        state = state_array[i] if i < len(state_array) else state_array[-1]
        ego_states.append(state_array_to_ego_state(state, timestamp, metadata))
    return ego_states


def state_array_to_coords_array(
    states: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.float64]:
    """
    Converts multi-dim array representation of ego states to bounding box coordinates.
    :param states: array representation of ego states (n_batch, n_time, len(StateIndex))
    :param metadata: vehicle metadata
    :return: multi-dim array of bounding box coordinates
    """
    n_batch, n_time, _ = states.shape

    half_length = metadata.half_length
    half_width = metadata.half_width
    rear_axle_to_center = metadata.rear_axle_to_center_longitudinal

    headings = states[..., StateIndex.HEADING]
    cos, sin = np.cos(headings), np.sin(headings)

    # calculate ego center from rear axle
    rear_axle_to_center_translate = np.stack([rear_axle_to_center * cos, rear_axle_to_center * sin], axis=-1)

    ego_centers: npt.NDArray[np.float64] = states[..., StateIndex.POINT] + rear_axle_to_center_translate

    coords_array: npt.NDArray[np.float64] = np.zeros((n_batch, n_time, len(BBCoordsIndex), 2), dtype=np.float64)

    coords_array[:, :, BBCoordsIndex.CENTER] = ego_centers
    coords_array[:, :, BBCoordsIndex.FRONT_LEFT] = translate_lon_and_lat(ego_centers, headings, half_length, half_width)
    coords_array[:, :, BBCoordsIndex.FRONT_RIGHT] = translate_lon_and_lat(
        ego_centers, headings, half_length, -half_width
    )
    coords_array[:, :, BBCoordsIndex.REAR_LEFT] = translate_lon_and_lat(ego_centers, headings, -half_length, half_width)
    coords_array[:, :, BBCoordsIndex.REAR_RIGHT] = translate_lon_and_lat(
        ego_centers, headings, -half_length, -half_width
    )

    return coords_array


def coords_array_to_polygon_array(
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.object_]:
    """
    Converts multi-dim array of bounding box coords to shapely polygons.
    :param coords: bounding box coords (including corners and center)
    :return: array of shapely polygons
    """
    # create coords copy and use center point for closed exterior
    coords_exterior: npt.NDArray[np.float64] = coords.copy()
    coords_exterior[..., BBCoordsIndex.CENTER, :] = coords_exterior[..., BBCoordsIndex.FRONT_LEFT, :]

    # load new coordinates into polygon array
    polygons = shapely.creation.polygons(coords_exterior)

    return polygons  # type: ignore


def state_array_to_center_state_array(
    state_array: npt.NDArray[np.float64], metadata: EgoStateSE3Metadata
) -> npt.NDArray[np.float64]:
    """
    Converts a rear-axle-referenced state array to a center-referenced one.
    :param state_array: array representation of ego states (..., len(StateIndex))
    :param metadata: vehicle metadata
    :return: center-referenced state array, same shape as input
    """
    assert state_array.shape[-1] == len(StateIndex)

    center_states = np.zeros(state_array.shape, dtype=np.float64)

    # coordinates
    center_states[..., StateIndex.STATE_SE2] = se2_array_translate_longitudinally(
        state_array[..., StateIndex.STATE_SE2], metadata.rear_axle_to_center_longitudinal
    )

    # velocity & acceleration
    displacement = np.zeros((1, 2), dtype=np.float64)
    displacement[..., PointIndex.X] = metadata.rear_axle_to_center_longitudinal

    center_states[..., StateIndex.VELOCITY_2D] = get_velocity_shifted(
        displacement,
        state_array[..., StateIndex.VELOCITY_2D],
        state_array[..., StateIndex.ANGULAR_VELOCITY],
    )
    center_states[..., StateIndex.ACCELERATION_2D] = get_acceleration_shifted(
        displacement,
        state_array[..., StateIndex.ACCELERATION_2D],
        state_array[..., StateIndex.ANGULAR_VELOCITY],
        state_array[..., StateIndex.ANGULAR_ACCELERATION],
    )

    # rest is copied
    center_states[..., StateIndex.STEERING_ANGLE] = state_array[..., StateIndex.STEERING_ANGLE]
    center_states[..., StateIndex.STEERING_RATE] = state_array[..., StateIndex.STEERING_RATE]
    center_states[..., StateIndex.ANGULAR_VELOCITY] = state_array[..., StateIndex.ANGULAR_VELOCITY]
    center_states[..., StateIndex.ANGULAR_ACCELERATION] = state_array[..., StateIndex.ANGULAR_ACCELERATION]

    return center_states
