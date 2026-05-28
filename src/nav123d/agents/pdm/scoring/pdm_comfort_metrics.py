from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE3Metadata
from scipy.signal import savgol_filter

from nav123d.agents.pdm.utils.pdm_array_representation import state_array_to_center_state_array
from nav123d.agents.pdm.utils.pdm_enums import StateIndex

# TODO: Refactor & add to config

# (1) ego_jerk_metric,
MAX_ABS_MAG_JERK: float = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
MAX_ABS_LAT_ACCEL: float = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
MAX_LON_ACCEL: float = 2.40  # [m/s^2]
MIN_LON_ACCEL: float = -4.05

# (4) ego_yaw_acceleration_metric
MAX_ABS_YAW_ACCEL: float = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
MAX_ABS_LON_JERK: float = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
MAX_ABS_YAW_RATE: float = 0.95  # [rad/s]


# Extended Comfort thresholds
acceleration_threshold: float = 0.7  # [m/s^2]
jerk_threshold: float = 0.5  # [m/s^3]
yaw_rate_threshold: float = 0.1  # [rad/s]
yaw_accel_threshold: float = 0.1  # [rad/s^2]


def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    metadata: EgoStateSE3Metadata,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float64]:
    """Extract acceleration of ego pose in simulation history over batch-dim.

    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param metadata: metadata of vehicle
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 8
    :raises ValueError: when coordinate not available
    :return: array containing acceleration values
    """

    n_batch, n_time, n_states = states.shape
    if acceleration_coordinate in {"x", "y"}:
        center_states = state_array_to_center_state_array(states, metadata)
        coordinate_index = StateIndex.ACCELERATION_X if acceleration_coordinate == "x" else StateIndex.ACCELERATION_Y
        acceleration: npt.NDArray[np.float64] = center_states[..., coordinate_index]

    elif acceleration_coordinate == "magnitude":
        acceleration: npt.NDArray[np.float64] = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available. "
            f"Available options are: x, y or magnitude"
        )

    acceleration = savgol_filter(
        acceleration,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """Extract jerk of ego pose in simulation history over batch-dim.

    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :param decimals: decimal precision, defaults to 8
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing jerk values
    """
    n_batch, n_time, n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(
        states,
        acceleration_coordinate=acceleration_coordinate,
        metadata=metadata,
    )
    jerk = _approximate_derivatives(
        ego_acceleration,  # type: ignore
        time_steps_s,  # type: ignore
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    jerk = np.round(jerk, decimals=decimals)
    return jerk


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """Extract yaw-rate of simulation history over batch-dim.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param decimals: decimal precision, defaults to 8
    :param window_length: window size for extraction, defaults to 15
    :return: array containing ego's yaw rate
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,  # type: ignore
        deriv_order=deriv_order,
        poly_order=poly_order,
    )  # convert to seconds
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def _phase_unwrap(headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Phase-unwraps heading angles so successive differences stay within pi radians.

    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value.

    :param headings: An array of headings (radians)
    :return: The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """Approximates the n-th derivative of the function interpolating equally-spaced (x, y) points.

    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's. We assume the x's are increasing and equally-spaced.

    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return: Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )  # type: ignore
    return derivative


def _within_bound(
    metric: Union[npt.NDArray[np.float64], npt.NDArray[np.float32]],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """Determines whether values in the batch-dim are within bounds.

    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: array of booleans whether metric values are within bounds
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _compute_lon_acceleration(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Compute longitudinal acceleration over batch-dim of simulated proposals.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: longitudinal acceleration within bound
    """
    lon_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="x", metadata=metadata)
    return _within_bound(lon_acceleration, min_bound=MIN_LON_ACCEL, max_bound=MAX_LON_ACCEL)


def _compute_lat_acceleration(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Compute lateral acceleration over batch-dim of simulated proposals.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: lateral acceleration within bound
    """
    lat_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="y", metadata=metadata)
    return _within_bound(lat_acceleration, min_bound=-MAX_ABS_LAT_ACCEL, max_bound=MAX_ABS_LAT_ACCEL)


def _compute_jerk_metric(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Compute absolute jerk over batch-dim of simulated proposals.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: absolute jerk within bound
    """
    jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        metadata=metadata,
    )
    return _within_bound(jerk_metric, min_bound=-MAX_ABS_MAG_JERK, max_bound=MAX_ABS_MAG_JERK)


def _compute_lon_jerk_metric(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Compute longitudinal jerk over batch-dim of simulated proposals.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: longitudinal jerk within bound
    """
    lon_jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        metadata=metadata,
    )
    return _within_bound(lon_jerk_metric, min_bound=-MAX_ABS_LON_JERK, max_bound=MAX_ABS_LON_JERK)


def _compute_yaw_accel(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Compute acceleration of yaw-angle over batch-dim of simulated proposals.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: acceleration of yaw-angle within bound
    """
    yaw_accel_metric = _extract_ego_yaw_rate(states, time_steps_s, deriv_order=2, poly_order=3)
    return _within_bound(yaw_accel_metric, min_bound=-MAX_ABS_YAW_ACCEL, max_bound=MAX_ABS_YAW_ACCEL)


def _compute_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Compute velocity of yaw-angle over batch-dim of simulated proposals.

    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: velocity of yaw-angle within bound
    """
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s)
    return _within_bound(yaw_rate_metric, min_bound=-MAX_ABS_YAW_RATE, max_bound=MAX_ABS_YAW_RATE)


def ego_is_comfortable(
    states: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> npt.NDArray[np.bool_]:
    """Accumulates all within-bound comfort metrics into a per-proposal, per-metric array.

    :param states: array representation of ego state values
    :param time_point_s: time steps [s] of time dim
    :param metadata: metadata of vehicle
    :return: boolean array (n_batch, n_metrics) flagging which comfort metrics are within bound
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == len(StateIndex)

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
    ]
    results: npt.NDArray[np.bool_] = np.zeros((n_batch, len(comfort_metric_functions)), dtype=np.bool_)
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, time_point_s, metadata)

    return results


def calculate_rms_difference(
    feature_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate RMS difference between consecutive frames for a given feature.

    :param feature_values: Array of shape (n_batch, n_time) containing feature values for each time step.
    :return: RMS differences for each trajectory in the batch.
    """
    differences = np.diff(feature_values, axis=1)  # Calculate frame-to-frame differences
    squared_differences = differences**2  # Square the differences
    mean_squared_diff = np.mean(squared_differences, axis=1)  # Take the mean over time
    rms = np.sqrt(mean_squared_diff)  # Compute the square root to get RMS
    return rms


def calculate_rms(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the Root Mean Square (RMS) of the given values along the time axis.

    :param values: Array containing values (n_batch, n_time).
    :return: RMS value per batch (n_batch,).
    """
    squared_values = values**2  # Square the differences
    mean_squared = np.mean(squared_values, axis=1)  # Compute mean along time axis
    rms_values = np.sqrt(mean_squared)  # Square root to get RMS
    return rms_values


def extract_features(
    states: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
    metadata: EgoStateSE3Metadata,
) -> dict:
    """Extract features needed for Extended Comfort evaluation.

    :param states: Array of ego states (n_batch, n_time, n_features).
    :param time_point_s: Array of time steps in seconds.
    :param metadata: metadata of vehicle
    :return: A dictionary of features.
    """
    return {
        "acceleration": _extract_ego_acceleration(states, "magnitude", metadata=metadata),
        "jerk": _extract_ego_jerk(states, "magnitude", time_point_s, metadata=metadata),
        "yaw_rate": _extract_ego_yaw_rate(states, time_point_s),
        "yaw_accel": _extract_ego_yaw_rate(states, time_point_s, deriv_order=2),
    }
