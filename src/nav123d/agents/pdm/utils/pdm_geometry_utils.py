# TODO: Move & rename this file for common usage (not specific for PDM)

from typing import List

import numpy as np
import numpy.typing as npt
from py123d.geometry import PoseSE2

from nav123d.agents.pdm.utils.pdm_enums import PointIndex, SE2Index


def normalize_angle(angle):
    """Map an angle into the range [-π, π].

    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def translate_lon_and_lat(
    centers: npt.NDArray[np.float64],
    headings: npt.NDArray[np.float64],
    lon: float,
    lat: float,
) -> npt.NDArray[np.float64]:
    """Translate the position component of a centers point array.

    :param centers: array to be translated
    :param headings: array with heading angles
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return: array of translated coordinates
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.stack(
        [
            (lat * np.cos(headings + half_pi)) + (lon * np.cos(headings)),
            (lat * np.sin(headings + half_pi)) + (lon * np.sin(headings)),
        ],
        axis=-1,
    )
    return centers + translation


def calculate_progress(path: List[PoseSE2]) -> List[float]:
    """Calculate the cumulative progress of a given path.

    :param path: a path consisting of PoseSE2 as waypoints
    :return: a cumulative list of progress
    """
    x_position = [point.x for point in path]
    y_position = [point.y for point in path]
    x_diff = np.diff(x_position)
    y_diff = np.diff(y_position)
    points_diff: npt.NDArray[np.float64] = np.concatenate(([x_diff], [y_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64)  # type: ignore


def se2_array_translate_longitudinally(se2_array: npt.NDArray[np.float64], distance: float) -> npt.NDArray[np.float64]:
    """Translates an SE2 array along the heading axis by distance.

    :param se2_array: array of SE2 states with (x,y,θ) in last dim
    :param distance: distance to translate [m]
    :return: Translated SE2 coords array.
    """
    assert se2_array.shape[-1] == len(SE2Index)
    translate_se2 = np.zeros(se2_array.shape, dtype=np.float64)
    translate_se2[..., SE2Index.X] = se2_array[..., SE2Index.X] + np.cos(se2_array[..., SE2Index.HEADING]) * distance
    translate_se2[..., SE2Index.Y] = se2_array[..., SE2Index.Y] + np.sin(se2_array[..., SE2Index.HEADING]) * distance
    translate_se2[..., SE2Index.HEADING] = se2_array[..., SE2Index.HEADING]
    return translate_se2


def get_velocity_shifted(
    displacement: npt.NDArray[np.float64],
    ref_velocity_2d: npt.NDArray[np.float64],
    ref_angular_vel: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes the velocity at a query point on the same planar rigid body as a reference point.

    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_velocity_2d: [m/s] The velocity vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :return: [m/s] The velocity vector at the given displacement.
    """
    assert displacement.shape[-1] == len(PointIndex)
    assert ref_velocity_2d.shape[-1] == len(PointIndex)
    assert ref_velocity_2d.shape[:-1] == ref_angular_vel.shape
    velocity_shift_term = np.zeros(ref_velocity_2d.shape, dtype=ref_velocity_2d.dtype)
    velocity_shift_term[..., PointIndex.X] = -displacement[..., PointIndex.Y] * ref_angular_vel
    velocity_shift_term[..., PointIndex.Y] = displacement[..., PointIndex.X] * ref_angular_vel
    return ref_velocity_2d + velocity_shift_term


def get_acceleration_shifted(
    displacement: npt.NDArray[np.float64],
    ref_accel_2d: npt.NDArray[np.float64],
    ref_angular_vel: npt.NDArray[np.float64],
    ref_angular_accel: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes the acceleration at a query point on the same planar rigid body as a reference point.

    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_accel_2d: [m/s^2] The acceleration vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :param ref_angular_accel: [rad/s^2] The angular acceleration of the body around the vertical axis
    :return: [m/s^2] The acceleration vector at the given displacement.
    """
    assert displacement.shape[-1] == len(PointIndex)
    assert ref_accel_2d.shape[-1] == len(PointIndex)
    assert ref_accel_2d.shape[:-1] == ref_angular_vel.shape
    assert ref_accel_2d.shape[:-1] == ref_angular_accel.shape
    centripetal_acceleration_term = displacement * ref_angular_vel[..., None] ** 2
    angular_acceleration_term = displacement * ref_angular_accel[..., None]
    return ref_accel_2d + centripetal_acceleration_term + angular_acceleration_term
