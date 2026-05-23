from __future__ import annotations

import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from py123d.geometry import PolylineSE2

PROXIMITY_ABS_TOL = 1e-10


class TrajectorySampling:
    """
    Trajectory sampling config. Provide any two of (num_poses, time_horizon, interval_length) and the third is
        deduced; passing all three is allowed if they are consistent. After construction all three attributes
        are guaranteed to be set.

    TODO: Update to use py123d conventions.
    """

    __slots__ = ("num_poses", "time_horizon", "interval_length")

    num_poses: int
    time_horizon: float
    interval_length: float

    def __init__(
        self,
        num_poses: Optional[int] = None,
        time_horizon: Optional[float] = None,
        interval_length: Optional[float] = None,
    ) -> None:
        if num_poses is not None and not isinstance(num_poses, int):
            raise ValueError(f"num_poses was defined but it is not int. Instead {type(num_poses)}!")
        time_horizon = float(time_horizon) if time_horizon is not None else None
        interval_length = float(interval_length) if interval_length is not None else None

        if num_poses is not None and time_horizon is not None and interval_length is None:
            interval_length = time_horizon / num_poses
        elif num_poses is not None and interval_length is not None and time_horizon is None:
            time_horizon = num_poses * interval_length
        elif time_horizon is not None and interval_length is not None and num_poses is None:
            remainder = math.fmod(time_horizon, interval_length)
            is_close_to_zero = math.isclose(remainder, 0, abs_tol=PROXIMITY_ABS_TOL)
            is_close_to_interval_length = math.isclose(remainder, interval_length, abs_tol=PROXIMITY_ABS_TOL)
            if not is_close_to_zero and not is_close_to_interval_length:
                raise ValueError(
                    "The time horizon must be a multiple of interval length! "
                    f"time_horizon = {time_horizon}, interval = {interval_length} and is {remainder}"
                )
            num_poses = int(time_horizon / interval_length)
        elif num_poses is not None and time_horizon is not None and interval_length is not None:
            if not math.isclose(num_poses, time_horizon / interval_length, abs_tol=PROXIMITY_ABS_TOL):
                raise ValueError(
                    "Not valid initialization of sampling class!"
                    f"time_horizon = {time_horizon}, "
                    f"interval = {interval_length}, num_poses = {num_poses}"
                )
        else:
            raise ValueError(
                f"Cant initialize class! num_poses = {num_poses}, "
                f"interval = {interval_length}, time_horizon = {time_horizon}"
            )

        self.num_poses = num_poses
        self.time_horizon = time_horizon
        self.interval_length = interval_length

    @property
    def step_time(self) -> float:
        """
        :return: [s] The time difference between two poses.
        """
        return self.interval_length

    def __hash__(self) -> int:
        return hash((self.num_poses, self.time_horizon, self.interval_length))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectorySampling):
            return NotImplemented
        return (
            math.isclose(other.time_horizon, self.time_horizon)
            and math.isclose(other.interval_length, self.interval_length)
            and other.num_poses == self.num_poses
        )


class TrajectorySE2:
    """Trajectory dataclass in NAVSIM."""

    pose_se2_array: npt.NDArray[np.float64]  # local coordinates
    timestamps: npt.NDArray[np.int64]  # [s] time of each pose in trajectory, relative to trajectory start time
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __init__(
        self,
        pose_se2_array: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.int64],
        # trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> None:

        self.pose_se2_array = pose_se2_array
        self.timestamps = timestamps
        # self.trajectory_sampling = trajectory_sampling

    # @classmethod
    # def from_arrays(
    #     cls,
    #     pose_se2_array: npt.NDArray[np.float64],
    #     timestamps: npt.NDArray[np.int64],
    #     copy: bool = True,
    # ) -> TrajectorySE2:
    #     pass

    @property
    def polyline_se2(self) -> PolylineSE2:
        return PolylineSE2.from_array(self.pose_se2_array)

    # def __post_init__(self):
    #     assert self.pose_se2_array.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
    #     assert self.pose_se2_array.shape[0] == self.trajectory_sampling.num_poses, (
    #         "Trajectory poses and sampling have unequal number of poses."
    #     )
    #     assert self.pose_se2_array.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."
