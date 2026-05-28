from typing import Optional

import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE2
from py123d.geometry import PoseSE2Index
from py123d.geometry.transform import rel_to_abs_se2_array

from nav123d.agents.pdm.scoring.pdm_scorer import PDMScorer
from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2


class PDMEmergencyBrake:
    """Class for emergency brake maneuver of PDM-Closed."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        time_to_infraction_threshold: float = 2.0,
        max_ego_speed: float = 5.0,
        max_long_accel: float = 2.40,
        min_long_accel: float = -4.05,
        infraction: str = "collision",
    ):
        """Constructor for PDMEmergencyBrake.

        :param trajectory_sampling: Sampling parameters for final trajectory
        :param time_to_infraction_threshold: threshold for applying brake, defaults to 2.0
        :param max_ego_speed: maximum speed to apply brake, defaults to 5.0
        :param max_long_accel: maximum longitudinal acceleration for braking, defaults to 2.40
        :param min_long_accel: min longitudinal acceleration for braking, defaults to -4.05
        :param infraction: infraction to determine braking (collision or ttc), defaults to "collision"
        """

        # trajectory parameters
        self._trajectory_sampling = trajectory_sampling

        # braking parameters
        self._max_ego_speed: float = max_ego_speed  # [m/s]
        self._max_long_accel: float = max_long_accel  # [m/s^2]
        self._min_long_accel: float = min_long_accel  # [m/s^2]

        # braking condition parameters
        self._time_to_infraction_threshold: float = time_to_infraction_threshold
        self._infraction: str = infraction

        assert self._infraction in {
            "collision",
            "ttc",
        }, f"PDMEmergencyBraking: Infraction {self._infraction} not available as brake condition!"

    def brake_if_emergency(
        self, ego_state_se2: EgoStateSE2, scores: npt.NDArray[np.float64], scorer: PDMScorer
    ) -> Optional[TrajectorySE2]:
        """Applies emergency brake only if an infraction is expected within horizon.

        :param ego_state_se2: state object of ego
        :param scores: array of proposal scores
        :param scorer: scorer class of PDM
        :return: brake trajectory or None
        """

        trajectory = None
        assert ego_state_se2.dynamic_state_se2 is not None, (
            "PDMEmergencyBraking: EgoStateSE2 must have dynamic state for brake decision!"
        )
        ego_speed: float = ego_state_se2.dynamic_state_se2.velocity_2d.magnitude

        proposal_idx = np.argmax(scores)

        # retrieve time to infraction depending on brake detection mode
        if self._infraction == "ttc":
            time_to_infraction = scorer.time_to_ttc_infraction(int(proposal_idx))

        elif self._infraction == "collision":
            time_to_infraction = scorer.time_to_at_fault_collision(int(proposal_idx))
        else:
            raise ValueError(f"PDMEmergencyBraking: Infraction {self._infraction} not available as brake condition!")

        # check time to infraction below threshold
        if time_to_infraction <= self._time_to_infraction_threshold and ego_speed <= self._max_ego_speed:
            trajectory = self._generate_trajectory(ego_state_se2)

        return trajectory

    def _generate_trajectory(self, ego_state_se2: EgoStateSE2) -> TrajectorySE2:
        """Generates a braking trajectory that decelerates ego to zero velocity.

        :param ego_state_se2: state object of ego
        :return: braking trajectory as SE2
        """
        current_time_point = ego_state_se2.timestamp
        assert ego_state_se2.dynamic_state_se2 is not None, (
            "PDMEmergencyBraking: EgoStateSE2 must have dynamic state for trajectory generation!"
        )
        current_velocity = ego_state_se2.dynamic_state_se2.velocity_2d.x
        current_acceleration = ego_state_se2.dynamic_state_se2.acceleration_2d.x

        target_velocity = 0.0

        if current_velocity > 0.2:
            k_p = 10.0
            k_d = 0.0

            error = -current_velocity
            dt_error = -current_acceleration
            u_t = k_p * error + k_d * dt_error

            error = max(min(u_t, self._max_long_accel), self._min_long_accel)
            correcting_velocity = 11 / 10 * (current_velocity + error)

        else:
            k_p = 4
            k_d = 1

            error = target_velocity - current_velocity
            dt_error = -current_acceleration

            u_t = k_p * error + k_d * dt_error

            correcting_velocity = max(min(u_t, self._max_long_accel), self._min_long_accel)

        pose_se2_array = np.zeros((self._trajectory_sampling.num_poses + 1, len(PoseSE2Index)), dtype=np.float64)
        timestamps = np.zeros((self._trajectory_sampling.num_poses + 1,), dtype=np.int64)

        # Propagate planned trajectory for set number of samples
        for time_idx in range(self._trajectory_sampling.num_poses + 1):
            time_t = self._trajectory_sampling.interval_length * time_idx
            pose_se2_array[time_idx, PoseSE2Index.X] = correcting_velocity * time_t
            timestamps[time_idx] = current_time_point.time_us + int(time_t * 1e6)

        # Transform to absolute coordinates
        pose_se2_array = rel_to_abs_se2_array(ego_state_se2.center_se2, pose_se2_array)
        return TrajectorySE2(pose_se2_array, timestamps)
