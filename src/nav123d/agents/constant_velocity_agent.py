import numpy as np

from nav123d.agents.base_agent import BaseAgent
from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2


class ConstantVelocityAgent(BaseAgent):
    """Constant velocity baseline agent."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.SENSOR

    def compute_trajectory(self, agent_api: AgentAPI) -> TrajectorySE2:
        """Inherited, see superclass."""
        # assert isinstance(agent_api, SensorAgentAPI), "agent_api should be of type BaseAgentAPI!"

        ego_state_se3 = agent_api.get_ego_state_se3_at_iteration(0)
        assert ego_state_se3 is not None, "Ego state should be available for trajectory computation!"
        assert ego_state_se3.dynamic_state_se3 is not None, (
            "Ego dynamic state should be available for trajectory computation!"
        )

        # Compute constant velocity poses in forward direction in local coordinates.
        num_poses, dt = (self._trajectory_sampling.num_poses, self._trajectory_sampling.interval_length)
        ego_planar_speed = ego_state_se3.dynamic_state_se3.velocity_2d.magnitude
        poses_se2 = np.array(
            [[(time_idx + 1) * dt * ego_planar_speed, 0.0, 0.0] for time_idx in range(num_poses)],
            dtype=np.float64,
        )
        timestamps = ego_state_se3.timestamp.time_us + np.arange(1, num_poses + 1) * int(dt * 1e6)
        return TrajectorySE2(pose_se2_array=poses_se2, timestamps=timestamps)
