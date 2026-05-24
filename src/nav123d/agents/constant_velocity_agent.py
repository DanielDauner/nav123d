import numpy as np
from py123d.datatypes import EgoStateSE2

from nav123d.agents.abstract_agent import AbstractAgent
from nav123d.common.dataclasses import AgentInput, SensorConfig
from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2


class ConstantVelocityAgent(AbstractAgent):
    """Constant velocity baseline agent."""

    requires_scene = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, *args, **kwargs) -> TrajectorySE2:
        """Inherited, see superclass."""

        assert "ego_state_se2" in kwargs, (
            "ConstantVelocityAgent requires ego_state_se2 in kwargs for trajectory computation!"
        )
        ego_state_se2 = kwargs["ego_state_se2"]
        assert isinstance(ego_state_se2, EgoStateSE2), "ego_state_se2 should be of type EgoStateSE2!"
        assert ego_state_se2.dynamic_state_se2 is not None, (
            "ego_state_se2 should have dynamic state for velocity extraction!"
        )
        ego_speed = ego_state_se2.dynamic_state_se2.velocity_2d.magnitude

        num_poses, dt = (
            self._trajectory_sampling.num_poses,
            self._trajectory_sampling.interval_length,
        )
        poses_se2 = np.array(
            [[(time_idx + 1) * dt * ego_speed, 0.0, 0.0] for time_idx in range(num_poses)],
            dtype=np.float64,
        )
        timestamps = ego_state_se2.timestamp.time_us + np.arange(1, num_poses + 1) * int(dt * 1e6)

        return TrajectorySE2(pose_se2_array=poses_se2, timestamps=timestamps)
