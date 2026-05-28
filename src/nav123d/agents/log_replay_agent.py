from nav123d.agents.base_agent import BaseAgent
from nav123d.agents.utils import sample_ego_trajectory_from_api
from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.geometry.trajectory import Trajectory, TrajectorySampling


class LogReplayAgent(BaseAgent):
    """Privileged agent interface of human operator."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """Initializes the log replay agent object.

        :param trajectory_sampling: trajectory sampling specification
        """
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.ORACLE

    def compute_trajectory(self, agent_api: AgentAPI) -> Trajectory:
        """Inherited, see superclass."""
        return sample_ego_trajectory_from_api(
            scene_api=agent_api,
            trajectory_sampling=self._trajectory_sampling,
            in_relative=True,
        )
