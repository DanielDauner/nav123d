from abc import ABC, abstractmethod

from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.datatypes.trajectory import Trajectory


class BaseAgent(ABC):
    """Base class for agents in NAVSIM."""

    @abstractmethod
    def name(self) -> str:
        """:return: string describing name of this agent."""

    @abstractmethod
    def initialize(self) -> None:
        """Initializes the agent before inference, e.g. loading model weights."""

    @abstractmethod
    def get_observation_type(self) -> ObservationType:
        """:return: ObservationType describing the required sensor inputs for this agent."""

    @abstractmethod
    def compute_trajectory(self, agent_api: AgentAPI) -> Trajectory:
        """Computes the ego vehicle trajectory.

        :param agent_api: API object providing access to the scene and current agent state.
        :return: Trajectory representing the predicted ego's position in future
        """
