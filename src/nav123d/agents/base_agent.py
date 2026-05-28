from abc import ABC, abstractmethod

from py123d.common.utils.enums import SerialIntEnum

from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.datatypes.trajectory import Trajectory


class TrajectoryFrame(SerialIntEnum):
    """Coordinate frame in which an agent expresses the trajectory it returns."""

    EGO_RELATIVE = 0  # poses relative to the initial ego rear-axle pose (origin at current ego, x forward)
    GLOBAL = 1  # poses in the absolute/global frame


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

    def get_trajectory_frame(self) -> TrajectoryFrame:
        """:return: coordinate frame of the trajectory returned by :meth:`compute_trajectory`.

        Defaults to ego-relative. Agents that emit absolute/global poses must override this and
        return ``TrajectoryFrame.GLOBAL``. The evaluation harness reads this to normalize every
        agent's output into a single canonical frame before scoring, so metrics never have to guess.
        """
        return TrajectoryFrame.EGO_RELATIVE

    @abstractmethod
    def compute_trajectory(self, agent_api: AgentAPI) -> Trajectory:
        """Computes the ego vehicle trajectory.

        :param agent_api: API object providing access to the scene and current agent state.
        :return: Trajectory representing the predicted ego's position in future, expressed in the
            frame declared by :meth:`get_trajectory_frame`.
        """
