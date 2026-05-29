import abc
from typing import Optional, Tuple

from py123d.api import SceneAPI

from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.datatypes.trajectory import Trajectory


class BaseSimulation(abc.ABC):
    """Base class for simulations."""

    @property
    @abc.abstractmethod
    def observation_type(self) -> ObservationType:
        """Returns the required observation type for this simulation."""

    @abc.abstractmethod
    def reset(self, scene_api: SceneAPI) -> AgentAPI:
        """Resets the simulation to the initial state of the given scene."""

    @abc.abstractmethod
    def step(self, agent_plan: Trajectory) -> Tuple[Optional[AgentAPI], bool]:
        """Steps the simulation forward by applying the given agent plan.

        NOTE: currently only implements trajectories for actions.

        :param agent_plan: The trajectory to apply to the agent.
        :return: A tuple containing the updated agent API and a boolean indicating if the simulation is done.
        """
