import abc
from typing import List

from py123d.api.scene.scene_api import SceneAPI
from py123d.common.utils.enums import SerialIntEnum

from nav123d.api.utils.route_utils import get_driving_command_heuristic_from_api, get_route_lane_group_ids_from_api
from nav123d.datatypes.driving_command import DrivingCommand


class ObservationType(SerialIntEnum):
    """Enum for the different types of agent observations."""

    SENSOR = 0
    PLANNER = 1
    ORACLE = 2


class AgentAPI(SceneAPI):
    """Base API for NAVSIM agents. IS-A :class:`SceneAPI` with restricted access semantics."""

    __slots__ = ()

    @property
    @abc.abstractmethod
    def observation_type(self) -> ObservationType:
        """Returns the name of the agent observation type."""

    def get_route_lane_group_ids(self) -> List[int]:
        """Returns the lane group ids corresponding to the route."""
        return get_route_lane_group_ids_from_api(self)

    def get_driving_command_heuristic(self) -> DrivingCommand:
        """Returns a heuristic high-level driving command for the current scene."""
        return get_driving_command_heuristic_from_api(self)


class SensorAgentAPI(AgentAPI):
    """API for agents that use sensor data."""

    __slots__ = ()

    @property
    def observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.SENSOR


class PlannerAgentAPI(AgentAPI):
    """API for agents that have privileged access to the scene."""

    __slots__ = ()

    @property
    def observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.PLANNER


class OracleAgentAPI(AgentAPI):
    """API for agents that have access to the oracle."""

    __slots__ = ()

    @property
    def observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.ORACLE
