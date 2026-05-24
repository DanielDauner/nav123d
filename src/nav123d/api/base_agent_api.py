import abc

from py123d.api.scene.scene_api import SceneAPI
from py123d.common.utils.enums import SerialIntEnum


class ObservationType(SerialIntEnum):
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
