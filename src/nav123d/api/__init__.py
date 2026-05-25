from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Optional, Tuple, Type, Union


from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI

from nav123d.api.arrow_agent_api import ArrowOracleAgentAPI, ArrowPlannerAgentAPI, ArrowSensorAgentAPI
from nav123d.api.base_agent_api import AgentAPI, ObservationType


def scene_api_to_agent_api(scene_api: SceneAPI, observation_type: ObservationType) -> "AgentAPI":
    """Helper function to convert from SceneAPI to AgentAPI for agent trajectory computation."""

    if isinstance(scene_api, ArrowSceneAPI):
        log_dir = scene_api._log_dir
        scene_metadata = scene_api._scene_metadata
        api_init = {
            observation_type.SENSOR: ArrowSensorAgentAPI,
            observation_type.PLANNER: ArrowPlannerAgentAPI,
            observation_type.ORACLE: ArrowOracleAgentAPI,
        }
        return api_init[observation_type](log_dir=log_dir, scene_metadata=scene_metadata)
    else:
        raise ValueError(f"Unsupported SceneAPI type: {type(scene_api)}")
