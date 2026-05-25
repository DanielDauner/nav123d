# logger = logging.getLogger(__name__)

# CONFIG_PATH = "config/pdm_scoring"
# CONFIG_NAME = "default_run_pdm_score"
from typing import Dict, List, Tuple

import pandas as pd
from py123d.api import SceneAPI, SceneFilter, get_filtered_scenes
from py123d.common.execution import RayExecutor, executor_map_chunked_list

from nav123d.agents.base_agent import BaseAgent
from nav123d.agents.constant_velocity_agent import ConstantVelocityAgent
from nav123d.agents.ego_status_mlp_agent import EgoStatusMLPAgent
from nav123d.agents.log_replay_agent import LogReplayAgent
from nav123d.agents.pdm.pdm_agent import PDMAgent
from nav123d.agents.transfuser.transfuser_agent import TransfuserAgent
from nav123d.agents.transfuser.transfuser_config import TransfuserConfig
from nav123d.api import scene_api_to_agent_api
from nav123d.geometry.trajectory import TrajectorySE2
from nav123d.metrics.pdm_metric import PDMMetric

EGO_MLP_SEED = 0
TRANSFUSER_SEED = 0
LTF_SEED = 0

AGENT_NAME = "ltf"


def _build_agent(name: str) -> BaseAgent:
    if name == "cv":
        return ConstantVelocityAgent()
    if name == "lr":
        return LogReplayAgent()
    if name == "es":
        return EgoStatusMLPAgent(
            checkpoint_path=f"/home/daniel/Downloads/ego_status_mlp_seed_{EGO_MLP_SEED}.ckpt",
            hidden_layer_dim=512,
            lr=1e-4,
        )
    if name == "tf":
        return TransfuserAgent(
            checkpoint_path=f"/home/daniel/Downloads/transfuser_seed_{TRANSFUSER_SEED}.ckpt",
            config=TransfuserConfig(latent=False),
            lr=1e-4,
        )
    if name == "ltf":
        return TransfuserAgent(
            checkpoint_path=f"/home/daniel/Downloads/ltf_seed_{LTF_SEED}.ckpt",
            config=TransfuserConfig(latent=True),
            lr=1e-4,
        )
    if name == "pdm":
        return PDMAgent(route_correction=False)
    raise ValueError(f"Unknown agent name: {name}")


def main():
    # 1. Load scene and agent trajectory.

    scene_filter = SceneFilter(
        datasets=["nuplan-mini"],
        # datasets=["nuscenes-interpolated-mini"],
        # datasets=["carla"],
        split_names=None,
        log_names=None,
        # target_iteration_duration_s=0.1,  # 10Hz iteration frequency
        future_duration_s=8.0,  # Look up to 8 seconds into the future.
        history_duration_s=0.0,  # Look up to 0.5 seconds into the past.
        timestamp_threshold_s=0.1,  # Allow for up to 50ms timestamp misalignment between modalities.
        required_scene_modalities=["ego_state_se3", "lidar.lidar_merged"],
        shuffle=False,
    )

    # executor = ThreadPoolExecutor()
    executor = RayExecutor()

    scenes = get_filtered_scenes(scene_filter)
    scene_dict = {scene.scene_uuid: scene for scene in scenes}

    trajectories_list: List[Tuple[str, TrajectorySE2]] = executor_map_chunked_list(
        executor,
        _agent_inference,
        scenes,
        name="Agent Inference",
    )

    scene_traj_pairs = [(scene_dict[scene_uuid], trajectory) for scene_uuid, trajectory in trajectories_list]

    metrics = executor_map_chunked_list(
        executor,
        _run_metrics,
        scene_traj_pairs,
        name="Agent Inference",
    )

    df = pd.DataFrame(metrics)
    numeric_cols = df.select_dtypes(include="number").columns
    average_row: Dict[str, object] = {"scene_uuid": "average"}
    average_row.update({col: df[col].mean() for col in numeric_cols})
    df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)
    df.to_csv("results.csv", index=False)


def _agent_inference(scenes: List[SceneAPI]) -> List[Tuple[str, TrajectorySE2]]:
    agent = _build_agent(AGENT_NAME)
    agent.initialize()
    trajectories = []
    for scene in scenes:
        scene_uuid = scene.scene_uuid
        agent_api = scene_api_to_agent_api(scene, observation_type=agent.get_observation_type())
        trajectory = agent.compute_trajectory(agent_api)
        assert isinstance(trajectory, TrajectorySE2), "Agent trajectory must be of type TrajectorySE2."
        trajectories.append((scene_uuid, trajectory))
    return trajectories


def _run_metrics(scene_traj_pairs: List[Tuple[SceneAPI, TrajectorySE2]]) -> List[Dict[str, List[float]]]:
    metric = PDMMetric()
    results = []
    for scene, trajectory in scene_traj_pairs:
        result = {"scene_uuid": scene.scene_uuid}
        score_dict = metric.compute_metric(scene, agent_trajectory=trajectory)
        result.update(score_dict)
        results.append(result)
    return results


if __name__ == "__main__":
    main()
