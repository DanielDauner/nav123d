import logging
from functools import partial
from pathlib import Path
from typing import Dict, List

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.common.execution import Executor, executor_map_chunked_list
from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.scene_builder_builder import build_scene_builder
from py123d.script.builders.scene_filter_builder import build_scene_filter
from py123d.script.builders.utils.utils_type import validate_type

from nav123d.agents.base_agent import BaseAgent
from nav123d.api import scene_api_to_agent_api
from nav123d.datatypes.trajectory import TrajectorySE2
from nav123d.metrics.trajectory_utils import to_ego_relative_trajectory_se2
from nav123d.script.builders.metric_builder import build_metrics

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/evaluation"
CONFIG_NAME = "default_evaluation"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for evaluating an agent.

    :param cfg: Hydra/omegaconf config (see config/evaluation/default_evaluation.yaml)
    """

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Executor and Scene Builder")
    executor: Executor = build_executor(cfg)
    scene_builder = build_scene_builder(cfg.scene_builder)

    logger.info("Building Scenes")
    scenes: List[SceneAPI] = []
    for scene_filter_cfg in cfg.test_scene_filter.values():
        scene_filter = build_scene_filter(scene_filter_cfg)
        scenes.extend(scene_builder.get_scenes(filter=scene_filter, executor=executor))
    logger.info("Num evaluation scenes: %d", len(scenes))

    logger.info("Running Evaluation")
    worker = partial(_evaluate_scenes, agent_cfg=cfg.agent, metrics_cfg=cfg.metrics)
    results: List[Dict[str, object]] = executor_map_chunked_list(
        executor,
        worker,
        scenes,
        name="Evaluation",
    )

    _save_results(results, cfg)


def _evaluate_scenes(
    scenes: List[SceneAPI],
    agent_cfg: DictConfig,
    metrics_cfg: DictConfig,
) -> List[Dict[str, object]]:
    """Run agent inference and all metrics for a chunk of scenes. Built per worker."""
    agent: BaseAgent = instantiate(agent_cfg)
    validate_type(agent, BaseAgent)
    agent.initialize()
    metrics = build_metrics(metrics_cfg)
    observation_type = agent.get_observation_type()

    results: List[Dict[str, object]] = []
    for scene in scenes:
        agent_api = scene_api_to_agent_api(scene, observation_type=observation_type)
        trajectory = agent.compute_trajectory(agent_api)
        assert isinstance(trajectory, TrajectorySE2), "Agent trajectory must be of type TrajectorySE2."

        # Normalize into the canonical ego-relative frame so every metric can assume a single
        # convention regardless of whether the agent planned in ego or global coordinates.
        initial_ego_state_se3 = scene.get_ego_state_se3_at_iteration(0)
        assert initial_ego_state_se3 is not None, "Initial ego state SE3 not found in SceneAPI."
        trajectory = to_ego_relative_trajectory_se2(
            trajectory=trajectory,
            frame=agent.get_trajectory_frame(),
            initial_ego_state_se2=initial_ego_state_se3.ego_state_se2,
        )

        result: Dict[str, object] = {"scene_uuid": scene.scene_uuid}
        for metric in metrics:
            result.update(metric.compute_metric(scene, agent_trajectory=trajectory))
        results.append(result)
    return results


def _save_results(results: List[Dict[str, object]], cfg: DictConfig) -> None:
    """Appends an average row and writes the per-scene results to disk.

    :param results: per-scene metric dicts
    :param cfg: Hydra/omegaconf config providing output_dir, results_file_stem and output_format
    """
    df = pd.DataFrame(results)
    numeric_cols = df.select_dtypes(include="number").columns
    average_row: Dict[str, object] = {"scene_uuid": "average"}
    average_row.update({col: df[col].mean() for col in numeric_cols})
    df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

    output_path = Path(cfg.output_dir) / f"{cfg.results_file_stem}.{cfg.output_format}"
    if cfg.output_format == "csv":
        df.to_csv(output_path, index=False)
    elif cfg.output_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output_format: {cfg.output_format}")
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
