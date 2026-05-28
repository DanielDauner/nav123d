import logging
from typing import List, Tuple

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.common.execution import Executor
from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.scene_builder_builder import build_scene_builder
from py123d.script.builders.scene_filter_builder import build_scene_filter
from torch.utils.data import DataLoader

from nav123d.agents.base_torch_agent import BaseTorchAgent
from nav123d.training.torch_agent_dataset import TorchAgentCachedDataset, TorchAgentDataset
from nav123d.training.torch_agent_lightning_module import TorchAgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    L.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    torch_agent: BaseTorchAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = TorchAgentLightningModule(agent=torch_agent)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert not cfg.force_cache_computation, (
            "force_cache_computation must be False when using cached data without building SceneLoader"
        )
        assert cfg.cache_path is not None, (
            "cache_path must be provided when using cached data without building SceneLoader"
        )
        train_data = TorchAgentCachedDataset(
            cache_path=cfg.cache_path,
            feature_builders=torch_agent.get_feature_builders(),
            target_builders=torch_agent.get_target_builders(),
            dataset_type="train",
        )
        val_data = TorchAgentCachedDataset(
            cache_path=cfg.cache_path,
            feature_builders=torch_agent.get_feature_builders(),
            target_builders=torch_agent.get_target_builders(),
            dataset_type="val",
        )
    else:
        executor = build_executor(cfg)
        train_data, val_data = build_datasets(cfg, torch_agent, executor)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.train_dataloader.params)
    logger.info("Num training samples: %d", len(train_data))  # type: ignore
    val_dataloader = DataLoader(val_data, **cfg.val_dataloader.params)
    logger.info("Num validation samples: %d", len(val_data))  # type: ignore

    logger.info("Building Trainer")
    trainer = L.Trainer(**cfg.trainer.params, callbacks=torch_agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def build_datasets(
    cfg: DictConfig, agent: BaseTorchAgent, executor: Executor
) -> Tuple[TorchAgentDataset, TorchAgentDataset]:
    """TODO"""

    # 1. Build training and validation scenes using scene builder and filter from hydra modules.
    scene_builder = build_scene_builder(cfg.scene_builder)
    train_scenes: List[SceneAPI] = []
    for _train_scene_filter_cfg in cfg.train_scene_filter.values():
        _train_scene_filter = build_scene_filter(_train_scene_filter_cfg)
        _train_scenes = scene_builder.get_scenes(filter=_train_scene_filter, executor=executor)
        train_scenes.extend(_train_scenes)

    val_scenes: List[SceneAPI] = []
    for _val_scene_filter_cfg in cfg.val_scene_filter.values():
        _val_scene_filter = build_scene_filter(_val_scene_filter_cfg)
        _val_scenes = scene_builder.get_scenes(filter=_val_scene_filter, executor=executor)
        val_scenes.extend(_val_scenes)

    train_data = TorchAgentDataset(
        scenes=train_scenes,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        observation_type=agent.get_observation_type(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        dataset_type="train",
    )

    val_data = TorchAgentDataset(
        scenes=val_scenes,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        observation_type=agent.get_observation_type(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        dataset_type="val",
    )

    return train_data, val_data


if __name__ == "__main__":
    main()
