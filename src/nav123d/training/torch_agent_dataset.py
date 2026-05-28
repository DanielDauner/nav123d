import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from py123d.api import SceneAPI
from torch.utils.data import Dataset
from tqdm import tqdm

from nav123d.agents.base_torch_agent import BaseFeatureBuilder, BaseTargetBuilder
from nav123d.api import scene_api_to_agent_api
from nav123d.api.base_agent_api import ObservationType
from nav123d.training.cache.gzip_cache_helper import (
    dump_feature_target_to_pickle,
    load_scene_from_cache,
    load_valid_caches,
)

logger = logging.getLogger(__name__)


class TorchAgentDataset(Dataset):
    def __init__(
        self,
        scenes: List[SceneAPI],
        feature_builders: List[BaseFeatureBuilder],
        target_builders: List[BaseTargetBuilder],
        observation_type: ObservationType,
        cache_path: Optional[Union[str, Path]] = None,
        force_cache_computation: bool = False,
        dataset_type: Literal["train", "val", "test"] = "train",
    ):
        assert dataset_type in {"train", "val", "test"}, "dataset_type must be either 'train' or 'val' or 'test'"

        self._scenes = scenes
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._observation_type = observation_type
        self._dataset_type = dataset_type

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = load_valid_caches(
            cache_path=self._cache_path,
            dataset_type=self._dataset_type,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
        )

        if self._cache_path is not None:
            self.cache_dataset()

    def _cache_scene_with_uuid(self, scene: SceneAPI) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param scene_uuid: unique identifier of scene to cache
        """
        assert self._cache_path is not None, "Dataset did not receive a cache path!"

        log_metadata = scene.get_log_metadata()
        scene_uuid_path = self._cache_path / self._dataset_type / log_metadata.log_name / scene.scene_uuid
        os.makedirs(scene_uuid_path, exist_ok=True)

        agent_api = scene_api_to_agent_api(scene, self._observation_type)
        for builder in self._feature_builders:
            data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_api)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[scene.scene_uuid] = scene_uuid_path

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        for scene in tqdm(self._scenes, desc="Caching Dataset"):
            self._cache_scene_with_uuid(scene)

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self._scenes)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        scene = self._scenes[idx]
        uuid = scene.scene_uuid
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            if (uuid not in self._valid_cache_paths) or self._force_cache_computation:
                self._cache_scene_with_uuid(scene)
            features, targets = load_scene_from_cache(
                scene_uuid_path=self._valid_cache_paths[uuid],
                feature_builders=self._feature_builders,
                target_builders=self._target_builders,
            )
        else:
            agent_input = scene_api_to_agent_api(scene, self._observation_type)
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)


class TorchAgentCachedDataset(Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[BaseFeatureBuilder],
        target_builders: List[BaseTargetBuilder],
        dataset_type: Literal["train", "val", "test"] = "train",
    ):
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        assert dataset_type in {"train", "val", "test"}, "dataset_type must be either 'train' or 'val' or 'test'"

        self._cache_path = Path(cache_path)
        self._dataset_type = dataset_type
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._valid_cache_paths: Dict[str, Path] = load_valid_caches(
            cache_path=self._cache_path,
            dataset_type=self._dataset_type,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
        )
        self._scene_uuids = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:  # type: ignore
        """
        :return: number of samples to load
        """
        return len(self._scene_uuids)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return load_scene_from_cache(
            scene_uuid_path=self._valid_cache_paths[self._scene_uuids[idx]],
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
        )
