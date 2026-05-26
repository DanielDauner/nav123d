import gzip
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from py123d.api import SceneAPI
from torch.utils.data import Dataset
from tqdm import tqdm

from nav123d.agents.base_torch_agent import BaseFeatureBuilder, BaseTargetBuilder
from nav123d.api import scene_api_to_agent_api
from nav123d.api.base_agent_api import ObservationType

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class TorchAgentDataset(Dataset):
    def __init__(
        self,
        scenes: List[SceneAPI],
        feature_builders: List[BaseFeatureBuilder],
        target_builders: List[BaseTargetBuilder],
        observation_type: ObservationType,
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        self._scenes = scenes
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._observation_type = observation_type

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[BaseFeatureBuilder],
        target_builders: List[BaseTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for scene_uuid_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[scene_uuid_path.name] = scene_uuid_path

        return valid_cache_paths

    def _cache_scene_with_uuid(self, scene: SceneAPI) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """
        assert self._cache_path is not None, "Dataset did not receive a cache path!"

        log_metadata = scene.get_log_metadata()
        scene_uuid_path = self._cache_path / log_metadata.log_name / scene.scene_uuid
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

    def _load_scene_with_uuid(self, uuid: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param uuid:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        scene_uuid_path = self._valid_cache_paths[uuid]
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

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
            features, targets = self._load_scene_with_uuid(uuid)
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
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:  # type: ignore
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[BaseFeatureBuilder],
        target_builders: List[BaseTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)
