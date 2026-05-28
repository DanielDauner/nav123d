import gzip
import logging
import pickle
from pathlib import Path
from typing import Dict, Final, List, Literal, Optional, Tuple, Union

import torch

from nav123d.agents.base_torch_agent import BaseFeatureBuilder, BaseTargetBuilder

# TODO@DanielDauner: Make cache mechanism modular.

logger = logging.getLogger(__name__)

GZIP_COMPRESSION_LEVEL: Final[int] = 1  # Use compresslevel = 1 to compress the size but also has fast write and read.


def load_feature_target_from_pickle(path: Union[Path, str]) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(
    path: Union[Path, str],
    data_dict: Dict[str, torch.Tensor],
    compresslevel: int = GZIP_COMPRESSION_LEVEL,
) -> None:
    """Helper function to save feature/target to pickle."""
    with gzip.open(path, "wb", compresslevel=compresslevel) as f:
        pickle.dump(data_dict, f)


def load_valid_caches(
    cache_path: Optional[Union[Path, str]],
    dataset_type: Literal["train", "val", "test"],
    feature_builders: List[BaseFeatureBuilder],
    target_builders: List[BaseTargetBuilder],
) -> Dict[str, Path]:
    """Scans the cache directory and returns scenes whose builder caches are all present.

    Scenes missing any feature/target builder cache are skipped with a warning.

    :param cache_path: root cache directory, or None to return an empty mapping
    :param dataset_type: dataset split ("train", "val" or "test") to scan
    :param feature_builders: feature builders whose caches must exist
    :param target_builders: target builders whose caches must exist
    :return: mapping from scene uuid to its cache directory
    """
    valid_cache_paths: Dict[str, Path] = {}
    if cache_path is not None:
        for split_name_path in (Path(cache_path) / dataset_type).iterdir():
            if not split_name_path.is_dir():
                continue
            for log_name_path in split_name_path.iterdir():
                if not log_name_path.is_dir():
                    continue
                for scene_uuid_path in log_name_path.iterdir():
                    if not scene_uuid_path.is_dir():
                        continue
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[scene_uuid_path.name] = scene_uuid_path
                    else:
                        logger.warning(f"Cache for scene {scene_uuid_path.name} is incomplete and will be ignored.")
    return valid_cache_paths


def load_scene_from_cache(
    scene_uuid_path: Path,
    feature_builders: List[BaseFeatureBuilder],
    target_builders: List[BaseTargetBuilder],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Loads the cached features and targets for a single scene.

    :param scene_uuid_path: cache directory of the scene
    :param feature_builders: feature builders whose caches to load
    :param target_builders: target builders whose caches to load
    :return: tuple of (feature dict, target dict)
    """
    feature_dict: Dict[str, torch.Tensor] = {}
    target_dict: Dict[str, torch.Tensor] = {}

    for builder in feature_builders:
        data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
        if data_dict_path.is_file():
            data_dict = load_feature_target_from_pickle(data_dict_path)
            feature_dict.update(data_dict)

    for builder in target_builders:
        data_dict_path = scene_uuid_path / (builder.get_unique_name() + ".gz")
        if data_dict_path.is_file():
            data_dict = load_feature_target_from_pickle(data_dict_path)
            target_dict.update(data_dict)

    return feature_dict, target_dict
