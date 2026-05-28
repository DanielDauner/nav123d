import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from py123d.script.builders.utils.utils_type import validate_type

from nav123d.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


def build_metrics(cfg_metrics: DictConfig) -> List[BaseMetric]:
    """Instantiates the list of metrics from config and validates their types.

    :param cfg_metrics: Hydra/omegaconf config group mapping metric names to metric configs
    :return: list of instantiated BaseMetric
    """
    logger.info("Building Metrics...")
    metrics: List[BaseMetric] = []
    for metric_cfg in cfg_metrics.values():
        metric: BaseMetric = instantiate(metric_cfg)
        validate_type(metric, BaseMetric)
        metrics.append(metric)
    logger.info("Building Metrics...DONE!")
    return metrics
