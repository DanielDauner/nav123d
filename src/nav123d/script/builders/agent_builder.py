import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from py123d.script.builders.utils.utils_type import validate_type

from nav123d.agents.base_agent import BaseAgent
from nav123d.agents.base_torch_agent import BaseTorchAgent

logger = logging.getLogger(__name__)


def build_agent(cfg: DictConfig) -> BaseAgent:
    logger.info("Building Agent...")
    agent: BaseAgent = instantiate(cfg.agent)
    validate_type(agent, BaseAgent)
    logger.info("Building Agent...DONE!")
    return agent


def build_torch_agent(cfg: DictConfig) -> BaseTorchAgent:
    logger.info("Building Torch Agent...")
    agent: BaseTorchAgent = instantiate(cfg.agent)
    validate_type(agent, BaseTorchAgent)
    logger.info("Building Torch Agent...DONE!")
    return agent
