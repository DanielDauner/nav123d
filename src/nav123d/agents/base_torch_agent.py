from __future__ import annotations

import abc
from typing import Dict, List, Union

import lightning as L
import numpy as np
import torch
from py123d.api import SceneAPI
from torch import Tensor

from nav123d.agents.base_agent import BaseAgent
from nav123d.api.base_agent_api import AgentAPI
from nav123d.geometry.trajectory import Trajectory, TrajectorySampling, TrajectorySE2


class BaseTorchAgent(torch.nn.Module, BaseAgent):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self, trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling

    @abc.abstractmethod
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the agent.

        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """

    @abc.abstractmethod
    def get_feature_builders(self) -> List[BaseFeatureBuilder]:
        """:return: List of feature builders run on the agent input at inference and training time."""

    @abc.abstractmethod
    def get_target_builders(self) -> List[BaseTargetBuilder]:
        """:return: List of target builders run on the ground-truth scene during training."""

    @abc.abstractmethod
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the loss for backpropagation from features, targets and model predictions."""

    @abc.abstractmethod
    def get_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],
    ]:
        """Returns the optimizers used by the lightning trainer.

        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """

    def compute_trajectory(self, agent_api: AgentAPI) -> Trajectory:
        """Inherited, see superclass."""
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_api))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses_se2_array = (
                predictions["trajectory"].squeeze(0).numpy().astype(np.float64)
            )  # (num_poses, 3) in local coordinates

        # Construct timestamped trajectory.
        ego_state_se3 = agent_api.get_ego_state_se3_at_iteration(0)
        assert ego_state_se3 is not None, "Ego state should be available for trajectory computation!"
        num_poses, dt = (self._trajectory_sampling.num_poses, self._trajectory_sampling.interval_length)
        timestamps = ego_state_se3.timestamp.time_us + np.arange(1, num_poses + 1) * int(dt * 1e6)
        return TrajectorySE2(pose_se2_array=poses_se2_array, timestamps=timestamps)

    def get_training_callbacks(self) -> List[L.Callback]:
        """Returns the lightning callbacks used during training; empty by default."""
        return []


class BaseFeatureBuilder(abc.ABC):
    """Abstract class of feature builder for agent training."""

    @abc.abstractmethod
    def get_unique_name(self) -> str:
        """:return: Unique name of created feature."""

    @abc.abstractmethod
    def compute_features(self, agent_api: AgentAPI) -> Dict[str, Tensor]:
        """Computes features from the agent API, i.e., without access to ground-truth.

        Outputs a dictionary where each item has a unique identifier and maps to a single feature tensor.
        One FeatureBuilder can return a dict with multiple FeatureTensors.
        """


class BaseTargetBuilder(abc.ABC):
    """Abstract class of target builder for agent training."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_unique_name(self) -> str:
        """:return: Unique name of created target."""

    @abc.abstractmethod
    def compute_targets(self, scene_api: SceneAPI) -> Dict[str, Tensor]:
        """Computes targets from the Scene object, i.e., with access to ground-truth.

        Outputs a dictionary where each item has a unique identifier and maps to a single target tensor.
        One TargetBuilder can return a dict with multiple TargetTensors.
        """
