from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from py123d.api import SceneAPI
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nav123d.agents.base_torch_agent import BaseFeatureBuilder, BaseTargetBuilder, BaseTorchAgent
from nav123d.agents.utils import sample_ego_trajectory_from_api
from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.datatypes.trajectory import TrajectorySampling


class EgoStatusMLPAgent(BaseTorchAgent):
    """EgoStatusMLP agent interface."""

    def __init__(
        self,
        hidden_layer_dim: int,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """Initializes the agent interface for EgoStatusMLP.

        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        :param trajectory_sampling: trajectory sampling specification.
        """
        super().__init__(trajectory_sampling)

        self._checkpoint_path = checkpoint_path
        self._lr = lr
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, self._trajectory_sampling.num_poses * 3),
        )

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        assert self._checkpoint_path is not None, "EgoStatusMLPAgent requires a checkpoint path for initialization!"
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.SENSOR

    def get_target_builders(self) -> List[BaseTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[BaseFeatureBuilder]:
        """Inherited, see superclass."""
        return [EgoStatusFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        poses: torch.Tensor = self._mlp(features["ego_status"].to(torch.float32))
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)


class EgoStatusFeatureBuilder(BaseFeatureBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self):
        """Initializes the feature builder."""

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "ego_status_feature"

    def compute_features(self, agent_api: AgentAPI) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        ego_state_se3 = agent_api.get_ego_state_se3_at_iteration(0)
        assert ego_state_se3 is not None, "Ego state should be available for feature computation!"
        dynamic_state_se3 = ego_state_se3.dynamic_state_se3
        assert dynamic_state_se3 is not None, "Ego dynamic state should be available for feature computation!"

        velocity = torch.tensor(dynamic_state_se3.velocity_2d.array, dtype=torch.float32)
        acceleration = torch.tensor(dynamic_state_se3.acceleration_2d.array, dtype=torch.float32)
        # FIXME: Need to implement driving command. Not available in current API. --- IGNORE ---
        driving_command = torch.zeros(4, dtype=torch.float32)
        driving_command[1] = 1.0  # index 0: left, 1: straight, 2: left, 3: unknown
        ego_status_feature = torch.cat([velocity, acceleration, driving_command], dim=-1)
        return {"ego_status": ego_status_feature}


class TrajectoryTargetBuilder(BaseTargetBuilder):
    """Trajectory target builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """Initializes the target builder.

        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene_api: SceneAPI) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        resampled_trajectory = sample_ego_trajectory_from_api(
            scene_api=scene_api,
            trajectory_sampling=self._trajectory_sampling,
            in_relative=True,
        )
        return {"trajectory": torch.tensor(resampled_trajectory.pose_se2_array, dtype=torch.float32)}
