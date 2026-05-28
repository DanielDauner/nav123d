from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nav123d.agents.base_torch_agent import BaseFeatureBuilder, BaseTargetBuilder, BaseTorchAgent
from nav123d.agents.transfuser.transfuser_callback import TransfuserCallback
from nav123d.agents.transfuser.transfuser_config import TransfuserConfig
from nav123d.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from nav123d.agents.transfuser.transfuser_loss import transfuser_loss
from nav123d.agents.transfuser.transfuser_model import TransfuserModel
from nav123d.api.base_agent_api import ObservationType
from nav123d.datatypes.trajectory import TrajectorySampling
from nav123d.training.callbacks.time_logging_callback import TimeLoggingCallback


class TransfuserAgent(BaseTorchAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """Initializes TransFuser agent.

        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(self._trajectory_sampling, config)

    def name(self) -> str:
        """Inherited, see superclass."""
        prefix = "Latent" if self._config.latent else ""
        return f"{prefix}{self.__class__.__name__}"

    def initialize(self) -> None:
        """Inherited, see superclass."""
        assert self._checkpoint_path is not None, "TransfuserAgent requires a checkpoint path for initialization!"
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
        return [TransfuserTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[BaseFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[L.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config), TimeLoggingCallback()]
