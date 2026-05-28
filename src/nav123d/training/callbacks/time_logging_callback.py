from time import perf_counter
from typing import Any, Optional

import pytorch_lightning as pl


class TimeLoggingCallback(pl.Callback):
    """Simple lightning callback to log training time."""

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        self.val_start = perf_counter()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        pl_module.log_dict(
            {
                "time_eval": perf_counter() - self.val_start,
                "step": pl_module.current_epoch,
            },
            rank_zero_only=True,
        )

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        self.test_start = perf_counter()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        pl_module.log_dict(
            {
                "time_test": perf_counter() - self.test_start,
                "step": pl_module.current_epoch,
            },
            rank_zero_only=True,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        self.train_start = perf_counter()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""
        pl_module.log_dict(
            {"time_epoch": perf_counter() - self.train_start, "step": pl_module.current_epoch},
            rank_zero_only=True,
        )
