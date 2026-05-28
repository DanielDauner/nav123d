from __future__ import annotations

import numpy as np
from py123d.api import SceneAPI
from py123d.geometry import PoseSE2Index

from nav123d.agents.utils import sample_ego_trajectory_from_api
from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2
from nav123d.metrics.base_metric import BaseMetric
from nav123d.metrics.trajectory_utils import resample_trajectory_se2


class DisplacementMetric(BaseMetric):
    def __init__(self) -> None:
        self._score_trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

        self._metrics = {"ade": _ade, "fde": _fde, "ahe": _ahe, "fhe": _fhe}

    def compute_metric(self, scene_api: SceneAPI, **kwargs) -> dict:
        """Inherited, see superclass."""

        assert "agent_trajectory" in kwargs, "Missing required argument: agent_trajectory"
        agent_trajectory_se2 = kwargs["agent_trajectory"]
        assert isinstance(agent_trajectory_se2, TrajectorySE2), (
            "Argument 'agent_trajectory' must be of type TrajectorySE2"
        )

        initial_ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
        assert initial_ego_state_se3 is not None, "Initial ego state SE3 not found in SceneAPI."
        initial_ego_state_se2 = initial_ego_state_se3.ego_state_se2

        ego_trajectory_se2 = sample_ego_trajectory_from_api(
            scene_api=scene_api,
            trajectory_sampling=self._score_trajectory_sampling,
            in_relative=True,
        )

        resampled_agent_trajectory = resample_trajectory_se2(
            trajectory=agent_trajectory_se2,
            sampling=self._score_trajectory_sampling,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=False,
            add_initial_ego_pose=False,
        )

        metrics_dict = {}
        for metric_name, metric_fn in self._metrics.items():
            metric_value = metric_fn(pred_traj=resampled_agent_trajectory, gt_traj=ego_trajectory_se2)
            metrics_dict[metric_name] = metric_value

        return metrics_dict


def _ade(pred_traj: TrajectorySE2, gt_traj: TrajectorySE2) -> float:
    """Compute Average Displacement Error (ADE) between predicted and ground truth trajectories."""
    assert pred_traj.pose_se2_array.shape == gt_traj.pose_se2_array.shape, "Trajectories must have the same shape."
    displacement_errors = np.linalg.norm(
        pred_traj.pose_se2_array[:, PoseSE2Index.XY] - gt_traj.pose_se2_array[:, PoseSE2Index.XY], axis=1
    )
    return float(np.mean(displacement_errors))


def _fde(pred_traj: TrajectorySE2, gt_traj: TrajectorySE2) -> float:
    """Compute Final Displacement Error (FDE) between predicted and ground truth trajectories."""
    assert pred_traj.pose_se2_array.shape == gt_traj.pose_se2_array.shape, "Trajectories must have the same shape."
    final_displacement_error = np.linalg.norm(
        pred_traj.pose_se2_array[-1, PoseSE2Index.XY] - gt_traj.pose_se2_array[-1, PoseSE2Index.XY]
    )
    return float(final_displacement_error)


def _ahe(pred_traj: TrajectorySE2, gt_traj: TrajectorySE2) -> float:
    """Compute Average Heading Error (AHE) between predicted and ground truth trajectories."""
    assert pred_traj.pose_se2_array.shape == gt_traj.pose_se2_array.shape, "Trajectories must have the same shape."
    heading_errors = np.abs(pred_traj.pose_se2_array[:, PoseSE2Index.YAW] - gt_traj.pose_se2_array[:, PoseSE2Index.YAW])
    return float(np.mean(heading_errors))


def _fhe(pred_traj: TrajectorySE2, gt_traj: TrajectorySE2) -> float:
    """Compute Final Heading Error (FHE) between predicted and ground truth trajectories."""
    assert pred_traj.pose_se2_array.shape == gt_traj.pose_se2_array.shape, "Trajectories must have the same shape."
    final_heading_error = np.abs(
        pred_traj.pose_se2_array[-1, PoseSE2Index.YAW] - gt_traj.pose_se2_array[-1, PoseSE2Index.YAW]
    )
    return float(final_heading_error)
