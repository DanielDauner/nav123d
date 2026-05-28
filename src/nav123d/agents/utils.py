import numpy as np
from py123d.api import SceneAPI
from py123d.datatypes import EgoStateSE3
from py123d.geometry.transform import abs_to_rel_se2_array

from nav123d.datatypes.trajectory import TrajectorySampling, TrajectorySE2


def sample_ego_trajectory_from_api(
    scene_api: SceneAPI,
    trajectory_sampling: TrajectorySampling,
    in_relative: bool = False,
    interpolation_addon_s: float = 0.5,
) -> TrajectorySE2:
    """Loads the logged ego poses and resamples them to the given sampling specification.

    :param scene_api: API providing access to the logged ego states.
    :param trajectory_sampling: target sampling specification of the returned trajectory.
    :param in_relative: if True, return poses relative to the initial ego pose, defaults to False.
    :param interpolation_addon_s: extra horizon (in seconds) of logged states to load so the final
        pose can be interpolated, defaults to 0.5.
    :return: future ego trajectory resampled to the target sampling, as SE2.
    """

    initial_ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state should be available for target computation!"

    # Load the full ego states of the logs
    pose_se2_ = []
    timestamps_ = []
    for ego_state_se3 in scene_api.get_modality_between_timestamps(
        start_timestamp=initial_ego_state_se3.timestamp.time_us,
        end_timestamp=initial_ego_state_se3.timestamp.time_us
        + int((trajectory_sampling.time_horizon + interpolation_addon_s) * 1e6),
        modality_type="ego_state_se3",
        inclusive="both",
    ):
        if isinstance(ego_state_se3, EgoStateSE3):
            pose_se2_.append(ego_state_se3.rear_axle_se2.array)
            timestamps_.append(ego_state_se3.timestamp.time_us)

    # Resample the full trajectory to the target trajectory sampling specification.
    full_trajectory = TrajectorySE2(
        pose_se2_array=np.array(pose_se2_, dtype=np.float64),
        timestamps=np.array(timestamps_, dtype=np.int64),
    )
    sampling = trajectory_sampling
    ego_timestamp = initial_ego_state_se3.timestamp.time_us
    sampling_timestamps = ego_timestamp + np.arange(1, sampling.num_poses + 1, dtype=np.int64) * int(
        sampling.interval_length * 1e6
    )
    resampled_se2_array = full_trajectory.interpolate(sampling_timestamps)

    # Convert to the relative poses.
    if in_relative:
        resampled_se2_array = abs_to_rel_se2_array(
            origin=initial_ego_state_se3.rear_axle_se2,
            pose_se2_array=resampled_se2_array,
        )

    return TrajectorySE2(pose_se2_array=resampled_se2_array, timestamps=sampling_timestamps)
