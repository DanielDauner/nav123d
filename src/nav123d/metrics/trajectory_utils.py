import numpy as np
from py123d.datatypes import EgoStateSE2
from py123d.geometry.transform import rel_to_abs_se2_array

from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2


def resample_trajectory_se2(
    trajectory: TrajectorySE2,
    sampling: TrajectorySampling,
    initial_ego_state_se2: EgoStateSE2,
    convert_to_absolute: bool = False,
    add_initial_ego_pose: bool = True,
) -> TrajectorySE2:
    """
    Resample trajectory to given sampling specification and return as SE2 array.
    :param trajectory: input trajectory
    :param sampling: sampling specification for resampling the trajectory
    :param initial_ego_state_se2: initial ego state as SE2
    :return: resampled trajectory as SE2 array
    """

    if convert_to_absolute:
        _trajectory = TrajectorySE2(
            pose_se2_array=rel_to_abs_se2_array(
                origin=initial_ego_state_se2.rear_axle_se2,
                pose_se2_array=trajectory.pose_se2_array,
            ),
            timestamps=trajectory.timestamps,
        )
    else:
        _trajectory = trajectory

    # NOTE @DanielDauner: The PDM modules expect the trajectory to start at the current ego timestamp/iteration.
    # If the first timestamp of the trajectory is larger than the ego timestamp, we concat the initial ego pose/timestamp.
    ego_timestamp = initial_ego_state_se2.timestamp.time_us
    if int(_trajectory.timestamps[0]) > initial_ego_state_se2.timestamp.time_us:
        initial_ego_se2_array = initial_ego_state_se2.rear_axle_se2.array
        new_se2_array = np.concatenate([initial_ego_se2_array[None, ...], _trajectory.pose_se2_array], axis=0)
        new_timestamps = np.concatenate(
            [np.array([initial_ego_state_se2.timestamp.time_us]), _trajectory.timestamps], axis=0
        )
        _trajectory = TrajectorySE2(pose_se2_array=new_se2_array, timestamps=new_timestamps)

    offset = 0 if add_initial_ego_pose else 1
    sampling_timestamps = ego_timestamp + np.arange(offset, sampling.num_poses + 1, dtype=np.int64) * int(
        sampling.interval_length * 1e6
    )
    resampled_se2_array = _trajectory.interpolate(sampling_timestamps)

    return TrajectorySE2(pose_se2_array=resampled_se2_array, timestamps=sampling_timestamps)
