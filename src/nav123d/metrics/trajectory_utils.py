import numpy as np
from py123d.datatypes import EgoStateSE2
from py123d.geometry.transform import abs_to_rel_se2_array, rel_to_abs_se2_array

from nav123d.agents.base_agent import TrajectoryFrame
from nav123d.datatypes.trajectory import TrajectorySampling, TrajectorySE2


def to_ego_relative_trajectory_se2(
    trajectory: TrajectorySE2,
    frame: TrajectoryFrame,
    initial_ego_state_se2: EgoStateSE2,
) -> TrajectorySE2:
    """Normalize a trajectory into the ego-relative frame given the frame it was produced in.

    This is the canonical-frame boundary: agents declare their output frame via
    ``BaseAgent.trajectory_frame()`` and the evaluation harness funnels every trajectory through
    here, so all downstream metrics can assume ego-relative poses.

    :param trajectory: input trajectory in the frame given by ``frame``
    :param frame: coordinate frame the input trajectory is expressed in
    :param initial_ego_state_se2: initial ego state, used as origin for the absolute<->relative transform
    :return: the trajectory expressed in the ego-relative frame
    """
    if frame == TrajectoryFrame.EGO_RELATIVE:
        return trajectory
    if frame == TrajectoryFrame.GLOBAL:
        return TrajectorySE2(
            pose_se2_array=abs_to_rel_se2_array(
                origin=initial_ego_state_se2.rear_axle_se2,
                pose_se2_array=trajectory.pose_se2_array,
            ),
            timestamps=trajectory.timestamps,
        )
    raise ValueError(f"Unsupported TrajectoryFrame: {frame}")


def to_absolute_trajectory_se2(
    trajectory: TrajectorySE2,
    frame: TrajectoryFrame,
    initial_ego_state_se2: EgoStateSE2,
) -> TrajectorySE2:
    """Normalize a trajectory into the absolute/global frame given the frame it was produced in.

    :param trajectory: input trajectory in the frame given by ``frame``
    :param frame: coordinate frame the input trajectory is expressed in
    :param initial_ego_state_se2: initial ego state, used as origin for the absolute<->relative transform
    :return: the trajectory expressed in the absolute/global frame
    """
    if frame == TrajectoryFrame.GLOBAL:
        return trajectory
    if frame == TrajectoryFrame.EGO_RELATIVE:
        return TrajectorySE2(
            pose_se2_array=rel_to_abs_se2_array(
                origin=initial_ego_state_se2.rear_axle_se2,
                pose_se2_array=trajectory.pose_se2_array,
            ),
            timestamps=trajectory.timestamps,
        )
    raise ValueError(f"Unsupported TrajectoryFrame: {frame}")


def resample_trajectory_se2(
    trajectory: TrajectorySE2,
    sampling: TrajectorySampling,
    initial_ego_state_se2: EgoStateSE2,
    convert_to_absolute: bool = False,
    add_initial_ego_pose: bool = True,
) -> TrajectorySE2:
    """Resample a trajectory to the given sampling specification.

    :param trajectory: input trajectory
    :param sampling: sampling specification for resampling the trajectory
    :param initial_ego_state_se2: initial ego state as SE2
    :param convert_to_absolute: if True, convert the input from ego-relative to absolute poses first, defaults to False
    :param add_initial_ego_pose: if True, include the initial ego pose as the first sample, defaults to True
    :return: resampled trajectory as SE2
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
