from typing import List

import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI
from py123d.datatypes import EgoStateSE2
from py123d.geometry.transform import rel_to_abs_se2_array

from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2
from nav123d.metrics.base_metric import BaseMetric
from nav123d.pdm.observation.pdm_observation import PDMObservation
from nav123d.pdm.pdm_closed_planner import PDMClosedInput, get_pdm_closed_planner
from nav123d.pdm.scoring.pdm_scorer import PDMScorer
from nav123d.pdm.simulation.pdm_simulator import PDMSimulator
from nav123d.pdm.utils.pdm_enums import StateIndex


class PDMMetric(BaseMetric):
    def __init__(self) -> None:
        self._score_trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        # self._pdm_planner = get_pdm_closed_planner()

    def compute_metric(self, scene_api: SceneAPI, **kwargs) -> dict:
        assert "agent_trajectory" in kwargs, "Missing required argument: agent_trajectory"
        agent_trajectory = kwargs["agent_trajectory"]
        assert isinstance(agent_trajectory, TrajectorySE2), "Argument 'agent_trajectory' must be of type TrajectorySE2"

        # 1. Run PDM-Closed to get trajectory.
        # 1.1 Initialize PDM-Closed planner with map and route information.
        pdm_planner = get_pdm_closed_planner()
        map_api = scene_api.get_map_api()
        assert map_api is not None, "MapAPI not found in SceneAPI."
        modality = scene_api.get_custom_modality_at_iteration(0, "scenario")
        assert modality is not None, "Scenario modality not found at iteration 2."
        lane_group_ids = [int(id_) for id_ in modality.data["route_roadblock_ids"]]
        pdm_planner.initialize(map_api, lane_group_ids)
        # 1.2 Run PDM-Closed inference
        current_input = PDMClosedInput.from_scene_api(scene_api)
        pdm_trajectory = pdm_planner.compute_planner_trajectory(current_input)

        # 2. Convert PDM + agent trajectory to state arrays well aligned.
        initial_ego_state_se2 = current_input.ego_state_se2
        resampled_pdm_trajectory = _resample_trajectory_se2(
            trajectory=pdm_trajectory,
            sampling=self._score_trajectory_sampling,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=True,
        )
        resampled_agent_trajectory = _resample_trajectory_se2(
            trajectory=agent_trajectory,
            sampling=self._score_trajectory_sampling,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=True,
        )
        states = _convert_trajectory_to_state_array(
            [
                resampled_pdm_trajectory,
                resampled_agent_trajectory,
            ]
        )  # for shape assertion

        # 3. Simulate PDM and agent trajectory.
        simulated_states = PDMSimulator(self._score_trajectory_sampling).simulate_proposals(
            states=states, initial_ego_state=initial_ego_state_se2
        )

        # 4. Evaluate PDM and agent trajectory with PDM-Closed scorer.
        # TODO@DanielDauner: This needs a cleaner solution, needs to be refactored together with PDMScorer.
        assert pdm_planner._drivable_area_map is not None, "PDM-Closed planner drivable area map is not initialized."
        assert pdm_planner._route_lane_group_dict is not None, (
            "PDM-Closed planner route lane groups are not initialized."
        )
        assert pdm_planner._centerline is not None, "PDM-Closed planner centerline is not initialized."
        drivable_area_map = pdm_planner._drivable_area_map
        route_lane_ids = list(pdm_planner._route_lane_group_dict.keys())
        centerline = pdm_planner._centerline

        log_replay_observation = PDMObservation(
            trajectory_sampling=TrajectorySampling(time_horizon=8, interval_length=0.1),
            proposal_sampling=TrajectorySampling(time_horizon=4, interval_length=0.1),
            map_radius=50.0,
            observation_sample_res=1,
            extend_observation_for_ttc=True,
        )
        log_replay_observation.update_replay(scene_api)

        pdm_scores = PDMScorer(proposal_sampling=self._score_trajectory_sampling).score_proposals(
            states=simulated_states,
            observation=log_replay_observation,
            centerline=centerline,
            route_lane_ids=route_lane_ids,
            drivable_area_map=drivable_area_map,
            ego_metadata=initial_ego_state_se2.metadata,
        )

        # 5. Return PDM sub-scores as dict.
        return pdm_scores[1]  # type: ignore


def _resample_trajectory_se2(
    trajectory: TrajectorySE2,
    sampling: TrajectorySampling,
    initial_ego_state_se2: EgoStateSE2,
    convert_to_absolute: bool = False,
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

    sampling_timestamps = ego_timestamp + np.arange(sampling.num_poses + 1, dtype=np.int64) * int(
        sampling.interval_length * 1e6
    )
    resampled_se2_array = _trajectory.interpolate(sampling_timestamps)

    return TrajectorySE2(pose_se2_array=resampled_se2_array, timestamps=sampling_timestamps)


def _convert_trajectory_to_state_array(trajectories: List[TrajectorySE2]) -> npt.NDArray[np.float64]:
    """
    Convert trajectory to state array representation for PDM modules.
    :param trajectory: input trajectory
    :return: trajectory as state array
    """
    # TODO@DanielDauner: This is a temporary solution, we should refactor the PDM modules to work with TrajectorySE2 directly.
    num_poses_ = trajectories[0].pose_se2_array.shape[0]

    _state_array = np.zeros(
        (
            len(trajectories),
            num_poses_,
            len(StateIndex),
        ),
        dtype=np.float64,
    )  # x, y, heading

    for traj_idx, trajectory in enumerate(trajectories):
        assert trajectory.pose_se2_array.shape[0] == num_poses_, "All trajectories must have the same number of poses."
        _state_array[traj_idx, :num_poses_, StateIndex.STATE_SE2] = trajectory.pose_se2_array

    return _state_array
