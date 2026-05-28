from typing import List

import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI

from nav123d.agents.pdm.observation.pdm_observation import PDMObservation
from nav123d.agents.pdm.pdm_agent import PDMAgent
from nav123d.agents.pdm.scoring.pdm_scorer import PDMScorer
from nav123d.agents.pdm.simulation.pdm_simulator import PDMSimulator
from nav123d.agents.pdm.utils.pdm_enums import StateIndex
from nav123d.api import scene_api_to_agent_api
from nav123d.datatypes.trajectory import TrajectorySampling, TrajectorySE2
from nav123d.metrics.base_metric import BaseMetric
from nav123d.metrics.trajectory_utils import resample_trajectory_se2


class PDMMetric(BaseMetric):
    """Scores an agent trajectory with the PDM-Closed closed-loop simulation and scorer."""

    def __init__(self, route_correction: bool = True) -> None:
        """Constructor of PDMMetric, fixing the scoring sampling to 4s at 0.1s intervals."""
        self._score_trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self._route_correction = route_correction

    def compute_metric(self, scene_api: SceneAPI, **kwargs) -> dict:
        """Inherited, see superclass."""
        assert "agent_trajectory" in kwargs, "Missing required argument: agent_trajectory"
        agent_trajectory = kwargs["agent_trajectory"]
        assert isinstance(agent_trajectory, TrajectorySE2), "Argument 'agent_trajectory' must be of type TrajectorySE2"

        # 1. Run PDM-Closed to get trajectory.
        # 1.1 Initialize PDM-Closed planner with map and route information.
        pdm_agent = PDMAgent(route_correction=False)
        map_api = scene_api.get_map_api()
        assert map_api is not None, "MapAPI not found in SceneAPI."
        pdm_agent.initialize()

        # 1.2 Run PDM-Closed inference
        agent_api = scene_api_to_agent_api(scene_api, observation_type=pdm_agent.get_observation_type())
        pdm_trajectory = pdm_agent.compute_trajectory(agent_api)
        assert isinstance(pdm_trajectory, TrajectorySE2), "PDM-Closed trajectory must be of type TrajectorySE2."

        # 2. Convert PDM + agent trajectory to state arrays well aligned.
        _ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
        assert _ego_state_se3 is not None, "Initial ego state SE3 not found in SceneAPI."
        initial_ego_state_se2 = _ego_state_se3.ego_state_se2

        # The PDM simulator/scorer operate in the absolute frame. PDM-Closed already plans in global
        # coordinates, whereas the evaluated agent trajectory reaches us in the canonical ego-relative
        # frame (normalized in run_evaluation), so only the latter needs converting to absolute.
        resampled_pdm_trajectory = resample_trajectory_se2(
            trajectory=pdm_trajectory,
            sampling=self._score_trajectory_sampling,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=False,
        )
        resampled_agent_trajectory = resample_trajectory_se2(
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
        assert pdm_agent._drivable_area_map is not None, "PDM-Closed planner drivable area map is not initialized."
        assert pdm_agent._route_lane_group_dict is not None, "PDM-Closed planner route lane groups are not initialized."
        assert pdm_agent._centerline is not None, "PDM-Closed planner centerline is not initialized."
        drivable_area_map = pdm_agent._drivable_area_map
        route_lane_ids = list(pdm_agent._route_lane_group_dict.keys())
        centerline = pdm_agent._centerline

        log_replay_observation = PDMObservation(
            trajectory_sampling=TrajectorySampling(time_horizon=8, interval_length=0.1),
            proposal_sampling=TrajectorySampling(time_horizon=4, interval_length=0.1),
            map_radius=100.0,
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
        return pdm_scores[1].iloc[0].to_dict()


def _convert_trajectory_to_state_array(trajectories: List[TrajectorySE2]) -> npt.NDArray[np.float64]:
    """Convert trajectories to a stacked state array representation for PDM modules.

    :param trajectories: input trajectories (all with the same number of poses)
    :return: trajectories as a state array
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
