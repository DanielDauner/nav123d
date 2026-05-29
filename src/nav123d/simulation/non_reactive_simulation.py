from typing import Dict, Optional, Tuple

from py123d.api import SceneAPI
from py123d.datatypes.detections.box_detections import BoxDetectionsSE2

from nav123d.agents.base_agent import TrajectoryFrame
from nav123d.agents.pdm.simulation.pdm_simulator import PDMSimulator
from nav123d.agents.pdm.utils.pdm_enums import StateIndex
from nav123d.agents.utils import sample_ego_trajectory_from_api
from nav123d.api import scene_api_to_agent_api
from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.datatypes.trajectory import Trajectory, TrajectorySampling, TrajectorySE2
from nav123d.metrics.pdm_metric import _convert_trajectory_to_state_array  # noqa: PLC2701
from nav123d.metrics.trajectory_utils import resample_trajectory_se2
from nav123d.simulation.base_simulation import BaseSimulation


class NonReactiveSimulationSE2(BaseSimulation):
    """A simulation that does not react to the ego's actions, i.e. all agents follow their predefined trajectories."""

    def __init__(self, observation_type: ObservationType, trajectory_frame: TrajectoryFrame) -> None:
        """Constructor of NonReactiveSimulation."""
        self._observation_type = observation_type
        self._trajectory_frame = trajectory_frame

        # Currently fixed
        self._simulation_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)

        # Internal state
        self._scene_api: Optional[SceneAPI] = None
        self._logs = {}

    @property
    def observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return self._observation_type

    def reset(self, scene_api: SceneAPI) -> AgentAPI:
        """Inherited, see superclass."""
        self._scene_api = scene_api
        self._logs = {}
        return scene_api_to_agent_api(scene_api, observation_type=self._observation_type)

    def step(self, agent_plan: Trajectory) -> Tuple[Optional[AgentAPI], bool]:
        """Inherited, see superclass."""
        assert isinstance(agent_plan, TrajectorySE2), (
            f"Expected agent_plan to be of type Trajectory, but got {type(agent_plan)}"
        )
        assert self._scene_api is not None, "Simulation must be reset with a SceneAPI before stepping."

        _ego_state_se3 = self._scene_api.get_ego_state_se3_at_iteration(0)
        assert _ego_state_se3 is not None, "Initial ego state SE3 not found in SceneAPI."
        initial_ego_state_se2 = _ego_state_se3.ego_state_se2

        ego_trajectory_se2 = sample_ego_trajectory_from_api(
            scene_api=self._scene_api,
            trajectory_sampling=self._simulation_sampling,
            in_relative=False,
        )
        resampled_ego_trajectory = resample_trajectory_se2(
            trajectory=ego_trajectory_se2,
            sampling=self._simulation_sampling,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=False,
        )
        resampled_agent_trajectory = resample_trajectory_se2(
            trajectory=agent_plan,
            sampling=self._simulation_sampling,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=True if self._trajectory_frame == TrajectoryFrame.EGO_RELATIVE else False,
        )
        states = _convert_trajectory_to_state_array(
            [
                resampled_ego_trajectory,
                resampled_agent_trajectory,
            ]
        )  # for shape assertion

        # 3. Simulate PDM and agent trajectory.
        simulated_states = PDMSimulator(self._simulation_sampling).simulate_proposals(
            states=states, initial_ego_state=initial_ego_state_se2
        )

        self._logs["initial_ego_state_se2"] = initial_ego_state_se2
        self._logs["replay_trajectory_se2"] = TrajectorySE2(
            pose_se2_array=simulated_states[0, :, StateIndex.STATE_SE2],
            timestamps=resampled_agent_trajectory.timestamps,
        )
        self._logs["simulated_trajectory_se2"] = TrajectorySE2(
            pose_se2_array=simulated_states[1, :, StateIndex.STATE_SE2],
            timestamps=resampled_agent_trajectory.timestamps,
        )
        self._logs["box_detections_se2"] = _get_log_replay_box_detections_se2(
            scene_api=self._scene_api, simulation_sampling=self._simulation_sampling
        )

        # In non-reactive simulation, we terminate after the first step.
        return None, False

    def get_logs(self) -> dict:
        """Inherited, see superclass."""
        return self._logs


def _get_log_replay_box_detections_se2(
    scene_api: SceneAPI,
    simulation_sampling: TrajectorySampling,
) -> Optional[Dict[int, BoxDetectionsSE2]]:
    """Helper function to get the bounding boxes of the replayed trajectory for visualization/debugging."""

    # 1. assert scene api uses roughly 10 Hz.
    simulation_iterations_s = simulation_sampling.step_time
    scene_iterations_s = scene_api.scene_metadata.iteration_duration_s
    assert abs(simulation_iterations_s - scene_iterations_s) < 0.05, (
        f"Expected scene iteration duration to be close to simulation sampling step time of {simulation_iterations_s}s, but got {scene_iterations_s}s"
    )
    box_detections_se2_dict: Optional[Dict[int, BoxDetectionsSE2]] = None

    if "box_detections_se3" in scene_api.get_all_modality_metadatas().keys():
        box_detections_se2_dict = {}
        for iteration in range(scene_api.scene_metadata.num_future_iterations + 1):
            box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(iteration)
            assert box_detections_se3 is not None, f"Box detections SE3 not found in SceneAPI at iteration {iteration}."
            box_detections_se2_dict[iteration] = box_detections_se3.box_detections_se2
    return box_detections_se2_dict
