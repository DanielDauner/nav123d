import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE2, Timestamp

from nav123d.agents.pdm.simulation.batch_kinematic_bicycle import BatchKinematicBicycleModel
from nav123d.agents.pdm.simulation.batch_lqr import BatchLQRTracker
from nav123d.agents.pdm.utils.pdm_array_representation import ego_state_to_state_array
from nav123d.geometry.trajectory import TrajectorySampling


class PDMSimulator:
    """Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation."""

    def __init__(self, proposal_sampling: TrajectorySampling):
        """Constructor of PDMSimulator.

        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self.proposal_sampling = proposal_sampling

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()
        self._tracker = BatchLQRTracker()

    def simulate_proposals(
        self, states: npt.NDArray[np.float64], initial_ego_state: EgoStateSE2
    ) -> npt.NDArray[np.float64]:
        """Simulate all proposals over batch-dim.

        :param states: proposal states as array
        :param initial_ego_state: ego-vehicle state at current iteration
        :return: simulated proposal states as array
        """

        proposal_states = states[:, : self.proposal_sampling.num_poses + 1]
        discretization_time = self.proposal_sampling.interval_length
        velocity_profile, curvature_profile = self._tracker.compute_reference_profiles(
            proposal_states=proposal_states,
            discretization_time=discretization_time,
        )

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)
        simulated_timestamps = np.zeros((self.proposal_sampling.num_poses + 1,), dtype=np.int64)
        simulated_timestamps[0] = initial_ego_state.timestamp.time_us

        current_time_point = Timestamp.from_us(initial_ego_state.timestamp.time_us)
        delta_time_point = int(self.proposal_sampling.interval_length * 1e6)  # convert from s to us
        sampling_time: Timestamp = Timestamp.from_us(delta_time_point)

        for time_idx in range(1, self.proposal_sampling.num_poses + 1):
            # 1. Track the trajectory with controller to get commands (steering rate and acceleration)
            command_states = self._tracker.track_trajectory(
                time_idx=time_idx - 1,
                initial_states=simulated_states[:, time_idx - 1],
                proposal_states=proposal_states,
                velocity_profile=velocity_profile,
                curvature_profile=curvature_profile,
                discretization_time=discretization_time,
                ego_wheel_base=initial_ego_state.metadata.wheel_base,
            )

            # 2. Propagate the state with the motion model and commands
            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,  # convert from us to s
                ego_metadata=initial_ego_state.metadata,
            )

            simulated_timestamps[time_idx] = current_time_point.time_us
            current_time_point = Timestamp.from_us(current_time_point.time_us + delta_time_point)

        return simulated_states
