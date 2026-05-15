import gc
import logging
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from nav123d.geometry.trajectory import TrajectorySampling
from nav123d.pdm.observation.pdm_observation import PDMObservation
from nav123d.pdm.observation.pdm_occupancy_map import PDMDrivableMap
from nav123d.pdm.proposal.batch_idm_policy import BatchIDMPolicy
from nav123d.pdm.proposal.pdm_generator import PDMGenerator
from nav123d.pdm.proposal.pdm_proposal import PDMProposalManager
from nav123d.pdm.scoring.pdm_scorer import PDMScorer
from nav123d.pdm.simulation.pdm_simulator import PDMSimulator
from nav123d.pdm.utils.pdm_closed_utils import (
    _build_route_dicts,
    _correct_route_roadblocks,
    _get_proposal_paths,
    _get_starting_lane,
)
from nav123d.pdm.utils.pdm_path import PDMPath

logger = logging.getLogger(__name__)


class PDMClosedPlanner(AbstractPlanner):
    """PDM-Closed planner."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
    ):
        """
        Constructor for PDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """
        assert trajectory_sampling.interval_length == proposal_sampling.interval_length, (
            "PDMClosedPlanner: Proposals and Trajectory must have equal interval length!"
        )

        # config parameters
        self._trajectory_sampling: TrajectorySampling = trajectory_sampling
        self._proposal_sampling: TrajectorySampling = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets
        self._map_radius: float = map_radius

        # observation/forecasting class
        self._observation = PDMObservation(trajectory_sampling, proposal_sampling, map_radius)

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)

        self._iteration: int = 0

        # lazy loaded
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_dict: Optional[Dict[str, RoadBlockGraphEdgeMapObject]] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None
        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None
        self._proposal_manager: Optional[PDMProposalManager] = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        gc.disable()
        ego_state, observation = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_dict, self._route_lane_dict = _correct_route_roadblocks(
                ego_state, self._map_api, self._route_roadblock_dict
            )

        # Update/Create drivable area polygon map
        self._drivable_area_map = PDMDrivableMap.from_simulation(self._map_api, ego_state, self._map_radius)

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(ego_state, self._observation, self._proposal_manager)

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(proposals_array, ego_state)

        # 5. Score proposals
        pdm_results = self._scorer.score_proposals(
            simulated_proposals_array,
            self._observation,
            self._centerline,
            list(self._route_lane_dict.keys()),
            self._drivable_area_map,
        )
        proposal_scores = np.array(pd.concat(pdm_results)["pdm_score"])

        trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))

        self._iteration += 1
        return trajectory

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route into self.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        self._route_roadblock_dict, self._route_lane_dict = _build_route_dicts(self._map_api, route_roadblock_ids)  # noqa: F821

    def _update_proposal_manager(self, ego_state: EgoState) -> None:
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """
        current_lane = _get_starting_lane(ego_state, self._route_lane_dict, self._drivable_area_map)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = _get_proposal_paths(
                current_lane,
                self._route_roadblock_dict,
                self._route_lane_dict,
                self._lateral_offsets,
            )
            self._centerline = proposal_paths[0]

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)


def get_pdm_closed_planner() -> PDMClosedPlanner:
    """
    Factory method to create PDMClosedPlanner with default parameters.
    """
    pdm_closed = PDMClosedPlanner(
        trajectory_sampling=TrajectorySampling(num_poses=80, interval_length=0.1),
        proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1),
        idm_policies=BatchIDMPolicy(
            speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
            fallback_target_velocity=15.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.5,
            decel_max=3.0,
        ),
        lateral_offsets=[-1.0, 1.0],
        map_radius=100,
    )
    return pdm_closed
