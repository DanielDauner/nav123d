from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from py123d.api import MapAPI
from py123d.datatypes import EgoStateSE2, Lane, LaneGroup
from py123d.geometry import OccupancyMap2D, PolylineSE2

from nav123d.agents.base_agent import BaseAgent
from nav123d.agents.pdm.observation.pdm_observation import PDMObservation
from nav123d.agents.pdm.proposal.batch_idm_policy import BatchIDMPolicy
from nav123d.agents.pdm.proposal.pdm_generator import PDMGenerator
from nav123d.agents.pdm.proposal.pdm_proposal import PDMProposalManager
from nav123d.agents.pdm.scoring.pdm_scorer import PDMScorer
from nav123d.agents.pdm.simulation.pdm_simulator import PDMSimulator
from nav123d.agents.pdm.utils.pdm_closed_utils import (
    build_drivable_area_occupancy_map,
    build_route_dicts,
    correct_route_lane_groups,
    get_centerline_as_polyline_se2,
    get_proposal_paths,
    get_starting_lane,
)
from nav123d.api.base_agent_api import AgentAPI, ObservationType
from nav123d.geometry.trajectory import Trajectory, TrajectorySampling

logger = logging.getLogger(__name__)


class PDMAgent(BaseAgent):
    """PDM-Closed planner."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(num_poses=80, interval_length=0.1),
        proposal_sampling: TrajectorySampling = TrajectorySampling(num_poses=40, interval_length=0.1),
        idm_policies: BatchIDMPolicy = BatchIDMPolicy(),
        lateral_offsets: Optional[List[float]] = [-1, 1],
        map_radius: float = 200,
        route_correction: bool = True,
    ):
        """Constructor for PDMAgent.

        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        :param route_correction: whether to correct the route lane groups on the first iteration
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
        self._route_correction: bool = route_correction

        # observation/forecasting class
        self._observation = PDMObservation(trajectory_sampling, proposal_sampling, map_radius)

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)

        self._iteration: int = 0

        # lazy loaded
        self._map_api: Optional[MapAPI] = None
        self._route_lane_group_dict: Optional[Dict[int, LaneGroup]] = None
        self._route_lane_dict: Optional[Dict[int, Lane]] = None
        self._centerline: Optional[PolylineSE2] = None
        self._drivable_area_map: Optional[OccupancyMap2D] = None
        self._proposal_manager: Optional[PDMProposalManager] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._iteration = 0

    def get_observation_type(self) -> ObservationType:
        """Inherited, see superclass."""
        return ObservationType.PLANNER

    def compute_trajectory(self, agent_api: AgentAPI) -> Trajectory:
        """Inherited, see superclass."""

        ego_state_se3 = agent_api.get_ego_state_se3_at_iteration(0)
        box_detections_se3 = agent_api.get_box_detections_se3_at_iteration(0)
        traffic_light_detections = agent_api.get_traffic_light_detections_at_iteration(0)
        # map_api = agent_api.get_map_api()
        # route_lane_group_ids = agent_api.get_route_lane_group_ids()

        if self._iteration == 0:
            self._map_api = agent_api.get_map_api()
            self._route_lane_group_ids = agent_api.get_route_lane_group_ids()
            assert self._map_api is not None, "Map API not found in Agent API."
            assert self._route_lane_group_ids is not None, "Route lane group ids not found in Agent API."

        assert ego_state_se3 is not None, "Ego state modality not found at iteration."
        assert box_detections_se3 is not None, "Box detections modality not found at iteration."
        assert self._map_api is not None, "Map API not found in Agent API."
        assert self._route_lane_group_ids is not None, "Route lane group ids not found in Agent API."

        self._route_lane_group_dict, self._route_lane_dict = build_route_dicts(
            self._map_api, self._route_lane_group_ids
        )
        gc.disable()

        assert self._map_api is not None and self._route_lane_group_dict is not None, (
            "Planner not initialized properly. Call initialize() before compute_planner_trajectory()."
        )

        ego_state_se2 = ego_state_se3.ego_state_se2
        box_detections_se2 = box_detections_se3.box_detections_se2

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            assert self._map_api is not None and self._route_lane_group_dict is not None, (
                "Planner not initialized properly."
            )
            if self._route_correction:
                self._route_lane_group_dict, self._route_lane_dict = correct_route_lane_groups(
                    ego_state_se2=ego_state_se2,
                    map_api=self._map_api,
                    route_lane_group_dict=self._route_lane_group_dict,
                )

        assert self._route_lane_group_dict is not None and self._route_lane_dict is not None, (
            "Route lane dicts not initialized."
        )

        # Update/Create drivable area polygon map
        self._drivable_area_map = build_drivable_area_occupancy_map(
            map_api=self._map_api,
            ego_state_se2=ego_state_se2,
            map_radius=self._map_radius,
        )

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state_se2=ego_state_se2,
            box_detections_se2=box_detections_se2,
            traffic_light_detections=traffic_light_detections,
            route_lane_dict=self._route_lane_dict,
        )

        # TODO: Refactor the rest and re-integrate the following steps:
        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state_se2)

        # 3. Generate/Unroll proposals
        assert self._proposal_manager is not None, "Proposal manager not initialized."
        proposals_array = self._generator.generate_proposals(ego_state_se2, self._observation, self._proposal_manager)

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(proposals_array, ego_state_se2)

        # 5. Score proposals
        assert self._centerline is not None, "Centerline not initialized."
        pdm_results = self._scorer.score_proposals(
            states=simulated_proposals_array,
            observation=self._observation,
            centerline=self._centerline,
            route_lane_ids=list(self._route_lane_dict.keys()),
            drivable_area_map=self._drivable_area_map,
            ego_metadata=ego_state_se2.metadata,
        )
        proposal_scores = np.array(pd.concat(pdm_results)["pdm_score"])

        trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))  # type: ignore

        self._iteration += 1
        return trajectory

    def _update_proposal_manager(self, ego_state_se2: EgoStateSE2) -> None:
        """Updates or initializes the PDMProposalManager.

        :param ego_state_se2: state of ego-vehicle
        """
        assert self._route_lane_dict is not None and self._drivable_area_map is not None, (
            "Planner not initialized properly."
        )
        current_lane = get_starting_lane(ego_state_se2, self._route_lane_dict, self._drivable_area_map)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0
        if create_new_proposals:
            assert self._route_lane_group_dict is not None, "Planner not initialized properly."
            self._centerline = get_centerline_as_polyline_se2(
                current_lane,
                self._route_lane_group_dict,
                self._route_lane_dict,
                ego_state_se2=ego_state_se2,
            )
            proposal_paths: List[PolylineSE2] = get_proposal_paths(self._centerline, self._lateral_offsets)
            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        assert self._proposal_manager is not None, "Proposal manager not initialized."
        self._proposal_manager.update(current_lane.speed_limit_mps)
