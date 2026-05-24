from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from py123d.api import MapAPI, SceneAPI
from py123d.datatypes import BoxDetectionsSE2, EgoStateSE2, Lane, LaneGroup, TrafficLightDetections
from py123d.geometry import OccupancyMap2D, PolylineSE2

from nav123d.api.agent_api import AgentAPI
from nav123d.geometry.trajectory import TrajectorySampling, TrajectorySE2
from nav123d.pdm.observation.pdm_observation import PDMObservation
from nav123d.pdm.proposal.batch_idm_policy import BatchIDMPolicy
from nav123d.pdm.proposal.pdm_generator import PDMGenerator
from nav123d.pdm.proposal.pdm_proposal import PDMProposalManager
from nav123d.pdm.scoring.pdm_scorer import PDMScorer
from nav123d.pdm.simulation.pdm_simulator import PDMSimulator
from nav123d.pdm.utils.pdm_closed_utils import (
    _get_proposal_paths,
    _get_starting_lane,
    build_drivable_area_occupancy_map,
    build_route_dicts,
    correct_route_lane_groups,
)

logger = logging.getLogger(__name__)


@dataclass
class PDMClosedInput:
    ego_state_se2: EgoStateSE2
    box_detections_se2: BoxDetectionsSE2
    traffic_light_detections: TrafficLightDetections

    @classmethod
    def from_agent_api(cls, agent_api: AgentAPI) -> PDMClosedInput:
        raise NotImplementedError

    @classmethod
    def from_scene_api(cls, scene_api: SceneAPI) -> PDMClosedInput:
        ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
        box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(0)
        traffic_light_detections = scene_api.get_traffic_light_detections_at_iteration(0)
        assert ego_state_se3 is not None, "Ego state modality not found at iteration."
        assert box_detections_se3 is not None, "Box detections modality not found at iteration."
        assert traffic_light_detections is not None, "Traffic light detections modality not found at iteration."
        return PDMClosedInput(
            ego_state_se2=ego_state_se3.ego_state_se2,
            box_detections_se2=box_detections_se3.box_detections_se2,
            traffic_light_detections=traffic_light_detections,
        )


class PDMClosedPlanner:
    """PDM-Closed planner."""

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
        self._map_api: Optional[MapAPI] = None
        self._route_lane_group_dict: Optional[Dict[int, LaneGroup]] = None
        self._route_lane_dict: Optional[Dict[int, Lane]] = None
        self._centerline: Optional[PolylineSE2] = None
        self._drivable_area_map: Optional[OccupancyMap2D] = None
        self._proposal_manager: Optional[PDMProposalManager] = None

    def initialize(self, map_api: MapAPI, route_lane_group_ids: List[int]) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = map_api
        self._route_lane_group_dict, self._route_lane_dict = build_route_dicts(self._map_api, route_lane_group_ids)
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def compute_planner_trajectory(self, current_input: PDMClosedInput) -> TrajectorySE2:
        """Inherited, see superclass."""
        gc.disable()

        assert self._map_api is not None and self._route_lane_group_dict is not None, (
            "Planner not initialized properly. Call initialize() before compute_planner_trajectory()."
        )

        ego_state_se2 = current_input.ego_state_se2
        # box_detections_se2 = current_input.box_detections_se2

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            assert self._map_api is not None and self._route_lane_group_dict is not None, (
                "Planner not initialized properly."
            )
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
            box_detections_se2=current_input.box_detections_se2,
            traffic_light_detections=current_input.traffic_light_detections,
            route_lane_dict=self._route_lane_dict,
        )

        # TODO: Refactor the rest and re-integrate the following steps:
        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state_se2)

        # 3. Generate/Unroll proposals
        assert self._proposal_manager is not None, "Proposal manager not initialized."
        proposals_array = self._generator.generate_proposals(ego_state_se2, self._observation, self._proposal_manager)

        # # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(proposals_array, ego_state_se2)

        # # 5. Score proposals
        assert self._centerline is not None and self._drivable_area_map is not None, "Planner not initialized properly."
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
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """
        assert self._route_lane_dict is not None and self._drivable_area_map is not None, (
            "Planner not initialized properly."
        )
        current_lane = _get_starting_lane(ego_state_se2, self._route_lane_dict, self._drivable_area_map)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            print("Plan new route!")
            assert self._route_lane_group_dict is not None, "Planner not initialized properly."
            proposal_paths: List[PolylineSE2] = _get_proposal_paths(
                current_lane,
                self._route_lane_group_dict,
                self._route_lane_dict,
                self._lateral_offsets,
                ego_state_se2=ego_state_se2,
            )
            self._centerline = proposal_paths[0]

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        assert self._proposal_manager is not None, "Proposal manager not initialized."
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
