from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from py123d.datatypes import EgoStateSE3Metadata, MapLayer
from py123d.geometry import OccupancyMap2D, PolylineSE2, PoseSE2
from shapely import creation

from nav123d.agents.pdm.observation.pdm_observation import PDMObservation
from nav123d.agents.pdm.scoring.pdm_comfort_metrics import ego_is_comfortable
from nav123d.agents.pdm.scoring.pdm_scorer_utils import (
    CollisionType,
    get_collision_type,
    is_agent_ahead,
    is_agent_behind,
)
from nav123d.agents.pdm.utils.pdm_array_representation import coords_array_to_polygon_array, state_array_to_coords_array
from nav123d.agents.pdm.utils.pdm_constants import DYNAMIC_OBJECT_LABELS
from nav123d.agents.pdm.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from nav123d.geometry.trajectory import TrajectorySampling


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    comfort: float

    pdm_score: float

    @classmethod
    def get_empty_results(cls) -> PDMResults:
        """
        Returns an instance of the class where all values are NaN.
        :return: empty PDM results dataclass.
        """
        return PDMResults(
            no_at_fault_collisions=np.nan,
            drivable_area_compliance=np.nan,
            driving_direction_compliance=np.nan,
            traffic_light_compliance=np.nan,
            ego_progress=np.nan,
            time_to_collision_within_bound=np.nan,
            comfort=np.nan,
            pdm_score=np.nan,
        )


@dataclass
class PDMScorerConfig:
    # weighted metric weights
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    comfort_weight: float = 2.0
    two_frame_extended_comfort_weight: float = 2.0

    # thresholds
    # comfort related config in navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
    driving_direction_horizon: float = 1.0  # [s] (driving direction) (nuplan)
    driving_direction_compliance_threshold: float = 2.0  # [m] (driving direction) (nuplan)
    driving_direction_violation_threshold: float = 6.0  # [m] (driving direction) (nuplan)

    stopped_speed_threshold: float = 5e-03  # [m/s] (ttc)
    future_collision_horizon_window: float = 1.0  # [s] (ttc)
    progress_distance_threshold: float = 5.0  # [m] (progress)
    lane_keeping_deviation_limit: float = 0.5  # [m] (lane keeping) (hydraMDP++)
    lane_keeping_horizon_window: float = 2.0  # [s] (lane keeping) (hydraMDP++)

    # human flag
    human_penalty_filter: Optional[bool] = None

    @property
    def weighted_metrics_array(self) -> npt.NDArray[np.float64]:
        weighted_metrics = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
        weighted_metrics[WeightedMetricIndex.PROGRESS] = self.progress_weight
        weighted_metrics[WeightedMetricIndex.TTC] = self.ttc_weight
        weighted_metrics[WeightedMetricIndex.COMFORT] = self.comfort_weight
        return weighted_metrics


@dataclass
class PDMScorerState:
    observation: PDMObservation
    centerline: PolylineSE2
    route_lane_ids: List[int]
    drivable_area_map: OccupancyMap2D
    num_proposals: int
    ego_metadata: EgoStateSE3Metadata

    states: npt.NDArray[np.float64]
    ego_coords: npt.NDArray[np.float64]
    ego_polygons: npt.NDArray[np.object_]
    ego_areas: npt.NDArray[np.bool_]
    multi_metrics: npt.NDArray[np.float64]
    weighted_metrics: npt.NDArray[np.float64]
    progress_raw: npt.NDArray[np.float64]
    collision_time_idcs: npt.NDArray[np.float64]
    ttc_time_idcs: npt.NDArray[np.float64]

    @classmethod
    def reset(
        cls,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PolylineSE2,
        route_lane_ids: List[int],
        drivable_area_map: OccupancyMap2D,
        ego_metadata: EgoStateSE3Metadata,
    ) -> PDMScorerState:
        assert states.ndim == 3
        assert states.shape[2] == len(StateIndex)

        num_proposals = states.shape[0]
        num_poses = states.shape[1] - 1

        # calculate coordinates of ego corners and center
        ego_coords = state_array_to_coords_array(states, ego_metadata)

        # initialize all ego polygons from corners
        ego_polygons = coords_array_to_polygon_array(ego_coords)

        # zero initialize all remaining arrays.
        ego_areas = np.zeros(
            (
                num_proposals,
                num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        multi_metrics = np.zeros((len(MultiMetricIndex), num_proposals), dtype=np.float64)
        weighted_metrics = np.zeros((len(WeightedMetricIndex), num_proposals), dtype=np.float64)
        progress_raw = np.zeros(num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        collision_time_idcs = np.zeros(num_proposals, dtype=np.float64)
        ttc_time_idcs = np.zeros(num_proposals, dtype=np.float64)
        collision_time_idcs.fill(np.inf)
        ttc_time_idcs.fill(np.inf)

        return PDMScorerState(
            observation=observation,
            centerline=centerline,
            route_lane_ids=route_lane_ids,
            drivable_area_map=drivable_area_map,
            num_proposals=num_proposals,
            ego_metadata=ego_metadata,
            states=states,
            ego_coords=ego_coords,
            ego_polygons=ego_polygons,
            ego_areas=ego_areas,
            multi_metrics=multi_metrics,
            weighted_metrics=weighted_metrics,
            progress_raw=progress_raw,
            collision_time_idcs=collision_time_idcs,
            ttc_time_idcs=ttc_time_idcs,
        )


class PDMScorer:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        config: PDMScorerConfig = PDMScorerConfig(),
    ):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self.proposal_sampling = proposal_sampling
        self._config = config

        # lazy loaded
        self._state: Optional[PDMScorerState] = None

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        assert self._state is not None, "PDMScorer not initialized properly."

        return float(self._state.collision_time_idcs[proposal_idx] * self.proposal_sampling.interval_length)

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        assert self._state is not None, "PDMScorer not initialized properly."
        return float(self._state.ttc_time_idcs[proposal_idx] * self.proposal_sampling.interval_length)

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PolylineSE2,
        route_lane_ids: List[int],
        drivable_area_map: OccupancyMap2D,
        ego_metadata: EgoStateSE3Metadata,
    ) -> List[pd.DataFrame]:
        """
        TODO: Update this docstring
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        :return: A List containing the PDMResult for each proposal
        """

        # initialize & lazy load class values
        self._state = PDMScorerState.reset(
            states=states,
            observation=observation,
            centerline=centerline,
            route_lane_ids=route_lane_ids,
            drivable_area_map=drivable_area_map,
            ego_metadata=ego_metadata,
        )

        # fill value ego-area array (used in multiple metrics)
        self._calculate_ego_area()

        # 1. multiplicative metrics
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        self._calculate_traffic_light_compliance()
        self._calculate_driving_direction_compliance()

        # 2. weighted metrics
        self._calculate_progress()
        self._calculate_ttc()
        self._calculate_comfort()

        pdm_scores = self._aggregate_pdm_scores()
        # multiplicative_metrics_prods, weighted_metrics_all = (
        #     self._state.multi_metrics.prod(axis=0),
        #     self._state.weighted_metrics,
        # )

        results: List[pd.DataFrame] = []
        for proposal_idx in range(self._state.num_proposals):
            no_at_fault_collisions = self._state.multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx]
            drivable_area_compliance = self._state.multi_metrics[MultiMetricIndex.DRIVABLE_AREA, proposal_idx]
            driving_direction_compliance = self._state.multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx]
            traffic_light_compliance = self._state.multi_metrics[
                MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE, proposal_idx
            ]

            ego_progress = self._state.weighted_metrics[WeightedMetricIndex.PROGRESS, proposal_idx]
            time_to_collision_within_bound = self._state.weighted_metrics[WeightedMetricIndex.TTC, proposal_idx]
            comfort = self._state.weighted_metrics[WeightedMetricIndex.COMFORT, proposal_idx]

            # multiplicative_metrics_prod = multiplicative_metrics_prods[proposal_idx]
            # weighted_metrics = weighted_metrics_all[:, proposal_idx]
            pdm_score = pdm_scores[proposal_idx]

            results.append(
                pd.DataFrame(
                    [
                        PDMResults(
                            no_at_fault_collisions=no_at_fault_collisions,
                            drivable_area_compliance=drivable_area_compliance,
                            driving_direction_compliance=driving_direction_compliance,
                            traffic_light_compliance=traffic_light_compliance,
                            ego_progress=ego_progress,
                            time_to_collision_within_bound=time_to_collision_within_bound,
                            comfort=comfort,
                            pdm_score=pdm_score,
                        )
                    ]
                )
            )
        return results

    def _aggregate_pdm_scores(self) -> npt.NDArray[np.float64]:
        """
        Score for PDM proposals, ignoring two-frame extended comfort.
        """
        assert self._state is not None, "PDMScorer not initialized properly."

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._state.multi_metrics.prod(axis=0)

        # normalize and fill progress values
        masked_progress = self._progress_raw * multiplicate_metric_scores
        norm_constant_progress = np.max(masked_progress)
        if norm_constant_progress > self._config.progress_distance_threshold:
            normalized_progress = np.clip(self._progress_raw / norm_constant_progress, 0.0, 1.0)
        else:
            normalized_progress = np.ones(len(masked_progress), dtype=np.float64)
        self._state.weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        weighted_metrics_array = self._config.weighted_metrics_array
        weighted_metric_scores = (self._state.weighted_metrics * weighted_metrics_array[..., None]).sum(axis=0)
        weighted_metric_scores /= weighted_metrics_array.sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    # TODO: Delete once refactoring final
    # def _reset(
    #     self,
    #     states: npt.NDArray[np.float64],
    #     observation: PDMObservation,
    #     centerline: PDMPath,
    #     route_lane_ids: List[str],
    #     drivable_area_map: PDMDrivableMap,
    # ) -> None:
    #     """
    #     Resets metric values and lazy loads input classes.
    #     :param states: array representation of simulated proposals
    #     :param observation: PDM's observation class
    #     :param centerline: path of the centerline
    #     :param route_lane_ids: list containing on-route lane ids
    #     :param drivable_area_map: Occupancy map of drivable are polygons
    #     """
    #     assert states.ndim == 3
    #     assert states.shape[1] == self.proposal_sampling.num_poses + 1
    #     assert states.shape[2] == len(StateIndex)

    #     self._state.observation = observation
    #     self._state.centerline = centerline
    #     self._route_lane_ids = route_lane_ids
    #     self._state.drivable_area_map = drivable_area_map
    #     self._human_past_trajectory = human_past_trajectory

    #     self._state.num_proposals = states.shape[0]

    #     # save ego state values
    #     self._state.states = states

    #     # calculate coordinates of ego corners and center
    #     self._state.ego_coords = state_array_to_coords_array(states, self._metadata)

    #     # initialize all ego polygons from corners
    #     self._state.ego_polygons = coords_array_to_polygon_array(self._state.ego_coords)

    #     # zero initialize all remaining arrays.
    #     self._state.ego_areas = np.zeros(
    #         (
    #             self._state.num_proposals,
    #             self.proposal_sampling.num_poses + 1,
    #             len(EgoAreaIndex),
    #         ),
    #         dtype=np.bool_,
    #     )
    #     self._state.multi_metrics = np.zeros((len(MultiMetricIndex), self._state.num_proposals), dtype=np.float64)
    #     self._state.weighted_metrics = np.zeros((len(WeightedMetricIndex), self._state.num_proposals), dtype=np.float64)
    #     self._progress_raw = np.zeros(self._state.num_proposals, dtype=np.float64)

    #     # initialize infraction arrays with infinity (meaning no infraction occurs)
    #     self._state.collision_time_idcs = np.zeros(self._state.num_proposals, dtype=np.float64)
    #     self._ttc_time_idcs = np.zeros(self._state.num_proposals, dtype=np.float64)
    #     self._state.collision_time_idcs.fill(np.inf)
    #     self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """
        assert self._state is not None, "PDMScorer not initialized properly."

        n_proposals, n_horizon, n_points, _ = self._state.ego_coords.shape

        in_polygons = self._state.drivable_area_map.contains_vectorized(self._state.ego_coords)
        in_polygons = in_polygons.transpose(1, 2, 0, 3)  # shape: n_proposals, n_horizon, n_polygons, n_points

        def _get_indices_of_map_layer(map_layer: MapLayer) -> List[int]:
            assert self._state is not None, "PDMScorer not initialized properly."
            map_object_ids: List[str] = self._state.drivable_area_map.ids  # type: ignore
            indices = []
            for index, map_object_id in enumerate(map_object_ids):
                if map_object_id.startswith(map_layer.serialize()):
                    indices.append(index)
            return indices

        drivable_lane_indices = _get_indices_of_map_layer(MapLayer.LANE)
        intersection_indices = _get_indices_of_map_layer(MapLayer.INTERSECTION)
        drivable_on_route_indices: List[int] = []
        for index, map_object_id in enumerate(self._state.drivable_area_map.ids):  # type: ignore
            assert isinstance(map_object_id, str), f"Expected map_object_id of type str, got {type(map_object_id)}"
            _layer, _id = map_object_id.split("-")
            if _layer == "lane_group" and int(_id) in self._state.route_lane_ids:
                drivable_on_route_indices.append(index)

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (corners_in_polygon[:, :, drivable_lane_indices].sum(axis=-1) > 0).sum(axis=-1) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(corners_in_polygon[:, :, drivable_lane_indices].sum(axis=-1) != 4, axis=-1)

        multiple_lanes_mask = np.logical_and(batch_multiple_lanes_mask, batch_not_single_lanes_mask)
        self._state.ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: if at least one corner is not within any drivable polygon
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_nondrivable_area_mask = (corners_in_polygon.sum(axis=-2) > 0).sum(axis=-1) < 4
        self._state.ego_areas[batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = center_in_polygon[..., drivable_on_route_indices].sum(axis=-1) == 0
        self._state.ego_areas[batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC] = True

        # in_intersection: if center is within any intersection polygon
        batch_intersection_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_intersection_mask = center_in_polygon[..., intersection_indices].sum(axis=-1) > 0
        self._state.ego_areas[batch_intersection_mask, EgoAreaIndex.INTERSECTION] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        assert self._state is not None, "PDMScorer not initialized properly."

        no_at_fault_collision_scores = np.ones(self._state.num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._state.observation.collided_track_ids)
            for proposal_idx in range(self._state.num_proposals)
        }

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            ego_polygons = self._state.ego_polygons[:, time_idx]
            intersecting = self._state.observation[time_idx].query(ego_polygons, predicate="intersects")  # type: ignore

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._state.observation[time_idx].ids[geometry_idx]
                if (self._state.observation.red_light_token in token) or (
                    token in proposal_collided_track_ids[proposal_idx]
                ):
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._state.ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._state.ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                )

                box_detection_se2 = self._state.observation.unique_objects[token]

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._state.states[proposal_idx, time_idx],
                    self._state.ego_polygons[proposal_idx, time_idx],
                    box_detection_se2,
                    self._state.observation[time_idx][token],  # type: ignore
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in {
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                }
                collision_at_lateral: bool = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = (
                        0.0 if box_detection_se2.attributes.default_label in DYNAMIC_OBJECT_LABELS else 0.5
                    )
                    no_at_fault_collision_scores[proposal_idx] = np.minimum(
                        no_at_fault_collision_scores[proposal_idx],
                        no_at_fault_collision_score,
                    )
                    self._state.collision_time_idcs[proposal_idx] = min(
                        time_idx, self._state.collision_time_idcs[proposal_idx]
                    )

                else:  # 2. no at fault collision
                    proposal_collided_track_ids[proposal_idx].append(token)

        self._state.multi_metrics[MultiMetricIndex.NO_COLLISION] = no_at_fault_collision_scores

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """
        assert self._state is not None, "PDMScorer not initialized properly."
        drivable_area_compliance_scores = np.ones(self._state.num_proposals, dtype=np.float64)
        off_road_mask = self._state.ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(axis=-1)
        drivable_area_compliance_scores[off_road_mask] = 0.0
        self._state.multi_metrics[MultiMetricIndex.DRIVABLE_AREA] = drivable_area_compliance_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        assert self._state is not None, "PDMScorer not initialized properly."
        center_coordinates = self._state.ego_coords[:, :, BBCoordsIndex.CENTER]
        oncoming_progress = np.zeros(
            (self._state.num_proposals, self.proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        oncoming_progress[:, 1:] = np.linalg.norm(center_coordinates[:, 1:] - center_coordinates[:, :-1], axis=-1)

        # mask out points that are not in oncoming traffic
        oncoming_traffic_masks = self._state.ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]

        # remove intersection
        for proposal_idx in range(self._state.num_proposals):
            for time_idx in range(self.proposal_sampling.num_poses + 1):
                is_in_intersection = self._state.ego_areas[proposal_idx, time_idx, EgoAreaIndex.INTERSECTION]
                if not oncoming_traffic_masks[proposal_idx, time_idx] or is_in_intersection:
                    oncoming_progress[proposal_idx, time_idx] = 0.0

        # aggregate
        driving_direction_compliance_scores = np.ones(self._state.num_proposals, dtype=np.float64)
        horizon = int(self._config.driving_direction_horizon / self.proposal_sampling.interval_length)

        oncoming_progress_over_horizon = np.concatenate(
            [
                oncoming_progress[:, max(0, time_idx - horizon) : time_idx + 1].sum(axis=-1)[..., None]
                for time_idx in range(oncoming_progress.shape[-1])
            ],
            dtype=np.float64,
            axis=-1,
        )

        for proposal_idx, progress in enumerate(oncoming_progress_over_horizon.max(axis=-1)):
            if progress < self._config.driving_direction_compliance_threshold:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif progress < self._config.driving_direction_violation_threshold:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._state.multi_metrics[MultiMetricIndex.DRIVING_DIRECTION] = driving_direction_compliance_scores

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """
        assert self._state is not None, "PDMScorer not initialized properly."

        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._state.num_proposals, dtype=np.float64)
        for proposal_idx in range(self._state.num_proposals):
            progress_start = self._state.centerline.project(
                self._state.ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER]
            )
            progress_end = self._state.centerline.project(
                self._state.ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER]
            )
            progress_in_meter[proposal_idx] = progress_end - progress_start

        self._progress_raw = np.clip(progress_in_meter, a_min=0, a_max=None)

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """
        assert self._state is not None, "PDMScorer not initialized properly."

        ttc_scores = np.ones(self._state.num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._state.observation.collided_track_ids)
            for proposal_idx in range(self._state.num_proposals)
        }

        # calculate TTC for specific time horizon (default:1s) in the future with less temporal resolution.
        future_time_idcs = np.arange(0, int(self._config.future_collision_horizon_window * 10), 3)
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and specific time horizon (default:1s) future projection
        coords_exterior = self._state.ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
        coords_exterior_time_steps = np.repeat(coords_exterior[:, :, None], n_future_steps, axis=2)

        speeds = np.hypot(
            self._state.states[..., StateIndex.VELOCITY_X],
            self._state.states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._state.states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._state.states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] += dxy_per_s[:, :, None] * delta_t

        polygons = creation.polygons(coords_exterior_time_steps)
        assert isinstance(polygons, np.ndarray), f"Expected polygons to be of type np.ndarray, got {type(polygons)}"

        # ttc needs to look future_time_idcs into the future,
        # so we can only calculate it for n_proposal_steps_to_evaluate steps

        n_proposal_steps_to_evaluate = self.proposal_sampling.num_poses - max(future_time_idcs)
        # check collision for each proposal and projection
        for time_idx in range(n_proposal_steps_to_evaluate + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._state.observation[current_time_idx].query(
                    polygons_at_time_step,  # type: ignore
                    predicate="intersects",
                )
                if len(intersecting) == 0:
                    continue
                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._state.observation[current_time_idx].ids[geometry_idx]
                    if (
                        (self._state.observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._state.ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                        or self._state.ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )
                    ego_rear_axle = PoseSE2.from_array(self._state.states[proposal_idx, time_idx, StateIndex.STATE_SE2])

                    centroid = self._state.observation[current_time_idx][token].centroid
                    track_heading = self._state.observation.unique_objects[token].center_se2.yaw
                    track_state = PoseSE2(centroid.x, centroid.y, track_heading)
                    # TODO: fix ego_area for intersection
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._state.ego_areas[proposal_idx, time_idx, EgoAreaIndex.INTERSECTION]
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_scores[proposal_idx] = np.minimum(ttc_scores[proposal_idx], 0.0)
                        self._state.ttc_time_idcs[proposal_idx] = min(time_idx, self._state.ttc_time_idcs[proposal_idx])
                    else:
                        temp_collided_track_ids[proposal_idx].append(token)

        self._state.weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _calculate_traffic_light_compliance(self) -> None:
        """
        Re-implementation of hydraMDP++'s traffic light compliance metric.
        """
        assert self._state is not None, "PDMScorer not initialized properly."
        # Initialize scores for all proposals to 1 (compliant by default)
        traffic_light_compliance_scores = np.ones(self._state.num_proposals, dtype=np.float64)

        # Iterate over each time step within the horizon
        for time_idx in range(self.proposal_sampling.num_poses + 1):
            # Get ego polygons (vehicle shapes) at the current time step
            ego_polygons = self._state.ego_polygons[:, time_idx]
            # Query objects intersecting with the ego polygons
            intersecting = self._state.observation[time_idx].query(ego_polygons, predicate="intersects")  # type: ignore

            # If no intersections, skip this time step
            if len(intersecting) == 0:
                continue

            # Iterate over each intersecting object
            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                # Skip if the score is already 0
                if traffic_light_compliance_scores[proposal_idx] == 0.0:
                    continue

                token = self._state.observation[time_idx].ids[geometry_idx]

                # Check if the intersecting object is a red light
                if token.startswith(self._state.observation.red_light_token):
                    traffic_light_compliance_scores[proposal_idx] = 0.0

        # Store the scores in the multi-metrics system for later evaluation
        self._state.multi_metrics[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE] = traffic_light_compliance_scores

    def _calculate_comfort(self) -> None:
        """
        Implementation of comfort metric, padded with past history states.
        """
        assert self._state is not None, "PDMScorer not initialized properly."
        time_point_s: npt.NDArray[np.float64] = (
            np.arange(0, self._state.states.shape[1]).astype(np.float64) * self.proposal_sampling.interval_length
        )
        is_comfortable = ego_is_comfortable(
            states=self._state.states, time_point_s=time_point_s, metadata=self._state.ego_metadata
        ).all(axis=-1)
        self._state.weighted_metrics[WeightedMetricIndex.COMFORT] = is_comfortable
