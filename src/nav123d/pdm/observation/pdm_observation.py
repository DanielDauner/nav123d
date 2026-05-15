from typing import Dict, List, Optional, Tuple

import numpy as np
from py123d.api import SceneAPI
from py123d.datatypes import (
    BoxDetectionSE2,
    BoxDetectionsSE2,
    EgoStateSE2,
    Lane,
    TrafficLightDetections,
    TrafficLightStatus,
)
from py123d.geometry import OccupancyMap2D
from py123d.geometry.utils.bounding_box_utils import bbse2_array_to_corners_array, corners_2d_array_to_polygon_array
from shapely.geometry import Polygon

from nav123d.geometry.trajectory import TrajectorySampling
from nav123d.pdm.observation.pdm_object_manager import PDMObjectManager
from nav123d.pdm.observation.pdm_occupancy_map import PDMOccupancyMap


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        map_radius: Optional[float] = 50,
        observation_sample_res: int = 2,
        extend_observation_for_ttc: bool = True,
    ):
        """
        Constructor of PDMObservation
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param map_radius: radius around ego to consider, defaults to 50
        :param observation_sample_res: sample resolution of forecast, defaults to 2
        :param extend_observation_for_ttc: extend observation for TTC metric, defaults to False
        """
        assert trajectory_sampling.interval_length == proposal_sampling.interval_length, (
            "PDMObservation: Proposals and Trajectory must have equal interval length!"
        )

        # observation needs length of trajectory horizon or proposal horizon +1s (for TTC metric)
        self._sample_interval: float = trajectory_sampling.interval_length  # [s]

        if extend_observation_for_ttc:
            self._observation_samples: int = (
                proposal_sampling.num_poses + int(1 / self._sample_interval)
                if proposal_sampling.num_poses + int(1 / self._sample_interval) > trajectory_sampling.num_poses
                else trajectory_sampling.num_poses
            )  # type: ignore
        else:
            self._observation_samples: int = max(trajectory_sampling.num_poses, proposal_sampling.num_poses)  # type: ignore

        self._map_radius: float = map_radius
        self._observation_sample_res: int = observation_sample_res

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: List[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._unique_objects: Optional[Dict[str, BoxDetectionSE2]] = None
        self._occupancy_maps_tl: Optional[List[Tuple[List[str], np.ndarray]]] = None

    def __getitem__(self, time_idx) -> PDMOccupancyMap:
        """
        Retrieves occupancy map for time_idx and adapt temporal resolution.
        :param time_idx: index for future simulation iterations [10Hz]
        :return: occupancy map
        """
        assert len(self._occupancy_maps) > 0, "PDMObservation: Has not been updated yet!"
        assert 0 <= time_idx < len(self._global_to_local_idcs), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    @property
    def collided_track_ids(self) -> List[str]:
        """
        Getter for past collided track tokens.
        :return: list of tokens
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """
        Getter for red light token indicator
        :return: string
        """
        return self._red_light_token

    @property
    def unique_objects(self) -> Dict[str, BoxDetectionSE2]:
        """
        Getter for unique tracked objects
        :return: dictionary of tokens, tracked objects
        """
        assert self._unique_objects is not None, "PDMObservation: Has not been updated yet!"
        return self._unique_objects

    @property
    def box_detections_se2(self) -> BoxDetectionsSE2:
        """
        Getter for detections tracks
        :return: list of detections tracks
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._box_detections_se2

    def update(
        self,
        ego_state_se2: EgoStateSE2,
        box_detections_se2: BoxDetectionsSE2,
        traffic_light_detections: TrafficLightDetections,
        route_lane_dict: Dict[int, Lane],
    ) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state_se2: state of ego vehicle
        :param box_detections_se2: input box detections of nuPlan
        :param traffic_light_detections: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps = []
        object_manager = self._get_object_manager(ego_state_se2, box_detections_se2)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_detections, route_lane_dict)

        (
            static_object_tokens,
            static_object_bbse2,
            dynamic_object_tokens,
            dynamic_object_bbse2,
            dynamic_object_dxy,
        ) = object_manager.get_nearest_objects(ego_state_se2.center_2d)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_bbse2.ndim == 1:
            static_object_bbse2 = static_object_bbse2[None, ...]

        if has_dynamic_object and dynamic_object_bbse2.ndim == 1:
            dynamic_object_bbse2 = dynamic_object_bbse2[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_polygons = corners_2d_array_to_polygon_array(
                bbse2_array_to_corners_array(static_object_bbse2)
            )

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_corners = bbse2_array_to_corners_array(dynamic_object_bbse2)
        else:
            dynamic_object_corners = np.array([], dtype=np.float64)
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_corners_t = dynamic_object_corners + delta_t * dynamic_object_dxy[:, None]
                dynamic_object_polygons = corners_2d_array_to_polygon_array(dynamic_object_corners_t)
            else:
                dynamic_object_polygons = np.array([], dtype=np.object_)

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state_se2.bounding_box_se2.shapely_polygon
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(self._occupancy_maps[0][intersecting_obstacle])
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        # TODO: these are only the current tracks. Other update functions also add future tracks
        self._box_detections_se2 = box_detections_se2
        self._collided_track_ids += new_collided_track_ids
        self._unique_objects = object_manager.unique_objects
        self._initialized = True

    def update_replay(self, scene_api: SceneAPI) -> None:
        # detection_tracks = scenario.get_future_tracked_objects(
        #     iteration_index, self._observation_samples * self._sample_interval
        # )

        # NOTE: @DanielDauner this onwards is where I stopped refactoring 2026-05-15

        occupancy_maps = []
        unique_objects = {}

        for iteration in range(self._observation_samples + self._observation_sample_res):
            box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(iteration)

            occupancy_dict = {}
            unique_objects = {}
            if box_detections_se3 is not None:
                for box_detection_se3 in box_detections_se3:
                    box_detection_se2 = box_detection_se3.box_detection_se2
                    token = box_detection_se2.attributes.track_token
                    polygon = box_detection_se2.shapely_polygon
                    occupancy_dict[token] = polygon

                    if token not in unique_objects.keys():
                        unique_objects[token] = box_detection_se2

            occupancy_maps.append(OccupancyMap2D.from_dict(occupancy_dict))

        occupancy_maps = []
        unique_objects = {}

        # for detection_track in detection_tracks:
        #     tokens, polygons = [], []
        #     for tracked_object in detection_track.tracked_objects:
        #         token, polygon = tracked_object.track_token, tracked_object.box.geometry
        #         tokens.append(token)
        #         polygons.append(polygon)

        #         if token not in unique_objects.keys():
        #             unique_objects[token] = tracked_object

        #     occupancy_map = PDMOccupancyMap(tokens, polygons)
        #     occupancy_maps.append(occupancy_map)

        assert len(occupancy_maps) == self._observation_samples + 1, (
            f"Expected observation length {self._observation_samples + 1}, but got {len(occupancy_maps)}"
        )

        # self._box_detections_se2 = detection_tracks
        self._occupancy_maps = occupancy_maps
        self._collided_track_ids = []
        self._unique_objects = unique_objects
        self._initialized = True

    def _get_object_manager(self, ego_state_se2: EgoStateSE2, box_detections_se2: BoxDetectionsSE2) -> PDMObjectManager:
        """
        Creates object manager class, but adding valid tracked objects.
        :param ego_state_se2: state of ego-vehicle of initial step
        :param box_detections_se2: input box detections of initial step
        :return: PDMObjectManager class
        """
        object_manager = PDMObjectManager()

        for box_detection_se2 in box_detections_se2:
            ego_box_distance = np.linalg.norm(
                ego_state_se2.center_2d.array - box_detection_se2.center_se2.point_2d.array
            )
            if (self._map_radius is not None and ego_box_distance > self._map_radius) or (
                box_detection_se2.attributes.track_token in self._collided_track_ids
            ):
                continue

            object_manager.add_object(box_detection_se2)

        return object_manager

    def _get_traffic_light_geometries(
        self,
        traffic_light_detections: TrafficLightDetections,
        route_lane_dict: Dict[int, Lane],
    ) -> Tuple[List[str], List[Polygon]]:
        """
        Collects red traffic lights along ego's route.
        :param traffic_light_detections: wrapper class for traffic light detections in 123D.
        :param route_lane_dict: dictionary of on-route lanes
        :return: tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens, traffic_light_polygons = [], []

        for data in traffic_light_detections:
            lane_id = int(data.lane_id)

            if (data.status == TrafficLightStatus.RED) and (lane_id in route_lane_dict.keys()):
                lane = route_lane_dict[lane_id]
                traffic_light_tokens.append(f"{self._red_light_token}_{lane_id}")
                traffic_light_polygons.append(lane.shapely_polygon)

        return traffic_light_tokens, traffic_light_polygons
