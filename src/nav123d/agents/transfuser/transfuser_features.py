from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import MapAPI, SceneAPI
from py123d.datatypes import BaseMapSurfaceObject, CameraID, DefaultBoxDetectionLabel, EgoStateSE3, LidarID, MapLayer
from py123d.geometry import BoundingBoxSE2Index, Point3DIndex, PoseSE2
from py123d.geometry.transform import abs_to_rel_se2_array
from py123d.geometry.utils.bounding_box_utils import bbse2_array_to_corners_array
from shapely import affinity
from shapely.geometry import LineString, Polygon
from torchvision import transforms

from nav123d.agents.base_torch_agent import BaseFeatureBuilder, BaseTargetBuilder
from nav123d.agents.transfuser.transfuser_config import TransfuserConfig
from nav123d.agents.utils import sample_ego_trajectory_from_api
from nav123d.api.base_agent_api import AgentAPI
from nav123d.geometry.trajectory import TrajectorySampling


class TransfuserFeatureBuilder(BaseFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_api: AgentAPI) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        # 1. Sensors
        features["camera_feature"] = self._get_camera_feature(agent_api)
        if not self._config.latent:
            features["lidar_feature"] = self._get_lidar_feature(agent_api)

        # 2. Status
        features["status_feature"] = self._get_status_feature(agent_api)
        return features

    def _get_camera_feature(self, agent_api: AgentAPI) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_api: input dataclass
        :return: stitched front view image as torch tensor
        """

        # NOTE: Hard-coded for nuplan dataset
        cam_f0 = agent_api.get_camera_at_iteration(0, CameraID.PCAM_F0)
        cam_l0 = agent_api.get_camera_at_iteration(0, CameraID.PCAM_L0)
        cam_r0 = agent_api.get_camera_at_iteration(0, CameraID.PCAM_R0)
        assert cam_f0 is not None and cam_l0 is not None and cam_r0 is not None, (
            "Front and side cameras should be available for feature computation!"
        )

        # Crop to ensure 4:1 aspect ratio
        l0 = cam_l0.image[28:-28, 416:-416]
        f0 = cam_f0.image[28:-28]
        r0 = cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_api: AgentAPI) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_api: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar = agent_api.get_lidar_at_iteration(0, LidarID.LIDAR_MERGED)
        if lidar is None:
            lidar = agent_api.get_lidar_at_iteration(0, LidarID.LIDAR_TOP)

        assert lidar is not None, "LiDAR should be available for feature computation!"
        lidar_xyz = lidar.xyz

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                start=self._config.lidar_min_x,
                stop=self._config.lidar_max_x,
                num=int((self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1),
            )
            ybins = np.linspace(
                start=self._config.lidar_min_y,
                stop=self._config.lidar_max_y,
                num=int((self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1),
            )
            hist = np.histogramdd(point_cloud[:, Point3DIndex.XY], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_xyz = lidar_xyz[lidar_xyz[..., Point3DIndex.Z] < self._config.max_height_lidar]
        below = lidar_xyz[lidar_xyz[..., Point3DIndex.Z] <= self._config.lidar_split_height]
        above = lidar_xyz[lidar_xyz[..., Point3DIndex.Z] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)

    def _get_status_feature(self, agent_api: AgentAPI) -> torch.Tensor:
        """
        Extract ego status and command from AgentAPI
        :param agent_api: input dataclass
        :return: ego status and command as torch tensor
        """
        ego_state_se3 = agent_api.get_ego_state_se3_at_iteration(0)
        assert ego_state_se3 is not None, "Ego state should be available for feature computation!"
        dynamic_state_se3 = ego_state_se3.dynamic_state_se3
        assert dynamic_state_se3 is not None, "Ego dynamic state should be available for feature computation!"

        velocity = torch.tensor(dynamic_state_se3.velocity_2d.array, dtype=torch.float32)
        acceleration = torch.tensor(dynamic_state_se3.acceleration_2d.array, dtype=torch.float32)
        # FIXME: Need to implement driving command. Not available in current API. --- IGNORE ---
        driving_command = torch.zeros(4, dtype=torch.float32)
        driving_command[1] = 1.0  # index 0: left, 1: straight, 2: left, 3: unknown

        status_feature = torch.concatenate(
            [
                torch.tensor(driving_command, dtype=torch.float32),
                torch.tensor(velocity, dtype=torch.float32),
                torch.tensor(acceleration, dtype=torch.float32),
            ],
        )
        return status_feature


class TransfuserTargetBuilder(BaseTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: TransfuserConfig,
    ):
        """
        Initializes target builder.
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config dataclass of TransFuser
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene_api: SceneAPI) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        # 1. Trajectory
        trajectory = self._compute_target_trajectory(scene_api=scene_api)

        ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
        assert ego_state_se3 is not None, "Ego state should be available for target computation!"

        bbse2_array, bbse2_labels = self._extract_relative_bounding_boxes(scene_api, ego_state_se3)
        agent_states, agent_labels = self._compute_agent_targets(bbse2_array, bbse2_labels)
        bev_semantic_map = self._compute_bev_semantic_map(scene_api, ego_state_se3, bbse2_array, bbse2_labels)

        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
        }

    def _compute_target_trajectory(self, scene_api: SceneAPI) -> torch.Tensor:
        """
        Extracts future trajectory in ego coordinates
        :param scene_api: input dataclass
        :return: future trajectory as torch tensor
        """
        resampled_trajectory = sample_ego_trajectory_from_api(
            scene_api=scene_api,
            trajectory_sampling=self._trajectory_sampling,
            in_relative=True,
        )
        return torch.tensor(resampled_trajectory.pose_se2_array, dtype=torch.float32)

    def _compute_agent_targets(
        self, bbse2_array: np.ndarray, bbse2_labels: List[DefaultBoxDetectionLabel]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param bbse2_array: array of bounding box values
        :param bbse2_labels: list of bounding box labels
        :return: tuple of bounding box values and labels (binary)
        """
        assert bbse2_array.shape[-1] == len(BoundingBoxSE2Index), (
            "Bounding box array should have correct number of dimensions!"
        )
        assert len(bbse2_labels) == bbse2_array.shape[0], "Number of labels should match number of bounding boxes!"

        max_agents = self._config.num_bounding_boxes
        config = self._config

        # Mask 1: VEHICLE labels.
        labels_arr = np.fromiter((int(label) for label in bbse2_labels), dtype=np.int64, count=len(bbse2_labels))
        vehicle_mask = labels_arr == int(DefaultBoxDetectionLabel.VEHICLE)

        # Mask 2: box centers inside the LiDAR window.
        xy = bbse2_array[:, BoundingBoxSE2Index.XY]
        in_lidar_mask = (
            (xy[:, 0] >= config.lidar_min_x)
            & (xy[:, 0] <= config.lidar_max_x)
            & (xy[:, 1] >= config.lidar_min_y)
            & (xy[:, 1] <= config.lidar_max_y)
        )

        agents_states_arr = bbse2_array[vehicle_mask & in_lidar_mask].astype(np.float32)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, len(BoundingBoxSE2Index)), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBoxSE2Index.XY], axis=-1)
            # stable sort so equal-distance ties keep input order deterministically
            argsort = np.argsort(distances, kind="stable")[:max_agents]

            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self,
        scene_api: SceneAPI,
        ego_state_se3: EgoStateSE3,
        bbse2_array: np.ndarray,
        bbse2_labels: List[DefaultBoxDetectionLabel],
    ) -> torch.Tensor:
        """
        Computes BEV semantic map with map and agent information
        """

        map_api = scene_api.get_map_api()
        assert map_api is not None, "Map API should be available for target computation!"

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_state_se3, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_state_se3, layers)
            else:
                entity_mask = self._compute_box_mask(bbse2_array, bbse2_labels, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: MapAPI, ego_state_se3: EgoStateSE3, layers: List[MapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_state_se3: ego state in SE3
        :param layers: map layers
        :return: binary mask as numpy array
        """

        ego_pose_se2 = ego_state_se3.rear_axle_se2

        map_object_dict = map_api.get_map_objects_in_radius(
            point=ego_state_se3.rear_axle_2d,
            radius=self._config.bev_radius,
            layers=layers,  # type: ignore
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                assert isinstance(map_object, BaseMapSurfaceObject), "Map object should be polygon type!"
                polygon: Polygon = self._geometry_local_coords(map_object.shapely_polygon, ego_pose_se2)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)  # type: ignore
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: MapAPI, ego_state_se3: EgoStateSE3, layers: List[MapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_state_se3: ego state in SE3
        :param layers: map layers
        :return: binary mask as numpy array
        """
        ego_pose_se2 = ego_state_se3.rear_axle_se2
        map_object_dict = map_api.get_map_objects_in_radius(
            point=ego_state_se3.rear_axle_2d,
            radius=self._config.bev_radius,
            layers=layers,  # type: ignore
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                # assert isinstance(map_object, Lane), "Map object should be polygon type!"
                linestring: LineString = self._geometry_local_coords(map_object.centerline_2d.linestring, ego_pose_se2)  # type: ignore
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(
                    map_linestring_mask,
                    [points],
                    isClosed=False,
                    color=255,  # type: ignore
                    thickness=2,
                )  # type: ignore
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self,
        bbse2_array: npt.NDArray[np.float64],
        bbse2_labels: List[DefaultBoxDetectionLabel],
        layers: List[DefaultBoxDetectionLabel],
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param bbse2_array: array of bounding boxes in SE2 (ego-relative)
        :param bbse2_labels: labels for each bounding box
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        assert bbse2_array.shape[-1] == len(BoundingBoxSE2Index), (
            "Bounding box array should have correct number of dimensions!"
        )
        assert len(bbse2_labels) == bbse2_array.shape[0], "Number of labels should match number of bounding boxes!"

        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        layer_set = set(layers)
        keep = np.fromiter((label in layer_set for label in bbse2_labels), dtype=bool, count=len(bbse2_labels))
        if keep.any():
            # Indexed by [Corners2DIndex, Point2DIndex] → shape (N, 4, 2) in ego frame.
            corners = bbse2_array_to_corners_array(bbse2_array[keep])

            for box_corners in corners:
                exterior = self._coords_to_pixel(box_corners.reshape((-1, 1, 2)))
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)  # type: ignore

        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    def _extract_relative_bounding_boxes(
        self, scene_api: SceneAPI, ego_state_se3: EgoStateSE3
    ) -> Tuple[npt.NDArray[np.float64], List[DefaultBoxDetectionLabel]]:
        """TODO"""

        box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(0)
        assert box_detections_se3 is not None, "Box detections should be available for target computation!"
        box_detections_se2 = box_detections_se3.box_detections_se2

        bbse2_array = np.zeros((len(box_detections_se2), len(BoundingBoxSE2Index)), dtype=np.float64)
        bbse2_labels = []
        for box_idx, box_detection in enumerate(box_detections_se2):
            bbse2_array[box_idx] = box_detection.bounding_box_se2.array
            bbse2_labels.append(box_detection.attributes.default_label)

        bbse2_array[..., BoundingBoxSE2Index.SE2] = abs_to_rel_se2_array(
            origin=ego_state_se3.rear_axle_se2,
            pose_se2_array=bbse2_array[..., BoundingBoxSE2Index.SE2],
        )

        return bbse2_array, bbse2_labels

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: PoseSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.yaw)
        b = np.sin(origin.yaw)
        d = -np.sin(origin.yaw)
        e = np.cos(origin.yaw)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)
