from typing import Dict, Tuple

import numpy as np
from py123d.datatypes import BoxDetectionSE2
from py123d.geometry import BoundingBoxSE2Index, Point2D
from py123d.geometry.utils.rotation_utils import normalize_angle

from nav123d.agents.pdm.utils.pdm_constants import DYNAMIC_OBJECT_LABELS

MAX_DYNAMIC_OBJECTS_PER_LABEL: Dict[str, int] = {
    "vehicle": 50,
    "person": 25,
    "two_wheeler": 10,
    "else": 10,
}

MAX_STATIC_OBJECTS: int = 50


class PDMObjectManager:
    """Class that stores and sorts tracked objects around the ego-vehicle."""

    def __init__(
        self,
        max_dynamic_objects_per_label: Dict[str, int] = MAX_DYNAMIC_OBJECTS_PER_LABEL,
        max_static_objects: int = MAX_STATIC_OBJECTS,
    ) -> None:
        """Constructor of PDMObjectManager.

        :param max_dynamic_objects_per_label: cap on tracked dynamic objects kept per label
        :param max_static_objects: cap on tracked static objects kept
        """

        # all objects
        self._unique_objects: Dict[str, BoxDetectionSE2] = {}

        # dynamic objects
        self._max_dynamic_objects_per_label = max_dynamic_objects_per_label
        self._dynamic_object_tokens = {key: [] for key in max_dynamic_objects_per_label.keys()}
        self._dynamic_object_bbse2 = {key: [] for key in max_dynamic_objects_per_label.keys()}
        self._dynamic_object_dxy = {key: [] for key in max_dynamic_objects_per_label.keys()}

        # static objects
        self._max_static_objects = max_static_objects
        self._static_object_tokens = []
        self._static_object_bbse2 = []

    @property
    def unique_objects(self) -> Dict[str, BoxDetectionSE2]:
        """:return: mapping from track token to every object added to the manager."""
        return self._unique_objects

    def add_object(self, box_detection_se2: BoxDetectionSE2) -> None:
        """Adds an object to the manager, sorting it into the dynamic or static category.

        :param box_detection_se2: any tracked object
        """

        bbse2_array = box_detection_se2.bounding_box_se2.array
        default_label = box_detection_se2.attributes.default_label
        track_token = box_detection_se2.attributes.track_token

        self._unique_objects[track_token] = box_detection_se2

        if default_label in DYNAMIC_OBJECT_LABELS:
            assert box_detection_se2.velocity_2d is not None, (
                f"Dynamic object {track_token} has no velocity information!"
            )
            velocity_2d = box_detection_se2.velocity_2d
            velocity_angle = np.arctan2(velocity_2d.y, velocity_2d.x)
            agent_drives_forward = (
                np.abs(normalize_angle(box_detection_se2.center_se2.yaw - velocity_angle)) < np.pi / 2
            )

            track_heading = (
                box_detection_se2.center_se2.yaw
                if agent_drives_forward
                else normalize_angle(box_detection_se2.center_se2.yaw + np.pi)
            )

            dxy = np.array(
                [
                    np.cos(track_heading) * velocity_2d.magnitude,
                    np.sin(track_heading) * velocity_2d.magnitude,
                ],
                dtype=np.float64,
            ).T  # x,y velocity [m/s]

            label = default_label.serialize()
            label = label if label in self._dynamic_object_tokens.keys() else "else"

            self._dynamic_object_tokens[label].append(track_token)
            self._dynamic_object_bbse2[label].append(bbse2_array)
            self._dynamic_object_dxy[label].append(dxy)

        else:
            self._static_object_tokens.append(track_token)
            self._static_object_bbse2.append(bbse2_array)

    def get_nearest_objects(self, position: Point2D) -> Tuple:
        """Retrieves the nearest objects per category, capped per label.

        :param position: global map position
        :return: tuple containing tokens, bbse2, and dynamic information of objects
        """
        dynamic_object_tokens, dynamic_object_bbse2_list, dynamic_object_dxy_list = (
            [],
            [],
            [],
        )

        for dynamic_object_type in self._dynamic_object_tokens.keys():
            (
                dynamic_object_tokens_,
                dynamic_object_bbse2_,
                dynamic_object_dxy_,
            ) = self._get_nearest_dynamic_objects(position, dynamic_object_type)

            if dynamic_object_bbse2_.ndim != 3:
                continue

            dynamic_object_tokens.extend(dynamic_object_tokens_)
            dynamic_object_bbse2_list.append(dynamic_object_bbse2_)
            dynamic_object_dxy_list.append(dynamic_object_dxy_)

        if len(dynamic_object_bbse2_list) > 0:
            dynamic_object_bbse2 = np.concatenate(dynamic_object_bbse2_list, axis=0, dtype=np.float64)
            dynamic_object_dxy = np.concatenate(dynamic_object_dxy_list, axis=0, dtype=np.float64)
        else:
            dynamic_object_bbse2 = np.array([], dtype=np.float64)
            dynamic_object_dxy = np.array([], dtype=np.float64)

        static_object_tokens, static_object_bbse2_array = self._get_nearest_static_objects(position)

        return (
            static_object_tokens,
            static_object_bbse2_array,
            dynamic_object_tokens,
            dynamic_object_bbse2,
            dynamic_object_dxy,
        )

    def _get_nearest_dynamic_objects(self, position: Point2D, label: str) -> Tuple:
        """Retrieves the nearest dynamic objects of the given label.

        :param position: Ego-vehicle position
        :param label: Object label to sort
        :return: Tuple of tokens, bbse2, and velocity of nearest objects.
        """
        position_coords = position.array[None, ...]  # shape: (1,2)

        object_tokens = self._dynamic_object_tokens[label]
        object_bbse2 = np.array(self._dynamic_object_bbse2[label], dtype=np.float64)
        object_dxy = np.array(self._dynamic_object_dxy[label], dtype=np.float64)

        if len(object_tokens) > 0:
            # add axis if single object found
            if object_bbse2.ndim == 1:
                object_bbse2 = object_bbse2[None, ...]
                object_dxy = object_dxy[None, ...]

            position_to_center_dist = ((object_bbse2[..., BoundingBoxSE2Index.XY] - position_coords) ** 2.0).sum(
                axis=-1
            ) ** 0.5

            object_argsort = np.argsort(position_to_center_dist)

            object_tokens = [object_tokens[i] for i in object_argsort][: self._max_dynamic_objects_per_label[label]]
            object_bbse2 = object_bbse2[object_argsort][: self._max_dynamic_objects_per_label[label]]
            object_dxy = object_dxy[object_argsort][: self._max_dynamic_objects_per_label[label]]

        return (object_tokens, object_bbse2, object_dxy)

    def _get_nearest_static_objects(self, position: Point2D) -> Tuple:
        """Retrieves the nearest static obstacles around ego's position.

        :param position: ego's position
        :return: tuple of tokens and coords of nearest objects
        """
        position_coords = position.array[None, ...]  # shape: (1,2)

        object_tokens = self._static_object_tokens
        object_bbse2 = np.array(self._static_object_bbse2, dtype=np.float64)

        if len(object_tokens) > 0:
            # add axis if single object found
            if object_bbse2.ndim == 1:
                object_bbse2 = object_bbse2[None, ...]

            position_to_center_dist = ((object_bbse2[..., BoundingBoxSE2Index.XY] - position_coords) ** 2.0).sum(
                axis=-1
            ) ** 0.5

            object_argsort = np.argsort(position_to_center_dist)

            object_tokens = [object_tokens[i] for i in object_argsort][: self._max_static_objects]
            object_bbse2 = object_bbse2[object_argsort][: self._max_static_objects]

        return (object_tokens, object_bbse2)
