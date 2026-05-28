import abc

from py123d.api import SceneAPI


class BaseMetric(abc.ABC):
    """Abstract base class for scene-level metrics."""

    @abc.abstractmethod
    def compute_metric(self, scene_api: SceneAPI, **kwargs) -> dict:
        """Compute metric for a given scene.

        :param scene_api: SceneAPI object containing scene information and data.
        :return: Dictionary containing metric scores.
        """
