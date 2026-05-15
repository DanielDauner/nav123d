from py123d.api import SceneAPI


class AgentAPI:
    def __init__(self, scene: SceneAPI) -> None:
        self._scene = scene
