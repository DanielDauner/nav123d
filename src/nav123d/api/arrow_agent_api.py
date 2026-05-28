from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from py123d.api.map.arrow.arrow_map_api import get_map_api_for_log
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Reader
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityReader
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Reader
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader
from py123d.api.scene.arrow.modalities.arrow_sync import get_timestamp_from_arrow_table
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import ArrowTrafficLightDetectionsReader
from py123d.api.scene.arrow.modalities.sync_utils import (
    _get_scene_sync_range,  # noqa: PLC2701
    get_all_modality_timestamps,
    get_modality_index_from_sync_index,
    get_modality_table,
    get_sync_table,
)
from py123d.api.scene.arrow.utils.arrow_scene_caches import _get_complete_log_scene_metadata  # noqa: PLC2701
from py123d.api.utils.arrow_metadata_utils import LogDirectoryMetadata, parse_log_directory_metadata
from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes import (
    BaseModality,
    BaseModalityMetadata,
    LogMetadata,
    MapMetadata,
    ModalityType,
    Timestamp,
    get_modality_key,
)
from py123d.datatypes.metadata import SceneMetadata

from nav123d.api.base_agent_api import AgentAPI, OracleAgentAPI, PlannerAgentAPI, SensorAgentAPI

MODALITY_READERS: Dict[ModalityType, Type[ArrowBaseModalityReader]] = {
    ModalityType.EGO_STATE_SE3: ArrowEgoStateSE3Reader,
    ModalityType.BOX_DETECTIONS_SE3: ArrowBoxDetectionsSE3Reader,
    ModalityType.TRAFFIC_LIGHT_DETECTIONS: ArrowTrafficLightDetectionsReader,
    ModalityType.CAMERA: ArrowCameraReader,
    ModalityType.LIDAR: ArrowLidarReader,
    ModalityType.CUSTOM: ArrowCustomModalityReader,
}

_SENSOR_MODALITIES: FrozenSet[ModalityType] = frozenset(
    {
        ModalityType.EGO_STATE_SE3,
        ModalityType.CAMERA,
        ModalityType.LIDAR,
    }
)
_PLANNER_MODALITIES: FrozenSet[ModalityType] = _SENSOR_MODALITIES | frozenset(
    {
        ModalityType.BOX_DETECTIONS_SE3,
        ModalityType.TRAFFIC_LIGHT_DETECTIONS,
    }
)


class ArrowAgentSceneAPI(AgentAPI):
    """Standalone Arrow-backed :class:`SceneAPI` implementation with access control.

    Does NOT inherit from :class:`ArrowSceneAPI`. ``super()`` in any of these methods reaches
    only :class:`SceneAPI`'s abstract methods (empty body, return ``None``) — eliminating the
    most innocent-looking cheat path. Read pipeline borrowed from py123d's module-level helpers.

    Policy is set at construction and exposed via read-only properties. Disallowed access
    raises :class:`PermissionError`; map / metadata accessors return ``None`` or filtered
    collections when the policy excludes them.
    """

    __slots__ = (
        "_log_dir",
        "_scene_metadata",
        "_allowed_modalities",
        "_allows_future",
        "_allows_map",
    )

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
        *,
        allowed_modalities: Optional[FrozenSet[ModalityType]],
        allows_future: bool,
        allows_map: bool,
    ) -> None:
        """Initializes the :class:`ArrowAgentSceneAPI`.

        :param log_dir: Path to the log directory containing per-modality Arrow files.
        :param scene_metadata: Scene metadata, defaults to None.
        :param allowed_modalities: Set of :class:`ModalityType` the agent may access, or ``None`` for all.
        :param allows_future: Whether the agent may access iterations / timestamps after the current one.
        :param allows_map: Whether :meth:`get_map_api` and :meth:`get_map_metadata` return the underlying map.
        """
        self._log_dir: Path = Path(log_dir)
        self._scene_metadata: Optional[SceneMetadata] = scene_metadata
        self._allowed_modalities = allowed_modalities
        self._allows_future = allows_future
        self._allows_map = allows_map

    def __reduce__(self):
        """Helper for pickling. Reconstructs via the leaf class's ``__init__(log_dir, scene_metadata)``
        and then restores the policy via :meth:`__setstate__`. Pickling the raw ``ArrowAgentSceneAPI``
        (rather than a concrete subclass) is not supported."""
        return (
            self.__class__,
            (self._log_dir, self._scene_metadata),
            {
                "_allowed_modalities": self._allowed_modalities,
                "_allows_future": self._allows_future,
                "_allows_map": self._allows_map,
            },
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore policy attributes after unpickling."""
        self._allowed_modalities = state["_allowed_modalities"]
        self._allows_future = state["_allows_future"]
        self._allows_map = state["_allows_map"]

    # ------------------------------------------------------------------------------------------------------------------
    # Read-only policy views
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def allowed_modalities(self) -> Optional[FrozenSet[ModalityType]]:
        """The set of allowed :class:`ModalityType`, or ``None`` if all are allowed."""
        return self._allowed_modalities

    @property
    def allows_future(self) -> bool:
        """Whether the agent may access future iterations / timestamps."""
        return self._allows_future

    @property
    def allows_map(self) -> bool:
        """Whether the agent may access the map API."""
        return self._allows_map

    # ------------------------------------------------------------------------------------------------------------------
    # Policy checks
    # ------------------------------------------------------------------------------------------------------------------

    def _check_iteration(self, iteration: int) -> None:
        """Raises PermissionError if a future iteration is accessed without future access."""
        if not self._allows_future and iteration > 0:
            raise PermissionError(f"Future iteration {iteration} not allowed for {type(self).__name__}.")

    def _check_modality(self, modality_type: Union[str, ModalityType]) -> None:
        """Raises PermissionError if the modality is outside the agent's allowed set."""
        if self._allowed_modalities is None:
            return
        mt = ModalityType.from_arbitrary(modality_type)
        if mt not in self._allowed_modalities:
            raise PermissionError(
                f"Modality {mt} not in allowed set "
                f"{sorted(m.serialize() for m in self._allowed_modalities)} for {type(self).__name__}."
            )

    def _check_timestamp_not_future(self, timestamp: Union[Timestamp, int]) -> None:
        """Raises PermissionError if the timestamp lies after the current iteration without future access."""
        if self._allows_future:
            return
        current_ts = self._get_timestamp_at_iteration_unchecked(0)
        ts_us = timestamp.time_us if isinstance(timestamp, Timestamp) else int(timestamp)
        if ts_us > current_ts.time_us:
            raise PermissionError(
                f"Future timestamp {ts_us} (current={current_ts.time_us}) not allowed for {type(self).__name__}."
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Internal Arrow helpers (lifted from py123d.api.scene.arrow.arrow_scene_api.ArrowSceneAPI)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_sync_index(self, iteration: int) -> int:
        """Resolve an iteration (which may be negative for history) to an absolute table index."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        metadata = self.get_scene_metadata()
        return metadata.initial_idx + iteration * metadata.target_iteration_stride

    def _get_log_dir_metadatas(self) -> LogDirectoryMetadata:
        """Parses and returns the log directory metadata (modality and log metadata)."""
        return parse_log_directory_metadata(self._log_dir)

    def _get_timestamp_at_iteration_unchecked(self, iteration: int) -> Timestamp:
        """Like :meth:`get_timestamp_at_iteration` but skips the policy check.

        Used by :meth:`_check_timestamp_not_future` to look up the current iteration's timestamp
        without recursing through the policy gate.
        """
        sync_table = get_sync_table(self._log_dir)
        return get_timestamp_from_arrow_table(sync_table, self._get_sync_index(iteration))

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Scene / Log Metadata
    # ------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        if self._scene_metadata is None:
            log_metadata = self.get_log_metadata()
            self._scene_metadata = _get_complete_log_scene_metadata(self._log_dir, log_metadata)
        return self._scene_metadata

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return self._get_log_dir_metadatas().log_metadata

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see superclass. Restricted by agent policy."""
        self._check_iteration(iteration)
        return self._get_timestamp_at_iteration_unchecked(iteration)

    def get_all_iteration_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        """Inherited, see superclass. Future timestamps are filtered when :attr:`allows_future` is ``False``."""
        sync_table = get_sync_table(self._log_dir)
        scene_metadata = self.get_scene_metadata()
        start_idx, end_idx = _get_scene_sync_range(scene_metadata, include_history)
        stride = scene_metadata.target_iteration_stride
        ts_column = sync_table["sync.timestamp_us"].to_numpy()
        ts = [Timestamp.from_us(ts_column[i]) for i in range(start_idx, end_idx, stride)]
        if self._allows_future:
            return ts
        current_us = self._get_timestamp_at_iteration_unchecked(0).time_us
        return [t for t in ts if t.time_us <= current_us]

    def get_scene_timestamp_boundaries(self, include_history: bool = False) -> Tuple[Timestamp, Timestamp]:
        """Inherited, see superclass. Upper bound is clamped to the current iteration's timestamp\
            when :attr:`allows_future` is ``False``."""
        sync_table = get_sync_table(self._log_dir)
        scene_metadata = self.get_scene_metadata()
        start_idx, end_idx = _get_scene_sync_range(scene_metadata, include_history)
        lo = get_timestamp_from_arrow_table(sync_table, start_idx)
        hi = get_timestamp_from_arrow_table(sync_table, end_idx - 1)
        if self._allows_future:
            return (lo, hi)
        current = self._get_timestamp_at_iteration_unchecked(0)
        return (lo, current)

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Map
    # ------------------------------------------------------------------------------------------------------------------

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Inherited, see superclass. Returns ``None`` when map access is not permitted."""
        if not self._allows_map:
            return None
        return self.get_log_metadata().map_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass. Returns ``None`` when map access is not permitted."""
        if not self._allows_map:
            return None
        return get_map_api_for_log(self._log_dir, self.get_log_metadata())

    # ------------------------------------------------------------------------------------------------------------------
    # 3. General modality access
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_modality_metadatas(self) -> Dict[str, BaseModalityMetadata]:
        """Inherited, see superclass. Restricted to allowed modalities."""
        all_md = self._get_log_dir_metadatas().modality_metadatas
        if self._allowed_modalities is None:
            return all_md
        return {k: v for k, v in all_md.items() if v.modality_type in self._allowed_modalities}

    def get_modality_metadata(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
    ) -> Optional[BaseModalityMetadata]:
        """Inherited, see superclass. Restricted by agent policy."""
        self._check_modality(modality_type)
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)
        return self._get_log_dir_metadatas().modality_metadatas.get(_modality_key)

    def get_all_modality_timestamps(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        include_history: bool = False,
    ) -> List[Timestamp]:
        """Inherited, see superclass. Restricted by agent policy.

        Future timestamps are filtered out when :attr:`allows_future` is ``False``.
        """
        self._check_modality(modality_type)
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)
        sync_table = get_sync_table(self._log_dir)
        modality_table = get_modality_table(self._log_dir, _modality_key)
        if modality_table is None:
            return []

        ts_col_name = f"{_modality_key}.timestamp_us"
        if ts_col_name not in modality_table.column_names:
            ts_col_name = next((c for c in modality_table.column_names if c.endswith("timestamp_us")), None)
        if ts_col_name is None:
            return []

        ts = get_all_modality_timestamps(
            self._log_dir,
            sync_table,
            self.get_scene_metadata(),
            _modality_key,
            ts_col_name,
            include_history,
        )
        if self._allows_future:
            return ts
        current_us = self._get_timestamp_at_iteration_unchecked(0).time_us
        return [t for t in ts if t.time_us <= current_us]

    def get_modality_at_iteration(
        self,
        iteration: int,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        **kwargs: Any,
    ) -> Optional[BaseModality]:
        """Inherited, see superclass. Restricted by agent policy."""
        self._check_iteration(iteration)
        self._check_modality(modality_type)
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        sync_table = get_sync_table(self._log_dir)
        sync_index = self._get_sync_index(iteration)

        modality: Optional[BaseModality] = None
        if _modality_key in sync_table.column_names:
            modality_table = get_modality_table(self._log_dir, _modality_key)
            modality_index = get_modality_index_from_sync_index(sync_table, _modality_key, sync_index)
            modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
            if modality_table is not None and modality_index is not None and modality_metadata is not None:
                modality = MODALITY_READERS[_modality_type].read_at_index(
                    index=modality_index,
                    table=modality_table,
                    metadata=modality_metadata,
                    dataset=self.dataset,
                    log_dir=self._log_dir,
                    **kwargs,
                )
        return modality

    def get_modality_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        **kwargs: Any,
    ) -> Optional[BaseModality]:
        """Inherited, see superclass. Restricted by agent policy."""
        self._check_timestamp_not_future(timestamp)
        self._check_modality(modality_type)
        _timestamp = Timestamp.from_us(timestamp) if not isinstance(timestamp, Timestamp) else timestamp
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        modality: Optional[BaseModality] = None
        modality_table = get_modality_table(self._log_dir, _modality_key)
        modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
        if modality_table is not None and modality_metadata is not None:
            modality = MODALITY_READERS[_modality_type].read_at_timestamp(
                timestamp=_timestamp,
                table=modality_table,
                metadata=modality_metadata,
                dataset=self.dataset,
                criteria=criteria,
                log_dir=self._log_dir,
                **kwargs,
            )
        return modality

    def get_modality_between_timestamps(
        self,
        start_timestamp: Union[Timestamp, int],
        end_timestamp: Union[Timestamp, int],
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        inclusive: Literal["left", "right", "both", "neither"] = "left",
        **kwargs: Any,
    ) -> Iterator[BaseModality]:
        """Inherited, see superclass. Restricted by agent policy."""
        self._check_timestamp_not_future(start_timestamp)
        self._check_timestamp_not_future(end_timestamp)
        self._check_modality(modality_type)
        start_us = start_timestamp.time_us if isinstance(start_timestamp, Timestamp) else int(start_timestamp)
        end_us = end_timestamp.time_us if isinstance(end_timestamp, Timestamp) else int(end_timestamp)

        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        modality_table = get_modality_table(self._log_dir, _modality_key)
        modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
        if modality_table is not None and modality_metadata is not None:
            sides: Dict[str, Tuple[Literal["left", "right"], Literal["left", "right"]]] = {
                "left": ("left", "left"),
                "right": ("right", "right"),
                "both": ("left", "right"),
                "neither": ("right", "left"),
            }
            start_side, end_side = sides[inclusive]

            ts_array = modality_table[f"{_modality_key}.timestamp_us"].to_numpy()
            lo = int(np.searchsorted(ts_array, start_us, side=start_side))
            hi = int(np.searchsorted(ts_array, end_us, side=end_side))

            reader = MODALITY_READERS[_modality_type]
            for index in range(lo, hi):
                modality = reader.read_at_index(
                    index=index,
                    table=modality_table,
                    metadata=modality_metadata,
                    dataset=self.dataset,
                    log_dir=self._log_dir,
                    **kwargs,
                )
                if modality is not None:
                    yield modality

    def get_modality_column_at_iteration(
        self,
        iteration: int,
        column: str,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        deserialize: bool = False,
    ) -> Optional[Any]:
        """Arrow-only column accessor (mirror of :meth:`ArrowSceneAPI.get_modality_column_at_iteration`).
        Restricted by agent policy.
        """
        self._check_iteration(iteration)
        self._check_modality(modality_type)
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        sync_table = get_sync_table(self._log_dir)
        sync_index = self._get_sync_index(iteration)

        modality: Optional[BaseModality] = None
        if _modality_key in sync_table.column_names:
            modality_table = get_modality_table(self._log_dir, _modality_key)
            modality_index = get_modality_index_from_sync_index(sync_table, _modality_key, sync_index)
            modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
            if modality_table is not None and modality_index is not None and modality_metadata is not None:
                modality = MODALITY_READERS[_modality_type].read_column_at_index(
                    index=modality_index,
                    table=modality_table,
                    metadata=modality_metadata,
                    column=column,
                    dataset=self.dataset,
                    deserialize=deserialize,
                    log_dir=self._log_dir,
                )
        return modality


# ----------------------------------------------------------------------------------------------------------------------
# Concrete agent APIs with fixed policies
# ----------------------------------------------------------------------------------------------------------------------


class ArrowSensorAgentAPI(SensorAgentAPI, ArrowAgentSceneAPI):
    """Arrow-backed Sensor agent: ego state + camera + lidar at history and current iteration only."""

    __slots__ = ()

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        super().__init__(
            log_dir,
            scene_metadata,
            allowed_modalities=_SENSOR_MODALITIES,
            allows_future=False,
            allows_map=False,
        )


class ArrowPlannerAgentAPI(PlannerAgentAPI, ArrowAgentSceneAPI):
    """Arrow-backed Planner agent: sensor modalities + box / traffic-light detections + map, no future access."""

    __slots__ = ()

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        super().__init__(
            log_dir,
            scene_metadata,
            allowed_modalities=_PLANNER_MODALITIES,
            allows_future=False,
            allows_map=True,
        )


class ArrowOracleAgentAPI(OracleAgentAPI, ArrowAgentSceneAPI):
    """Arrow-backed Oracle agent: unrestricted access to all modalities, map, and future iterations."""

    __slots__ = ()

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        super().__init__(
            log_dir,
            scene_metadata,
            allowed_modalities=None,
            allows_future=True,
            allows_map=True,
        )
