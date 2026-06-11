"""Robot-side sensor + action descriptors.

Each descriptor is the *single* place that knows how one logical signal
crosses the robot ↔ spec boundary:

* ``topics`` — the dexdata Spec topic(s) this descriptor produces.
* ``is_present(robot)`` — whether the live hardware can satisfy it.
* ``enable_in_config(robot_configs)`` — set the dexcontrol flags needed
  to wire this sensor up at ``Robot(...)`` construction time. Default
  no-op; cameras override.
* ``collect(robot | action, keep)`` — yield one :class:`Reading` per
  enabled output for this tick.

The recorder no longer enumerates body parts. It iterates the registries
(:data:`STATE_SENSORS`, :data:`ACTION_PRODUCERS`), restricts each to the
topics in the active Spec, and pumps the resulting Readings into the
Writer. Adding a new body part = one new row here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Reading:
    """One signal's value for one tick."""

    topic: str
    value: np.ndarray
    publish_ts_ns: int  # sensor-claimed capture time; 0 for synthetic/action


# ────────────────────────────────────────────────────────────────────────────
# State sensors — collect from the live robot.
# ────────────────────────────────────────────────────────────────────────────


class StateSensor(Protocol):
    @property
    def topics(self) -> frozenset[str]: ...
    def is_present(self, robot: Any) -> bool: ...
    def enable_in_config(self, robot_configs: Any) -> None: ...
    def collect(self, robot: Any, keep: set[str]) -> list[Reading]: ...


@dataclass(frozen=True)
class JointStateSensor:
    """qpos and (optionally) qvel for one dexcontrol joint component."""

    component: str
    qvel: bool = False

    @property
    def topics(self) -> frozenset[str]:
        out = {f"/robot/state/{self.component}/qpos"}
        if self.qvel:
            out.add(f"/robot/state/{self.component}/qvel")
        return frozenset(out)

    def is_present(self, robot: Any) -> bool:
        return robot.has_component(self.component)

    def enable_in_config(self, robot_configs: Any) -> None:
        # Joint components are always wired in dexcontrol; no flag to flip.
        return None

    def collect(self, robot: Any, keep: set[str]) -> list[Reading]:
        part = getattr(robot, self.component)
        ts = int(part.get_timestamp_ns())
        out: list[Reading] = []
        qpos_topic = f"/robot/state/{self.component}/qpos"
        if qpos_topic in keep:
            out.append(
                Reading(qpos_topic, np.asarray(part.get_joint_pos(), dtype=np.float32), ts)
            )
        if self.qvel:
            qvel_topic = f"/robot/state/{self.component}/qvel"
            if qvel_topic in keep:
                out.append(
                    Reading(qvel_topic, np.asarray(part.get_joint_vel(), dtype=np.float32), ts)
                )
        return out


@dataclass(frozen=True)
class WrenchSensor:
    """6-DOF wrench (force + torque) at one arm's wrist FT sensor.

    Source on the robot is under the *arm* (``robot.left_arm.wrench_sensor``),
    but the Spec puts these topics under the *hand* — encoding the physical
    mounting location. This sensor bridges that side/long-name gap in one
    place instead of inline per call site.
    """

    side: str  # "left" or "right"

    @property
    def _arm(self) -> str:
        return f"{self.side}_arm"

    @property
    def _hand(self) -> str:
        return f"{self.side}_hand"

    @property
    def topics(self) -> frozenset[str]:
        return frozenset({
            f"/robot/state/{self._hand}/wrench/force",
            f"/robot/state/{self._hand}/wrench/torque",
        })

    def is_present(self, robot: Any) -> bool:
        if not robot.has_component(self._arm):
            return False
        return getattr(robot, self._arm).wrench_sensor is not None

    def enable_in_config(self, robot_configs: Any) -> None:
        return None

    def collect(self, robot: Any, keep: set[str]) -> list[Reading]:
        sensor = getattr(robot, self._arm).wrench_sensor
        w = np.asarray(sensor.get_wrench_state(), dtype=np.float32)
        ts = int(sensor.get_timestamp_ns())
        out: list[Reading] = []
        force_topic = f"/robot/state/{self._hand}/wrench/force"
        torque_topic = f"/robot/state/{self._hand}/wrench/torque"
        if force_topic in keep:
            out.append(Reading(force_topic, w[:3].copy(), ts))
        if torque_topic in keep:
            out.append(Reading(torque_topic, w[3:].copy(), ts))
        return out


@dataclass(frozen=True)
class HeadCamera:
    """One Zenoh sensor producing up to three modality streams.

    Batches the requested modalities into one ``get_obs`` call so toggling
    on left_rgb + right_rgb + depth costs one network hop, not three.
    Depth is passed through at native resolution and dtype (float32 meters,
    0 = invalid); resizing would corrupt the invalid mask.
    """

    image_resolution: tuple[int, int]  # (width, height), cv2.resize convention

    _MODALITY_TOPIC = {
        "left_rgb":  "/camera/head_left/rgb/video",
        "right_rgb": "/camera/head_right/rgb/video",
        "depth":     "/camera/head_left/depth",
    }

    @property
    def topics(self) -> frozenset[str]:
        return frozenset(self._MODALITY_TOPIC.values())

    def is_present(self, robot: Any) -> bool:
        return getattr(robot.sensors, "head_camera", None) is not None

    def enable_in_config(self, robot_configs: Any) -> None:
        if "head_camera" in robot_configs.sensors:
            robot_configs.sensors["head_camera"].enabled = True

    def collect(self, robot: Any, keep: set[str]) -> list[Reading]:
        wanted = [mod for mod, topic in self._MODALITY_TOPIC.items() if topic in keep]
        if not wanted:
            return []
        obs = robot.sensors.head_camera.get_obs(
            obs_keys=wanted, include_timestamp=True
        )
        out: list[Reading] = []
        for mod, value in obs.items():
            if value is None:
                continue
            topic = self._MODALITY_TOPIC[mod]
            ts = int(value["timestamp_ns"])
            payload = value["data"]
            if mod == "depth":
                out.append(Reading(topic, payload, ts))
            else:
                img = cv2.resize(payload, self.image_resolution)
                out.append(Reading(topic, img, ts))
        return out


@dataclass(frozen=True)
class WristCamera:
    """Single-stream wrist RGB camera."""

    side: str  # "left" or "right"
    image_resolution: tuple[int, int]

    @property
    def _sensor_attr(self) -> str:
        return f"{self.side}_wrist_camera"

    @property
    def topic(self) -> str:
        return f"/camera/{self.side}_wrist/rgb/video"

    @property
    def topics(self) -> frozenset[str]:
        return frozenset({self.topic})

    def is_present(self, robot: Any) -> bool:
        return getattr(robot.sensors, self._sensor_attr, None) is not None

    def enable_in_config(self, robot_configs: Any) -> None:
        if self._sensor_attr in robot_configs.sensors:
            robot_configs.sensors[self._sensor_attr].enabled = True

    def collect(self, robot: Any, keep: set[str]) -> list[Reading]:
        if self.topic not in keep:
            return []
        value = getattr(robot.sensors, self._sensor_attr).get_obs(include_timestamp=True)
        if value is None:
            return []
        img = cv2.resize(value["data"], self.image_resolution)
        return [Reading(self.topic, img, int(value["timestamp_ns"]))]


def state_sensors(image_resolution: tuple[int, int]) -> tuple[StateSensor, ...]:
    """The full state-side registry. Returned as a tuple so each caller's
    list is independent (sensors carry per-recorder config like resolution)."""
    return (
        JointStateSensor("left_arm", qvel=True),
        JointStateSensor("right_arm", qvel=True),
        JointStateSensor("head"),
        JointStateSensor("torso"),
        JointStateSensor("left_hand"),
        JointStateSensor("right_hand"),
        WrenchSensor(side="left"),
        WrenchSensor(side="right"),
        HeadCamera(image_resolution=image_resolution),
        WristCamera(side="left", image_resolution=image_resolution),
        WristCamera(side="right", image_resolution=image_resolution),
    )


# ────────────────────────────────────────────────────────────────────────────
# Action producers — read from the caller-supplied resolved-action dict.
# ────────────────────────────────────────────────────────────────────────────


class ActionProducer(Protocol):
    @property
    def topics(self) -> frozenset[str]: ...
    def collect(self, action: dict[str, Any], keep: set[str]) -> list[Reading]: ...


@dataclass(frozen=True)
class JointActionProducer:
    """Reads ``action[component]['pos']`` → ``/robot/action/{component}/qpos``."""

    component: str

    @property
    def topic(self) -> str:
        return f"/robot/action/{self.component}/qpos"

    @property
    def topics(self) -> frozenset[str]:
        return frozenset({self.topic})

    def collect(self, action: dict[str, Any], keep: set[str]) -> list[Reading]:
        if self.topic not in keep:
            return []
        data = action.get(self.component)
        if not isinstance(data, dict) or "pos" not in data:
            return []
        return [Reading(self.topic, np.asarray(data["pos"], dtype=np.float32), 0)]


@dataclass(frozen=True)
class ChassisActionProducer:
    """Reads ``action['chassis']`` (``{vx, vy, wz}`` dict) → planar twist."""

    topic: str = "/chassis/action/velocity"

    @property
    def topics(self) -> frozenset[str]:
        return frozenset({self.topic})

    def collect(self, action: dict[str, Any], keep: set[str]) -> list[Reading]:
        if self.topic not in keep:
            return []
        chassis = action.get("chassis")
        if not chassis:
            return []
        value = np.asarray(
            [
                float(chassis.get("vx", 0.0)),
                float(chassis.get("vy", 0.0)),
                float(chassis.get("wz", 0.0)),
            ],
            dtype=np.float32,
        )
        return [Reading(self.topic, value, 0)]


ACTION_PRODUCERS: tuple[ActionProducer, ...] = (
    JointActionProducer("left_arm"),
    JointActionProducer("right_arm"),
    JointActionProducer("head"),
    JointActionProducer("torso"),
    JointActionProducer("left_hand"),
    JointActionProducer("right_hand"),
    ChassisActionProducer(),
)
