#!/usr/bin/env python3
"""Real-time telemetry viewer for robot commands using Rerun.

This script subscribes to robot telemetry data and visualizes:
- Raw commands (before filtering)
- Filtered commands (after filtering)
- Actual robot positions
- Exoskeleton joint positions (DYN)
"""

import time
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from loguru import logger

class RerunTelemetryViewer:
    """Real-time telemetry visualization using Rerun."""

    def __init__(
        self,
        namespace: str = "",
        telemetry_topic: str = "robot/telemetry",
        exo_joints_topic: str = "exo/joints",
        recording_id: str = "robot_telemetry",
    ):
        """Initialize telemetry viewer.

        Args:
            namespace: Optional namespace prefix for topics.
            telemetry_topic: Topic to subscribe for telemetry data.
            exo_joints_topic: Topic to subscribe for exo joints data.
            recording_id: Rerun recording ID.
        """
        # Initialize Node
        self.node = Node(name="rerun_telemetry_viewer", namespace=namespace)

        self.telemetry_topic = telemetry_topic
        self.exo_joints_topic = exo_joints_topic
        self.recording_id = recording_id

        # Components and joint counts are inferred dynamically from telemetry
        self.components = []  # populated after first message
        self.component_joints: Dict[str, int] = {}
        self._structure_ready = False

        # Subscriber for telemetry data
        self.subscriber = None
        # Subscriber for exo joints data
        self.exo_joints_subscriber = None

        # Latest exo joint data
        self.latest_exo_joints: Dict[str, Any] = {}

        # Start time for relative timestamps
        self.start_time = time.time()

        # Track connection status
        self.last_data_time = None
        self.frame_count = 0

    def initialize(self):
        """Initialize subscriber and Rerun."""
        # Initialize Rerun
        rr.init(self.recording_id, spawn=True)

        # Log initial structure
        self._setup_rerun_structure()

        # Create subscriber through Node with dexcomm's pickle deserializer
        self.subscriber = self.node.create_subscriber(
            self.telemetry_topic,
            callback=self._on_telemetry,
            decoder=DictDataCodec.decode,
        )

        # Create subscriber for exo joints data
        self.exo_joints_subscriber = self.node.create_subscriber(
            self.exo_joints_topic,
            callback=self._on_exo_joints,
            decoder=DictDataCodec.decode,
        )

    def _setup_rerun_structure(self):
        """Set up the Rerun logging structure with organized views."""
        # Create entity structure
        rr.log("telemetry", rr.Clear(recursive=True))

        # Series keywords and colors centralized
        @dataclass(frozen=True)
        class SeriesKeys:
            RAW: str = "raw_cmd"
            FILTERED: str = "filtered_cmd"
            ROBOT: str = "actual_pos"
            DYN: str = "dyn_pos"
            RAW_VEL: str = "raw_vel"
            FILTERED_VEL: str = "filtered_vel"
            ROBOT_VEL: str = "actual_vel"

        @dataclass(frozen=True)
        class SeriesColors:
            RAW: tuple = (0, 212, 255)
            FILTERED: tuple = (255, 75, 193)
            ROBOT: tuple = (255, 213, 0)
            DYN: tuple = (0, 255, 0)  # Green for DYN
            RAW_VEL: tuple = (100, 180, 255)  # Light blue for raw velocity
            FILTERED_VEL: tuple = (255, 150, 220)  # Light pink for filtered velocity
            ROBOT_VEL: tuple = (255, 240, 100)  # Light yellow for actual velocity
            # RAW: tuple = (31,119,180)
            # FILTERED: tuple = (44,160,44)
            # ROBOT: tuple = (214,39,40)

        self.SERIES_KEYS = SeriesKeys()
        self.SERIES_COLORS = SeriesColors()

        # Defer entity creation until first message (inferred DOF)

        # Log connection status
        rr.log("telemetry/status", rr.TextDocument("Waiting for data..."))

        # Blueprint will be created dynamically after first message
        self._structure_ready = False

    def _setup_blueprint(self):
        """Create a per-joint blueprint: one TimeSeries view per joint for positions and velocities."""
        component_containers = []

        # Build a container per component, with all joint views inside
        for comp in self.components:
            n_joints = self.component_joints.get(comp, 0)
            position_views = []
            velocity_views = []

            for i in range(n_joints):
                joint_origin = "/"

                # Position paths
                raw_path = f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.RAW}"
                filtered_path = (
                    f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.FILTERED}"
                )
                robot_path = f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.ROBOT}"

                # Velocity paths
                raw_vel_path = f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.RAW_VEL}"
                filtered_vel_path = (
                    f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.FILTERED_VEL}"
                )
                robot_vel_path = (
                    f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.ROBOT_VEL}"
                )

                # Add DYN path for left_arm and right_arm (position only)
                pos_contents = [raw_path, filtered_path, robot_path]
                if comp in ["left_arm", "right_arm"]:
                    dyn_path = f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.DYN}"
                    pos_contents.append(dyn_path)

                # Position view
                position_views.append(
                    rrb.TimeSeriesView(
                        name=f"{comp} / joint_{i} / pos",
                        origin=joint_origin,
                        contents=pos_contents,
                    )
                )

                # Velocity view
                vel_contents = [raw_vel_path, filtered_vel_path, robot_vel_path]
                velocity_views.append(
                    rrb.TimeSeriesView(
                        name=f"{comp} / joint_{i} / vel",
                        origin=joint_origin,
                        contents=vel_contents,
                    )
                )

            # Group positions and velocities into separate sub-containers
            component_containers.append(
                rrb.Grid(
                    name=comp,
                    contents=[
                        rrb.Grid(name="Positions", contents=position_views),
                        rrb.Grid(name="Velocities", contents=velocity_views),
                    ],
                )
            )

        # Keep entity panel open for toggling entire components or joints
        blueprint = rrb.Blueprint(*component_containers, collapse_panels=False)

        rr.send_blueprint(blueprint)

    def _infer_component_joints(self, msg: Dict[str, Any]) -> Dict[str, int]:
        """Infer joint counts from telemetry message."""
        inferred: Dict[str, int] = {}

        robot_state = msg.get("robot_state") or {}
        if isinstance(robot_state, dict):
            # Handle new nested structure: robot_state["positions"][component]
            positions = robot_state.get("positions", {})
            if isinstance(positions, dict):
                for comp, arr in positions.items():
                    try:
                        inferred[comp] = int(len(arr))
                    except Exception:
                        pass
            # Also support old flat structure for backward compatibility
            else:
                for comp, arr in robot_state.items():
                    if comp not in ["positions", "velocities"]:
                        try:
                            inferred[comp] = int(len(arr))
                        except Exception:
                            pass

        components = msg.get("components") or {}
        if isinstance(components, dict):
            for comp, data in components.items():
                if comp not in inferred and isinstance(data, dict) and "pos" in data:
                    try:
                        inferred[comp] = int(len(data["pos"]))
                    except Exception:
                        pass

        raw = msg.get("raw_command") or {}
        if isinstance(raw, dict):
            for comp, data in raw.items():
                if comp not in inferred and isinstance(data, dict) and "pos" in data:
                    try:
                        inferred[comp] = int(len(data["pos"]))
                    except Exception:
                        pass

        return inferred

    def _ensure_structure_from_msg(self, msg: Dict[str, Any]) -> None:
        """Initialize/rebuild component structure based on telemetry message."""
        inferred = self._infer_component_joints(msg)
        if not inferred:
            return

        # First-time setup
        if not self._structure_ready:
            self.component_joints = inferred
            self.components = sorted(self.component_joints.keys())

            # Create entities and styling for each joint
            for comp, n_joints in self.component_joints.items():
                for i in range(n_joints):
                    # Position series
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.RAW}",
                        rr.Scalars(0.0),
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.FILTERED}",
                        rr.Scalars(0.0),
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.ROBOT}",
                        rr.Scalars(0.0),
                    )

                    # Position styling
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.RAW}",
                        rr.SeriesLines(colors=list(self.SERIES_COLORS.RAW)),
                        static=True,
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.FILTERED}",
                        rr.SeriesLines(colors=list(self.SERIES_COLORS.FILTERED)),
                        static=True,
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.ROBOT}",
                        rr.SeriesLines(colors=list(self.SERIES_COLORS.ROBOT)),
                        static=True,
                    )

                    # Velocity series
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.RAW_VEL}",
                        rr.Scalars(0.0),
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.FILTERED_VEL}",
                        rr.Scalars(0.0),
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.ROBOT_VEL}",
                        rr.Scalars(0.0),
                    )

                    # Velocity styling
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.RAW_VEL}",
                        rr.SeriesLines(colors=list(self.SERIES_COLORS.RAW_VEL)),
                        static=True,
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.FILTERED_VEL}",
                        rr.SeriesLines(colors=list(self.SERIES_COLORS.FILTERED_VEL)),
                        static=True,
                    )
                    rr.log(
                        f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.ROBOT_VEL}",
                        rr.SeriesLines(colors=list(self.SERIES_COLORS.ROBOT_VEL)),
                        static=True,
                    )

                    # Add DYN series for left_arm and right_arm (position only)
                    if comp in ["left_arm", "right_arm"]:
                        rr.log(
                            f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.DYN}",
                            rr.Scalars(0.0),
                        )
                        rr.log(
                            f"telemetry/{comp}/joint_{i}/{self.SERIES_KEYS.DYN}",
                            rr.SeriesLines(colors=list(self.SERIES_COLORS.DYN)),
                            static=True,
                        )

            # Build blueprint now that we know DOF
            self._setup_blueprint()
            self._structure_ready = True
            return

        # If inferred structure changed, rebuild
        if inferred != self.component_joints:
            self.component_joints = inferred
            self.components = sorted(self.component_joints.keys())
            # Clear and reinitialize base, then rebuild
            rr.log("telemetry", rr.Clear(recursive=True))
            self._setup_rerun_structure()
            self._structure_ready = False
            self._ensure_structure_from_msg(msg)

    def _on_telemetry(self, msg: Dict[str, Any]):
        """Handle incoming telemetry data.

        Args:
            msg: Telemetry message containing timestamp, components, robot_state.
        """
        try:
            # Update connection status
            self.last_data_time = time.time()
            self.frame_count += 1

            # Get timestamp
            if "timestamp_ns" in msg:
                timestamp = msg["timestamp_ns"] / 1e9  # Convert nanoseconds to seconds
            else:
                timestamp = msg.get("timestamp", time.time())

            # Set Rerun time (use wall-clock timestamps for x-axis)
            rr.set_time("time", timestamp=timestamp)

            # Log connection status
            rr.log("telemetry/status", rr.TextDocument("✅ Connected"))

            # Ensure structure is initialized from first message
            self._ensure_structure_from_msg(msg)

            # Process each component
            if self._structure_ready:
                for comp in self.components:
                    self._log_component_data(comp, msg)

            # Log overall metrics
            self._log_metrics(msg)

        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
            rr.log("telemetry/errors", rr.TextDocument(f"Error: {e}"))

    def _on_exo_joints(self, msg: Dict[str, Any]):
        """Handle incoming exo joints data.

        Args:
            msg: Exo joints message with left_arm_pos and right_arm_pos arrays.
        """
        # Store latest exo joints data
        self.latest_exo_joints = msg

    def _log_component_data(self, component: str, msg: Dict[str, Any]):
        """Log data for a specific component.

        Args:
            component: Component name (e.g., "left_arm").
            msg: Telemetry message.
        """
        # Get raw command (before filtering) - positions and velocities
        raw_cmd = None
        raw_vel = None
        if "raw_command" in msg and component in msg["raw_command"]:
            raw_data = msg["raw_command"][component]
            if isinstance(raw_data, dict):
                if "pos" in raw_data:
                    raw_cmd = np.array(raw_data["pos"])
                if "vel" in raw_data:
                    raw_vel = np.array(raw_data["vel"])

        # Get filtered command (after filtering) - positions and velocities
        filtered_cmd = None
        filtered_vel = None
        if "components" in msg and component in msg["components"]:
            filtered_data = msg["components"][component]
            if isinstance(filtered_data, dict):
                if "pos" in filtered_data:
                    filtered_cmd = np.array(filtered_data["pos"])
                if "vel" in filtered_data:
                    filtered_vel = np.array(filtered_data["vel"])

        # Get actual robot position and velocity (handle new nested structure)
        robot_pos = None
        robot_vel = None
        if "robot_state" in msg:
            robot_state = msg["robot_state"]
            # Handle new nested structure
            if isinstance(robot_state, dict):
                if "positions" in robot_state and component in robot_state["positions"]:
                    robot_pos = np.array(robot_state["positions"][component])
                if (
                    "velocities" in robot_state
                    and component in robot_state["velocities"]
                ):
                    robot_vel = np.array(robot_state["velocities"][component])
                # Backward compatibility: old flat structure
                elif component in robot_state and "positions" not in robot_state:
                    robot_pos = np.array(robot_state[component])

        # Log position data using the unified per-joint structure
        if raw_cmd is not None:
            for i, val in enumerate(raw_cmd):
                rr.log(
                    f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.RAW}",
                    rr.Scalars(float(val)),
                )

        if filtered_cmd is not None:
            for i, val in enumerate(filtered_cmd):
                rr.log(
                    f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.FILTERED}",
                    rr.Scalars(float(val)),
                )

        if robot_pos is not None:
            for i, val in enumerate(robot_pos):
                rr.log(
                    f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.ROBOT}",
                    rr.Scalars(float(val)),
                )

        # Log velocity data
        if raw_vel is not None:
            for i, val in enumerate(raw_vel):
                rr.log(
                    f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.RAW_VEL}",
                    rr.Scalars(float(val)),
                )

        if filtered_vel is not None:
            for i, val in enumerate(filtered_vel):
                rr.log(
                    f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.FILTERED_VEL}",
                    rr.Scalars(float(val)),
                )

        if robot_vel is not None:
            for i, val in enumerate(robot_vel):
                rr.log(
                    f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.ROBOT_VEL}",
                    rr.Scalars(float(val)),
                )

        # Log DYN (exo joint) data for left_arm and right_arm (position only)
        if component in ["left_arm", "right_arm"] and self.latest_exo_joints:
            dyn_key = f"{component}_pos"
            if dyn_key in self.latest_exo_joints:
                dyn_pos = np.array(self.latest_exo_joints[dyn_key])
                for i, val in enumerate(dyn_pos):
                    rr.log(
                        f"telemetry/{component}/joint_{i}/{self.SERIES_KEYS.DYN}",
                        rr.Scalars(float(val)),
                    )

        # Log tracking error if both filtered and robot positions exist
        if filtered_cmd is not None and robot_pos is not None:
            error = np.abs(filtered_cmd - robot_pos)
            mean_error = np.mean(error)
            max_error = np.max(error)

            rr.log(
                f"telemetry/{component}/tracking_error", rr.Scalars(float(mean_error))
            )
            rr.log(f"telemetry/{component}/max_error", rr.Scalars(float(max_error)))

    def _log_metrics(self, msg: Dict[str, Any]):
        """Log overall system metrics.

        Args:
            msg: Telemetry message.
        """
        # Count active components
        active_components = 0
        total_joints = 0

        if "components" in msg:
            for comp in self.components:
                if comp in msg["components"]:
                    comp_data = msg["components"][comp]
                    if isinstance(comp_data, dict) and "pos" in comp_data:
                        active_components += 1
                        total_joints += len(comp_data["pos"])

        # Log metrics
        rr.log(
            "telemetry/metrics/active_components", rr.Scalars(float(active_components))
        )
        rr.log("telemetry/metrics/total_joints", rr.Scalars(float(total_joints)))

        # Log summary text
        summary = f"""
        Active Components: {active_components}/{len(self.components)}
        Total Joints: {total_joints}
        Frame: {self.frame_count}
        """
        rr.log("telemetry/summary", rr.TextDocument(summary))

    def run(self):
        """Run the telemetry viewer."""
        self.initialize()
        logger.info("Telemetry viewer started with empty canvas")
        logger.info("📊 Drag components or joints from the left panel to create plots")

        try:
            # Keep running
            while True:
                time.sleep(0.1)

                # Check connection status
                if self.last_data_time:
                    time_since_data = time.time() - self.last_data_time
                    if time_since_data > 2.0:
                        rr.log(
                            "telemetry/status", rr.TextDocument("⚠️ No data (timeout)")
                        )

        except KeyboardInterrupt:
            logger.info("Telemetry viewer stopped by user")
        finally:
            self.node.shutdown()

def main(
    namespace: str = "",
    telemetry_topic: str = "robot/telemetry",
    exo_joints_topic: str = "exo/joints",
    recording_id: str = "robot_telemetry",
):
    """Main entry point.

    Args:
        namespace: Optional namespace prefix for topics.
        telemetry_topic: Topic to subscribe for telemetry data.
        exo_joints_topic: Topic to subscribe for exo joints data.
        recording_id: Rerun recording ID.
    """
    logger.info(
        f"Starting Rerun Telemetry Viewer{f' (namespace: {namespace})' if namespace else ''}"
    )

    viewer = RerunTelemetryViewer(
        namespace=namespace,
        telemetry_topic=telemetry_topic,
        exo_joints_topic=exo_joints_topic,
        recording_id=recording_id,
    )

    try:
        viewer.run()
    except KeyboardInterrupt:
        logger.info("Telemetry viewer stopped by user")
    except Exception as e:
        logger.error(f"Error running telemetry viewer: {e}")

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
