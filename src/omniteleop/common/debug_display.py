#!/usr/bin/env python3
"""Efficient debug display using rich for real-time teleoperation data."""

import sys
from typing import Dict, Any, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED
import time

# Global default precision - change this to affect all debug output
DEFAULT_PRECISION = 2  # Number of decimal places for float values

class DebugDisplay:
    """Flicker-free debug display using rich Live updates.

    Features:
    - Clean table formatting with borders
    - Smooth in-place updates using rich.Live
    - Rate-limited updates to reduce CPU usage
    - Configurable float precision
    """

    def __init__(
        self,
        name: str,
        rate: int = 40,
        refresh_rate: float = 10.0,
        precision: Optional[int] = None,
    ):
        """Initialize debug display.

        Args:
            name: Process name for display
            rate: Data rate in Hz (shown in the table title)
            refresh_rate: Display refresh rate in Hz (how often to update the table)
            precision: Number of decimal places for float values (uses DEFAULT_PRECISION if None)
        """
        self.name = name
        self.rate = rate
        self.refresh_rate = min(refresh_rate, 20.0)  # Cap at 20Hz to prevent flashing
        self.precision = precision if precision is not None else DEFAULT_PRECISION
        self.precision = int(self.precision)

        # Format strings based on precision
        self.float_fmt = f"{{:7.{self.precision}f}}"
        self.float_fmt_sign = (
            f"{{:+6.{self.precision}f}}"  # For signed values like joystick
        )

        # Console for output
        self.console = Console(
            force_terminal=True,
            force_interactive=False,
            no_color=False,
            highlight=False,  # Disable syntax highlighting for speed
            log_time=False,
            log_path=False,
            soft_wrap=False,
        )

        # Track if we've printed before (for cursor positioning)
        self._first_print = True
        self._last_line_count = 0

        # Rate limiting based on refresh_rate
        self._last_print = 0
        self._print_interval = 1.0 / self.refresh_rate  # Convert Hz to seconds

    def start(self):
        """Start the display (compatibility method)."""
        # For compatibility - no longer needed with direct printing
        pass

    def stop(self):
        """Stop the display (compatibility method)."""
        # For compatibility - no longer needed with direct printing
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def _clear_previous_output(self):
        """Clear previous output by moving cursor up and clearing lines."""
        if not self._first_print and self._last_line_count > 0:
            # Move cursor up and clear lines
            for _ in range(self._last_line_count):
                sys.stdout.write("\033[1A")  # Move up one line
                sys.stdout.write("\033[2K")  # Clear entire line
        self._first_print = False

    def should_print(self) -> bool:
        """Check if enough time has passed for next print."""
        now = time.perf_counter()
        if now - self._last_print >= self._print_interval:
            self._last_print = now
            return True
        return False

    def print_leader_arm(self, joints: Dict[str, float]):
        """Print leader arm data in a structured table."""
        if not self.should_print():
            return

        # Clear previous output
        self._clear_previous_output()

        # Create table with borders
        table = Table(
            title=f"[bold cyan]{self.name}[/bold cyan] @ {self.rate}Hz",
            box=ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        # Group by arm
        left_joints = {k: v for k, v in joints.items() if k.startswith("L_")}
        right_joints = {k: v for k, v in joints.items() if k.startswith("R_")}

        # Add columns
        table.add_column("Arm", style="yellow", width=8)
        table.add_column("Joint", style="cyan", width=12)
        table.add_column("Value", style="green", justify="right", width=10)

        # Add left arm joints
        for i, (joint, value) in enumerate(left_joints.items()):
            arm_label = "Left" if i == 0 else ""
            table.add_row(arm_label, joint, self.float_fmt.format(value))

        # Add separator if both arms present
        if left_joints and right_joints:
            table.add_row("", "---", "---")

        # Add right arm joints
        for i, (joint, value) in enumerate(right_joints.items()):
            arm_label = "Right" if i == 0 else ""
            table.add_row(arm_label, joint, self.float_fmt.format(value))

        # Print table and track line count
        with self.console.capture() as capture:
            self.console.print(table)
        output = capture.get()
        self._last_line_count = output.count("\n")
        sys.stdout.write(output)
        sys.stdout.flush()

    def print_joycon(self, data: Dict[str, Any]):
        """Print JoyCon data in a structured table."""
        if not self.should_print():
            return

        # Clear previous output
        self._clear_previous_output()

        ls = data["left"]["stick"]
        rs = data["right"]["stick"]

        # Create table with borders
        table = Table(
            title=f"[bold cyan]{self.name}[/bold cyan] @ {self.rate}Hz",
            box=ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Controller", style="yellow", width=12)
        table.add_column("Input", style="cyan", width=15)
        table.add_column("Value", style="green", width=30)

        # Left controller - stick then buttons
        table.add_row(
            "Left",
            "Stick (X, Y)",
            f"({self.float_fmt_sign.format(ls['x'])}, {self.float_fmt_sign.format(ls['y'])})",
        )

        lb = [k for k, v in data["left"]["buttons"].items() if v]
        table.add_row("", "Buttons", ", ".join(lb) if lb else "None")

        # Add separator row
        table.add_row("", "---", "---")

        # Right controller - stick then buttons
        table.add_row(
            "Right",
            "Stick (X, Y)",
            f"({self.float_fmt_sign.format(rs['x'])}, {self.float_fmt_sign.format(rs['y'])})",
        )

        rb = [k for k, v in data["right"]["buttons"].items() if v]
        table.add_row("", "Buttons", ", ".join(rb) if rb else "None")

        # Print table and track line count
        with self.console.capture() as capture:
            self.console.print(table)
        output = capture.get()
        self._last_line_count = output.count("\n")
        sys.stdout.write(output)
        sys.stdout.flush()

    def print_vr(self, data: Dict[str, Any]):
        """Print VR PoseData in a structured table."""
        if not self.should_print():
            return

        # Clear previous output
        self._clear_previous_output()

        # Create table with borders
        table = Table(
            title=f"[bold cyan]{self.name}[/bold cyan] @ {self.rate}Hz",
            box=ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Controller", style="yellow", width=12)
        table.add_column("Input", style="cyan", width=15)
        table.add_column("Value", style="green", width=35)

        # Left controller data
        left = data.get("left", {})

        # Thumbstick
        thumbstick = left.get("thumbstick", [0.0, 0.0])
        table.add_row(
            "Left",
            "Thumbstick (X,Y)",
            f"({self.float_fmt_sign.format(thumbstick['x'])}, {self.float_fmt_sign.format(thumbstick['y'])})",
        )

        # Triggers
        index_trigger = left.get("index_trigger", 0.0)
        grip_trigger = left.get("grip_trigger", 0.0)
        table.add_row("", "Index Trigger", f"{self.float_fmt.format(index_trigger)}")
        table.add_row("", "Gripper Trigger", f"{self.float_fmt.format(grip_trigger)}")

        # Thumbstick click
        thumbstick_click = left.get("thumbstick_click", False)
        table.add_row("", "Thumbstick Click", "Yes" if thumbstick_click else "No")

        # Wrist pose (position only for brevity)
        wrist = left.get("wrist", None)
        if wrist is not None:
            # Handle both numpy array and list formats
            if hasattr(wrist, "shape") and len(wrist.shape) == 2:
                pos = wrist[:3, 3]  # Extract position from numpy transformation matrix
            elif isinstance(wrist, list) and len(wrist) == 4 and len(wrist[0]) == 4:
                pos = [
                    wrist[0][3],
                    wrist[1][3],
                    wrist[2][3],
                ]  # Extract position from list matrix
            else:
                pos = None

            if pos is not None:
                table.add_row(
                    "",
                    "Wrist Position",
                    f"({self.float_fmt_sign.format(pos[0])}, {self.float_fmt_sign.format(pos[1])}, {self.float_fmt_sign.format(pos[2])})",
                )

        # Add separator row
        table.add_row("", "---", "---")

        # Right controller data
        right = data.get("right", {})

        # Thumbstick
        thumbstick = right.get("thumbstick", [0.0, 0.0])
        table.add_row(
            "Right",
            "Thumbstick (X,Y)",
            f"({self.float_fmt_sign.format(thumbstick['x'])}, {self.float_fmt_sign.format(thumbstick['y'])})",
        )

        # Triggers
        index_trigger = right.get("index_trigger", 0.0)
        grip_trigger = right.get("grip_trigger", 0.0)
        table.add_row("", "Index Trigger", f"{self.float_fmt.format(index_trigger)}")
        table.add_row("", "Gripper Trigger", f"{self.float_fmt.format(grip_trigger)}")

        # Thumbstick click
        thumbstick_click = right.get("thumbstick_click", False)
        table.add_row("", "Thumbstick Click", "Yes" if thumbstick_click else "No")

        # Wrist pose (position only for brevity)
        wrist = right.get("wrist", None)
        if wrist is not None:
            # Handle both numpy array and list formats
            if hasattr(wrist, "shape") and len(wrist.shape) == 2:
                pos = wrist[:3, 3]  # Extract position from numpy transformation matrix
            elif isinstance(wrist, list) and len(wrist) == 4 and len(wrist[0]) == 4:
                pos = [
                    wrist[0][3],
                    wrist[1][3],
                    wrist[2][3],
                ]  # Extract position from list matrix
            else:
                pos = None

            if pos is not None:
                table.add_row(
                    "",
                    "Wrist Position",
                    f"({self.float_fmt_sign.format(pos[0])}, {self.float_fmt_sign.format(pos[1])}, {self.float_fmt_sign.format(pos[2])})",
                )

        # Print table and track line count
        with self.console.capture() as capture:
            self.console.print(table)
        output = capture.get()
        self._last_line_count = output.count("\n")
        sys.stdout.write(output)
        sys.stdout.flush()

    def print_robot_command(
        self,
        components: Dict[str, Dict[str, np.ndarray]],
        safety_flags: Optional[Dict] = None,
    ):
        """Print robot command data in a structured table."""
        if not self.should_print():
            return

        # Clear previous output
        self._clear_previous_output()

        # Create table with borders
        table = Table(
            title=f"[bold cyan]{self.name}[/bold cyan] @ {self.rate}Hz",
            box=ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Component", style="yellow", width=15)
        table.add_column("Type", style="cyan", width=8)
        table.add_column("Values", style="green")

        # Print each component's array
        for comp_name, comp_data in components.items():
            if "pos" in comp_data:
                pos = comp_data["pos"]
                # Format array efficiently with configurable precision
                if len(pos) <= 10:
                    vals_str = " ".join(self.float_fmt.format(v) for v in pos)
                else:
                    # Show first 7 and last 3 for arrays > 10 elements
                    vals_str = " ".join(self.float_fmt.format(v) for v in pos[:7])
                    vals_str += " ... "
                    vals_str += " ".join(self.float_fmt.format(v) for v in pos[-3:])

                table.add_row(comp_name, "pos", f"[{vals_str}]")

                # Add velocity if present and non-zero
                if "vel" in comp_data:
                    vel = comp_data["vel"]
                    if np.any(vel != 0):
                        # Same truncation logic for velocity
                        if len(vel) <= 10:
                            vel_str = " ".join(self.float_fmt.format(v) for v in vel)
                        else:
                            vel_str = " ".join(
                                self.float_fmt.format(v) for v in vel[:7]
                            )
                            vel_str += " ... "
                            vel_str += " ".join(
                                self.float_fmt.format(v) for v in vel[-3:]
                            )

                        table.add_row("", "vel", f"[{vel_str}]")

            # for base controller, print vx, vy, wz
            if comp_name == "chassis" and "vx" in comp_data:
                table.add_row(
                    comp_name,
                    "vx,vy,wz",
                    f"[{self.float_fmt.format(comp_data['vx'])} {self.float_fmt.format(comp_data['vy'])} {self.float_fmt.format(comp_data['wz'])}]",
                )

        # Safety flags if any are active
        if safety_flags:
            active = [k for k, v in safety_flags.items() if v]
            if active:
                table.add_row(
                    "Safety", "Flags", Text(", ".join(active), style="bold red")
                )

        # Print table and track line count
        with self.console.capture() as capture:
            self.console.print(table)
        output = capture.get()
        self._last_line_count = output.count("\n")
        sys.stdout.write(output)
        sys.stdout.flush()

# Singleton instance for each process
_debug_display: Optional[DebugDisplay] = None

def get_debug_display(
    name: str,
    rate: int = 40,
    refresh_rate: float = 10.0,
    precision: Optional[int] = None,
) -> DebugDisplay:
    """Get or create debug display singleton.

    Args:
        name: Process name for display
        rate: Data rate in Hz (shown in the table title)
        refresh_rate: Display refresh rate in Hz (how often to update the table)
        precision: Number of decimal places for float values (uses DEFAULT_PRECISION if None)
    """
    global _debug_display
    if _debug_display is None:
        _debug_display = DebugDisplay(name, rate, refresh_rate, precision=precision)
    return _debug_display
