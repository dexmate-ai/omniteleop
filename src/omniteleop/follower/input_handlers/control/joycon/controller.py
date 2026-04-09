"""Centralized JoyCon controller for teleoperation."""

from typing import Dict, Any, Optional
from loguru import logger

from omniteleop.follower.input_handlers.control.controller import AbstractController
from omniteleop.follower.input_handlers.control.joycon.base_controller import (
    BaseController,
)
from omniteleop.follower.input_handlers.control.joycon.torso_controller import (
    TorsoController,
)
from omniteleop.follower.input_handlers.control.joycon.hand_controller import (
    HandController,
)
from omniteleop.follower.input_handlers.control.joycon.head_controller import (
    HeadController,
)
from omniteleop.follower.input_handlers.control.commands import RobotCommands
from omniteleop.follower.input_handlers.utils.button_manager import (
    ButtonManager,
    ButtonEvent,
)

class JoyConController(AbstractController):
    """Centralized interpreter for JoyCon inputs.

    Manages priority and routing of JoyCon inputs to appropriate
    component controllers based on button combinations.

    Toggle-based activation system with priority:
    - Base control: +/- buttons (single) toggle activation (highest priority)
    - Head control: +/- buttons (simultaneous) toggle manual head control
    - Torso control: ZL/ZR buttons toggle activation (medium priority)
    - Fine adjustment: L/R buttons toggle activation (lowest priority)

    Priority order: exit > estop > base > head > torso > hands
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize JoyCon controller.

        Args:
            config: Configuration dictionary with component settings
        """
        super().__init__(config)

        self.config = config or {}

        # Initialize component controllers
        self._init_component_controllers()

        # Initialize button manager with debouncing
        self._init_button_manager()

        # Control state
        self.base_active = False
        self.head_active = False
        self.torso_active = False
        self.fine_adjustment_active = False
        self.estop_active = True  # Start with estop active for safety
        self.exit_requested = False
        self.recording_active = False  # Track recording state
        self._last_recording_state = (
            False  # Track previous recording state for change detection
        )

        # Statistics
        self.stats = {
            "total_updates": 0,
            "base_activations": 0,
            "head_activations": 0,
            "torso_activations": 0,
            "hand_activations": 0,
            "estop_toggles": 0,
            "recording_toggles": 0,  # Add recording toggle stats
        }

        logger.info("JoyCon controller initialized with debounced button management")

    def _init_component_controllers(self) -> None:
        """Initialize all component controllers."""
        # Extract configurations
        base_config = self.config.get("base", {})
        torso_config = self.config.get("torso", {})

        hands_config = self.config.get("hands", {})

        # Determine hand type from ROBOT_CONFIG env var
        import os

        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        if "gripper" in robot_config:
            hand_type = "gripper"
        elif "f5d6" in robot_config:
            hand_type = "hand_f5d6"
        else:
            hand_type = "none"

        # Get left and right configurations directly from hands config (flat structure)
        left_config = hands_config.get("left", {})
        right_config = hands_config.get("right", {})

        # Add type and sensitivity to each config for hand controller
        left_config["type"] = hand_type
        right_config["type"] = hand_type
        left_config["sensitivity"] = hands_config.get("sensitivity", 0.05)
        right_config["sensitivity"] = hands_config.get("sensitivity", 0.05)

        # Get deadzones
        stick_deadzone = self.config.get("stick_deadzone", 0.1)

        # Create controllers
        self.base_controller = BaseController(
            linear_velocity=base_config.get("linear_velocity", 0.5),
            angular_velocity=base_config.get("angular_velocity", 1.0),
            deadzone=stick_deadzone,
        )

        self.torso_controller = TorsoController(
            sensitivity=torso_config.get("sensitivity", 0.01), deadzone=stick_deadzone
        )

        # Get head configuration
        head_config = self.config.get("head", {})
        self.head_controller = HeadController(
            sensitivity=head_config.get("sensitivity", 0.1), deadzone=stick_deadzone
        )

        self.hand_controller = HandController(
            left_config=left_config,
            right_config=right_config,
            stick_deadzone=stick_deadzone,
        )

    def _init_button_manager(self) -> None:
        """Initialize button manager with all buttons and combos."""
        self.button_manager = ButtonManager()

        # Get button timing configs
        button_config = self.config.get("button_timings", {})
        default_debounce = button_config.get(
            "default_debounce", 0.01
        )  # Back to original for responsiveness
        stick_debounce = button_config.get(
            "stick_debounce", 0.03
        )  # Slightly higher for sticks but still responsive

        # Add individual buttons with configurable debounce times
        # Left controller buttons - use slightly higher debounce for sticks
        self.button_manager.add_button("left_stick", debounce_time=stick_debounce)
        self.button_manager.add_button("left_l", debounce_time=default_debounce)
        self.button_manager.add_button("left_zl", debounce_time=default_debounce)
        self.button_manager.add_button("left_minus", debounce_time=default_debounce)
        self.button_manager.add_button(
            "left_capture", debounce_time=default_debounce
        )  # Capture button

        # Right controller buttons - use slightly higher debounce for sticks
        self.button_manager.add_button("right_stick", debounce_time=stick_debounce)
        self.button_manager.add_button("right_r", debounce_time=default_debounce)
        self.button_manager.add_button("right_zr", debounce_time=default_debounce)
        self.button_manager.add_button("right_plus", debounce_time=default_debounce)
        self.button_manager.add_button(
            "right_home", debounce_time=default_debounce
        )  # Home button

        # Add button combos for safety features
        estop_duration = button_config.get("estop_hold_duration", 1.0)
        exit_duration = button_config.get("exit_hold_duration", 1.0)
        combo_grace_period = button_config.get(
            "combo_grace_period", 0.1
        )  # Moderate grace period for combos

        # Emergency stop combo: both sticks for 2 seconds
        self.button_manager.add_combo(
            name="estop_toggle",
            buttons=["left_stick", "right_stick"],
            hold_duration=estop_duration,
            require_simultaneous=True,
            grace_period=combo_grace_period,
            on_triggered=self._on_estop_triggered,
        )

        # Exit combo: capture + home buttons for 1 second
        self.button_manager.add_combo(
            name="exit",
            buttons=["left_capture", "right_home"],
            hold_duration=exit_duration,
            require_simultaneous=True,
            grace_period=combo_grace_period,
            on_triggered=self._on_exit_triggered,
        )

        # Recording combo: L + R buttons for 3 seconds
        recording_duration = button_config.get("recording_hold_duration", 3.0)
        self.button_manager.add_combo(
            name="recording_toggle",
            buttons=["left_l", "right_r"],
            hold_duration=recording_duration,
            require_simultaneous=True,
            grace_period=combo_grace_period,
            on_triggered=self._on_recording_triggered,
        )

        # Head control toggle combo: + and - buttons simultaneously (shorter duration for quick toggle)
        head_toggle_duration = button_config.get("head_toggle_duration", 0.3)
        self.button_manager.add_combo(
            name="head_toggle",
            buttons=["left_minus", "right_plus"],
            hold_duration=head_toggle_duration,
            require_simultaneous=True,
            grace_period=combo_grace_period,
            on_triggered=self._on_head_toggle_triggered,
        )

    def _on_estop_triggered(self) -> None:
        """Handle emergency stop toggle."""
        self.estop_active = not self.estop_active
        self.stats["estop_toggles"] += 1
        state = "ACTIVATED" if self.estop_active else "DEACTIVATED"
        logger.warning(f"Emergency stop {state}")

    def _on_exit_triggered(self) -> None:
        """Handle exit request."""
        self.estop_active = True
        self.exit_requested = True
        logger.critical("EMERGENCY EXIT ACTIVATED - Shutting down all systems")

    def _on_recording_triggered(self) -> None:
        """Handle recording toggle."""
        self.recording_active = not self.recording_active
        self.stats["recording_toggles"] += 1
        state = "STARTED" if self.recording_active else "STOPPED"
        logger.info(f"Recording {state} via JoyCon combo (L + R)")

    def _on_head_toggle_triggered(self) -> None:
        """Handle head control toggle."""
        self._toggle_mode("head")

    def _toggle_mode(self, mode: str) -> None:
        """Toggle a control mode and deactivate others."""
        # Deactivate all modes first
        prev_base = self.base_active
        prev_head = self.head_active
        prev_torso = self.torso_active
        prev_fine = self.fine_adjustment_active

        self.base_active = False
        self.head_active = False
        self.torso_active = False
        self.fine_adjustment_active = False

        # Toggle the requested mode
        if mode == "base":
            self.base_active = not prev_base
            if self.base_active:
                self.stats["base_activations"] += 1
                logger.info("Base control ACTIVATED")
            else:
                logger.info("Base control DEACTIVATED")

        elif mode == "head":
            self.head_active = not prev_head
            if self.head_active:
                self.stats["head_activations"] += 1
                logger.info("Manual head control ACTIVATED")
            else:
                logger.info("Manual head control DEACTIVATED")

        elif mode == "torso":
            self.torso_active = not prev_torso
            if self.torso_active:
                self.stats["torso_activations"] += 1
                logger.info("Torso control ACTIVATED")
            else:
                logger.info("Torso control DEACTIVATED")

        elif mode == "fine":
            self.fine_adjustment_active = not prev_fine
            if self.fine_adjustment_active:
                logger.info("Fine adjustment ACTIVATED")
            else:
                logger.info("Fine adjustment DEACTIVATED")

    def process(self, joycon_data: Dict[str, Any]) -> RobotCommands:
        """Process JoyCon data and return commands for all components.

        Priority is enforced: exit > estop > base > torso > hands

        Args:
            joycon_data: Raw JoyCon data with 'left' and 'right' controller info

        Returns:
            RobotCommands with appropriate component commands
        """
        self.stats["total_updates"] += 1

        # Initialize result
        commands = RobotCommands()

        # Extract button states
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})
        left_buttons = left_data.get("buttons", {})
        right_buttons = right_data.get("buttons", {})

        # Prepare raw button states for button manager
        raw_states = {
            "left_stick": left_buttons.get("stick", False),
            "left_l": left_buttons.get("l", False),
            "left_zl": left_buttons.get("zl", False),
            "left_minus": left_buttons.get(
                "minus", False
            ),  # Minus button for recording combo
            "left_capture": left_buttons.get("capture", False),
            "right_stick": right_buttons.get("stick", False),
            "right_r": right_buttons.get("r", False),
            "right_zr": right_buttons.get("zr", False),
            "right_plus": right_buttons.get(
                "plus", False
            ),  # Plus button for recording combo
            "right_home": right_buttons.get("home", False),
        }

        # Update button manager and get events
        events = self.button_manager.update(raw_states)

        # Debug: log button press events
        for button_name, event in events.items():
            if event == ButtonEvent.PRESSED:
                logger.info(f"Button {button_name} PRESSED")
            elif event == ButtonEvent.RELEASED:
                logger.debug(f"Button {button_name} RELEASED")

        # Check for exit first (highest priority)
        if self.exit_requested:
            commands.estop = True
            commands.exit_requested = True
            commands.priority = "exit"
            return commands

        # Check for active estop
        if self.estop_active:
            commands.estop = True
            commands.priority = "estop"

            # Check if we're currently holding estop combo for progress feedback
            estop_combo = self.button_manager.get_combo("estop_toggle")
            if (
                estop_combo
                and estop_combo.progress is not None
                and estop_combo.progress < 1.0
            ):
                remaining = estop_combo.time_remaining
                if remaining and int((2.0 - remaining) * 10) % 5 == 0:  # Log every 0.5s
                    logger.info(
                        f"Hold both sticks for {remaining:.1f}s more to toggle estop"
                    )

            return commands

        # Process mode toggles based on button events
        # We check for button press events to toggle modes
        # Check if both +/- are pressed (head control combo), if so don't toggle base
        both_plus_minus_pressed = raw_states.get(
            "left_minus", False
        ) and raw_states.get("right_plus", False)

        if not both_plus_minus_pressed:
            if "left_minus" in events and events["left_minus"] == ButtonEvent.PRESSED:
                self._toggle_mode("base")
            elif "right_plus" in events and events["right_plus"] == ButtonEvent.PRESSED:
                self._toggle_mode("base")

        if "left_zl" in events and events["left_zl"] == ButtonEvent.PRESSED:
            self._toggle_mode("torso")
        elif "right_zr" in events and events["right_zr"] == ButtonEvent.PRESSED:
            self._toggle_mode("torso")

        # Check for fine adjustment toggle (but avoid conflict with recording combo)
        # Only toggle fine adjustment if not both L and R are pressed (recording combo)
        left_l_pressed = "left_l" in events and events["left_l"] == ButtonEvent.PRESSED
        right_r_pressed = (
            "right_r" in events and events["right_r"] == ButtonEvent.PRESSED
        )
        both_lr_pressed = raw_states.get("left_l", False) and raw_states.get(
            "right_r", False
        )

        if left_l_pressed and not both_lr_pressed:
            self._toggle_mode("fine")
        elif right_r_pressed and not both_lr_pressed:
            self._toggle_mode("fine")

        # Process active control mode
        if self.base_active:
            commands.base = self.base_controller.process(joycon_data)
            commands.priority = "base"
            return commands

        if self.head_active:
            commands.head = self.head_controller.process(joycon_data)
            commands.priority = "head"
            return commands

        if self.torso_active:
            commands.torso = self.torso_controller.process(joycon_data)
            commands.priority = "torso"
            return commands

        # Hand control (includes fine adjustment mode)
        if self.hand_controller.is_active(joycon_data) or self.fine_adjustment_active:
            joycon_data["fine_adjustment_active"] = self.fine_adjustment_active
            commands.hands = self.hand_controller.process(joycon_data)
            commands.priority = "hands"
            self.stats["hand_activations"] += 1
            return commands

        # No active control
        commands.priority = "none"
        return commands

    def get_control_mode(self, joycon_data: Dict[str, Any]) -> str:
        """Get the current control mode without processing.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            String indicating active control mode: "base", "head", "torso", "hands", or "none"
        """
        if self.base_active:
            return "base"
        elif self.head_active:
            return "head"
        elif self.torso_active:
            return "torso"
        elif self.hand_controller.is_active(joycon_data) or self.fine_adjustment_active:
            return "hands"
        else:
            return "none"

    def get_stats(self) -> Dict[str, int]:
        """Get controller statistics.

        Returns:
            Dictionary with activation counts
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset controller statistics."""
        for key in self.stats:
            self.stats[key] = 0

    def get_hand_controller(self) -> HandController:
        """Get the hand controller for direct access."""
        return self.hand_controller

    def get_torso_controller(self) -> TorsoController:
        """Get the torso controller for direct access."""
        return self.torso_controller

    def get_base_controller(self) -> BaseController:
        """Get the base controller for direct access."""
        return self.base_controller

    def get_head_controller(self) -> HeadController:
        """Get the head controller for direct access."""
        return self.head_controller

    def is_estop_active(self) -> bool:
        """Check if emergency stop is currently active.

        Returns:
            True if estop is active
        """
        return self.estop_active

    def is_exit_requested(self) -> bool:
        """Check if exit has been requested via estop.

        Returns:
            True if exit was requested
        """
        return self.exit_requested

    def reset_estop(self):
        """Reset emergency stop state."""
        self.estop_active = False
        self.exit_requested = False
        logger.info("Emergency stop reset")

    def get_activation_states(self) -> Dict[str, bool]:
        """Get current activation states of all components.

        Returns:
            Dictionary with activation states
        """
        return {
            "base": self.base_active,
            "head": self.head_active,
            "torso": self.torso_active,
            "fine_adjustment": self.fine_adjustment_active,
        }

    def get_recording_command(self) -> Optional[str]:
        """Get the recording command to send based on current state.

        Returns:
            "start" if recording should start, "stop" if recording should stop, None otherwise
        """
        # Only return command on state change
        if self._last_recording_state != self.recording_active:
            self._last_recording_state = self.recording_active
            return "start" if self.recording_active else "stop"

        return None
