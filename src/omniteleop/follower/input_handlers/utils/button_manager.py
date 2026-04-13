"""Button debouncing and state management for JoyCon controller."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Callable
from loguru import logger

class ButtonEvent(Enum):
    """Button event types."""

    NONE = auto()
    PRESSED = auto()  # Button just pressed (rising edge)
    RELEASED = auto()  # Button just released (falling edge)
    HELD = auto()  # Button held for required duration
    HOLDING = auto()  # Button being held but not yet triggered

@dataclass
class DebouncedButton:
    """
    Debounced button with configurable hold time.

    Handles button debouncing, edge detection, and timed holds.
    """

    name: str
    hold_duration: float = 0.0  # Time required to register as "held" (0 = instant)
    debounce_time: float = 0.05  # Debounce time to filter noise
    on_held: Optional[Callable[[], None]] = None  # Callback when held long enough
    on_pressed: Optional[Callable[[], None]] = None  # Callback on press
    on_released: Optional[Callable[[], None]] = None  # Callback on release

    # Internal state
    _raw_state: bool = field(default=False, init=False)
    _debounced_state: bool = field(default=False, init=False)
    _previous_state: bool = field(default=False, init=False)
    _state_change_time: Optional[float] = field(default=None, init=False)
    _hold_start_time: Optional[float] = field(default=None, init=False)
    _hold_triggered: bool = field(default=False, init=False)
    _last_event: ButtonEvent = field(default=ButtonEvent.NONE, init=False)

    def update(self, raw_state: bool) -> ButtonEvent:
        """
        Update button state and return current event.

        Args:
            raw_state: Raw button state from hardware

        Returns:
            ButtonEvent indicating what happened
        """
        current_time = time.time()
        self._raw_state = raw_state

        # Handle debouncing
        if raw_state != self._debounced_state:
            if self._state_change_time is None:
                self._state_change_time = current_time
            elif current_time - self._state_change_time >= self.debounce_time:
                # State has been stable for debounce time
                self._previous_state = self._debounced_state
                self._debounced_state = raw_state
                self._state_change_time = None

                # Handle state transitions
                if self._debounced_state and not self._previous_state:
                    # Rising edge - button pressed
                    self._hold_start_time = current_time
                    self._hold_triggered = False
                    self._last_event = ButtonEvent.PRESSED

                    if self.on_pressed:
                        self.on_pressed()

                    # Always return PRESSED on rising edge
                    # HELD will be triggered on subsequent updates if hold_duration > 0
                    return ButtonEvent.PRESSED

                elif not self._debounced_state and self._previous_state:
                    # Falling edge - button released
                    hold_time = (
                        current_time - self._hold_start_time
                        if self._hold_start_time
                        else 0
                    )
                    self._hold_start_time = None
                    self._hold_triggered = False
                    self._last_event = ButtonEvent.RELEASED

                    if self.on_released:
                        self.on_released()

                    if hold_time > 0 and hold_time < self.hold_duration:
                        logger.debug(
                            f"{self.name} released after {hold_time:.2f}s (needed {self.hold_duration}s)"
                        )

                    return ButtonEvent.RELEASED
        else:
            # Reset state change time if state matches
            self._state_change_time = None

        # Check for held state (only if hold_duration > 0)
        if (
            self._debounced_state
            and self._hold_start_time
            and not self._hold_triggered
            and self.hold_duration > 0
        ):
            hold_time = current_time - self._hold_start_time

            if hold_time >= self.hold_duration:
                # Button held for required duration
                self._hold_triggered = True
                self._last_event = ButtonEvent.HELD

                if self.on_held:
                    self.on_held()

                logger.info(f"{self.name} held for {hold_time:.1f}s - triggered")
                return ButtonEvent.HELD
            else:
                # Still holding but not triggered yet
                self._last_event = ButtonEvent.HOLDING
                return ButtonEvent.HOLDING

        # No event
        return ButtonEvent.NONE

    @property
    def is_pressed(self) -> bool:
        """Check if button is currently pressed (debounced)."""
        return self._debounced_state

    @property
    def is_held(self) -> bool:
        """Check if button has been held for required duration."""
        return self._hold_triggered and self._debounced_state

    @property
    def hold_progress(self) -> Optional[float]:
        """
        Get hold progress as a fraction (0.0 to 1.0).

        Returns None if not being held.
        """
        if not self._debounced_state or not self._hold_start_time:
            return None

        if self.hold_duration == 0:
            return 1.0

        hold_time = time.time() - self._hold_start_time
        return min(hold_time / self.hold_duration, 1.0)

    @property
    def hold_time_remaining(self) -> Optional[float]:
        """Get time remaining until hold triggers."""
        if (
            not self._debounced_state
            or not self._hold_start_time
            or self._hold_triggered
        ):
            return None

        hold_time = time.time() - self._hold_start_time
        return max(self.hold_duration - hold_time, 0.0)

    def reset(self) -> None:
        """Reset button to initial state."""
        self._raw_state = False
        self._debounced_state = False
        self._previous_state = False
        self._state_change_time = None
        self._hold_start_time = None
        self._hold_triggered = False
        self._last_event = ButtonEvent.NONE

@dataclass
class ButtonCombo:
    """
    Handle multi-button combinations with timing.

    Useful for safety features that require multiple buttons.
    """

    name: str
    buttons: list[str]  # List of button names that must be pressed
    hold_duration: float = 0.0  # Time all buttons must be held together
    require_simultaneous: bool = True  # If True, all must be pressed within window
    simultaneous_window: float = 0.5  # Time window for "simultaneous" presses
    grace_period: float = 0.1  # Grace period for brief button release (to handle noise)
    on_triggered: Optional[Callable[[], None]] = None

    # Internal state
    _combo_start_time: Optional[float] = field(default=None, init=False)
    _combo_triggered: bool = field(default=False, init=False)
    _first_press_time: Optional[float] = field(default=None, init=False)
    _last_all_pressed_time: Optional[float] = field(default=None, init=False)

    def update(self, button_states: Dict[str, bool]) -> ButtonEvent:
        """
        Update combo state based on button states.

        Args:
            button_states: Dictionary of button name to pressed state

        Returns:
            ButtonEvent indicating combo state
        """
        current_time = time.time()

        # Check if all required buttons are pressed
        all_pressed = all(button_states.get(btn, False) for btn in self.buttons)

        if all_pressed:
            self._last_all_pressed_time = current_time

            # Only check simultaneous requirement if combo not started yet
            if self._combo_start_time is None and self.require_simultaneous:
                if self._first_press_time is None:
                    self._first_press_time = current_time
                elif current_time - self._first_press_time > self.simultaneous_window:
                    # Buttons not pressed simultaneously
                    self._reset()
                    return ButtonEvent.NONE

            # Start or continue timing
            if self._combo_start_time is None:
                self._combo_start_time = current_time
                logger.info(
                    f"{self.name} combo started - hold for {self.hold_duration}s"
                )

                if self.hold_duration == 0:
                    # Instant trigger
                    self._combo_triggered = True
                    if self.on_triggered:
                        self.on_triggered()
                    return ButtonEvent.HELD

                return ButtonEvent.PRESSED

            # Check if held long enough
            hold_time = current_time - self._combo_start_time

            if not self._combo_triggered and hold_time >= self.hold_duration:
                self._combo_triggered = True
                logger.info(f"{self.name} combo triggered after {hold_time:.1f}s")

                if self.on_triggered:
                    self.on_triggered()

                # Reset the combo so it doesn't trigger again until buttons are released
                self._combo_start_time = None
                return ButtonEvent.HELD
            elif not self._combo_triggered:
                return ButtonEvent.HOLDING

        else:
            # Not all buttons pressed - check if we should apply grace period
            if self._combo_start_time is not None:
                # We're in the middle of a combo
                if self._last_all_pressed_time is not None:
                    time_since_all_pressed = current_time - self._last_all_pressed_time

                    # If within grace period, ignore the button release
                    if time_since_all_pressed <= self.grace_period:
                        # Continue as if buttons are still pressed
                        return ButtonEvent.HOLDING

                # Grace period expired or never had all pressed - cancel combo
                hold_time = current_time - self._combo_start_time
                if hold_time < self.hold_duration and not self._combo_triggered:
                    logger.info(f"{self.name} combo cancelled after {hold_time:.1f}s")
                self._reset()
                return ButtonEvent.RELEASED

            # Check if we should reset first press time
            any_pressed = any(button_states.get(btn, False) for btn in self.buttons)
            if not any_pressed:
                self._first_press_time = None

        return ButtonEvent.NONE

    def _reset(self) -> None:
        """Reset combo state."""
        self._combo_start_time = None
        self._combo_triggered = False
        self._first_press_time = None
        self._last_all_pressed_time = None

    @property
    def progress(self) -> Optional[float]:
        """Get combo progress as a fraction (0.0 to 1.0)."""
        if self._combo_start_time is None:
            return None

        if self.hold_duration == 0:
            return 1.0 if self._combo_triggered else 0.0

        hold_time = time.time() - self._combo_start_time
        return min(hold_time / self.hold_duration, 1.0)

    @property
    def time_remaining(self) -> Optional[float]:
        """Get time remaining until combo triggers."""
        if self._combo_start_time is None or self._combo_triggered:
            return None

        hold_time = time.time() - self._combo_start_time
        return max(self.hold_duration - hold_time, 0.0)

class ButtonManager:
    """
    Centralized button management with debouncing and combos.

    Manages all button states, debouncing, and combo detection.
    """

    def __init__(self):
        """Initialize button manager."""
        self.buttons: Dict[str, DebouncedButton] = {}
        self.combos: Dict[str, ButtonCombo] = {}
        self._raw_states: Dict[str, bool] = {}

    def add_button(
        self,
        name: str,
        hold_duration: float = 0.0,
        debounce_time: float = 0.05,
        on_held: Optional[Callable[[], None]] = None,
        on_pressed: Optional[Callable[[], None]] = None,
        on_released: Optional[Callable[[], None]] = None,
    ) -> DebouncedButton:
        """
        Add a button to manage.

        Args:
            name: Button identifier
            hold_duration: Time required for "held" event
            debounce_time: Debounce duration
            on_held: Callback when held
            on_pressed: Callback when pressed
            on_released: Callback when released

        Returns:
            The created DebouncedButton
        """
        button = DebouncedButton(
            name=name,
            hold_duration=hold_duration,
            debounce_time=debounce_time,
            on_held=on_held,
            on_pressed=on_pressed,
            on_released=on_released,
        )
        self.buttons[name] = button
        return button

    def add_combo(
        self,
        name: str,
        buttons: list[str],
        hold_duration: float = 0.0,
        require_simultaneous: bool = True,
        grace_period: float = 0.1,
        on_triggered: Optional[Callable[[], None]] = None,
    ) -> ButtonCombo:
        """
        Add a button combination.

        Args:
            name: Combo identifier
            buttons: List of button names in combo
            hold_duration: Time combo must be held
            require_simultaneous: If buttons must be pressed together
            grace_period: Grace period for brief button release
            on_triggered: Callback when combo triggers

        Returns:
            The created ButtonCombo
        """
        combo = ButtonCombo(
            name=name,
            buttons=buttons,
            hold_duration=hold_duration,
            require_simultaneous=require_simultaneous,
            grace_period=grace_period,
            on_triggered=on_triggered,
        )
        self.combos[name] = combo
        return combo

    def update(self, raw_states: Dict[str, bool]) -> Dict[str, ButtonEvent]:
        """
        Update all buttons and combos.

        Args:
            raw_states: Dictionary of button name to raw state

        Returns:
            Dictionary of button/combo name to event
        """
        self._raw_states = raw_states
        events = {}

        # Update individual buttons
        for name, button in self.buttons.items():
            raw_state = raw_states.get(name, False)
            event = button.update(raw_state)
            if event != ButtonEvent.NONE:
                events[name] = event

        # Get debounced states for combos
        debounced_states = {name: btn.is_pressed for name, btn in self.buttons.items()}

        # Update combos
        for name, combo in self.combos.items():
            event = combo.update(debounced_states)
            if event != ButtonEvent.NONE:
                events[name] = event

        return events

    def get_button(self, name: str) -> Optional[DebouncedButton]:
        """Get a button by name."""
        return self.buttons.get(name)

    def get_combo(self, name: str) -> Optional[ButtonCombo]:
        """Get a combo by name."""
        return self.combos.get(name)

    def is_pressed(self, name: str) -> bool:
        """Check if a button is pressed."""
        button = self.buttons.get(name)
        return button.is_pressed if button else False

    def is_held(self, name: str) -> bool:
        """Check if a button is held."""
        button = self.buttons.get(name)
        return button.is_held if button else False

    def reset(self) -> None:
        """Reset all buttons and combos."""
        for button in self.buttons.values():
            button.reset()
        for combo in self.combos.values():
            combo._reset()
