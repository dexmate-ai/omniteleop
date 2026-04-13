#!/usr/bin/env python3
"""Signal filtering utilities for robot command smoothing."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from typing import Dict, Optional, Any

class ButterworthFilter:
    """Multi-channel Butterworth low-pass filter for robot joint commands."""

    def __init__(self, cutoff_freq: float = 10.0, order: int = 2, fs: float = 250.0):
        """Initialize Butterworth filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order (default 2 for smooth response)
            fs: Sampling frequency in Hz
        """
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.fs = fs

        # Design filter
        nyquist = fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        self.sos = butter(order, normalized_cutoff, btype="low", output="sos")

        # Filter states for each component and joint
        self.filter_states = {}

    def _get_state_key(self, component: str, joint_idx: int) -> str:
        """Generate unique key for filter state storage."""
        return f"{component}_joint_{joint_idx}"

    def filter_component(
        self, component: str, data: np.ndarray, field: str = "pos"
    ) -> np.ndarray:
        """Apply filter to a component's joint data.

        Args:
            component: Component name (e.g., 'left_arm', 'right_arm')
            data: Joint data array
            field: Data field ('pos' or 'vel')

        Returns:
            Filtered data array
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Handle scalar or 1D array
        if data.ndim == 0:
            data = data.reshape(1)

        filtered = np.zeros_like(data)

        # Filter each joint independently
        for i in range(len(data)):
            state_key = f"{component}_{field}_{i}"

            if state_key not in self.filter_states:
                # Initialize filter state for this joint
                self.filter_states[state_key] = sosfilt_zi(self.sos) * data[i]

            # Apply filter
            filtered_val, self.filter_states[state_key] = sosfilt(
                self.sos, [data[i]], zi=self.filter_states[state_key]
            )
            filtered[i] = filtered_val[0]

        return filtered

    def reset(self, component: Optional[str] = None):
        """Reset filter states.

        Args:
            component: If specified, only reset this component's states
        """
        if component:
            keys_to_remove = [k for k in self.filter_states if k.startswith(component)]
            for key in keys_to_remove:
                del self.filter_states[key]
        else:
            self.filter_states.clear()

class MultiChannelFilter:
    """Wrapper for applying filters to multiple robot components."""

    def __init__(
        self, filter_config: Optional[Dict] = None, control_rate: float = 250.0
    ):
        """Initialize multi-channel filter with per-component configuration.

        Args:
            filter_config: Dictionary with filter configuration from YAML.
            control_rate: Control loop rate in Hz (for Butterworth filters).
        """
        self.control_rate = control_rate
        self.component_filters = {}
        self.previous_values = {}  # For EMA filters

        if filter_config is None:
            # No filtering if no config provided
            self.default_filter_type = "none"
            return

        # Parse default filter settings
        default_config = filter_config.get("default", {})
        self.default_filter_type = default_config.get("type", "none").lower()
        self.default_filter_params = default_config

        # Create filters for each component
        component_configs = filter_config.get("components", {})

        # Get all possible components (will be created on demand)
        self.component_configs = component_configs

    def _get_or_create_filter(self, component: str):
        """Get or create a filter for a specific component.

        Args:
            component: Component name.

        Returns:
            Filter instance or None.
        """
        if component in self.component_filters:
            return self.component_filters[component]

        # Get component-specific config or use default
        if component in self.component_configs:
            config = self.component_configs[component]
        else:
            config = self.default_filter_params

        filter_type = config.get("type", self.default_filter_type).lower()

        if filter_type == "butterworth":
            filter_obj = ButterworthFilter(
                cutoff_freq=config.get("cutoff_freq", 10.0),
                order=config.get("order", 2),
                fs=self.control_rate,
            )
        elif filter_type == "ema":
            # For EMA, we store the alpha value
            filter_obj = {"type": "ema", "alpha": config.get("alpha", 0.1)}
        else:
            filter_obj = None

        self.component_filters[component] = filter_obj
        return filter_obj

    def apply(self, components: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply filtering to all components using their specific filters.

        Args:
            components: Dictionary of component data.

        Returns:
            Filtered components dictionary.
        """
        filtered_components = {}

        for component, data in components.items():
            filter_obj = self._get_or_create_filter(component)

            if filter_obj is None:
                # No filtering for this component
                filtered_components[component] = data
                continue

            filtered_data = data.copy()

            if isinstance(filter_obj, ButterworthFilter):
                # Apply Butterworth filter
                if "pos" in data:
                    filtered_data["pos"] = filter_obj.filter_component(
                        component, data["pos"], "pos"
                    )
                if "vel" in data:
                    filtered_data["vel"] = filter_obj.filter_component(
                        component, data["vel"], "vel"
                    )
                # Handle chassis velocity fields (vx, vy, wz)
                if "vx" in data and "vy" in data and "wz" in data:
                    velocity_array = np.array([data["vx"], data["vy"], data["wz"]])
                    filtered_velocity = filter_obj.filter_component(
                        component, velocity_array, "chassis_vel"
                    )
                    filtered_data["vx"] = float(filtered_velocity[0])
                    filtered_data["vy"] = float(filtered_velocity[1])
                    filtered_data["wz"] = float(filtered_velocity[2])

            elif isinstance(filter_obj, dict) and filter_obj["type"] == "ema":
                # Apply EMA filter
                alpha = filter_obj["alpha"]
                for field in ["pos", "vel"]:
                    if field in data:
                        key = f"{component}_{field}"
                        current = np.array(data[field])

                        if key in self.previous_values:
                            filtered_value = (
                                alpha * current
                                + (1 - alpha) * self.previous_values[key]
                            )
                        else:
                            filtered_value = current

                        self.previous_values[key] = filtered_value
                        filtered_data[field] = filtered_value

                # Handle chassis velocity fields (vx, vy, wz)
                if "vx" in data and "vy" in data and "wz" in data:
                    for vel_field in ["vx", "vy", "wz"]:
                        key = f"{component}_{vel_field}"
                        current = float(data[vel_field])

                        if key in self.previous_values:
                            filtered_value = (
                                alpha * current
                                + (1 - alpha) * self.previous_values[key]
                            )
                        else:
                            filtered_value = current

                        self.previous_values[key] = filtered_value
                        filtered_data[vel_field] = filtered_value

            filtered_components[component] = filtered_data

        return filtered_components
