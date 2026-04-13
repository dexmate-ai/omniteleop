#!/usr/bin/env python3
"""Trajectory interpolation utilities for smooth motion control.

This module provides a standalone interpolator class that handles trajectory
interpolation and velocity computation for robot control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from scipy.interpolate import PchipInterpolator, interp1d
from collections import deque

class TrajectoryInterpolator:
    """Standalone trajectory interpolator with velocity computation.

    This class maintains a history of trajectory points and provides smooth
    interpolation with automatic or manual velocity computation.

    Optimized for real-time control with minimal overhead.

    Attributes:
        method: Interpolation method ('linear' or 'cubic')
        history_size: Maximum number of points to keep in history
    """

    __slots__ = (
        "method",
        "history_size",
        "_times",
        "_positions",
        "_interpolators",
        "_interpolators_dirty",
        "_first_time",
        "_prev_positions",
        "_prev_time",
        "_component_times",
    )

    def __init__(
        self,
        method: Literal["linear", "cubic"] = "cubic",
        history_size: int = 4,
    ):
        """Initialize trajectory interpolator.

        Args:
            method: Interpolation method - 'linear' or 'cubic'
            history_size: Maximum number of trajectory points to keep
        """
        self.method = method
        self.history_size = history_size

        # Optimized storage: separate times and positions
        self._times: deque = deque(maxlen=history_size)
        self._positions: Dict[str, deque] = {}  # component -> deque of arrays
        self._component_times: Dict[str, deque] = {}  # component -> deque of timestamps

        # Cached interpolators (rebuilt when history changes)
        self._interpolators: Optional[Dict[str, object]] = None
        self._interpolators_dirty = True
        self._first_time: float = 0.0

        # Cache for velocity computation via finite difference
        self._prev_positions: Optional[Dict[str, np.ndarray]] = None
        self._prev_time: float = 0.0

    def add_point(
        self,
        timestamp: float,
        positions: Dict[str, np.ndarray | List[float]],
    ):
        """Add a trajectory point to history.

        Args:
            timestamp: Time of this point in seconds
            positions: Dictionary mapping component names to position arrays
        """
        # Track first timestamp for normalization
        if not self._times:
            self._first_time = timestamp

        self._times.append(timestamp)

        # Add positions for each component
        for component, pos in positions.items():
            # Initialize deques for new components
            if component not in self._positions:
                self._positions[component] = deque(maxlen=self.history_size)
                self._component_times[component] = deque(maxlen=self.history_size)

            # Store timestamp for this component
            self._component_times[component].append(timestamp)

            # Store as numpy array for efficiency
            if isinstance(pos, list):
                self._positions[component].append(np.array(pos, dtype=np.float64))
            else:
                self._positions[component].append(pos.astype(np.float64))

        self._interpolators_dirty = True

    def clear(self):
        """Clear all history and cached interpolators."""
        self._times.clear()
        self._positions.clear()
        self._component_times.clear()
        self._interpolators = None
        self._interpolators_dirty = True
        self._first_time = 0.0
        self._prev_positions = None
        self._prev_time = 0.0

    def _build_interpolators(self) -> Optional[Dict[str, object]]:
        """Build scipy interpolators from current history.

        Returns:
            Dictionary mapping component names to interpolator objects,
            or None if insufficient data.
        """
        if len(self._times) < 2:
            return None

        # Build interpolators for each component using its own timestamps
        interpolators = {}

        try:
            for component, pos_deque in self._positions.items():
                # Skip if insufficient data for this component
                if len(pos_deque) < 2:
                    continue

                # Get timestamps specific to this component
                component_times = np.array(
                    list(self._component_times[component]), dtype=np.float64
                )

                # Normalize times relative to THIS component's first time (not global)
                # This ensures each component is independent and survives deque wraparound
                component_first_time = component_times[0]
                normalized_times = component_times - component_first_time

                # Stack positions into 2D array: (n_points, n_joints)
                positions = np.stack(list(pos_deque), axis=0)

                # Determine if we can use cubic (need at least 4 points for this component)
                use_cubic = self.method == "cubic" and len(normalized_times) >= 4

                if use_cubic:
                    # PCHIP for smooth C1 continuous interpolation
                    interpolators[component] = {
                        "interpolator": PchipInterpolator(
                            normalized_times,
                            positions,
                            axis=0,
                            extrapolate=False,
                        ),
                        "first_time": component_first_time,
                        "last_time": component_times[-1],
                    }
                else:
                    # Linear interpolation
                    interpolators[component] = {
                        "interpolator": interp1d(
                            normalized_times,
                            positions,
                            kind="linear",
                            axis=0,
                            bounds_error=False,
                            fill_value=(positions[0], positions[-1]),
                        ),
                        "first_time": component_first_time,
                        "last_time": component_times[-1],
                    }

        except Exception:
            return None

        return interpolators if interpolators else None

    def interpolate(
        self,
        query_time: float,
        compute_velocity: bool = True,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        """Interpolate positions and optionally velocities at given time.

        Args:
            query_time: Time to query (in same units as added points)
            compute_velocity: Whether to compute velocities using cached previous positions

        Returns:
            Tuple of (positions_dict, velocities_dict) or (positions_dict, None)
            Returns (None, None) if interpolation fails

        Note:
            Velocity is computed using finite difference with cached previous position:
            vel = (pos_current - pos_previous) / (t_current - t_previous)
            For cubic interpolation, analytical derivative is used.
        """
        # Rebuild interpolators if needed
        if self._interpolators_dirty or self._interpolators is None:
            self._interpolators = self._build_interpolators()
            self._interpolators_dirty = False

        if self._interpolators is None or len(self._times) < 2:
            return None, None

        # Pre-allocate result dictionaries
        positions = {}
        velocities = {} if compute_velocity else None

        try:
            for component, interp_data in self._interpolators.items():
                # Extract interpolator and time range for this component
                interp = interp_data["interpolator"]
                comp_first_time = interp_data["first_time"]
                comp_last_time = interp_data["last_time"]

                # Compute relative time normalized to THIS component's first time
                relative_time = query_time - comp_first_time
                comp_last_relative = comp_last_time - comp_first_time

                # Clamp to valid range for this component with small margin
                min_margin = 0.001
                upper_bound = max(comp_last_relative - min_margin, min_margin + 0.0001)
                clamped_time = np.clip(relative_time, min_margin, upper_bound)

                # Interpolate position at clamped time
                pos = interp(clamped_time)
                positions[component] = pos

                # Compute velocity if requested
                if compute_velocity:
                    if isinstance(interp, PchipInterpolator):
                        # Analytical derivative (most efficient for cubic)
                        vel = interp(clamped_time, 1)
                    else:
                        # Finite difference using cached previous position
                        if (
                            self._prev_positions is not None
                            and component in self._prev_positions
                            and self._prev_time > 0
                        ):
                            # Use cached previous position from last interpolation call
                            dt = query_time - self._prev_time
                            if dt > 0:
                                vel = (pos - self._prev_positions[component]) / dt
                            else:
                                vel = np.zeros_like(pos)
                        else:
                            # No cached previous position - use backward finite difference
                            # Query position at a small time step before current time
                            dt = 0.01  # 10ms lookback
                            t_prev = max(clamped_time - dt, min_margin)
                            actual_dt = clamped_time - t_prev

                            if actual_dt > 0:
                                pos_prev = interp(t_prev)
                                vel = (pos - pos_prev) / actual_dt
                            else:
                                vel = np.zeros_like(pos)

                    velocities[component] = vel

            # Cache current positions and time for next iteration (linear only)
            if compute_velocity and self._interpolators:
                first_interp = next(iter(self._interpolators.values()))["interpolator"]
                if not isinstance(first_interp, PchipInterpolator):
                    self._prev_positions = positions.copy()
                    self._prev_time = query_time

        except Exception:
            return None, None

        return positions, velocities

    def get_latest_positions(self) -> Optional[Dict[str, np.ndarray]]:
        """Get positions from the most recent point in history.

        Returns:
            Dictionary of positions, or None if no history
        """
        if not self._times or not self._positions:
            return None

        # Return the last position for each component
        latest = {}
        for component, pos_deque in self._positions.items():
            if pos_deque:
                latest[component] = pos_deque[-1].copy()

        return latest if latest else None

    def has_sufficient_data(self) -> bool:
        """Check if there is enough data for interpolation.

        Returns:
            True if at least 2 points in history
        """
        return len(self._times) >= 2

    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """Get the time range of current history.

        Returns:
            Tuple of (start_time, end_time) or None if no history
        """
        if len(self._times) < 2:
            return None

        return self._times[0], self._times[-1]

# Convenience function for single-use interpolation
def interpolate_trajectory(
    timestamps: List[float],
    positions_history: Dict[str, List[np.ndarray]],
    query_time: float,
    method: Literal["linear", "cubic"] = "cubic",
    compute_velocity: bool = True,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """One-shot trajectory interpolation without maintaining state.

    Args:
        timestamps: List of timestamps for each point
        positions_history: Dict mapping component names to list of position arrays
        query_time: Time to query
        method: Interpolation method ('linear' or 'cubic')
        compute_velocity: Whether to compute velocities

    Returns:
        Tuple of (positions_dict, velocities_dict)

    Example:
        >>> timestamps = [0.0, 0.1, 0.2, 0.3]
        >>> positions = {
        ...     'left_arm': [np.array([0.1]*7), np.array([0.2]*7), ...]
        ... }
        >>> pos, vel = interpolate_trajectory(timestamps, positions, 0.15)
    """
    interpolator = TrajectoryInterpolator(method=method, history_size=len(timestamps))

    # Add all points
    for t, idx in zip(timestamps, range(len(timestamps))):
        point_positions = {
            component: positions_history[component][idx]
            for component in positions_history
        }
        interpolator.add_point(t, point_positions)

    # Interpolate at query time
    return interpolator.interpolate(query_time, compute_velocity)
