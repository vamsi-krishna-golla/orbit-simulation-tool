"""
Observational tracking of invariants during simulation.

InvariantChecker records the value of an invariant at different timesteps,
then provides methods to query how it changed. It is purely read-only:
it observes but never modifies the simulation.
"""

import numpy as np
from typing import List, Tuple, Any
from solar_system.invariants.invariant import Invariant


class InvariantChecker:
    """
    Records observations of an invariant over time.
    
    This is OBSERVATIONAL ONLY:
        - Does not modify bodies
        - Does not affect simulation
        - Does not enforce any constraint
        - Just records what happened
    
    Usage:
        checker = InvariantChecker(energy_invariant)
        
        # During simulation
        for step in range(num_steps):
            world.step(integrator, dt)
            checker.observe(world.bodies, world.time)
        
        # After simulation
        print(f"Energy change: {checker.relative_change_percent():.2e}%")
    
    Attributes:
        invariant: The Invariant being tracked
        history: List of (time, value) tuples recorded during simulation
    """
    
    def __init__(self, invariant: Invariant):
        """
        Initialize checker for a specific invariant.
        
        Args:
            invariant: The Invariant to observe
        """
        self.invariant = invariant
        self.history: List[Tuple[float, Any]] = []
    
    def observe(self, bodies: List, time: float) -> None:
        """
        Record the invariant's value at this timestep.
        
        This is a READ-ONLY operation: bodies are not modified.
        
        Args:
            bodies: List of Body objects (not modified)
            time: Current simulation time
        
        Notes:
            - Bodies are passed to the measurement function but never mutated
            - The measurement function should also be read-only
            - Multiple observations can be made at any time
        """
        value = self.invariant.measure(bodies)
        self.history.append((time, value))
    
    def initial_value(self) -> Any:
        """
        Get the first recorded value.
        
        Returns:
            The value from the first observation
        
        Raises:
            IndexError: If no observations have been made
        """
        return self.history[0][1]
    
    def final_value(self) -> Any:
        """
        Get the most recent recorded value.
        
        Returns:
            The value from the last observation
        
        Raises:
            IndexError: If no observations have been made
        """
        return self.history[-1][1]
    
    def absolute_change(self) -> Any:
        """
        Compute the absolute change from initial to final value.
        
        Returns:
            final_value - initial_value
        
        Notes:
            - Works for scalars or arrays (uses numpy subtraction)
            - Returns the same type as the invariant's measure function
        """
        initial = self.initial_value()
        final = self.final_value()
        
        # Handle both scalar and array quantities
        if isinstance(initial, np.ndarray):
            return final - initial
        else:
            return final - initial
    
    def relative_change_percent(self) -> float:
        """
        Compute relative change as a percentage.
        
        Returns:
            100 * (final - initial) / |initial|
        
        Notes:
            - For vector quantities, uses magnitude (L2 norm)
            - Returns percentage (not fraction)
            - Sign indicates direction: positive = increase, negative = decrease
        """
        initial = self.initial_value()
        final = self.final_value()
        
        # For vector quantities, use magnitude
        if isinstance(initial, np.ndarray):
            initial_mag = np.linalg.norm(initial)
            final_mag = np.linalg.norm(final)
            return 100 * (final_mag - initial_mag) / initial_mag
        else:
            # For scalar quantities
            return 100 * (final - initial) / abs(initial)
    
    def get_history(self) -> List[Tuple[float, Any]]:
        """
        Get the complete observation history.
        
        Returns:
            List of (time, value) tuples
        
        Notes:
            - Useful for plotting or detailed analysis
            - Returns a reference (not a copy) for efficiency
        """
        return self.history

