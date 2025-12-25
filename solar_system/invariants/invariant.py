"""
Definition of an observable invariant.

An invariant is simply a named measurement function that can be applied
to a list of bodies to extract some quantity of interest.

This is pure data — no logic, no expectations, no enforcement.
"""

from dataclasses import dataclass
from typing import Callable, List, Any


@dataclass
class Invariant:
    """
    Represents a quantity that can be measured from a system of bodies.
    
    This is OBSERVATIONAL: it defines what to measure, not what should happen.
    The experiment decides which invariants to track and interprets the results.
    
    Attributes:
        name: Human-readable identifier (e.g., "Total Energy")
        description: Explanation of what this quantity represents
        measure: Function that takes bodies and returns a measurement
                 Signature: (List[Body]) -> Any (typically float or np.ndarray)
    
    Example:
        energy_invariant = Invariant(
            name="Total Energy",
            description="Sum of kinetic and gravitational potential energy",
            measure=total_energy
        )
    
    Notes:
        - No tolerance or expectation encoded here
        - No claim about who should preserve this
        - Just a definition of what to observe
        - Experiments choose which invariants to track
    """
    name: str
    description: str
    measure: Callable[[List, Any], Any]  # List[Body] -> value

