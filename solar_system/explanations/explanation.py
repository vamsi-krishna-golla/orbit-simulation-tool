"""
Minimal Explanation abstraction.

An Explanation is a declarative bundle of:
  - Physics law (what forces exist)
  - Integrator (how time evolves)
  - Claimed invariants (what should be preserved)

This is PURE DATA: no logic, no methods, no enforcement.
Experiments use explanations to test whether claims match observed behavior.
"""

from dataclasses import dataclass
from typing import Callable, List
import numpy as np
from solar_system.physics.bodies import Body
from solar_system.invariants.invariant import Invariant


@dataclass
class Explanation:
    """
    A testable claim about how a universe behaves.
    
    This is DECLARATIVE ONLY: it states what an explanation claims,
    but does not enforce or test anything. Experiments coordinate
    testing by running simulations and observing invariants.
    
    Attributes:
        name: Human-readable identifier (e.g., "Newtonian + Verlet")
        description: Explanation of what this combination represents
        force_law: Function that computes force between two bodies
                   Signature: (Body, Body) -> np.ndarray
        integrator: Integration function for time evolution
                    Signature: (state, derivative_func, t, dt) -> new_state
        claimed_invariants: List of Invariant objects that this explanation
                           claims should be preserved
    
    Example:
        newtonian_verlet = Explanation(
            name="Newtonian + Verlet",
            description="Newton's gravity with symplectic time evolution",
            force_law=gravitational_force,
            integrator=verlet_step,
            claimed_invariants=[energy_invariant, momentum_invariant]
        )
    
    Notes:
        - This is pure data: no methods, no logic
        - No enforcement: doesn't modify simulations
        - No testing: doesn't run simulations
        - Just a declaration of claims
        - Experiments decide how to test these claims
    """
    name: str
    description: str
    force_law: Callable[[Body, Body], np.ndarray]
    integrator: Callable
    claimed_invariants: List[Invariant]

