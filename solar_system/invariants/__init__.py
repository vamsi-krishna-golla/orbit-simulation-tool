"""
Observational invariant checking for simulation auditing.

This module provides tools to observe whether simulations preserve
quantities they claim to preserve, without modifying the simulation itself.

Design principle:
    Invariants are OBSERVATIONS, not enforcement.
    We audit explanations, we don't correct them.
"""

from solar_system.invariants.invariant import Invariant
from solar_system.invariants.checker import InvariantChecker

__all__ = ['Invariant', 'InvariantChecker']

