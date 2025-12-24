"""
Equations of motion for Newtonian mechanics.

All quantities in SI units (meters, kilograms, seconds).
"""

import numpy as np
from solar_system.physics.bodies import Body


def acceleration(body: Body, net_force: np.ndarray) -> np.ndarray:
    """
    Compute the acceleration of a body given the net force acting on it.
    
    Implements Newton's second law of motion:
        a = F_net / m
    
    where:
        - a is the acceleration vector (m/s²)
        - F_net is the net force vector (N)
        - m is the mass (kg)
    
    Physical meaning:
        Acceleration is the rate of change of velocity. It describes how
        the body's motion is changing at this instant due to the forces
        acting on it. Acceleration is not part of the state; it is derived
        from the current forces and mass.
    
    Args:
        body: The body experiencing the force
        net_force: The net force vector acting on the body (Newtons)
    
    Returns:
        3D acceleration vector (m/s²)
    
    Notes:
        - This is a pure function: no side effects, no state mutation
        - Acceleration is an instantaneous quantity, valid only for the
          current force and mass
        - In inertial frames, this is the time derivative of velocity
    """
    # Newton's second law: a = F / m
    accel = net_force / body.mass
    
    return accel

