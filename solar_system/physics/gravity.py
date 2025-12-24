"""
Newtonian gravitational interaction between point masses.

All quantities in SI units (meters, kilograms, seconds).
"""

import numpy as np
from solar_system.physics.bodies import Body


# Gravitational constant in SI units
G = 6.674e-11  # m^3 kg^-1 s^-2


def gravitational_force(body1: Body, body2: Body) -> np.ndarray:
    """
    Compute the gravitational force on body1 due to body2.
    
    Implements Newton's law of universal gravitation:
        F_12 = -G * m1 * m2 / r^3 * r_12
    
    where:
        - F_12 is the force vector on body1 due to body2
        - r_12 = position1 - position2 (displacement from body2 to body1)
        - r = |r_12| (distance between bodies)
    
    The force is attractive: it points from body1 toward body2.
    
    Args:
        body1: The body experiencing the force
        body2: The body exerting the force
    
    Returns:
        3D force vector (Newtons) acting on body1 due to body2
    
    Raises:
        ValueError: If the two bodies are at the same position (r = 0),
                   which creates a mathematical singularity. In the point-mass
                   approximation, bodies cannot overlap.
    
    Notes:
        - This is a pure function: no side effects, no state mutation
        - Returns instantaneous force based on current positions and masses
        - Does not depend on velocities (Newtonian gravity is velocity-independent)
        - By Newton's third law: F_21 = -F_12
    """
    # Displacement vector from body2 to body1
    r_12 = body1.position - body2.position
    
    # Distance between bodies
    r = np.linalg.norm(r_12)
    
    # Check for singularity
    if r == 0:
        raise ValueError(
            f"Gravitational singularity: {body1.name} and {body2.name} "
            f"are at the same position. Point masses cannot overlap."
        )
    
    # Gravitational force magnitude: F = G * m1 * m2 / r^2
    force_magnitude = G * body1.mass * body2.mass / (r * r)
    
    # Direction: from body1 toward body2 (attractive)
    # Unit vector from body1 to body2 is -r_12 / r
    force_direction = -r_12 / r
    
    # Force vector
    force = force_magnitude * force_direction
    
    return force

