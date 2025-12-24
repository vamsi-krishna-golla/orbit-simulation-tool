"""
Physical body representation for Newtonian mechanics.

All quantities in SI units (meters, kilograms, seconds).
"""

import numpy as np


class Body:
    """
    Represents a point-mass body in 3D space.
    
    This is a pure state representation: it stores what the body *is* 
    (its mass) and what state it's *in* (position and velocity), 
    but not how it behaves.
    
    Attributes:
        name: Human-readable identifier (e.g., "Sun", "Earth")
        mass: Mass in kilograms (kg). Determines inertial resistance 
              and gravitational field strength.
        position: 3D position vector in meters (m). Location in 
                  inertial Cartesian coordinates (x, y, z).
        velocity: 3D velocity vector in meters per second (m/s). 
                  Rate of change of position.
    """
    
    def __init__(self, name: str, mass: float, position: np.ndarray, velocity: np.ndarray):
        """
        Initialize a physical body.
        
        Args:
            name: Identifier for the body
            mass: Mass in kg (must be positive)
            position: Position vector [x, y, z] in meters
            velocity: Velocity vector [vx, vy, vz] in m/s
        """
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        
        # Basic validation
        if mass <= 0:
            raise ValueError(f"Mass must be positive, got {mass}")
        if self.position.shape != (3,):
            raise ValueError(f"Position must be 3D vector, got shape {self.position.shape}")
        if self.velocity.shape != (3,):
            raise ValueError(f"Velocity must be 3D vector, got shape {self.velocity.shape}")
    
    def __repr__(self):
        """
        String representation showing the body's current state.
        """
        return (f"Body('{self.name}', "
                f"mass={self.mass:.3e} kg, "
                f"pos={self.position}, "
                f"vel={self.velocity})")

