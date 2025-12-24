"""
Measurement of conserved quantities for gravitational N-body systems.

These are pure observation functions that compute derived physical quantities
from the state of bodies. They are read-only and have no side effects on the
simulation.

All quantities use SI units (meters, kilograms, seconds, Joules, etc.).
"""

import numpy as np
from typing import List
from solar_system.physics.bodies import Body
from solar_system.physics.gravity import G


def total_energy(bodies: List[Body]) -> float:
    """
    Compute the total mechanical energy of the system.
    
    Total energy = Kinetic energy + Gravitational potential energy
    
    Kinetic energy:
        E_k = sum over all bodies of (1/2 * m * v²)
    
    Gravitational potential energy (for all pairs):
        E_p = -sum over all pairs (i<j) of (G * m_i * m_j / r_ij)
    
    Physical meaning:
        In an isolated Newtonian gravitational system, total energy is
        a conserved quantity. It should remain constant over time.
    
    What Euler does:
        Euler integration does NOT conserve energy. The total energy will
        drift systematically (increase or decrease) over time. This is the
        primary signature of Euler's failure for orbital mechanics.
        
        Expected drift rate: O(dt) over long integrations.
        
        Consequences:
            - Energy increasing → orbits expand, bodies move apart
            - Energy decreasing → orbits contract, bodies fall together
    
    Args:
        bodies: List of Body objects (read-only, not modified)
    
    Returns:
        Total mechanical energy in Joules (J)
    
    Notes:
        - This is a pure function with no side effects
        - Bodies are not modified
        - The same input always produces the same output
    """
    # Kinetic energy: sum of (1/2) * m * v²
    kinetic_energy = 0.0
    for body in bodies:
        v_squared = np.dot(body.velocity, body.velocity)
        kinetic_energy += 0.5 * body.mass * v_squared
    
    # Gravitational potential energy: -sum of G * m_i * m_j / r_ij for all pairs
    potential_energy = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1, n):  # Only count each pair once (i < j)
            # Displacement vector from body i to body j
            r_ij = bodies[j].position - bodies[i].position
            # Distance between bodies
            distance = np.linalg.norm(r_ij)
            # Potential energy for this pair (negative)
            potential_energy -= G * bodies[i].mass * bodies[j].mass / distance
    
    # Total energy
    total = kinetic_energy + potential_energy
    
    return total


def total_momentum(bodies: List[Body]) -> np.ndarray:
    """
    Compute the total linear momentum of the system.
    
    Total momentum = sum over all bodies of (m * v)
    
    Physical meaning:
        In an isolated system with no external forces, total linear momentum
        is conserved. This follows from Newton's third law: internal forces
        come in equal and opposite pairs, so they cancel in the total.
    
    What Euler does:
        Euler integration SHOULD conserve total momentum exactly (to machine
        precision), regardless of timestep size.
        
        Why? Because gravitational forces are internal and obey Newton's third
        law: F_12 = -F_21. When summing forces, they cancel exactly.
        
        If momentum is NOT conserved:
            This indicates a bug in the force calculation or force summation,
            NOT a problem with the Euler method.
    
    Args:
        bodies: List of Body objects (read-only, not modified)
    
    Returns:
        Total momentum vector in kg⋅m/s (3D array)
    
    Notes:
        - This is a pure function with no side effects
        - Bodies are not modified
        - Momentum conservation is a sanity check for implementation correctness
    """
    # Sum of m * v for all bodies
    momentum = np.zeros(3)
    for body in bodies:
        momentum += body.mass * body.velocity
    
    return momentum


def total_angular_momentum(bodies: List[Body]) -> np.ndarray:
    """
    Compute the total angular momentum of the system about the origin.
    
    Total angular momentum = sum over all bodies of (r × (m * v))
    
    where × denotes the cross product.
    
    Physical meaning:
        For central forces (forces that point along the line connecting bodies),
        angular momentum is conserved. Gravity is a central force, so in a
        Newtonian gravitational system, angular momentum should be constant.
        
        Conservation of angular momentum implies:
            - Kepler's second law (equal areas in equal times)
            - The orbital plane remains fixed
            - No artificial precession
    
    What Euler does:
        Euler integration approximately conserves angular momentum, but not
        perfectly. Small numerical errors introduce fictitious torques.
        
        Expected drift rate: Much smaller than energy drift, typically O(dt²)
        or better.
        
        Consequences of drift:
            - Orbital plane may slowly precess
            - Small violations of Kepler's second law
            - Usually a minor effect compared to energy drift
    
    Args:
        bodies: List of Body objects (read-only, not modified)
    
    Returns:
        Total angular momentum vector in kg⋅m²/s (3D array)
        Direction: perpendicular to the plane of motion (right-hand rule)
        Magnitude: relates to orbital angular velocity
    
    Notes:
        - This is a pure function with no side effects
        - Bodies are not modified
        - Angular momentum is computed about the origin (0, 0, 0)
        - For a planar orbit, L points perpendicular to the orbital plane
    """
    # Sum of r × (m * v) for all bodies
    angular_momentum = np.zeros(3)
    for body in bodies:
        # r × p where p = m * v
        r = body.position
        p = body.mass * body.velocity
        # Cross product: L = r × p
        L_body = np.cross(r, p)
        angular_momentum += L_body
    
    return angular_momentum

