"""
Explanation audit experiment: Testing honesty of physics+numerics combinations.

This experiment demonstrates how to use Explanations to test whether
combinations of physics laws and integrators preserve the invariants
they claim to preserve.

Key concept:
    An explanation is HONEST if it preserves all invariants it claims.
    An explanation is DISHONEST if it violates any claimed invariant.

This is observational auditing: we test claims, we don't enforce them.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.physics.gravity import gravitational_force
from solar_system.physics.dynamics import acceleration
from solar_system.numerics.integrators import euler_step, verlet_step
from solar_system.analysis.conserved import total_energy, total_momentum, total_angular_momentum
from solar_system.invariants import Invariant, InvariantChecker
from solar_system.explanations import Explanation


# Physical constants (SI units)
M_SUN = 1.989e30      # kg
M_EARTH = 5.972e24    # kg
AU = 1.496e11         # m
G = 6.674e-11         # m^3 kg^-1 s^-2

# Orbital period
T_ORBIT = 2 * np.pi * np.sqrt(AU**3 / (G * M_SUN))

# Circular orbital velocity
V_CIRCULAR = np.sqrt(G * M_SUN / AU)


def create_sun_earth_system():
    """Create Sun-Earth system with circular orbit."""
    sun = Body("Sun", M_SUN, 
               np.array([0.0, 0.0, 0.0]), 
               np.array([0.0, 0.0, 0.0]))
    
    earth = Body("Earth", M_EARTH,
                 np.array([AU, 0.0, 0.0]),
                 np.array([0.0, V_CIRCULAR, 0.0]))
    
    return [sun, earth]


def make_derivative_func_with_force(bodies, force_law):
    """
    Create derivative function using a specific force law.
    
    This allows us to use any force law with any integrator.
    The derivative function encapsulates the physics.
    
    Args:
        bodies: List of Body objects (for mass and name extraction)
        force_law: Function (body1, body2) -> force vector
    
    Returns:
        Function (state, t) -> derivative
    """
    n = len(bodies)
    masses = np.array([body.mass for body in bodies])
    names = [body.name for body in bodies]
    
    def derivative_func(state, t):
        """Compute time derivative using specified force law."""
        derivative = np.zeros_like(state)
        
        # Extract positions and velocities
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        
        for i in range(n):
            positions[i] = state[3*i : 3*i + 3]
            velocities[i] = state[3*n + 3*i : 3*n + 3*i + 3]
        
        # Create temporary Body objects
        temp_bodies = [
            Body(names[i], masses[i], positions[i], velocities[i])
            for i in range(n)
        ]
        
        # Compute accelerations using specified force law
        accelerations = np.zeros((n, 3))
        
        for i in range(n):
            net_force = np.zeros(3)
            for j in range(n):
                if i != j:
                    force = force_law(temp_bodies[i], temp_bodies[j])
                    net_force += force
            
            accel = acceleration(temp_bodies[i], net_force)
            accelerations[i] = accel
        
        # Pack derivative
        for i in range(n):
            derivative[3*i : 3*i + 3] = velocities[i]
        
        for i in range(n):
            derivative[3*n + 3*i : 3*n + 3*i + 3] = accelerations[i]
        
        return derivative
    
    return derivative_func


def test_explanation(explanation: Explanation, dt, num_orbits=10):
    """
    Test whether an explanation preserves its claimed invariants.
    
    This function:
        1. Runs a simulation using the explanation's force law and integrator
        2. Observes all claimed invariants during the simulation
        3. Returns checkers that can be queried for deviation
    
    Args:
        explanation: The Explanation to test
        dt: Timestep in seconds
        num_orbits: Number of orbits to simulate
    
    Returns:
        Dictionary of InvariantChecker objects (keyed by invariant name)
    """
    duration = num_orbits * T_ORBIT
    num_steps = int(duration / dt)
    
    print(f"\nTesting: {explanation.name}")
    print(f"  Description: {explanation.description}")
    print(f"  Claims: {[inv.name for inv in explanation.claimed_invariants]}")
    print(f"  Duration: {num_orbits} orbits ({num_steps} steps)")
    
    # Create initial system
    bodies = create_sun_earth_system()
    
    # Pack initial state
    state = np.zeros(12)  # 2 bodies × 6 components
    for i, body in enumerate(bodies):
        state[3*i : 3*i + 3] = body.position
        state[6 + 3*i : 6 + 3*i + 3] = body.velocity
    
    # Create derivative function using explanation's force law
    derivative_func = make_derivative_func_with_force(bodies, explanation.force_law)
    
    # Create checkers for claimed invariants only
    checkers = {inv.name: InvariantChecker(inv) 
                for inv in explanation.claimed_invariants}
    
    # Initial observation
    for i, body in enumerate(bodies):
        body.position = state[3*i : 3*i + 3].copy()
        body.velocity = state[6 + 3*i : 6 + 3*i + 3].copy()
    
    for checker in checkers.values():
        checker.observe(bodies, 0.0)
    
    # Simulation loop using explanation's integrator
    time = 0.0
    for step in range(num_steps):
        # Time evolution using explanation's integrator
        state = explanation.integrator(state, derivative_func, time, dt)
        time += dt
        
        # Update bodies for observation
        for i, body in enumerate(bodies):
            body.position = state[3*i : 3*i + 3].copy()
            body.velocity = state[6 + 3*i : 6 + 3*i + 3].copy()
        
        # Observe claimed invariants
        for checker in checkers.values():
            checker.observe(bodies, time)
    
    print(f"  Complete.")
    
    return checkers


def is_honest(explanation: Explanation, checkers, tolerance=0.1):
    """
    Determine if an explanation is honest about its claims.
    
    An explanation is honest if all claimed invariants are preserved
    within the tolerance threshold.
    
    Args:
        explanation: The Explanation being tested
        checkers: Dictionary of InvariantChecker objects
        tolerance: Maximum allowed relative deviation (percent)
    
    Returns:
        bool: True if all claims are respected, False otherwise
    """
    for inv in explanation.claimed_invariants:
        checker = checkers[inv.name]
        deviation = abs(checker.relative_change_percent())
        if deviation > tolerance:
            return False
    return True


def main():
    """
    Audit multiple explanations and report their honesty.
    """
    print("="*70)
    print("EXPLANATION AUDIT: Testing Honesty of Physics+Numerics Combinations")
    print("="*70)
    print()
    print("This experiment tests whether explanations preserve the invariants")
    print("they claim to preserve. An explanation is HONEST if it delivers on")
    print("its claims, DISHONEST if it violates them.")
    print()
    print("="*70)
    
    # Define invariants (reusable across explanations)
    energy_invariant = Invariant(
        name="Energy",
        description="Total mechanical energy (kinetic + potential)",
        measure=total_energy
    )
    
    momentum_invariant = Invariant(
        name="Momentum",
        description="Total linear momentum",
        measure=total_momentum
    )
    
    ang_momentum_invariant = Invariant(
        name="Angular Momentum",
        description="Total angular momentum about origin",
        measure=total_angular_momentum
    )
    
    # Define explanations to test
    explanations = [
        Explanation(
            name="Newtonian + Verlet",
            description="Newton's gravity with symplectic time evolution",
            force_law=gravitational_force,
            integrator=verlet_step,
            claimed_invariants=[energy_invariant, momentum_invariant, ang_momentum_invariant]
        ),
        
        Explanation(
            name="Newtonian + Euler (Modest)",
            description="Newton's gravity with Euler, honest about limitations",
            force_law=gravitational_force,
            integrator=euler_step,
            claimed_invariants=[momentum_invariant]  # Only claims momentum
        ),
        
        Explanation(
            name="Newtonian + Euler (Dishonest)",
            description="Newton's gravity with Euler, claims energy (but won't preserve it)",
            force_law=gravitational_force,
            integrator=euler_step,
            claimed_invariants=[energy_invariant, momentum_invariant]  # Claims energy!
        ),
    ]
    
    # Simulation parameters
    dt = 3600  # 1 hour
    num_orbits = 10
    
    # Test each explanation
    results = []
    for explanation in explanations:
        checkers = test_explanation(explanation, dt, num_orbits)
        honest = is_honest(explanation, checkers, tolerance=0.1)
        results.append((explanation, checkers, honest))
    
    # Report findings
    print()
    print("="*70)
    print("AUDIT RESULTS:")
    print("="*70)
    print()
    
    for explanation, checkers, honest in results:
        status = "[HONEST]" if honest else "[DISHONEST]"
        print(f"{explanation.name}: {status}")
        print(f"  Claims: {[inv.name for inv in explanation.claimed_invariants]}")
        
        for inv in explanation.claimed_invariants:
            checker = checkers[inv.name]
            deviation = checker.relative_change_percent()
            print(f"    {inv.name:20s} -> {deviation:+10.6e} %")
        
        print()
    
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    print()
    print("Newtonian + Verlet:")
    print("  Claims energy, momentum, angular momentum")
    print("  Preserves all three -> HONEST")
    print()
    print("Newtonian + Euler (Modest):")
    print("  Claims only momentum")
    print("  Preserves momentum -> HONEST (limited claims, but truthful)")
    print()
    print("Newtonian + Euler (Dishonest):")
    print("  Claims energy and momentum")
    print("  Violates energy -> DISHONEST (makes claims it can't keep)")
    print()
    print("Key insight:")
    print("  An explanation is honest if it preserves what it claims.")
    print("  Modest claims that are kept are better than ambitious claims that fail.")
    print()
    print("="*70)


if __name__ == "__main__":
    main()

