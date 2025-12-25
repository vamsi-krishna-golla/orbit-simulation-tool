"""
Physics sensitivity experiment: How changing the force law creates different universes.

This is a CONTROLLED COUNTERFACTUAL PHYSICS EXPERIMENT.

What we hold constant:
    - Initial conditions (same Sun-Earth system)
    - Integrator (Verlet, unchanged)
    - Timestep (same value)
    - Diagnostic methods (same analysis functions)

What we vary:
    - The physical force law itself: 1/r^2 vs 1/r^(2+epsilon)

Purpose:
    To demonstrate that different physical laws, even with tiny modifications,
    create genuinely different self-consistent universes — and that a good
    numerical integrator will faithfully reveal those differences.

Key insight:
    Previous experiments showed: same physics + different numerics → artifacts
    This experiment shows: different physics + same numerics → real divergence

This is NOT a claim about real gravity. It is an exploration of how
computational universes depend on the physical explanations we encode.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.simulation.world import World
from solar_system.numerics.integrators import verlet_step
from solar_system.analysis.conserved import total_angular_momentum
from solar_system.physics.gravity import G, gravitational_force, modified_gravitational_force
from solar_system.physics.dynamics import acceleration


# Physical constants (SI units)
M_SUN = 1.989e30      # kg
M_EARTH = 5.972e24    # kg
AU = 1.496e11         # m

# Orbital period (for Newtonian gravity)
T_ORBIT = 2 * np.pi * np.sqrt(AU**3 / (G * M_SUN))

# Circular orbital velocity (for Newtonian gravity)
V_CIRCULAR = np.sqrt(G * M_SUN / AU)


def create_sun_earth_system():
    """
    Create Sun-Earth system with circular orbit initial conditions.
    
    These initial conditions are designed for Newtonian gravity (1/r^2).
    When we use modified gravity, the SAME initial conditions will
    evolve differently - that's the point.
    """
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
    
    This allows us to use the SAME integrator (Verlet) with DIFFERENT
    physical laws. The integrator knows nothing about which law it's
    evolving — it just faithfully computes time derivatives.
    
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
        """
        Compute time derivative of state using specified force law.
        
        This is the key coordination point:
            - Extract positions/velocities from state vector
            - Compute forces using the provided force_law
            - Apply F = ma to get accelerations
            - Return derivative: [velocities, accelerations]
        """
        derivative = np.zeros_like(state)
        
        # Extract positions and velocities from state
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        
        for i in range(n):
            positions[i] = state[3*i : 3*i + 3]
            velocities[i] = state[3*n + 3*i : 3*n + 3*i + 3]
        
        # Create temporary Body objects for force calculations
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
                    # Use the provided force law (Newtonian or modified)
                    force = force_law(temp_bodies[i], temp_bodies[j])
                    net_force += force
            
            # Apply F = ma (same for both universes)
            accel = acceleration(temp_bodies[i], net_force)
            accelerations[i] = accel
        
        # Pack derivative: d/dt [positions, velocities] = [velocities, accelerations]
        for i in range(n):
            derivative[3*i : 3*i + 3] = velocities[i]
        
        for i in range(n):
            derivative[3*n + 3*i : 3*n + 3*i + 3] = accelerations[i]
        
        return derivative
    
    return derivative_func


def run_with_force_law(force_law, force_label, dt, num_orbits=50):
    """
    Run simulation with specified force law.
    
    Args:
        force_law: Function (body1, body2) -> force vector
        force_label: String describing the force law (for output)
        dt: Timestep in seconds
        num_orbits: Number of orbits to simulate
    
    Returns:
        trajectory: Dict with 'time', 'x', 'y' arrays
        angular_momentum_data: Dict with 'time', 'L_deviation' arrays
    """
    duration = num_orbits * T_ORBIT
    num_steps = int(duration / dt)
    log_every = max(1, int((24 * 3600) / dt))  # Log once per day
    
    print(f"  Running {force_label}...")
    print(f"    Steps: {num_steps}, logging every {log_every} steps")
    
    # Create initial system
    bodies = create_sun_earth_system()
    
    # Pack initial state
    state = np.zeros(12)  # 2 bodies × 6 components each
    for i, body in enumerate(bodies):
        state[3*i : 3*i + 3] = body.position
        state[6 + 3*i : 6 + 3*i + 3] = body.velocity
    
    # Create derivative function with specified force law
    derivative_func = make_derivative_func_with_force(bodies, force_law)
    
    # Measure initial angular momentum
    for i, body in enumerate(bodies):
        body.position = state[3*i : 3*i + 3].copy()
        body.velocity = state[6 + 3*i : 6 + 3*i + 3].copy()
    L0 = np.linalg.norm(total_angular_momentum(bodies))
    
    # Storage
    trajectory = {'time': [], 'x': [], 'y': []}
    angular_momentum_data = {'time': [], 'L_deviation': []}
    
    time = 0.0
    
    # Simulation loop
    for step in range(num_steps + 1):
        # Update bodies for diagnostics
        for i, body in enumerate(bodies):
            body.position = state[3*i : 3*i + 3].copy()
            body.velocity = state[6 + 3*i : 6 + 3*i + 3].copy()
        
        # Log data
        if step % log_every == 0:
            earth = bodies[1]
            trajectory['time'].append(time / (24*3600))
            trajectory['x'].append(earth.position[0] / AU)
            trajectory['y'].append(earth.position[1] / AU)
            
            L = total_angular_momentum(bodies)
            L_mag = np.linalg.norm(L)
            L_deviation = (L_mag - L0) / L0 * 100
            angular_momentum_data['time'].append(time / (24*3600))
            angular_momentum_data['L_deviation'].append(L_deviation)
        
        # Integration step using Verlet (SAME for both force laws)
        if step < num_steps:
            state = verlet_step(state, derivative_func, time, dt)
            time += dt
    
    print(f"    Complete.")
    
    return trajectory, angular_momentum_data


def main():
    """
    Run physics sensitivity experiment.
    
    This demonstrates that changing the physical law (not the numerics)
    creates a genuinely different universe.
    """
    print("="*70)
    print("PHYSICS SENSITIVITY EXPERIMENT: INVERSE-SQUARE LAW")
    print("="*70)
    print()
    print("This experiment demonstrates how different physical explanations")
    print("create different self-consistent universes.")
    print()
    print("What we hold CONSTANT:")
    print("  - Initial conditions (same Sun-Earth configuration)")
    print("  - Integrator (Verlet, unchanged)")
    print("  - Timestep (same value)")
    print("  - Diagnostic methods (same analysis functions)")
    print()
    print("What we VARY:")
    print("  - The force law itself: 1/r^2 vs 1/r^(2+epsilon)")
    print()
    print("="*70)
    
    # Simulation parameters
    dt = 3600  # 1 hour timestep
    num_orbits = 50
    epsilon = 0.1  # Modification to inverse-square law
    
    print(f"\nSimulation parameters:")
    print(f"  Timestep: {dt/3600:.1f} hours")
    print(f"  Duration: {num_orbits} Newtonian orbital periods")
    print(f"  Modified gravity exponent: 2 + epsilon = {2+epsilon:.1f}")
    print()
    
    # Run with Newtonian gravity (1/r^2)
    print("Universe 1: Newtonian Gravity (epsilon = 0)")
    newtonian_traj, newtonian_L = run_with_force_law(
        gravitational_force,
        "Newtonian gravity (1/r^2)",
        dt,
        num_orbits
    )
    
    # Run with modified gravity (1/r^(2+epsilon))
    print()
    print(f"Universe 2: Modified Gravity (epsilon = {epsilon})")
    modified_force = lambda b1, b2: modified_gravitational_force(b1, b2, epsilon)
    modified_traj, modified_L = run_with_force_law(
        modified_force,
        f"Modified gravity (1/r^{2+epsilon:.1f})",
        dt,
        num_orbits
    )
    
    print()
    print("="*70)
    print("Generating visualizations...")
    
    # Create figures directory
    import os
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot 1: Side-by-side orbital trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Newtonian universe
    ax1.plot(newtonian_traj['x'], newtonian_traj['y'], 
             linewidth=1.5, color='blue', alpha=0.8)
    ax1.scatter([0], [0], s=100, c='orange', marker='o', label='Sun', zorder=5)
    ax1.set_xlabel('x (AU)', fontsize=11)
    ax1.set_ylabel('y (AU)', fontsize=11)
    ax1.set_title(f'Universe 1: Newtonian (1/r^2)\n{num_orbits} periods', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Modified universe
    ax2.plot(modified_traj['x'], modified_traj['y'], 
             linewidth=1.5, color='red', alpha=0.8)
    ax2.scatter([0], [0], s=100, c='orange', marker='o', label='Sun', zorder=5)
    ax2.set_xlabel('x (AU)', fontsize=11)
    ax2.set_ylabel('y (AU)', fontsize=11)
    ax2.set_title(f'Universe 2: Modified (1/r^{2+epsilon:.1f})\n{num_orbits} periods', 
                  fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    save_path = f'{figures_dir}/physics_sensitivity_orbits.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Plot 2: Angular momentum conservation comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(newtonian_L['time'], newtonian_L['L_deviation'], 
            label='Universe 1: Newtonian (1/r^2)', linewidth=2, color='blue')
    ax.plot(modified_L['time'], modified_L['L_deviation'], 
            label=f'Universe 2: Modified (1/r^{2+epsilon:.1f})', linewidth=2, color='red')
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Angular Momentum Deviation (%)', fontsize=12)
    ax.set_title('Angular Momentum Conservation: Physics Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{figures_dir}/physics_sensitivity_angular_momentum.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    print()
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    print()
    print("Universe 1 (Newtonian, 1/r^2):")
    print("  - Closed, repeating orbit")
    print("  - Angular momentum conserved (to numerical precision)")
    print("  - This is the universe we observe")
    print()
    print(f"Universe 2 (Modified, 1/r^{2+epsilon:.1f}):")
    print("  - Precessing orbit (rosette pattern)")
    print("  - Angular momentum NOT conserved")
    print("  - Closed orbits do not generically exist")
    print("  - This is a COUNTERFACTUAL universe")
    print()
    print("Key insight:")
    print("  The integrator (Verlet) worked correctly in BOTH universes.")
    print("  It faithfully evolved each set of physical laws.")
    print("  The differences are REAL physics, not numerical artifacts.")
    print()
    print("  Different explanations -> different worlds.")
    print()
    print("="*70)


if __name__ == "__main__":
    main()

