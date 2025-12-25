"""
Invariant checking experiment: Euler vs Verlet.

This experiment demonstrates observational auditing of simulations.
We track which invariants are preserved by which integrators, without
enforcing anything — just observing and reporting.

Key insight:
    Same physics + different time evolution = different invariant preservation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.simulation.world import World
from solar_system.numerics.integrators import euler_step, verlet_step
from solar_system.analysis.conserved import total_energy, total_angular_momentum
from solar_system.invariants import Invariant, InvariantChecker


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


def run_with_invariant_tracking(integrator, integrator_name, dt, num_orbits=10):
    """
    Run simulation and track invariants.
    
    Args:
        integrator: Integration function (euler_step or verlet_step)
        integrator_name: String name for reporting
        dt: Timestep in seconds
        num_orbits: Number of orbits to simulate
    
    Returns:
        Dictionary of InvariantChecker objects (by invariant name)
    """
    duration = num_orbits * T_ORBIT
    num_steps = int(duration / dt)
    
    print(f"\nRunning {integrator_name}...")
    print(f"  Duration: {num_orbits} orbits ({num_steps} steps)")
    
    # Create world
    bodies = create_sun_earth_system()
    world = World(bodies, time=0.0)
    
    # Define invariants to observe
    invariants = [
        Invariant(
            name="Energy",
            description="Total mechanical energy (kinetic + potential)",
            measure=total_energy
        ),
        Invariant(
            name="Angular Momentum",
            description="Total angular momentum about origin",
            measure=total_angular_momentum
        ),
    ]
    
    # Create checkers for each invariant
    checkers = {inv.name: InvariantChecker(inv) for inv in invariants}
    
    # Initial observation
    for checker in checkers.values():
        checker.observe(world.bodies, world.time)
    
    # Simulation loop
    for step in range(num_steps):
        # Time evolution
        world.step(integrator, dt)
        
        # Observe invariants (every step for accurate tracking)
        for checker in checkers.values():
            checker.observe(world.bodies, world.time)
    
    print(f"  Complete.")
    
    return checkers


def visualize_invariant_comparison(euler_checkers, verlet_checkers, num_orbits):
    """
    Visualize invariant deviation over time for Euler vs Verlet.
    
    Extracts history from InvariantChecker objects and plots deviation
    as plain arrays. Rendering consumes only data, not checker objects.
    
    Args:
        euler_checkers: Dict of InvariantChecker objects from Euler run
        verlet_checkers: Dict of InvariantChecker objects from Verlet run
        num_orbits: Number of orbits simulated (for titles)
    """
    import os
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Extract history data for energy
    euler_energy_history = euler_checkers["Energy"].get_history()
    verlet_energy_history = verlet_checkers["Energy"].get_history()
    
    # Convert to plain arrays with relative deviation
    E0_euler = euler_energy_history[0][1]
    E0_verlet = verlet_energy_history[0][1]
    
    euler_energy_time = np.array([t for t, _ in euler_energy_history]) / (24*3600)  # to days
    euler_energy_dev = np.array([(E - E0_euler) / abs(E0_euler) * 100 
                                   for _, E in euler_energy_history])
    
    verlet_energy_time = np.array([t for t, _ in verlet_energy_history]) / (24*3600)
    verlet_energy_dev = np.array([(E - E0_verlet) / abs(E0_verlet) * 100 
                                    for _, E in verlet_energy_history])
    
    # Extract history data for angular momentum
    euler_L_history = euler_checkers["Angular Momentum"].get_history()
    verlet_L_history = verlet_checkers["Angular Momentum"].get_history()
    
    # Convert to plain arrays with relative deviation (using magnitude for vectors)
    L0_euler = np.linalg.norm(euler_L_history[0][1])
    L0_verlet = np.linalg.norm(verlet_L_history[0][1])
    
    euler_L_time = np.array([t for t, _ in euler_L_history]) / (24*3600)
    euler_L_dev = np.array([(np.linalg.norm(L) - L0_euler) / L0_euler * 100 
                             for _, L in euler_L_history])
    
    verlet_L_time = np.array([t for t, _ in verlet_L_history]) / (24*3600)
    verlet_L_dev = np.array([(np.linalg.norm(L) - L0_verlet) / L0_verlet * 100 
                               for _, L in verlet_L_history])
    
    # Plot 1: Energy deviation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(euler_energy_time, euler_energy_dev, 
            label='Euler', color='red', linewidth=2, alpha=0.8)
    ax.plot(verlet_energy_time, verlet_energy_dev, 
            label='Verlet', color='blue', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Energy Deviation (%)', fontsize=12)
    ax.set_title(f'Energy Conservation: Euler vs Verlet ({num_orbits} orbits)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{figures_dir}/invariant_energy_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Plot 2: Angular momentum deviation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(euler_L_time, euler_L_dev, 
            label='Euler', color='red', linewidth=2, alpha=0.8)
    ax.plot(verlet_L_time, verlet_L_dev, 
            label='Verlet', color='blue', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Angular Momentum Deviation (%)', fontsize=12)
    ax.set_title(f'Angular Momentum Conservation: Euler vs Verlet ({num_orbits} orbits)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{figures_dir}/invariant_angular_momentum_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """
    Run Euler and Verlet with invariant tracking, then report findings.
    """
    print("="*70)
    print("INVARIANT OBSERVATIONAL AUDIT: EULER vs VERLET")
    print("="*70)
    print()
    print("This experiment observes which invariants are preserved by which")
    print("integrators. We do not enforce invariants — we simply measure them.")
    print()
    print("Setup:")
    print("  - Physics: Newtonian gravity (claims: energy, momentum, ang. momentum)")
    print("  - System: Sun-Earth, circular orbit")
    print("  - Duration: 10 orbital periods")
    print("  - Observed: Energy, Angular Momentum")
    print()
    print("="*70)
    
    # Simulation parameters
    dt = 3600  # 1 hour
    num_orbits = 10
    
    # Run with Euler
    euler_checkers = run_with_invariant_tracking(
        euler_step, "Euler", dt, num_orbits
    )
    
    # Run with Verlet
    verlet_checkers = run_with_invariant_tracking(
        verlet_step, "Verlet", dt, num_orbits
    )
    
    # Report findings
    print()
    print("="*70)
    print(f"INVARIANT REPORT ({num_orbits} orbits):")
    print("="*70)
    print()
    
    # Energy
    print("Energy:")
    euler_energy_change = euler_checkers["Energy"].relative_change_percent()
    verlet_energy_change = verlet_checkers["Energy"].relative_change_percent()
    print(f"  Euler   -> {euler_energy_change:+.6f} %")
    print(f"  Verlet  -> {verlet_energy_change:+.10f} %")
    print()
    
    # Angular Momentum
    print("Angular Momentum:")
    euler_L_change = euler_checkers["Angular Momentum"].relative_change_percent()
    verlet_L_change = verlet_checkers["Angular Momentum"].relative_change_percent()
    print(f"  Euler   -> {euler_L_change:+.6f} %")
    print(f"  Verlet  -> {verlet_L_change:+.10f} %")
    print()
    
    # Interpretation
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    print()
    print("Energy Conservation:")
    print("  - Newtonian gravity CLAIMS this should be conserved")
    print(f"  - Euler VIOLATES: {abs(euler_energy_change):.2f}% drift")
    print(f"  - Verlet RESPECTS: {abs(verlet_energy_change):.2e}% bounded oscillation")
    print()
    print("Angular Momentum Conservation:")
    print("  - Newtonian gravity CLAIMS this should be conserved")
    print(f"  - Euler VIOLATES: {abs(euler_L_change):.2e}% drift")
    print(f"  - Verlet RESPECTS: {abs(verlet_L_change):.2e}% (numerical precision)")
    print()
    print("Key finding:")
    print("  Euler violates BOTH energy and angular momentum conservation.")
    print("  Verlet preserves BOTH (it is symplectic and time-reversible).")
    print("  Euler creates a self-consistent but NON-PHYSICAL universe where")
    print("  conserved quantities drift systematically.")
    print()
    print("="*70)
    
    # Visualize invariant evolution
    print()
    print("Generating invariant deviation plots...")
    visualize_invariant_comparison(euler_checkers, verlet_checkers, num_orbits)
    print()
    print("Done!")


if __name__ == "__main__":
    main()

