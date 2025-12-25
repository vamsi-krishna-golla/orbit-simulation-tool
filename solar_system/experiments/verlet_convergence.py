"""
Verlet integrator convergence study: timestep vs energy error.

This experiment demonstrates how the Verlet integrator's energy conservation
improves as timestep decreases. We expect error to scale as O(dt^2) for a
second-order accurate method.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.simulation.world import World
from solar_system.numerics.integrators import verlet_step
from solar_system.analysis.conserved import total_energy


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


def run_simulation(dt, num_orbits=10):
    """
    Run Sun-Earth simulation with given timestep.
    
    Args:
        dt: Timestep in seconds
        num_orbits: Number of orbits to simulate
    
    Returns:
        max_energy_error: Maximum relative energy error (as percentage)
    """
    duration = num_orbits * T_ORBIT
    num_steps = int(duration / dt)
    
    # Create world
    bodies = create_sun_earth_system()
    world = World(bodies, time=0.0)
    
    # Initial energy
    E0 = total_energy(world.bodies)
    
    # Track maximum energy deviation
    max_deviation = 0.0
    
    # Simulation loop
    for step in range(num_steps):
        # Measure energy
        E = total_energy(world.bodies)
        deviation = abs((E - E0) / E0) * 100
        max_deviation = max(max_deviation, deviation)
        
        # Integration step
        world.step(verlet_step, dt)
    
    # Final energy check
    E_final = total_energy(world.bodies)
    final_deviation = abs((E_final - E0) / E0) * 100
    max_deviation = max(max_deviation, final_deviation)
    
    return max_deviation


def main():
    """Run convergence study with multiple timesteps."""
    print("="*70)
    print("VERLET INTEGRATOR CONVERGENCE STUDY")
    print("="*70)
    print("\nTesting multiple timestep values...\n")
    
    # Timestep values to test (in hours)
    timesteps_hours = [0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0]
    
    # Convert to seconds
    timesteps_seconds = [dt_hr * 3600 for dt_hr in timesteps_hours]
    
    # Storage for results
    max_errors = []
    
    print(f"{'Timestep (hrs)':<15} {'Steps/Orbit':<15} {'Max Energy Error (%)'}")
    print("-"*70)
    
    # Run simulations
    for dt_hr, dt_sec in zip(timesteps_hours, timesteps_seconds):
        steps_per_orbit = T_ORBIT / dt_sec
        
        print(f"{dt_hr:<15.1f} {steps_per_orbit:<15.0f} ", end="", flush=True)
        
        max_error = run_simulation(dt_sec, num_orbits=10)
        max_errors.append(max_error)
        
        print(f"{max_error:.6e}")
    
    print("\n" + "="*70)
    print("Generating convergence plot...")
    
    # Create log-log plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.loglog(timesteps_hours, max_errors, 'o-', linewidth=2, markersize=8, 
              label='Verlet integrator')
    
    # Add reference line for O(dt^2) scaling
    # Pick a point in the middle for reference
    ref_idx = len(timesteps_hours) // 2
    ref_dt = timesteps_hours[ref_idx]
    ref_error = max_errors[ref_idx]
    
    # Generate O(dt^2) reference line
    dt_range = np.array(timesteps_hours)
    reference_line = ref_error * (dt_range / ref_dt)**2
    
    ax.loglog(dt_range, reference_line, '--', color='gray', linewidth=1.5,
              label='O(dt²) reference')
    
    # Formatting
    ax.set_xlabel('Timestep (hours)', fontsize=12)
    ax.set_ylabel('Maximum Energy Error (%)', fontsize=12)
    ax.set_title('Verlet Integrator: Convergence Study (10 orbits)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    # Save plot
    import os
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    save_path = f'{figures_dir}/verlet_convergence.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Error decreases as timestep decreases (smaller is better)")
    print("2. Verlet is a 2nd-order method: error scales as O(dt²)")
    print("3. Halving the timestep reduces error by ~4x")
    print("4. Even with large timesteps, Verlet shows bounded errors")
    print("="*70)


if __name__ == "__main__":
    main()

