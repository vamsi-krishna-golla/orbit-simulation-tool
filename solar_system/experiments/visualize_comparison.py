"""
Visualization comparison: Euler vs Verlet integration.

This experiment demonstrates the difference between Euler and Verlet methods
by visualizing both the orbital trajectories and energy conservation.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.simulation.world import World
from solar_system.numerics.integrators import euler_step, verlet_step
from solar_system.analysis.conserved import total_energy
from solar_system.rendering.plot_2d import plot_orbit_xy, plot_energy_vs_time


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


def main():
    """Run Sun-Earth simulation and visualize Euler vs Verlet comparison."""
    
    # Simulation parameters
    dt = 3600  # 1 hour timestep
    num_orbits = 10
    duration = num_orbits * T_ORBIT
    num_steps = int(duration / dt)
    log_every = max(1, int((24 * 3600) / dt))  # Log once per day
    
    print(f"Running Sun-Earth simulation")
    print(f"Timestep: {dt/3600:.1f} hours")
    print(f"Duration: {num_orbits} orbits")
    print(f"Total steps: {num_steps}")
    
    # Create two independent worlds
    bodies_euler = create_sun_earth_system()
    bodies_verlet = create_sun_earth_system()
    
    world_euler = World(bodies_euler, time=0.0)
    world_verlet = World(bodies_verlet, time=0.0)
    
    # Initial energy
    E0 = total_energy(world_euler.bodies)
    
    # Data storage for Euler
    euler_trajectory = {
        'time': [],
        'earth_x': [],
        'earth_y': [],
    }
    euler_energy = {
        'time': [],
        'energy': [],
    }
    
    # Data storage for Verlet
    verlet_trajectory = {
        'time': [],
        'earth_x': [],
        'earth_y': [],
    }
    verlet_energy = {
        'time': [],
        'energy': [],
    }
    
    # Simulation loop
    print("Simulating...")
    for step in range(num_steps + 1):
        # Log data
        if step % log_every == 0:
            # Euler data
            earth_euler = world_euler.bodies[1]
            euler_trajectory['time'].append(world_euler.time / (24*3600))
            euler_trajectory['earth_x'].append(earth_euler.position[0] / AU)
            euler_trajectory['earth_y'].append(earth_euler.position[1] / AU)
            
            E_euler = total_energy(world_euler.bodies)
            euler_energy['time'].append(world_euler.time / (24*3600))
            euler_energy['energy'].append((E_euler - E0) / abs(E0) * 100)
            
            # Verlet data
            earth_verlet = world_verlet.bodies[1]
            verlet_trajectory['time'].append(world_verlet.time / (24*3600))
            verlet_trajectory['earth_x'].append(earth_verlet.position[0] / AU)
            verlet_trajectory['earth_y'].append(earth_verlet.position[1] / AU)
            
            E_verlet = total_energy(world_verlet.bodies)
            verlet_energy['time'].append(world_verlet.time / (24*3600))
            verlet_energy['energy'].append((E_verlet - E0) / abs(E0) * 100)
        
        # Integration step
        if step < num_steps:
            world_euler.step(euler_step, dt)
            world_verlet.step(verlet_step, dt)
    
    print("Simulation complete. Generating plots...")
    
    # Create figures directory if it doesn't exist
    import os
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot 1: Orbital trajectories
    trajectories = [
        {
            'label': 'Euler',
            'x': euler_trajectory['earth_x'],
            'y': euler_trajectory['earth_y'],
        },
        {
            'label': 'Verlet',
            'x': verlet_trajectory['earth_x'],
            'y': verlet_trajectory['earth_y'],
        },
    ]
    
    plot_orbit_xy(
        trajectories,
        title=f'Earth Orbit: Euler vs Verlet ({num_orbits} orbits)',
        xlabel='x (AU)',
        ylabel='y (AU)',
        save_path=f'{figures_dir}/orbit_euler_vs_verlet.png',
    )
    print(f"  Saved: {figures_dir}/orbit_euler_vs_verlet.png")
    
    # Plot 2: Energy drift for Euler
    energy_data_euler = {
        'time': euler_energy['time'],
        'energy': euler_energy['energy'],
    }
    
    plot_energy_vs_time(
        energy_data_euler,
        title=f'Energy Drift: Euler Method ({num_orbits} orbits)',
        xlabel='Time (days)',
        ylabel='Energy Error (%)',
        color='red',
        save_path=f'{figures_dir}/energy_euler.png',
    )
    print(f"  Saved: {figures_dir}/energy_euler.png")
    
    # Plot 3: Energy drift for Verlet
    energy_data_verlet = {
        'time': verlet_energy['time'],
        'energy': verlet_energy['energy'],
    }
    
    plot_energy_vs_time(
        energy_data_verlet,
        title=f'Energy Conservation: Verlet Method ({num_orbits} orbits)',
        xlabel='Time (days)',
        ylabel='Energy Error (%)',
        color='green',
        save_path=f'{figures_dir}/energy_verlet.png',
    )
    print(f"  Saved: {figures_dir}/energy_verlet.png")
    
    print("\nDone! All plots saved to solar_system/figures/ directory.")


if __name__ == "__main__":
    main()

