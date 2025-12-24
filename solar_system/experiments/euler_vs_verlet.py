"""
Direct comparison: Euler vs. Velocity-Verlet integration.

This experiment runs the SAME Sun-Earth system with both integrators
to demonstrate the dramatic difference in energy conservation.

Both use the same timestep, same initial conditions, same physics.
The ONLY difference is the numerical integration method.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.simulation.world import World
from solar_system.numerics.integrators import euler_step, verlet_step
from solar_system.analysis.conserved import (
    total_energy,
    total_momentum,
    total_angular_momentum
)


# Physical constants (SI units)
M_SUN = 1.989e30      # kg
M_EARTH = 5.972e24    # kg
AU = 1.496e11         # m
G = 6.674e-11         # m^3 kg^-1 s^-2

# Orbital period
T_ORBIT = 2 * np.pi * np.sqrt(AU**3 / (G * M_SUN))

# Circular orbital velocity
V_CIRCULAR = np.sqrt(G * M_SUN / AU)


def create_initial_system():
    """Create Sun-Earth system in circular orbit."""
    sun = Body("Sun", M_SUN, 
               np.array([0.0, 0.0, 0.0]), 
               np.array([0.0, 0.0, 0.0]))
    
    earth = Body("Earth", M_EARTH,
                 np.array([AU, 0.0, 0.0]),
                 np.array([0.0, V_CIRCULAR, 0.0]))
    
    return [sun, earth]


def run_comparison(dt, num_orbits=10):
    """
    Run both Euler and Verlet with the same timestep and compare.
    
    Args:
        dt: Timestep in seconds
        num_orbits: Number of orbits to simulate
    """
    duration = num_orbits * T_ORBIT
    num_steps = int(duration / dt)
    log_every = max(1, int((24 * 3600) / dt))  # Log once per day
    
    print(f"\n{'='*70}")
    print(f"EULER vs. VERLET COMPARISON")
    print(f"{'='*70}")
    print(f"Timestep: {dt/3600:.2f} hours ({dt:.0f} seconds)")
    print(f"Duration: {num_orbits} orbits ({duration/(24*3600):.1f} days)")
    print(f"Total steps: {num_steps}")
    print(f"Steps per orbit: {T_ORBIT/dt:.0f}")
    
    # Create two independent worlds (Euler and Verlet)
    bodies_euler = create_initial_system()
    bodies_verlet = create_initial_system()
    
    world_euler = World(bodies_euler, time=0.0)
    world_verlet = World(bodies_verlet, time=0.0)
    
    # Measure initial state (should be identical)
    E0 = total_energy(world_euler.bodies)
    P0 = total_momentum(world_euler.bodies)
    L0 = total_angular_momentum(world_euler.bodies)
    
    print(f"\nInitial conditions:")
    print(f"  Energy:    {E0:.6e} J")
    print(f"  Momentum:  {np.linalg.norm(P0):.6e} kg*m/s")
    print(f"  Ang. mom.: {np.linalg.norm(L0):.6e} kg*m^2/s")
    
    # Storage for diagnostics
    results = {
        'time': [],
        'orbit': [],
        'euler_E': [],
        'euler_dE': [],
        'verlet_E': [],
        'verlet_dE': [],
        'euler_r': [],
        'verlet_r': [],
    }
    
    print(f"\nRunning simulation...")
    print(f"{'Orbit':<8} {'Euler dE/E0':<15} {'Verlet dE/E0':<15} {'Ratio'}")
    print("-"*70)
    
    # Simulation loop
    for step in range(num_steps + 1):
        # Log diagnostics
        if step % log_every == 0:
            # Measure both systems
            E_euler = total_energy(world_euler.bodies)
            E_verlet = total_energy(world_verlet.bodies)
            
            dE_euler_pct = (E_euler - E0) / abs(E0) * 100
            dE_verlet_pct = (E_verlet - E0) / abs(E0) * 100
            
            r_euler = np.linalg.norm(world_euler.bodies[1].position) / AU
            r_verlet = np.linalg.norm(world_verlet.bodies[1].position) / AU
            
            orbit_num = world_euler.time / T_ORBIT
            
            # Store
            results['time'].append(world_euler.time / (24*3600))
            results['orbit'].append(orbit_num)
            results['euler_E'].append(E_euler)
            results['euler_dE'].append(dE_euler_pct)
            results['verlet_E'].append(E_verlet)
            results['verlet_dE'].append(dE_verlet_pct)
            results['euler_r'].append(r_euler)
            results['verlet_r'].append(r_verlet)
            
            # Print progress
            if step % (5 * log_every) == 0 or step == num_steps:
                ratio = abs(dE_euler_pct / dE_verlet_pct) if dE_verlet_pct != 0 else float('inf')
                print(f"{orbit_num:6.2f}   {dE_euler_pct:+8.5f}%      {dE_verlet_pct:+8.5f}%      {ratio:6.0f}x")
        
        # Take integration steps (skip on last iteration)
        if step < num_steps:
            world_euler.step(euler_step, dt)
            world_verlet.step(verlet_step, dt)
    
    # Final comparison
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS after {num_orbits} orbits:")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<15} {'Energy Drift':<15} {'Distance':<12} {'Status'}")
    print("-"*70)
    
    euler_final = results['euler_dE'][-1]
    verlet_final = results['verlet_dE'][-1]
    
    euler_status = "OK" if abs(euler_final) < 0.1 else "FAIL"
    verlet_status = "OK" if abs(verlet_final) < 0.1 else "FAIL"
    
    print(f"{'Euler':<15} {euler_final:+8.5f}%      {results['euler_r'][-1]:.4f} AU   {euler_status}")
    print(f"{'Verlet':<15} {verlet_final:+8.5f}%      {results['verlet_r'][-1]:.4f} AU   {verlet_status}")
    
    # Improvement factor
    if verlet_final != 0:
        improvement = abs(euler_final / verlet_final)
        print(f"\nVerlet is {improvement:.0f}x better at conserving energy!")
    
    print(f"{'='*70}")
    
    return results


def main():
    """Run comparison experiments with different timesteps."""
    print("\n" + "="*70)
    print("EULER vs. VELOCITY-VERLET: ENERGY CONSERVATION COMPARISON")
    print("="*70)
    
    # Test with a moderate timestep (1 hour)
    print("\nTest 1: 1-hour timestep, 10 orbits")
    results_1hr = run_comparison(dt=3600, num_orbits=10)
    
    # Test with a larger timestep (6 hours)
    print("\n\nTest 2: 6-hour timestep, 10 orbits")
    results_6hr = run_comparison(dt=21600, num_orbits=10)
    
    print("\n\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Euler shows SYSTEMATIC energy drift (monotonic increase/decrease)")
    print("2. Verlet shows BOUNDED energy oscillation (no long-term drift)")
    print("3. The improvement factor is typically 100x-10000x")
    print("4. Verlet is SYMPLECTIC: preserves Hamiltonian structure")
    print("5. Both integrators still conserve momentum perfectly")
    print("="*70)
    
    return results_1hr, results_6hr


if __name__ == "__main__":
    results = main()

