"""
Controlled experiment demonstrating Euler integration failure for orbital mechanics.

This script runs the Sun-Earth system with various timesteps to show:
1. Energy conservation violation (primary failure mode)
2. Momentum conservation (should hold)
3. Angular momentum drift (small)
4. Orbital degradation (spiraling)

The experiment compares 5 timestep choices from fine (10 min) to catastrophic (3 days).
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from solar_system.physics.bodies import Body
from solar_system.simulation.world import World
from solar_system.numerics.integrators import euler_step
from solar_system.analysis.conserved import (
    total_energy,
    total_momentum,
    total_angular_momentum
)


# Physical constants (SI units)
M_SUN = 1.989e30      # kg
M_EARTH = 5.972e24    # kg
AU = 1.496e11         # m (1 astronomical unit)
G = 6.674e-11         # m^3 kg^-1 s^-2

# Orbital period for circular orbit at 1 AU
T_ORBIT = 2 * np.pi * np.sqrt(AU**3 / (G * M_SUN))  # seconds (~365.25 days)

# Circular orbital velocity at 1 AU
V_CIRCULAR = np.sqrt(G * M_SUN / AU)  # m/s (~29.78 km/s)


def create_initial_system():
    """
    Create Sun-Earth system with Earth in circular orbit at 1 AU.
    
    Returns:
        List of two Body objects [Sun, Earth]
    """
    sun = Body(
        name="Sun",
        mass=M_SUN,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    earth = Body(
        name="Earth",
        mass=M_EARTH,
        position=np.array([AU, 0.0, 0.0]),      # 1 AU on +x axis
        velocity=np.array([0.0, V_CIRCULAR, 0.0])  # Moving in +y direction
    )
    
    return [sun, earth]


def run_experiment(name, dt, duration, log_interval):
    """
    Run a single experiment with given timestep.
    
    Args:
        name: Descriptive name for this test case
        dt: Timestep in seconds
        duration: Total simulation time in seconds
        log_interval: How often to log diagnostics (seconds)
    
    Returns:
        Dictionary of diagnostic data
    """
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Timestep: {dt/3600:.2f} hours ({dt:.0f} seconds)")
    print(f"Duration: {duration/T_ORBIT:.2f} orbits ({duration/(24*3600):.1f} days)")
    print(f"Steps per orbit: {T_ORBIT/dt:.0f}")
    
    # Create initial system
    bodies = create_initial_system()
    world = World(bodies, time=0.0)
    
    # Initial measurements
    E0 = total_energy(world.bodies)
    P0 = total_momentum(world.bodies)
    L0 = total_angular_momentum(world.bodies)
    
    print(f"\nInitial conditions:")
    print(f"  Total energy:    {E0:.6e} J")
    print(f"  Total momentum:  {np.linalg.norm(P0):.6e} kg*m/s")
    print(f"  Angular momentum: {np.linalg.norm(L0):.6e} kg*m^2/s")
    
    # Storage for diagnostics
    diagnostics = {
        'time': [],      # days
        'orbit': [],     # orbit number
        'E': [],         # total energy (J)
        'dE_percent': [], # energy drift (%)
        'P_drift': [],   # momentum drift magnitude
        'L_drift': [],   # angular momentum drift magnitude
        'x_AU': [],      # Earth x position (AU)
        'y_AU': [],      # Earth y position (AU)
        'r_AU': [],      # Earth distance from Sun (AU)
        'v_kms': [],     # Earth speed (km/s)
    }
    
    # Simulation loop
    num_steps = int(duration / dt)
    log_every = max(1, int(log_interval / dt))
    
    print(f"\nRunning {num_steps} steps...")
    
    for step in range(num_steps + 1):  # +1 to include final state
        # Log diagnostics at specified intervals
        if step % log_every == 0:
            # Measure conserved quantities
            E = total_energy(world.bodies)
            P = total_momentum(world.bodies)
            L = total_angular_momentum(world.bodies)
            
            # Earth's state
            earth_pos = world.bodies[1].position
            earth_vel = world.bodies[1].velocity
            
            # Compute derived quantities
            time_days = world.time / (24 * 3600)
            orbit_num = world.time / T_ORBIT
            dE_percent = (E - E0) / abs(E0) * 100
            P_drift = np.linalg.norm(P - P0)
            L_drift = np.linalg.norm(L - L0)
            r_AU = np.linalg.norm(earth_pos) / AU
            v_kms = np.linalg.norm(earth_vel) / 1000
            
            # Store
            diagnostics['time'].append(time_days)
            diagnostics['orbit'].append(orbit_num)
            diagnostics['E'].append(E)
            diagnostics['dE_percent'].append(dE_percent)
            diagnostics['P_drift'].append(P_drift)
            diagnostics['L_drift'].append(L_drift)
            diagnostics['x_AU'].append(earth_pos[0] / AU)
            diagnostics['y_AU'].append(earth_pos[1] / AU)
            diagnostics['r_AU'].append(r_AU)
            diagnostics['v_kms'].append(v_kms)
            
            # Print periodic progress
            if step % (5 * log_every) == 0 or step == num_steps:
                print(f"  Orbit {orbit_num:5.2f} | "
                      f"dE/E0 = {dE_percent:+8.4f}% | "
                      f"r = {r_AU:.4f} AU | "
                      f"v = {v_kms:.2f} km/s")
        
        # Take integration step (skip on last iteration)
        if step < num_steps:
            world.step(euler_step, dt)
    
    # Final summary
    print(f"\n{'-'*70}")
    print(f"FINAL STATE after {diagnostics['orbit'][-1]:.2f} orbits:")
    print(f"  Energy drift:       {diagnostics['dE_percent'][-1]:+.4f}%")
    print(f"  Momentum drift:     {diagnostics['P_drift'][-1]:.6e} kg*m/s")
    print(f"  Ang. mom. drift:    {diagnostics['L_drift'][-1]:.6e} kg*m^2/s")
    print(f"  Earth distance:     {diagnostics['r_AU'][-1]:.4f} AU")
    print(f"  Earth speed:        {diagnostics['v_kms'][-1]:.2f} km/s")
    print(f"{'-'*70}")
    
    return diagnostics


def main():
    """
    Run the complete experiment with multiple timestep values.
    """
    print("\n" + "="*70)
    print("EULER INTEGRATION FAILURE EXPERIMENT")
    print("Sun-Earth System | Circular Orbit at 1 AU")
    print("="*70)
    print(f"\nOrbital parameters:")
    print(f"  Radius: {AU:.6e} m (1 AU)")
    print(f"  Velocity: {V_CIRCULAR:.6e} m/s ({V_CIRCULAR/1000:.2f} km/s)")
    print(f"  Period: {T_ORBIT:.6e} s ({T_ORBIT/(24*3600):.2f} days)")
    
    # Define test cases: (name, dt, duration, log_interval)
    test_cases = [
        ("CASE A: Timestep = 10 minutes", 
         600, 2 * T_ORBIT, 24 * 3600),
        
        ("CASE B: Timestep = 1 hour", 
         3600, 2 * T_ORBIT, 24 * 3600),
        
        ("CASE C: Timestep = 6 hours", 
         21600, 2 * T_ORBIT, 6 * 3600),
        
        ("CASE D: Timestep = 1 day", 
         86400, 1 * T_ORBIT, 6 * 3600),
        
        ("CASE E: Timestep = 3 days (CATASTROPHIC)", 
         259200, 0.5 * T_ORBIT, 259200),
    ]
    
    results = {}
    
    # Run each test case
    for name, dt, duration, log_interval in test_cases:
        try:
            diagnostics = run_experiment(name, dt, duration, log_interval)
            results[name] = diagnostics
        except Exception as e:
            print(f"\nWARNING: {name} FAILED: {e}")
            results[name] = None
    
    # Summary comparison
    print("\n\n" + "="*70)
    print("SUMMARY: Energy Drift Comparison")
    print("="*70)
    print(f"{'Case':<35} {'Timestep':<12} {'Energy Drift':<15} {'Status'}")
    print("-"*70)
    
    for name, dt, duration, log_interval in test_cases:
        if results[name] is not None:
            final_drift = results[name]['dE_percent'][-1]
            status = "OK" if abs(final_drift) < 1.0 else "FAIL"
            print(f"{name:<35} {dt/3600:>6.2f} hours   {final_drift:>+8.4f}%       {status}")
        else:
            print(f"{name:<35} {dt/3600:>6.2f} hours   {'CRASHED':<15} FAIL")
    
    print("="*70)
    print("\nKEY OBSERVATIONS:")
    print("  * Energy drift increases with larger timesteps (O(dt) error)")
    print("  * Momentum is conserved in all cases (Newton's 3rd law)")
    print("  * Large timesteps cause orbit to spiral in/out")
    print("  * Euler is fundamentally unsuitable for long-term orbital mechanics")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()

