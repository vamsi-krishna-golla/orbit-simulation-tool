"""
Numerical integration schemes for ordinary differential equations.

These are pure mathematical methods, independent of any specific physics.
All integrators solve dy/dt = f(y, t) for various numerical schemes.
"""

import numpy as np
from typing import Callable


def euler_step(
    state: np.ndarray,
    derivative_func: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float
) -> np.ndarray:
    """
    Advance state by one timestep using the Euler method.
    
    Mathematical formulation:
        y(t + dt) = y(t) + f(y(t), t) * dt
    
    where:
        - y is the state vector (abstract, can represent anything)
        - f(y, t) is the derivative function (dy/dt)
        - dt is the timestep
    
    Physical interpretation:
        Assumes the derivative (rate of change) is CONSTANT throughout
        the timestep. This is equivalent to linear extrapolation along
        the tangent to the true trajectory.
    
    What Euler assumes:
        1. The state changes linearly during dt
        2. The derivative at t is valid for the entire interval [t, t+dt]
        3. Higher-order terms in the Taylor series are negligible
    
    What errors it introduces:
        1. Local truncation error: O(dt²) per step
           - The true solution has curvature; Euler follows a straight line
        2. Global error: O(dt) over long integration
           - Errors accumulate linearly with the number of steps
        3. Energy drift: For Hamiltonian systems, energy is NOT conserved
           - Orbits spiral in or out over time
        4. No symplectic structure: Phase space volume not preserved
    
    When to use Euler:
        - Educational purposes (shows what discretization means)
        - Very short integrations where accuracy isn't critical
        - When you want to see numerical integration fail instructively
    
    When NOT to use Euler:
        - Long-term orbital mechanics (energy drift is catastrophic)
        - Any problem requiring energy conservation
        - Production simulations (use Verlet, RK4, or adaptive methods)
    
    Args:
        state: Current state vector y(t), shape (n,)
        derivative_func: Function f(y, t) that computes dy/dt
        t: Current time
        dt: Timestep (must be positive and small enough for stability)
    
    Returns:
        New state vector y(t + dt), shape (n,)
    
    Notes:
        - This is a pure function: no side effects, no mutation
        - The derivative_func encapsulates all physics/dynamics
        - This integrator is completely agnostic to what the state represents
        - No validation is performed on dt (caller's responsibility)
    
    Example:
        # Solve dy/dt = -y (exponential decay)
        def f(y, t):
            return -y
        
        y0 = np.array([1.0])
        y1 = euler_step(y0, f, t=0.0, dt=0.1)
        # y1 ≈ [0.9], true solution is exp(-0.1) ≈ 0.9048
    """
    # Evaluate derivative at current state
    dydt = derivative_func(state, t)
    
    # Euler update: straight-line extrapolation
    new_state = state + dydt * dt
    
    return new_state


def verlet_step(
    state: np.ndarray,
    derivative_func: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float
) -> np.ndarray:
    """
    Advance state by one timestep using the velocity-Verlet method.
    
    Mathematical formulation (for second-order ODEs):
        v(t + dt/2) = v(t) + a(t) * dt/2           [half-step velocity]
        r(t + dt) = r(t) + v(t + dt/2) * dt        [full-step position]
        a(t + dt) = compute from r(t + dt)         [new acceleration]
        v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2 [full-step velocity]
    
    where:
        - r is position, v is velocity, a is acceleration
        - This is the "leapfrog" scheme: positions and velocities
          are updated in a staggered manner
    
    Physical interpretation:
        Velocity-Verlet is a SYMPLECTIC integrator. It preserves the
        Hamiltonian structure of mechanical systems, which means:
        - Energy is approximately conserved over long times
        - Phase space volume is preserved (Liouville's theorem)
        - Orbits remain bounded (no artificial spiraling)
    
    What velocity-Verlet assumes:
        1. The system is governed by second-order ODEs (r'=v, v'=a)
        2. Acceleration depends only on position, not velocity
        3. The symplectic structure matters (Hamiltonian mechanics)
    
    What errors it introduces:
        1. Local truncation error: O(dt³) per step
           - Better than Euler's O(dt²)
        2. Global error: O(dt²) over long integration
           - Much better than Euler's O(dt)
        3. Energy oscillates slightly but does NOT drift systematically
           - This is the key advantage for orbital mechanics
        4. Time-reversible: running backwards recovers the initial state
    
    When to use velocity-Verlet:
        - Long-term orbital mechanics (planets, satellites)
        - Molecular dynamics simulations
        - Any Hamiltonian system where energy conservation matters
        - When stability is more important than raw accuracy
    
    When NOT to use velocity-Verlet:
        - Systems with velocity-dependent forces (drag, friction)
        - Non-Hamiltonian systems
        - When you need very high accuracy (use higher-order methods)
    
    Args:
        state: Current state vector [positions..., velocities...], shape (2N,)
               where N is the total number of position coordinates
        derivative_func: Function f(y, t) that returns [velocities..., accelerations...]
        t: Current time
        dt: Timestep
    
    Returns:
        New state vector [new_positions..., new_velocities...], shape (2N,)
    
    Notes:
        - This is a pure function: no side effects, no mutation
        - Requires TWO calls to derivative_func per step (vs. one for Euler)
        - The derivative_func must return velocities and accelerations separately
        - State structure MUST be [positions, velocities] (first half, second half)
        - This integrator is completely agnostic to the physics
    
    Implementation note:
        The state vector is structured as:
            [x1, y1, z1, ..., xN, yN, zN, vx1, vy1, vz1, ..., vxN, vyN, vzN]
        
        The derivative_func returns:
            [vx1, vy1, vz1, ..., vxN, vyN, vzN, ax1, ay1, az1, ..., axN, ayN, azN]
        
        We split these in half to extract positions, velocities, and accelerations.
    """
    # Determine size of state (must be even: half positions, half velocities)
    n = len(state)
    if n % 2 != 0:
        raise ValueError(f"State vector must have even length (got {n})")
    
    half = n // 2
    
    # Extract current positions and velocities
    r_current = state[:half]      # First half: positions
    v_current = state[half:]      # Second half: velocities
    
    # Step 1: Compute current accelerations
    # derivative_func returns [velocities, accelerations]
    derivative = derivative_func(state, t)
    a_current = derivative[half:]  # Second half: accelerations
    
    # Step 2: Half-step velocity update
    v_half = v_current + a_current * (dt / 2.0)
    
    # Step 3: Full-step position update (using half-step velocity)
    r_new = r_current + v_half * dt
    
    # Step 4: Build intermediate state for computing new accelerations
    # We need accelerations at the new positions
    intermediate_state = np.concatenate([r_new, v_half])
    
    # Step 5: Compute new accelerations at new positions
    derivative_new = derivative_func(intermediate_state, t + dt)
    a_new = derivative_new[half:]  # Second half: accelerations
    
    # Step 6: Complete velocity update (second half-step)
    v_new = v_half + a_new * (dt / 2.0)
    
    # Step 7: Assemble new state [positions, velocities]
    new_state = np.concatenate([r_new, v_new])
    
    return new_state

