"""
Simulation orchestration layer for N-body systems.

This module bridges the physics layer (bodies, forces) and the numerics
layer (integrators, state vectors). It manages:
  - Conversion between Body objects and abstract state vectors
  - Construction of derivative functions from physics laws
  - Time evolution using numerical integrators

No physics is implemented here (delegated to physics layer).
No numerical methods are implemented here (delegated to numerics layer).
This is pure coordination and translation.
"""

import numpy as np
from typing import List, Callable
from solar_system.physics.bodies import Body
from solar_system.physics.gravity import gravitational_force
from solar_system.physics.dynamics import acceleration


class World:
    """
    Manages the state and evolution of an N-body gravitational system.
    
    Responsibilities:
        - Store the list of bodies
        - Pack bodies into state vectors for integrators
        - Unpack state vectors back into bodies
        - Construct derivative functions from physics laws
        - Advance time using numerical integrators
    
    State vector convention:
        For N bodies, the state is a flat array of 6N numbers:
            [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN,     # All positions
             vx1, vy1, vz1, vx2, vy2, vz2, ..., vxN, vyN, vzN]  # All velocities
        
        First 3N numbers: positions of all bodies (body1.pos, body2.pos, ...)
        Last 3N numbers: velocities of all bodies (body1.vel, body2.vel, ...)
    
    Attributes:
        bodies: List of Body objects in the system
        time: Current simulation time (seconds)
    """
    
    def __init__(self, bodies: List[Body], time: float = 0.0):
        """
        Initialize the world with a list of bodies.
        
        Args:
            bodies: List of Body objects to simulate
            time: Initial simulation time (default: 0.0 seconds)
        """
        self.bodies = bodies
        self.time = time
    
    def pack_state(self) -> np.ndarray:
        """
        Convert the current bodies into a state vector.
        
        Packs positions and velocities of all bodies into a flat array
        according to the state convention: [all positions, all velocities].
        
        Returns:
            State vector of shape (6N,) where N is the number of bodies
        
        Notes:
            - Masses are NOT included in the state (they're constants)
            - The ordering must match what unpack_state() expects
            - This is a pure function (bodies are not modified)
        """
        n = len(self.bodies)
        state = np.zeros(6 * n)
        
        # Pack all positions first
        for i, body in enumerate(self.bodies):
            state[3*i : 3*i + 3] = body.position
        
        # Pack all velocities second
        for i, body in enumerate(self.bodies):
            state[3*n + 3*i : 3*n + 3*i + 3] = body.velocity
        
        return state
    
    def unpack_state(self, state: np.ndarray) -> None:
        """
        Update bodies from a state vector.
        
        Extracts positions and velocities from the state vector and updates
        the Body objects in place. This is the ONLY place where bodies are
        mutated during simulation.
        
        Args:
            state: State vector of shape (6N,) where N is the number of bodies
        
        Notes:
            - Masses are unchanged (not part of the state)
            - Names are unchanged (not part of the state)
            - The ordering must match what pack_state() produces
            - Bodies are mutated in place for efficiency
        """
        n = len(self.bodies)
        
        # Unpack positions
        for i, body in enumerate(self.bodies):
            body.position = state[3*i : 3*i + 3].copy()
        
        # Unpack velocities
        for i, body in enumerate(self.bodies):
            body.velocity = state[3*n + 3*i : 3*n + 3*i + 3].copy()
    
    def make_derivative_func(self) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Construct a derivative function for use with numerical integrators.
        
        The derivative function computes d(state)/dt given the current state,
        using the physics laws (gravity, Newton's second law) but operating
        on abstract state vectors as required by integrators.
        
        Returns:
            A function f(state, t) -> derivative that:
                - Takes state vector (6N,) and time (float)
                - Returns derivative vector (6N,)
                - Encapsulates all physics (forces, accelerations)
                - Can be passed to any integrator
        
        Derivative structure:
            d/dt [positions, velocities] = [velocities, accelerations]
            
            Because:
                - d(position)/dt = velocity (by definition)
                - d(velocity)/dt = acceleration (from F = ma)
        
        Notes:
            - This is a closure that captures self.bodies (for masses)
            - Physics is delegated to gravitational_force() and acceleration()
            - No physics math is performed here, only coordination
        """
        n = len(self.bodies)
        
        # Extract masses (constant parameters, not part of state)
        masses = np.array([body.mass for body in self.bodies])
        names = [body.name for body in self.bodies]
        
        def derivative_func(state: np.ndarray, t: float) -> np.ndarray:
            """
            Compute the time derivative of the state.
            
            Args:
                state: Current state vector [positions, velocities]
                t: Current time (not used for Newtonian gravity, 
                   but required by integrator interface)
            
            Returns:
                Derivative vector [velocities, accelerations]
            """
            derivative = np.zeros_like(state)
            
            # Extract positions and velocities from state
            positions = np.zeros((n, 3))
            velocities = np.zeros((n, 3))
            
            for i in range(n):
                positions[i] = state[3*i : 3*i + 3]
                velocities[i] = state[3*n + 3*i : 3*n + 3*i + 3]
            
            # Create temporary Body objects for physics calculations
            # (These are not the "real" bodies, just carriers for physics functions)
            temp_bodies = [
                Body(names[i], masses[i], positions[i], velocities[i])
                for i in range(n)
            ]
            
            # Compute accelerations using physics layer
            accelerations = np.zeros((n, 3))
            
            for i in range(n):
                # Compute net gravitational force on body i from all other bodies
                net_force = np.zeros(3)
                for j in range(n):
                    if i != j:
                        # Delegate to physics layer: no force computation here
                        force = gravitational_force(temp_bodies[i], temp_bodies[j])
                        net_force += force
                
                # Compute acceleration using Newton's second law
                # Delegate to physics layer: no F=ma here
                accel = acceleration(temp_bodies[i], net_force)
                accelerations[i] = accel
            
            # Pack derivative: [velocities, accelerations]
            # d(position)/dt = velocity
            for i in range(n):
                derivative[3*i : 3*i + 3] = velocities[i]
            
            # d(velocity)/dt = acceleration
            for i in range(n):
                derivative[3*n + 3*i : 3*n + 3*i + 3] = accelerations[i]
            
            return derivative
        
        return derivative_func
    
    def step(self, integrator: Callable, dt: float) -> None:
        """
        Advance the simulation by one timestep using the provided integrator.
        
        This method:
            1. Packs the current bodies into a state vector
            2. Constructs a derivative function from physics laws
            3. Calls the integrator to compute the new state
            4. Unpacks the new state back into the bodies
            5. Advances the time
        
        Args:
            integrator: A numerical integration function with signature
                       integrator(state, derivative_func, t, dt) -> new_state
                       Examples: euler_step, verlet_step, rk4_step
            dt: Timestep size (seconds)
        
        Notes:
            - The integrator is completely agnostic to the physics
            - The derivative function encapsulates all physics
            - Bodies are mutated in place with the new state
            - This is the only method that advances time
        """
        # Pack bodies → state vector
        state = self.pack_state()
        
        # Construct derivative function (encapsulates physics)
        derivative_func = self.make_derivative_func()
        
        # Use integrator to compute new state (pure numerics, no physics)
        new_state = integrator(state, derivative_func, self.time, dt)
        
        # Unpack state vector → bodies (mutation happens here)
        self.unpack_state(new_state)
        
        # Advance time
        self.time += dt
    
    def __repr__(self):
        """String representation of the world state."""
        body_names = [body.name for body in self.bodies]
        return f"World(t={self.time:.2e}s, bodies={body_names})"

