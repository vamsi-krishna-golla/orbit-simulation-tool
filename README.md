# Universal Simulation Tool

## What this project is

**Universal Simulation Tool** is an explanation-first simulation framework that treats physical laws as *executable ideas*, not just equations.

Instead of focusing on visual realism or performance, the project focuses on **what assumptions a simulation makes**, **which physical laws it preserves**, and **how those choices shape the resulting "universe."**

The goal is not just to simulate systems, but to **understand why simulations succeed or fail**.

---

## Core idea: equations are not enough

In physics, we often write laws like:

- Newton's law of gravitation  
- Newton's second law  

But a simulation needs **one more rule**:

> **How the present becomes the future.**

That rule is not part of the equations themselves.  
It is encoded in the **numerical integrator**.

This project makes that rule explicit.

---

## Architecture (separation of concerns)

```
physics/ → what the laws are
numerics/ → how time evolution is approximated
simulation/ → how physics and numerics are connected
analysis/ → how results are observed (read-only)
```

### Physics layer
- Defines bodies, forces, and acceleration
- No time-stepping
- No numerical assumptions

### Numerics layer
- Defines integrators (Euler, Verlet, etc.)
- Pure mathematics
- Knows nothing about gravity, planets, or energy

### Simulation layer
- Translates physical state into abstract vectors
- Orchestrates time evolution
- Does not define laws or integrators

### Analysis layer
- Measures energy, momentum, angular momentum
- Read-only observation
- No influence on the simulation

---

## A concrete experiment: Euler vs Verlet

To make the consequences of integrator choice visible, we ran a controlled experiment:

- Same Sun–Earth system
- Same gravity law
- Same initial conditions
- Same diagnostics

The **only difference** was the integrator.

### Euler integrator

Euler assumes:
> "The current velocity and acceleration stay constant during a timestep."

Observed results:
- Total energy drifts systematically
- Earth's orbit slowly spirals outward
- Momentum remains conserved
- Long-term motion becomes unphysical

This produces a self-consistent but **non-physical universe**:
- Gravity exists
- Energy is slowly created from nowhere
- Stable orbits are impossible

### Velocity-Verlet integrator

Verlet treats motion symmetrically in time.

Observed results:
- Total energy remains bounded (small oscillations, no drift)
- Orbits remain stable over many years
- Momentum and angular momentum are conserved
- Motion is time-reversible (approximately)

This produces a universe that matches the **structural constraints** of real Newtonian mechanics.

---

## Key insight

> **Two simulations can use identical equations and still describe different universes.**

The difference is not the force law.  
It is **which invariants the time evolution respects**.

Integrator choice is not a numerical detail — it is a *physical commitment*.

---

## What this project demonstrates

- Numerical methods encode assumptions about reality
- Conservation laws are not automatically preserved
- Accuracy is not the same as physical honesty
- Long-term behavior matters more than short-term precision

This tool is designed to make those facts visible.

---

## What this project is not (yet)

- Not a graphics engine
- Not an astrophysics toolkit
- Not optimized for performance
- Not a full solar system model

Those are downstream concerns.

---

## How to extend responsibly

When adding new components, the guiding questions should be:

- What physical laws should be preserved?
- What invariants does this method respect or violate?
- What kind of universe does this choice allow?

Every new integrator, model, or approximation should make those answers explicit.

---

## Status

Current implementation includes:
- Newtonian gravity
- Euler and velocity-Verlet integrators
- Energy, momentum, and angular momentum diagnostics
- A clean Sun–Earth experiment demonstrating integrator-dependent universes

The framework is intentionally minimal and evolving.

---

### One-sentence summary

> **Universal Simulation Tool explores how different rules of time evolution, applied to the same equations, create different self-consistent worlds.**
