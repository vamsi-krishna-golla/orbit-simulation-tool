"""
2D plotting functions for simulation visualization.

This module provides pure rendering functions that accept plain data structures
(lists, dicts, arrays) and produce static matplotlib plots.

Design constraints:
    - Read-only: receives data, does not modify it
    - No simulation: receives results, does not run simulations
    - No domain imports: does not import physics, simulation, or analysis layers
    - Static only: no animations, interactive plots, or real-time updates

All plotting functions are agnostic to the source of the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_orbit_xy(trajectories: List[Dict], **kwargs) -> None:
    """
    Plot x-y projection of multiple trajectories.
    
    Each trajectory is treated as a generic labeled curve. This function
    does not know or care what the curves represent (orbits, paths, etc.).
    
    Args:
        trajectories: List of trajectory dictionaries, each containing:
            {
                'label': str,        # Curve label for legend
                'x': array-like,     # x-coordinates
                'y': array-like,     # y-coordinates (same length as x)
            }
        
        **kwargs (optional):
            title: str - Plot title (default: 'Trajectories')
            xlabel: str - X-axis label (default: 'x')
            ylabel: str - Y-axis label (default: 'y')
            figsize: tuple - Figure size (default: (8, 8))
            save_path: str - If provided, save to file instead of showing
            equal_aspect: bool - Force equal x/y scaling (default: True)
            grid: bool - Show grid (default: True)
    
    Returns:
        None (displays plot via plt.show() or saves to file)
    
    Example:
        trajectories = [
            {'label': 'Earth', 'x': [1, 2, 3], 'y': [0, 1, 0]},
            {'label': 'Sun', 'x': [0, 0, 0], 'y': [0, 0, 0]},
        ]
        plot_orbit_xy(trajectories, xlabel='x (AU)', ylabel='y (AU)')
    
    Notes:
        - All curves are plotted on the same axes
        - No unit conversion is performed (caller's responsibility)
        - Empty trajectory lists are handled gracefully
    """
    # Extract kwargs with defaults
    title = kwargs.get('title', 'Trajectories')
    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'y')
    figsize = kwargs.get('figsize', (8, 8))
    save_path = kwargs.get('save_path', None)
    equal_aspect = kwargs.get('equal_aspect', True)
    grid = kwargs.get('grid', True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each trajectory
    for traj in trajectories:
        label = traj.get('label', 'unlabeled')
        x = np.asarray(traj['x'])
        y = np.asarray(traj['y'])
        ax.plot(x, y, label=label, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if equal_aspect:
        ax.set_aspect('equal')
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    if trajectories:  # Only show legend if there are trajectories
        ax.legend()
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_energy_vs_time(energy_data: Dict, **kwargs) -> None:
    """
    Plot energy values over time.
    
    This is a generic line plot function that plots whatever arrays it receives.
    It does not interpret the physical meaning of the data or perform any
    transformations (e.g., absolute vs relative energy).
    
    Args:
        energy_data: Dictionary containing:
            {
                'time': array-like,   # Time values (x-axis)
                'energy': array-like, # Energy values (y-axis, same length)
            }
        
        **kwargs (optional):
            title: str - Plot title (default: 'Energy vs Time')
            xlabel: str - X-axis label (default: 'Time')
            ylabel: str - Y-axis label (default: 'Energy')
            figsize: tuple - Figure size (default: (10, 6))
            save_path: str - If provided, save to file instead of showing
            grid: bool - Show grid (default: True)
            color: str - Line color (default: 'blue')
    
    Returns:
        None (displays plot via plt.show() or saves to file)
    
    Example:
        energy_data = {
            'time': [0, 1, 2, 3],
            'energy': [100, 101, 99, 100],
        }
        plot_energy_vs_time(energy_data, xlabel='Time (days)', ylabel='Energy (J)')
    
    Notes:
        - No unit conversion is performed (caller's responsibility)
        - Caller decides whether to pass absolute or relative energy values
        - Empty arrays are handled gracefully
    """
    # Extract kwargs with defaults
    title = kwargs.get('title', 'Energy vs Time')
    xlabel = kwargs.get('xlabel', 'Time')
    ylabel = kwargs.get('ylabel', 'Energy')
    figsize = kwargs.get('figsize', (10, 6))
    save_path = kwargs.get('save_path', None)
    grid = kwargs.get('grid', True)
    color = kwargs.get('color', 'blue')
    
    # Extract data
    time = np.asarray(energy_data['time'])
    energy = np.asarray(energy_data['energy'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    ax.plot(time, energy, color=color, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

