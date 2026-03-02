# GOPH 547 Lab 01: forward modelling of corrected gravity data and the concepts of gravity potential and gravity effect.
# Matthew Davidson UCID: 30182729
# February 09, 2025
# Description:
# This file generates several set of mass anomalies (saved in .mat files) that have the same total mass as the single anomaly in Part A and generates contour plots of their gravity potential and effect.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

# Variables
# mass (kg)
mass = 1.0e7
# center of mass (m)
centre_of_mass = np.array([0, 0, -10])
# number of masses
number_of_masses = 5

# Provided from the lab (Part B (1))
U_m = mass / 5
sigma_m = mass / 100
U_x = 0
U_y = 0
sigma_x = 20
sigma_y = 20
U_z = -10
sigma_z = 2
# Must be below or equal to -1m for all masses
max_z = -1 

z_values = [0, 10, 100]
grid_spacing = [5, 25]





def x_y_grid(dx):
    """Creates a grid of x and y coordinates from -100 to 100 meters with a specified spacing (dx).
    Parameters
    ----------
    dx : float
        Spacing between grid points in meters.
    
    Returns
    -------
    X : 2D array (ndarray)
        X coordinates of the grid points.
    Y : 2D array (ndarray)
        Y coordinates of the grid points.
    """
    x = np.arange(-100, 100 + dx, dx)
    y = np.arange(-100, 100 + dx, dx)
    X, Y = np.meshgrid(x, y)
    return X, Y


def one_mass(max_t=10_000):
    """Generates a set of 5 masses and their locations that satisfy the constraints of having a total mass of 10 million metric tonnes, a center of mass at (0, 0, -10) m, and all masses located at or below -1 m in the z direction.
    Parameters
    ----------
    max_t : int, optional, default=10_000
        Maximum number of attempts to generate valid masses and locations before raising an error.

    Returns
    -------
    masses : 1D array (ndarray)
        Array of 5 mass values in kg.
    locations : 2D array (ndarray)
        Array of shape (5, 3) containing the x, y, z coordinates of each mass in meters.
    """

    for i in range(max_t):
        four_masses = np.random.normal(U_m, sigma_m, size = 4)
        if np.any(four_masses <= 0):
            continue
        fifth_mass = mass - np.sum(four_masses)
        if fifth_mass <= 0:
            continue
        masses = np.concatenate((four_masses, [fifth_mass]))

        x_locations = np.random.normal(U_x, sigma_x, size = 4)
        y_locations = np.random.normal(U_y, sigma_y, size = 4)
        z_locations = np.random.normal(U_z, sigma_z, size = 4)

        if np.any(z_locations > max_z):
            continue

        x_5th_location = (mass * centre_of_mass[0] - np.sum(four_masses * x_locations)) / fifth_mass
        y_5th_location = (mass * centre_of_mass[1] - np.sum(four_masses * y_locations)) / fifth_mass
        z_5th_location = (mass * centre_of_mass[2] - np.sum(four_masses * z_locations)) / fifth_mass

        if z_5th_location > max_z:
            continue

        locations = np.column_stack([np.concatenate([x_locations, [x_5th_location]]), np.concatenate([y_locations, [y_5th_location]]), np.concatenate([z_locations, [z_5th_location]])])


        total_mass = masses.sum()
        center_mass = (masses[:, None] * locations).sum(axis=0) / mass

        if not np.isclose(total_mass, mass, atol=1e-6):
            continue
        if not np.allclose(center_mass, centre_of_mass, atol=1e-6):
            continue
        if not np.all(locations[:, 2] <= max_z):
            continue

        return masses, locations
    
    raise RuntimeError(f"Could not generate valid masses and locations after {max_t} attempts.")




def U_gz_calc(X, Y, z, masses, locations):
    """ Calculates the gravity potential (U) and gravity effect (gz) at a grid of points defined by X, Y, and z due to a set of masses located at specified locations.
    Parameters
    ----------
    X : 2D array (ndarray)
        X coordinates of the grid points.
    Y : 2D array (ndarray)
        Y coordinates of the grid points.
    z : float
        Z coordinate of the grid points (constant for all points).
    masses : 1D array (ndarray)
        Array of mass values in kg.
    locations : 2D array (ndarray)
        Array of shape (5, 3) containing the x, y, z coordinates of each mass in meters.
    
    Returns
    -------
    U_total : 2D array (ndarray)
        Gravity potential at each grid point due to all masses.
    gz_total : 2D array (ndarray)
        Gravity effect at each grid point due to all masses.
    """
    Z = np.full_like(X, z, dtype=float)
    points = np.stack([X, Y, Z], axis=-1)
    U_total = np.zeros_like(X, dtype=float)
    gz_total = np.zeros_like(X, dtype=float)

    for mass_k, location in zip(masses, locations):
        U_total += gravity_potential_point(points, location, mass_k)
        gz_total += gravity_effect_point(points, location, mass_k)

    return U_total, gz_total


def plotting_set(set_index, masses, locations, dx):
    """Generates contour plots of the gravity potential (U) and gravity effect (gz) for a given set of masses and their locations, and saves the figure as a PNG file.
    Parameters
    ----------
    set_index : int
        Index of the mass set (used for naming the output file).
    masses : 1D array (ndarray)
        Array of mass values in kg.
    locations : 2D array (ndarray)
        Array of shape (5, 3) containing the x, y, z coordinates of each mass in meters.
    dx : float
        Spacing between grid points in meters.
    
    Returns
    -------
    None
        Saves a figure to the "figures" directory with a filename indicating the mass set index and grid spacing used."""
    x = np.arange(-100, 100 + dx, dx)
    y = np.arange(-100, 100 + dx, dx)
    X, Y = np.meshgrid(x, y)
    U_list = []
    gz_list = []
    for z in z_values:
        U, gz = U_gz_calc(X, Y, z, masses, locations)
        U_list.append(U)
        gz_list.append(gz)
    
    U_min = min(U.min() for U in U_list)
    U_max = max(U.max() for U in U_list)
    gz_min = min(gz.min() for gz in gz_list)
    gz_max = max(gz.max() for gz in gz_list)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    for i, z in enumerate(z_values):
        axU = axes[i, 0]
        cU = axU.contourf(X, Y, U_list[i], levels=30, vmin=U_min, vmax=U_max, cmap='viridis')
        axU.plot(X.ravel(), Y.ravel(), "kx", markersize=2)
        axU.set_title(f"Gravity Potential (U) at z={z} m")
        axU.set_xlabel("X (m)")
        axU.set_ylabel("Y (m)")
        plt.colorbar(cU, ax=axU, label="Gravity Potential (U)")
        axU.set_aspect('equal')


        axgz = axes[i, 1]
        cgz = axgz.contourf(X, Y, gz_list[i], levels=30, cmap="plasma")
        axgz.plot(X.ravel(), Y.ravel(), "kx", markersize=2)
        axgz.set_title(f"Gravity Effect (gz) at z={z} m")
        axgz.set_xlabel("X (m)")
        axgz.set_ylabel("Y (m)")
        plt.colorbar(cgz, ax=axgz, label="Gravity Effect (gz)")
        axgz.set_aspect('equal')

    

    fig.suptitle(f"Mass set {set_index} (dx = {dx} m)", fontsize=14)
    filename = f"figures/mass_set_{set_index}_dx_{dx:.1f}.png"
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def main():
    for i in range (1, 4):
        masses, locations = one_mass()
        total_mass = masses.sum()
        center_mass = (masses[:, None] * locations).sum(axis=0) / total_mass
        print(f"total mass: {total_mass:.6e} kg (target {mass:.6e} kg)")
        print(f"center of mass: {center_mass} m (target {centre_of_mass} m)")
        print(f"max z = {locations[:, 2].max()} m (must be <= {max_z} m)")


        savemat(f"mass_set_{i}.mat", {"masses": masses, "locations": locations})
        for dx in grid_spacing:
            plotting_set(i, masses, locations, dx)
        
    
if __name__ == "__main__":
    main()