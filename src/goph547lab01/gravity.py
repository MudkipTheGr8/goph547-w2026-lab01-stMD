# GOPH 547 Lab 01:
# Matthew Davidson UCID: 30182729
# January 04, 2025

# This file has two functions (gravity_potential_point() and gravity_effect_point())
# that compute the gravity potential and vertical gravity effect due to a point mass
# anomaly.

import numpy as np


def gravity_potential_point(x, xm, m, G=6.67e-11):
    """ Compute the gravity potential due to a point mass.
    
    Parameters
    ----------
    x : array_like, shape=(3,)
    Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.67e-11
        Constant of gravitation in m^3 kg^-1 s^-2.
        Default in SI units.
        Allows user to modify for different unit systems.
    
    Returns
    -------
    U : float
        Gravity potential at survey point due to point mass anomaly.
    """

    # Convert inputs to numpy arrays just incase they are lists.
    x = np.asarray(x, dtype = float)
    xm = np.asarray(xm, dtype = float)

    # Compute distance r between survey point and point mass
    r = x - xm
    r_norm = np.linalg.norm(r, axis = -1)

    # Compute gravity potential U.
    U = G * m / r_norm
    return U








def gravity_effect_point(x, xm, m, G=6.67e-11):
    """Compute the vertical gravity effect due to a point
    mass (positive downward).

    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation.
        Default in SI units.
        Allows user to modify if using different unit.

    Returns
    -------
    gz : float
        Gravity effect at x due to anomaly at xm.
    """

    # Convert inputs to numpy arrays just incase they are lists
    x = np.asarray(x, dtype = float)
    xm = np.asarray(xm, dtype = float)

    # Compute distance r between survey point and point mass
    r = x - xm
    r_norm = np.linalg.norm(r, axis = -1)

    # Extract vertical component dz
    dz = r[..., 2]

    # Compute vertical gravity effect gz.
    gz = G * m * dz / (r_norm**3)
    return gz


