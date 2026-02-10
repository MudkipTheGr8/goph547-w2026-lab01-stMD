# GOPH 547 Lab 01: forward modelling of corrected gravity data and the concepts of gravity potential and gravity effect.
# Matthew Davidson UCID: 30182729
# February 04, 2025
# Description:
# This file imports gravity_potential_point and gravity_effect_point from goph547lab01.gravity and uses them to generate a contour plot (2 main figures, based off of spacing).

import numpy as np
import matplotlib.pyplot as plt
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point



def contour_plot_creation():

    # Variables:
    # Mass anomaly = 10 million metric tonnes = 10^10kg
    m = 1.0e10
    # Centroid at 0i + oj - 10k [m]
    xm = np.array([0.0, 0.0, -10.0])
    #Heights that are being computed at
    z_values = [0,10,100]
    # Grid spacing in x and y directions [meters]
    grid_spacing = [5.0,25.0]

    # Creating plots for each grid spacing and height.
    for spacing in grid_spacing:
        # Creating figure with 2 columns (U and gz) with 3 rows where one is for each height
        fig, axes = plt.subplots(3,2,figsize=(14,16))
        plt.subplots_adjust(hspace=0.35, wspace=0.3)
        # Tracking maximum and minimum values for colorbar limits
        U_min, U_max = np.inf, -np.inf
        gz_min, gz_max = np.inf, -np.inf
        # Holding the data to be plotted
        data_plotted = {}

        # Plotting U and gz on grid for each height (z_values)
        for i, z in enumerate(z_values):
            # Grid of points from -100 meters to 100 meters in both the x and y axis
            x = np.arange(-100, 101, spacing)
            y = np.arange(-100, 101, spacing)
            X, Y = np.meshgrid(x, y)

            # Creating arrays
            U_grid = np.zeros_like(X)
            gz_grid = np.zeros_like(X)

            # Calculation at each grid point
            for row in range (X.shape[0]):
                for col in range (X.shape[1]):
                    x = [X[row,col], Y[row,col], z]
                    U_grid[row,col] = gravity_potential_point(x, xm, m)
                    gz_grid[row,col] = gravity_effect_point(x, xm, m)
            
            # Minimum and maximum colorbar limit updates for U and gz
            U_min = min(U_min, U_grid.min())
            U_max = max(U_max, U_grid.max())
            gz_min = min(gz_min, gz_grid.min())
            gz_max = max(gz_max, gz_grid.max())

            # Storing data for plotting after loop
            data_plotted[i] = {'X': X, 'Y': Y, 'U_grid': U_grid, 'gz_grid': gz_grid}

            # Contour plots creating
        for i, z in enumerate(z_values):
            data = data_plotted[i]
            X, Y, U_grid, gz_grid = data['X'], data['Y'], data['U_grid'], data['gz_grid']

            # Left column of figure (Gravitational Potential (U))
            lc = axes[i,0]
            contour_lc = lc.contourf(X, Y, U_grid, levels=20, cmap='viridis', vmin = U_min, vmax = U_max)
            lc.plot(X.flatten(), Y.flatten(), 'xk', markersize = 2)
            plt.colorbar(contour_lc, ax = lc, label = 'Gravitational Potential (U) [J/kg (SI units)]')
            lc.set_xlabel('x [m]')
            lc.set_ylabel('y [m]')
            lc.set_title(f'Gravitational Potential (U) at z = {z} m')
            lc.set_aspect('equal')

            # Right column of figure (Vertical Gravity Effect (gz))
            rc = axes[i,1]
            contour_rc = rc.contourf(X, Y, gz_grid, levels=20, cmap='viridis', vmin = gz_min, vmax = gz_max)
            rc.plot(X.flatten(), Y.flatten(), 'xk', markersize = 2)
            plt.colorbar(contour_rc, ax = rc, label = 'Vertical Gravity Effect (gz) [m/s^2 (SI units)]')
            rc.set_xlabel('x [m]')
            rc.set_ylabel('y [m]')
            rc.set_title(f'Vertical Gravity Effect (gz) at z = {z} m')
            rc.set_aspect('equal')
            
        # Figure title
        fig.suptitle(f'Gravitational Potential (U) and Vertical Gravity Effect (gz) for Mass Anomaly of {m:.1e} kg with Grid Spacing of {spacing} m')

        plt.show()

if __name__ == "__main__":
    contour_plot_creation()