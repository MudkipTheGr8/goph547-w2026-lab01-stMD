# GOPH 547 Lab 01: forward modelling of corrected gravity data and the concepts of gravity potential and gravity effect.
# Matthew Davidson UCID: 30182729
# February 11, 2025
# Description: 
# This file loads a 3D grid of density values from a .mat file, computes the total mass, barycentre, and density statistics of the mass distribution, and generates contour plots of the mean density in different planes. It also performs forward modelling of the vertical gravity effect (gz) at different observation heights and computes the second derivative of gz with respect to z. Finally, it generates contour plots of gz and its second derivative at different heights.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from goph547lab01.gravity import gravity_effect_point



data = loadmat("anomaly_data.mat")

x= data["x"]
y = data["y"]
z = data["z"]
rho = data["rho"]

dx = dy = dz = 2
cell_volume = dx * dy * dz

total_mass = np.sum(rho) * cell_volume

x_coord = np.sum(x * rho) * cell_volume / total_mass
y_coord = np.sum(y * rho) * cell_volume / total_mass
z_coord = np.sum(z * rho) * cell_volume / total_mass

max_rho = np.max(rho)
mean_rho = np.mean(rho)


print("Distributed Mass Properties:")
print(f"Total Mass: {total_mass:.3e} kg")
print(f"Barycentre: x = {x_coord:.3f} m, y = {y_coord:.3f} m, z = {z_coord:.3f} m")
print(f"Maximum Density: {max_rho:.3e} kg/m^3")
print(f"Mean Density: {mean_rho:.3e} kg/m^3")






rho_yz = np.mean(rho, axis=0)
rho_xz = np.mean(rho, axis=1)
rho_xy = np.mean(rho, axis=2)

vmin = min(rho_yz.min(), rho_xz.min(), rho_xy.min())
vmax = max(rho_yz.max(), rho_xz.max(), rho_xy.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)




c0 = axes[0].contourf(x[0, :, 0], z[0, 0, :], rho_yz.T, levels=30, vmin=vmin, vmax=vmax, cmap = "viridis")
axes[0].plot(y_coord, z_coord, "xk", markersize = 3)
axes[0].set_title("Mean Density (rho) in XZ plane")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("z (m)")

c1 = axes[1].contourf(y[:, 0, 0], z[0, 0, :], rho_xz.T, levels=30, vmin=vmin, vmax=vmax, cmap = "viridis")
axes[1].plot(x_coord, z_coord, "xk", markersize = 3)
axes[1].set_title("Mean Density (rho) in YZ plane")
axes[1].set_xlabel("y (m)")
axes[1].set_ylabel("z (m)")


c2 = axes[2].contourf(x[0, :, 0], y[:, 0, 0], rho_xy, levels=30, vmin=vmin, vmax=vmax, cmap = "viridis")
axes[2].plot(x_coord, y_coord, "xk", markersize = 3)
axes[2].set_title("Mean Density (rho) in XY plane")
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("y (m)")


fig.colorbar(c2, ax = axes, label = "Density (rho) [kg/m^3]")
plt.savefig("figures/density_slices.png", dpi=200)
plt.show()








thres = 0.1 * max_rho

mask = rho > thres

rho_masked = rho[mask]

mean_rho_masked = np.mean(rho_masked)

print(f" Density threshold: {thres:.3e} kg/m^3")
print(f"Mean Density of Masked Region: {mean_rho_masked:.3e} kg/m^3")

x1d = x[0, :, 0]
y1d = y[:, 0, 0]
z1d = z[0, 0, :]

i_nz, j_nz, k_nz = np.where(mask)

print(f"x range: [{x1d[j_nz.min()]:.1f}, {x1d[j_nz.max()]:.1f}] m")
print(f"y range: [{y1d[i_nz.min()]:.1f}, {y1d[i_nz.max()]:.1f}] m")
print(f"z range: [{z1d[k_nz.min()]:.1f}, {z1d[k_nz.max()]:.1f}] m")










def foward_modelling_of_gz(z_obs):
    """Does forward modelling of the gravity effect (gz) for a given observation height z_obs

    Parameters
    ----------
    z_obs : float
        Observation height in meters.
    
    Returns
    -------
    XX : numpy.ndarray
        X coordinates of observation points.
    YY : numpy.ndarray
        Y coordinates of observation points.
    gz_total : numpy.ndarray
        Total gravity effect (gz) at each observation point.
    """

    X_observation = x[0, :, 0]
    Y_observation = y[:, 0, 0]
    XX, YY = np.meshgrid(X_observation, Y_observation)
    gz_total = np.zeros_like(XX, dtype=float)

    points = np.stack([XX, YY, np.full_like(XX, z_obs)], axis=-1)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if rho[i, j, k] == 0:
                    continue
                mass_cell = rho[i, j, k] * cell_volume
                location_cell = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                gz_total += gravity_effect_point(points, location_cell, mass_cell)

    return XX, YY, gz_total

XX0, YY0, gz_0 = foward_modelling_of_gz(0)
XX100, YY100, gz_100 = foward_modelling_of_gz(100)







XX1, YY1, gz_1 = foward_modelling_of_gz(1)
XX110, YY110, gz_110 = foward_modelling_of_gz(110)

dgz_dz_0 = (gz_1 - gz_0) / 1
dgz_dz_100 = (gz_110 - gz_100) / 10










fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)




cf = axes[0, 0].contourf(XX0, YY0, gz_0, levels=30, cmap='plasma')
plt.colorbar(cf, ax=axes[0, 0], label="Gravity Effect (gz) [m/s^2]")
axes[0, 0].set_title("Gravitational Effect (gz) at z = 0 m")

cf = axes[0, 1].contourf(XX1, YY1, gz_1, levels=30, cmap='plasma')
plt.colorbar(cf, ax=axes[0, 1], label="Gravity Effect (gz) [m/s^2]")
axes[0, 1].set_title("Gravitational Effect (gz) at z = 1 m")

cf = axes[1, 0].contourf(XX100, YY100, gz_100, levels=30, cmap='plasma')
plt.colorbar(cf, ax=axes[1, 0], label="Gravity Effect (gz) [m/s^2]")
axes[1, 0].set_title("Gravitational Effect (gz) at z = 100 m")

cf = axes[1, 1].contourf(XX110, YY110, gz_110, levels=30, cmap='plasma')
plt.colorbar(cf, ax=axes[1, 1], label="Gravity Effect (gz) [m/s^2]")
axes[1, 1].set_title("Gravitational Effect (gz) at z = 110 m")




for ax in axes.ravel():
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal', adjustable='box')


plt.savefig("figures/gz_slices.png", dpi=200)
plt.show()












dx = x[0, 1, 0] - x[0, 0, 0]
dy = y[1, 0, 0] - y[0, 0, 0]

d2gz_dx2 = (gz_0[:, 2:] - 2 * gz_0[:, 1:-1] + gz_0[:, :-2]) / dx**2
d2gz_dy2 = (gz_0[2:, :] - 2 * gz_0[1:-1, :] + gz_0[:-2, :]) / dy**2

d2gz_dx2_full = np.zeros_like(gz_0)
d2gz_dx2_full[:, 1:-1] = d2gz_dx2

d2gz_dy2_full = np.zeros_like(gz_0)
d2gz_dy2_full[1:-1, :] = d2gz_dy2

d2gz_dz2_0 = -(d2gz_dx2_full + d2gz_dy2_full)


d2gz_dx2 = (gz_100[:, 2:] - 2 * gz_100[:, 1:-1] + gz_100[:, :-2]) / dx**2
d2gz_dy2 = (gz_100[2:, :] - 2 * gz_100[1:-1, :] + gz_100[:-2, :]) / dy**2

d2gz_dx2_full = np.zeros_like(gz_100)
d2gz_dx2_full[:, 1:-1] = d2gz_dx2

d2gz_dy2_full = np.zeros_like(gz_0)
d2gz_dy2_full[1:-1, :] = d2gz_dy2

d2gz_dz2_100 = -(d2gz_dx2_full + d2gz_dy2_full)


fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

cf0 = axes[0].contourf(XX0, YY0, d2gz_dz2_0, levels=30, cmap = "viridis")
plt.colorbar(cf0, ax=axes[0], label="(d2gz/dz2) [m^-1s^-2]")
axes[0].set_title("d2gz/dz2 at z = 0m")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")

cf100 = axes[1].contourf(XX100, YY100, d2gz_dz2_100, levels=30, cmap = "viridis")
plt.colorbar(cf100, ax=axes[1], label="(d2gz/dz2) [m^-1s^-2]")
axes[1].set_title("d2gz/dz2 at z = 100m")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")

plt.savefig("figures/d2gz_dz2_slices.png", dpi=200)
