# plotting.py

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_observed_vs_fitted(x_obs, y_obs, sigma_x, sigma_y, 
                            x_proj_act, y_proj_act, 
                            x_proj_fit, y_proj_fit):
    """
    Plots observed positions with errors and fitted orbit projection.

    Parameters:
        x_obs (numpy.ndarray): Observed x positions (meters)
        y_obs (numpy.ndarray): Observed y positions (meters)
        sigma_x (numpy.ndarray): Uncertainties in x (meters)
        sigma_y (numpy.ndarray): Uncertainties in y (meters)
        x_proj_act (numpy.ndarray): Actual orbit projection x (meters)
        y_proj_act (numpy.ndarray): Actual orbit projection y (meters)
        x_proj_fit (numpy.ndarray): Fitted orbit projection x (meters)
        y_proj_fit (numpy.ndarray): Fitted orbit projection y (meters)
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_obs, y_obs, xerr=sigma_x, yerr=sigma_y,
                 fmt='o', label='Observed positions with errors', capsize=2)
    
    plt.plot(x_proj_act, y_proj_act, label='Orbit projection with literature parameters')
    
    plt.plot(x_proj_fit, y_proj_fit, label='Fitted orbit projection')
    plt.axis('equal')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.title('Observed Positions with Errors and Fitted Orbit Projection')
    plt.show()

def plot_3d_orbit(x_orbit, y_orbit, z_orbit, x_obs, y_obs):
    """
    Plots the 3D visualization of the orbit.

    Parameters:
        x_orbit (numpy.ndarray): Orbit x positions (meters)
        y_orbit (numpy.ndarray): Orbit y positions (meters)
        z_orbit (numpy.ndarray): Orbit z positions (meters)
        x_obs (numpy.ndarray): Observed x positions (meters)
        y_obs (numpy.ndarray): Observed y positions (meters)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_orbit, y_orbit, z_orbit, label='Fitted Orbit', color='blue')
    ax.scatter(0, 0, 0, color='orange', label='Planet', s=100)
    # Plot observed positions at z=0
    ax.scatter(x_obs, y_obs, np.zeros_like(x_obs), color='red', label='Observed Positions')
    # Optional: Plot projection lines
    for xi, yi, zi in zip(x_orbit, y_orbit, z_orbit):
        ax.plot([xi, xi], [yi, yi], [zi, 0], color='gray', linestyle='--', linewidth=0.5)
    # Also, plot the projected orbit onto the XY plane
    ax.plot(x_orbit, y_orbit, zs=0, label='Projected Orbit', color='green', linestyle='--')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    ax.set_title("3D Visualization of the Moon's Orbit around the Planet")
    plt.show()

def plot_chi_squared_surface(I_grid_deg, Omega_grid_deg, delta_chi_sq, chi_squared):
    """
    Plots the chi-squared surface for inclination and longitude of ascending node.

    Parameters:
        I_grid_deg (numpy.ndarray): Inclination grid in degrees
        Omega_grid_deg (numpy.ndarray): Longitude of ascending node grid in degrees
        delta_chi_sq (numpy.ndarray): Delta chi-squared values
        chi_squared (float): Minimum chi-squared value
    """
    import matplotlib.colors as mcolors

    # Define your vmin and vmax
    vmin, vmax = 0, chi_squared * 200

    # Mask the delta_chi_sq data outside [vmin, vmax]
    masked_delta_chi_sq = np.ma.masked_outside(delta_chi_sq, vmin, vmax)

    # Retrieve and customize the colormap
    cmap = plt.get_cmap('jet_r').copy()
    cmap.set_bad(color='grey')

    # Create the plot
    plt.figure(figsize=(8,6))
    extent = [I_grid_deg.min(), I_grid_deg.max(), Omega_grid_deg.min(), Omega_grid_deg.max()]
    
    plt.imshow(
        masked_delta_chi_sq,  # Transpose for correct orientation
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.9
    )
    
    # Add colorbar
    plt.colorbar(label=r'$\Delta\chi^2$')
    
    # Labeling
    plt.xlabel(r'Inclination $i$ (degrees)')
    plt.ylabel(r'Longitude of Ascending Node $\Omega$ (degrees)')
    plt.title('Chi-squared Surface for $i$ and $\Omega$')
    
    # Confidence levels for two parameters (chi-squared distribution)
    levels = [2.30, 6.17, 11.8]
    CS = plt.contour(
        I_grid_deg, 
        Omega_grid_deg, 
        delta_chi_sq, 
        levels=levels, 
        colors=['black', 'black', 'black']
    )
    
    plt.grid(True, alpha=0.2)
    plt.xticks(np.arange(180, 361, 30), np.arange(0, 181, 30))
    
    plt.show()

