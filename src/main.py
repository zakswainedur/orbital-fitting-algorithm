# main.py

import numpy as np
from .data_processing import read_observations, prepare_data
from .constants import JED, VIEWING_RA_DEG, VIEWING_DEC_DEG, R0, MU
from .utils import define_unit_vector, compute_projection_axes, construct_rotation_matrix, rotate_and_project
from .orbital_fitting import orbital_elements_to_positions, residuals, compute_alpha_delta
from .plotting import plot_observed_vs_fitted, plot_3d_orbit, plot_chi_squared_surface
from scipy.optimize import least_squares

def main():
    # Paths
    csv_path = 'data/observations.csv'

    # Read and prepare data
    observations = read_observations(csv_path)
    data = prepare_data(observations)

    # Prepare orthonormal basis relative to viewing direction
    viewing_dir = define_unit_vector(VIEWING_RA_DEG, VIEWING_DEC_DEG)
    viewing_dir /= np.linalg.norm(viewing_dir)
    print(f"Viewing Direction Vector: {viewing_dir}")

    x_axis, y_axis, z_axis = compute_projection_axes(VIEWING_RA_DEG, VIEWING_DEC_DEG)

    # Julian centuries since Reference Epoch
    T = np.array([(ot - JED).total_seconds() / (36525.0 * 86400.0) for ot in observations['observation_times']])

    # Compute alpha and delta for each observation
    alphas, deltas = compute_alpha_delta(T)

    # Construct a single rotation matrix based on average alpha and delta
    mean_alpha = np.mean(alphas)
    mean_delta = np.mean(deltas)
    C = construct_rotation_matrix(mean_alpha, mean_delta)

    # Initial Guesses for Orbital Elements
    a_guess = 3.54e8  # meters (example value)
    e_guess = 0.0             # Eccentricity
    i_guess_deg = 180 + 157    # Inclination in degrees
    Omega_guess_deg = 250      # Longitude of ascending node in degrees
    omega_guess_deg = 149.5    # Argument of periapsis in degrees
    M0_guess_deg = 200         # Mean anomaly at epoch in degrees

    # Convert angles from degrees to radians
    i_guess = np.radians(i_guess_deg)
    Omega_guess = np.radians(Omega_guess_deg)
    omega_guess = np.radians(omega_guess_deg)
    M0_guess = np.radians(M0_guess_deg)

    # Combine initial parameters into an array
    initial_params = [a_guess, e_guess, i_guess, Omega_guess, omega_guess, M0_guess]

    # Bounds for parameters to ensure they stay within physical limits
    bounds = (
        [0, 0.0, np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0],  # Lower bounds
        [np.inf, 1.0, np.deg2rad(360), np.deg2rad(360), np.deg2rad(360), 2*np.pi]    # Upper bounds
    )

    # Run weighted least squares optimization
    result = least_squares(
        residuals, 
        initial_params,
        args=(
            data['t_obs'], 
            data['x_obs'], 
            data['y_obs'], 
            data['sigma_x'], 
            data['sigma_y'], 
            C, 
            x_axis, 
            y_axis
        ),
        bounds=bounds,
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=10000
    )

    # Extract fitted parameters
    a_fit, e_fit, i_fit, Omega_fit, omega_fit, M0_fit = result.x

    # Compute covariance matrix
    n_obs = 2 * len(data['t_obs'])  # x and y observations
    n_params = len(result.x)
    dof = n_obs - n_params

    # Estimate the variance of the residuals
    residual = result.fun
    chi_squared = np.sum(residual**2)
    reduced_chi_squared = chi_squared / dof

    print(f"Chi-squared: {chi_squared}")
    print(f"Reduced Chi-squared: {reduced_chi_squared}")

    # Compute the covariance matrix using pseudoinverse
    J = result.jac
    cov = np.linalg.pinv(J.T @ J) * reduced_chi_squared
    parameter_errors = np.sqrt(np.diag(cov)) 

    # Normalize inclination
    i_fit = i_fit % np.pi

    # Print fitted parameters with uncertainties
    print("\nFitted orbital parameters with uncertainties:")
    print(f"Semi-major axis (a): {a_fit:.6f} ± {parameter_errors[0]:.6f} meters")
    print(f"Eccentricity (e): {e_fit:.6f} ± {parameter_errors[1]:.6f}")
    print(f"Inclination (i): {np.degrees(i_fit):.6f}° ± {np.degrees(parameter_errors[2]):.6f}°")
    print(f"Longitude of ascending node (Ω): {np.degrees(Omega_fit):.6f}° ± {np.degrees(parameter_errors[3]):.6f}°")
    print(f"Argument of periapsis (ω): {np.degrees(omega_fit):.6f}° ± {np.degrees(parameter_errors[4]):.6f}°")
    print(f"Mean anomaly at epoch (M0): {np.degrees(M0_fit):.6f}° ± {np.degrees(parameter_errors[5]):.6f}°")

    # Generate fitted orbit positions over a fine time grid for plotting
    num_plot_points = 1000
    t_fine = np.linspace(min(data['t_obs']), max(data['t_obs']) * 2, num_plot_points)

    # Actual parameters (assuming literature values or known parameters)
    actual_param_projection = orbital_elements_to_positions(
        3.54e8, 0, np.deg2rad(180 - 157), np.deg2rad(352 - 180), np.deg2rad(3), np.deg2rad(150), t_fine, MU)
    x_proj_act, y_proj_act = rotate_and_project(actual_param_projection, R0, C, x_axis, y_axis)

    # Fitted projection
    fitted_projection = orbital_elements_to_positions(
        a_fit, e_fit, i_fit, Omega_fit, omega_fit, M0_fit, t_fine, MU)
    x_proj_fit, y_proj_fit = rotate_and_project(fitted_projection, R0, C, x_axis, y_axis)

    # Plot observed data and fitted orbit
    plot_observed_vs_fitted(
        data['x_obs'], data['y_obs'], 
        data['sigma_x'], data['sigma_y'], 
        x_proj_act, y_proj_act, 
        x_proj_fit, y_proj_fit
    )

    # 3D Plot of the Orbit
    # Generate positions over one full orbital period for 3D visualization
    P_fit = 2 * np.pi * np.sqrt(a_fit**3 / MU)
    t_full_orbit = np.linspace(0, P_fit, num_plot_points)
    r2 = orbital_elements_to_positions(
        a_fit, e_fit, i_fit, Omega_fit, omega_fit, M0_fit, t_full_orbit, MU)
    x_orbit, y_orbit, z_orbit = r2
    plot_3d_orbit(x_orbit, y_orbit, z_orbit, data['x_obs'], data['y_obs'])

    # Define ranges for i and Omega
    delta_angle = np.radians(10)  # +/- 10 degrees around the best-fit
    i_values = np.linspace(i_fit - delta_angle, i_fit + delta_angle, 100)
    Omega_values = np.linspace(Omega_fit - delta_angle, Omega_fit + delta_angle, 100)

    # Create meshgrid
    I_grid, Omega_grid = np.meshgrid(i_values, Omega_values)

    # Initialize chi-squared grid
    chi_sq_grid = np.zeros_like(I_grid)

    # Fixed parameters
    a_fixed = a_fit
    e_fixed = e_fit
    omega_fixed = omega_fit
    M0_fixed = M0_fit

    # Compute chi-squared over the grid
    for idx_i in range(I_grid.shape[0]):
        for idx_j in range(I_grid.shape[1]):
            params = [a_fixed, e_fixed, I_grid[idx_i, idx_j], Omega_grid[idx_i, idx_j], omega_fixed, M0_fixed]
            res = residuals(params, data['t_obs'], data['x_obs'], data['y_obs'], 
                           data['sigma_x'], data['sigma_y'], C, x_axis, y_axis)
            chi_sq = np.sum(res**2)
            chi_sq_grid[idx_i, idx_j] = chi_sq

    # Compute delta chi-squared
    chi_sq_min = chi_squared  # From your fitting result
    delta_chi_sq = chi_sq_grid - chi_sq_min

    # Convert grid to degrees for plotting
    I_grid_deg = np.degrees(I_grid)
    Omega_grid_deg = np.degrees(Omega_grid)

    # Plot chi-squared surface
    plot_chi_squared_surface(I_grid_deg, Omega_grid_deg, delta_chi_sq, chi_squared)

if __name__ == "__main__":
    main()
