# orbital_fitting.py

import numpy as np
from scipy.optimize import least_squares
from .constants import MU
from .utils import rotate_and_project

def solve_kepler(M, e, tol=1e-12, max_iter=200):
    """
    Solves Kepler's Equation M = E - e*sin(E) for E using the Newton-Raphson method.

    Parameters:
        M (numpy.ndarray): Mean anomaly (radians)
        e (float): Eccentricity
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations

    Returns:
        numpy.ndarray: Eccentric anomaly (radians)
    """
    E = M.copy()
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta_E = -f / f_prime
        E += delta_E
        if np.max(np.abs(delta_E)) < tol:
            break
    return E

def orbital_elements_to_positions(a, e, i, Omega, omega, M0, t, mu):
    """
    Computes the 3D position vectors at times t based on orbital elements.

    Parameters:
        a (float): Semi-major axis (meters)
        e (float): Eccentricity
        i (float): Inclination (radians)
        Omega (float): Longitude of ascending node (radians)
        omega (float): Argument of periapsis (radians)
        M0 (float): Mean anomaly at epoch (radians)
        t (numpy.ndarray): Observation times (seconds)
        mu (float): Gravitational parameter (m^3 s^-2)

    Returns:
        numpy.ndarray: Position vectors (3, N)
    """
    # Compute mean motion
    n = np.sqrt(mu / a**3)
    # Compute mean anomaly at times t
    M = M0 + n * t
    M = np.mod(M, 2 * np.pi)
    # Solve Kepler's equation to get eccentric anomaly E
    E = solve_kepler(M, e)
    # Compute true anomaly nu
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    # Compute distance r
    r = a * (1 - e * np.cos(E))
    # Compute argument of latitude u = omega + nu
    u = omega + nu
    # Position in orbital plane coordinates
    x_orb = r * np.cos(u)
    y_orb = r * np.sin(u)

    # Rotation matrices
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    # Transformation to inertial coordinates
    x = (cos_Omega * x_orb - sin_Omega * y_orb * cos_i)
    y = (sin_Omega * x_orb + cos_Omega * y_orb * cos_i)
    z = y_orb * sin_i
    positions = np.array([y, x, z])
    return positions 

def compute_alpha_delta(T):
    """
    Computes the right ascension (alpha) and declination (delta) of Neptune's pole
    based on Julian centuries since JED2447763.5.

    Parameters:
        T (numpy.ndarray): Time in Julian centuries from JED2447763.5

    Returns:
        tuple: Arrays of alpha and delta in degrees
    """
    psi = 352.099 + 52.318 * T
    alpha_p = 298.953  # Right ascension of the pole of the invariable plane
    delta_p = 43.312    # Declination of the pole of the invariable plane

    alpha = alpha_p + 0.696 * np.sin(np.deg2rad(psi)) - 0.003 * np.sin(2 * np.deg2rad(psi)) 
    delta = delta_p + 0.506 * np.cos(np.deg2rad(psi)) + 0.001 * np.cos(2 * np.deg2rad(psi))

    return alpha, delta

def residuals(params, t, x_obs, y_obs, sigma_x, sigma_y, C, x_axis, y_axis):
    """
    Computes residuals for weighted least squares optimization.

    Parameters:
        params (list): [a, e, i, Omega, omega, M0]
        t (numpy.ndarray): Observation times (seconds)
        x_obs (numpy.ndarray): Observed x positions (meters)
        y_obs (numpy.ndarray): Observed y positions (meters)
        sigma_x (numpy.ndarray): Uncertainties in x (meters)
        sigma_y (numpy.ndarray): Uncertainties in y (meters)
        C (numpy.ndarray): Rotation matrix
        x_axis (numpy.ndarray): x-axis unit vector
        y_axis (numpy.ndarray): y-axis unit vector

    Returns:
        numpy.ndarray: Concatenated residuals for x and y
    """
    a, e, i_angle, Omega, omega, M0 = params
    if not (0 < a and 0 <= e < 1):
        return np.inf * np.ones(2 * len(x_obs))
    ri = orbital_elements_to_positions(a, e, i_angle, Omega, omega, M0, t, MU)
    
    x_proj, y_proj = rotate_and_project(ri, np.zeros(3), C, x_axis, y_axis)
    
    res_x = (x_proj - x_obs) / sigma_x
    res_y = (y_proj - y_obs) / sigma_y
    res = np.concatenate([res_x, res_y])
    return res
