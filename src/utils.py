# utils.py

import numpy as np

def define_unit_vector(RA_deg, Dec_deg):
    """
    Converts Right Ascension and Declination in degrees to a unit vector.
    """
    RA_rad = np.deg2rad(RA_deg)
    Dec_rad = np.deg2rad(Dec_deg)
    return np.array([
        np.cos(Dec_rad) * np.cos(RA_rad),
        np.cos(Dec_rad) * np.sin(RA_rad),
        np.sin(Dec_rad)
    ], dtype=np.float64)

def compute_projection_axes(RA_deg, Dec_deg):
    """
    Computes the x_axis and y_axis unit vectors based on the viewing direction defined by RA and Dec.
    """
    RA_rad = np.deg2rad(RA_deg)
    Dec_rad = np.deg2rad(Dec_deg)
    
    # Z-axis: Viewing direction
    z_axis = np.array([
        np.cos(Dec_rad) * np.cos(RA_rad),
        np.cos(Dec_rad) * np.sin(RA_rad),
        np.sin(Dec_rad)
    ])
    z_axis /= np.linalg.norm(z_axis)
    
    # X-axis: Direction of increasing RA (East)
    x_axis = np.array([
        -np.sin(RA_rad),
        np.cos(RA_rad),
        0
    ])
    x_axis /= np.linalg.norm(x_axis)
    
    # Y-axis: Direction of increasing Dec (North)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    return x_axis, y_axis, z_axis

def construct_rotation_matrix(alpha_deg, delta_deg):
    """
    Constructs the rotation matrix to convert vectors from Neptune's equatorial frame
    to Earth's equatorial frame.

    Parameters:
        alpha_deg (float): Right ascension of Neptune's pole in degrees
        delta_deg (float): Declination of Neptune's pole in degrees

    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    alpha = np.deg2rad(alpha_deg)
    delta = np.deg2rad(delta_deg)

    # Rotation matrix about Z-axis by alpha
    R_z = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0,              0,             1]
    ])
    
    # Rotation matrix about Y-axis by delta
    R_y = np.array([
        [ np.cos(delta), 0, np.sin(delta)],
        [0,             1,             0],
        [-np.sin(delta), 0, np.cos(delta)]
    ])

    # Combined rotation matrix
    C = R_z @ R_y
    return C

def rotate_and_project(ri, r0, C, x_axis, y_axis):
    """
    Rotates the vector (ri - r0) into Earth's equatorial frame and projects onto 2D image.

    Parameters:
        ri (numpy.ndarray): Observed position vectors (3, N) in Earth's frame
        r0 (numpy.ndarray): Reference position vector of Neptune in Earth's frame (3,)
        C (numpy.ndarray): 3x3 rotation matrix from Neptune's to Earth's frame
        x_axis (numpy.ndarray): x-axis unit vector of projection plane
        y_axis (numpy.ndarray): y-axis unit vector of projection plane

    Returns:
        tuple: Projected 2D positions (x_proj, y_proj)
    """
    # Translate positions relative to r0 (Neptune's position in Earth's frame)
    delta_r = ri - r0.reshape(3, 1)
    # Apply rotation matrix to convert to Earth's equatorial frame
    rotated_r = C @ delta_r
    # Project onto 2D plane using the new axes
    x_proj = x_axis @ rotated_r
    y_proj = y_axis @ rotated_r

    return x_proj, y_proj
