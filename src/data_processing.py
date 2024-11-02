# data_processing.py

import numpy as np
import datetime
import csv
from .constants import AU_IN_METERS

def read_observations(csv_path):
    """
    Reads observational data from a CSV file and preprocesses it.

    Parameters:
        csv_path (str): Path to the observations CSV file.

    Returns:
        dict: Dictionary containing processed observational data.
    """
    # Initialize lists to store data
    observation_times = []
    delta_ra_arcsec = []
    delta_dec_arcsec = []
    sigma_ra_arcsec = []
    sigma_dec_arcsec = []
    dec_planet_deg_list = []
    rotation_angles = []
    plate_scales = []
    distance_to_planet_AU = None

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Read observation time and convert to datetime object
            observation_time_str = row['observation_time']
            observation_time = datetime.datetime.strptime(observation_time_str, '%Y-%m-%dT%H:%M:%S')
            observation_times.append(observation_time)

            # Read delta RA and delta Dec in arcseconds
            delta_ra_arcsec.append(float(row['delta_ra_arcsec']))
            delta_dec_arcsec.append(float(row['delta_dec_arcsec']))

            # Read plate scale and rotation angle
            plate_scale = float(row['plate_scale'])  # arcsec/pixel
            rotation_angle_deg = float(row['rotation_angle'])
            plate_scales.append(plate_scale)
            rotation_angles.append(rotation_angle_deg)

            # For uncertainties, convert pixel errors to arcseconds using the plate scale
            planet_error_x_pix = float(row['planet_error_x'])
            planet_error_y_pix = float(row['planet_error_y'])
            moon_error_x_pix = float(row['moon_error_x'])
            moon_error_y_pix = float(row['moon_error_y'])

            # Combine errors from planet and moon
            sigma_x_arcsec = np.sqrt(planet_error_x_pix**2 + moon_error_x_pix**2) * plate_scale
            sigma_y_arcsec = np.sqrt(planet_error_y_pix**2 + moon_error_y_pix**2) * plate_scale

            # For RA, divide by cos(Dec) to account for convergence of meridians
            dec_planet_deg = float(row['dec_planet_deg'])
            dec_planet_deg_list.append(dec_planet_deg)
            cos_dec = np.cos(np.radians(dec_planet_deg))

            sigma_ra_arcsec.append(sigma_x_arcsec / cos_dec)
            sigma_dec_arcsec.append(sigma_y_arcsec)

            # Store distance to planet (assuming it's constant for all observations)
            if distance_to_planet_AU is None:
                distance_to_planet_AU = float(row['distance_to_planet_AU'])

    # Convert lists to numpy arrays
    observation_times = np.array(observation_times)
    delta_ra_arcsec = np.array(delta_ra_arcsec)
    delta_dec_arcsec = np.array(delta_dec_arcsec)
    sigma_ra_arcsec = np.array(sigma_ra_arcsec)
    sigma_dec_arcsec = np.array(sigma_dec_arcsec)
    dec_planet_deg_list = np.array(dec_planet_deg_list)
    rotation_angles = np.array(rotation_angles)
    plate_scales = np.array(plate_scales)
    distance_to_planet_AU = distance_to_planet_AU if distance_to_planet_AU is not None else 1.0

    return {
        'observation_times': observation_times,
        'delta_ra_arcsec': delta_ra_arcsec,
        'delta_dec_arcsec': delta_dec_arcsec,
        'sigma_ra_arcsec': sigma_ra_arcsec,
        'sigma_dec_arcsec': sigma_dec_arcsec,
        'dec_planet_deg_list': dec_planet_deg_list,
        'rotation_angles': rotation_angles,
        'plate_scales': plate_scales,
        'distance_to_planet_AU': distance_to_planet_AU
    }

def prepare_data(observations):
    """
    Converts observational data to positions and uncertainties in meters.

    Parameters:
        observations (dict): Dictionary containing observational data.

    Returns:
        dict: Dictionary with processed positions, uncertainties, and time.
    """
    # Convert observation times to seconds since the first observation
    epoch = observations['observation_times'][0]
    t_obs = np.array([(ot - epoch).total_seconds() for ot in observations['observation_times']])

    # Convert delta RA and delta Dec from arcseconds to radians
    delta_ra_rad = np.deg2rad(observations['delta_ra_arcsec'] / 3600.0)
    delta_dec_rad = np.deg2rad(observations['delta_dec_arcsec'] / 3600.0)
    sigma_ra_rad = np.deg2rad(observations['sigma_ra_arcsec'] / 3600.0)
    sigma_dec_rad = np.deg2rad(observations['sigma_dec_arcsec'] / 3600.0)

    # Convert angular separations to positions in meters
    D = observations['distance_to_planet_AU'] * AU_IN_METERS  # meters

    # Compute observed positions in meters
    x_obs = D * delta_ra_rad
    y_obs = D * delta_dec_rad

    # Compute uncertainties in meters
    sigma_x = observations['sigma_ra_arcsec'] * (D / 3600.0) * (np.pi / 180.0)
    sigma_y = observations['sigma_dec_arcsec'] * (D / 3600.0) * (np.pi / 180.0)

    return {
        't_obs': t_obs,
        'x_obs': x_obs,
        'y_obs': y_obs,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y
    }
