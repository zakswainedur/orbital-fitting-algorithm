# constants.py

import numpy as np
import datetime

# =====================================================
# Constants and Parameters
# =====================================================

# Gravitational constant (G) in m^3 kg^-1 s^-2
G = 6.67430e-11

# Mass of the planet (M_planet)
# For Neptune: M_planet = 1.02413e26 kg
M_PLANET = 1.02413e26  # kg

# Gravitational parameter (mu = G * M_planet)
MU = G * M_PLANET  # m^3 s^-2

# 1 AU in meters
AU_IN_METERS = 1.495978707e11

# Neptune Center in Neptune frame
R0 = np.array([0, 0, 0]) 

# Reference Epoch for pole precession JED2447763.5 
JED = datetime.datetime(1989, 8, 25, 0, 0, 0) # From nasa jpl

# Viewing angle 
VIEWING_RA_DEG = 358.27870   # in decimal format 
VIEWING_DEC_DEG = -2.19055   # in decimal format 
