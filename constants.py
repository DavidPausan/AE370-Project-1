# constants.py
# Physical constants and mission parameters for AE 370 Project 1.
# Units: km, s, rad (unless the name says deg).
# Source for Earth constants: Vallado, Fundamentals of Astrodynamics and Applications.

import numpy as np

# Earth
MU = 398600.4418        # gravitational parameter [km^3/s^2]
R_E = 6378.137          # equatorial radius [km]
J2 = 1.08263e-3         # oblateness coefficient [-]
OMEGA_E = 7.2921159e-5  # rotation rate [rad/s]

# Constellation
ALT = 1000.0            # altitude [km]
A = R_E + ALT           # orbit radius (circular) [km]

# Coverage
EPS_MIN_DEG = 10.0      # elevation mask [deg]
LAT_MIN, LAT_MAX = 33.0, 43.0     # region of interest [deg N]
LON_MIN, LON_MAX = 124.0, 131.0   # region of interest [deg E]
GRID_SPACING_DEG = 1.0            # ground grid spacing [deg]

# Time
DAY = 86400.0           # [s]
