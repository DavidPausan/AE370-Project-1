# dynamics.py
# Right-hand side f(u) of the IVP: point-mass gravity + J2 (no drag).
# State of one satellite: [x, y, z, vx, vy, vz] in the ECI frame (km, km/s).
# For a constellation, u has shape (number of satellites, 6).

import numpy as np
from constants import MU, R_E, J2


def accel_point_mass(r):
    """Point-mass gravity, a = -mu r / |r|^3  (Eq. 1). r has shape (N, 3)."""
    r_norm = np.linalg.norm(r, axis=1)
    return -MU * r / r_norm[:, None] ** 3


def accel_j2(r):
    """J2 acceleration from a = -grad(U)  (Eq. 3). r has shape (N, 3)."""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r_norm = np.linalg.norm(r, axis=1)
    factor = -1.5 * MU * J2 * R_E**2 / r_norm**5
    zz = 5.0 * z**2 / r_norm**2

    ax = factor * x * (1.0 - zz)
    ay = factor * y * (1.0 - zz)
    az = factor * z * (3.0 - zz)
    return np.column_stack((ax, ay, az))


def f(u, t, use_j2=True):
    """du/dt for every satellite. u has shape (N, 6). t is unused (f is autonomous)."""
    r = u[:, 0:3]
    v = u[:, 3:6]
    a = accel_point_mass(r)
    if use_j2:
        a = a + accel_j2(r)
    return np.hstack((v, a))


def f_two_body(u, t):
    """Same as f but without J2 (used for the Kepler validation case)."""
    return f(u, t, use_j2=False)
