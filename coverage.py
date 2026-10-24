# coverage.py
# Which satellites can see which ground points (Step 10).

import numpy as np
from constants import (R_E, OMEGA_E, EPS_MIN_DEG, LAT_MIN, LAT_MAX,
                       LON_MIN, LON_MAX, GRID_SPACING_DEG)


def region_grid():
    """Earth-fixed unit vectors g_hat of every grid point, shape (n_points, 3)."""
    lats = np.deg2rad(np.arange(LAT_MIN, LAT_MAX + 1e-9, GRID_SPACING_DEG))
    lons = np.deg2rad(np.arange(LON_MIN, LON_MAX + 1e-9, GRID_SPACING_DEG))
    points = []
    for lat in lats:
        for lon in lons:
            points.append([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon),
                           np.sin(lat)])
    return np.array(points)


def grid_in_eci(grid, t):
    """Rotate the Earth-fixed grid into ECI at time t: g_hat(t) = R_z(omega_E t) g_hat."""
    th = OMEGA_E * t
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return grid @ R.T


def sats_in_view(r_sats, t, grid):
    """Number of satellites above the elevation mask at each grid point.

    r_sats : (N, 3) satellite positions in ECI [km]
    returns: (n_points,) integer counts
    """
    g = grid_in_eci(grid, t)                         # (n_points, 3)
    sin_eps = np.sin(np.deg2rad(EPS_MIN_DEG))
    counts = np.zeros(len(g), dtype=int)
    for p in range(len(g)):
        d = r_sats - R_E * g[p]                       # line of sight, (N, 3)
        sin_elev = (d @ g[p]) / np.linalg.norm(d, axis=1)
        counts[p] = np.sum(sin_elev > sin_eps)
    return counts


def min_in_view(u_history, times, grid, sample_every=1):
    """N(t): fewest satellites in view over the whole grid, at each sampled time.

    u_history : (n_times, N, 6) states from rk4
    returns   : sampled times and N(t)
    """
    t_out, n_out = [], []
    for k in range(0, len(times), sample_every):
        counts = sats_in_view(u_history[k, :, 0:3], times[k], grid)
        t_out.append(times[k])
        n_out.append(counts.min())
    return np.array(t_out), np.array(n_out)


def stereo_availability(n_min):
    """Fraction of samples where every point has at least 2 satellites in view."""
    return np.mean(n_min >= 2)


def footprint_half_angle(alt):
    """Earth central angle lambda (rad) seen by one satellite (Step 10)."""
    eps = np.deg2rad(EPS_MIN_DEG)
    return np.arccos(R_E * np.cos(eps) / (R_E + alt)) - eps
