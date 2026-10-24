# walker.py
# Initial conditions from a Walker-delta pattern i:T/P/F.

import numpy as np
from constants import MU, A
from rk4 import rk4
from dynamics import f


def walker_angles(T, P, F):
    """RAAN and argument of latitude (rad) of every satellite (Eqs. 8 and 9)."""
    S = T // P                          # satellites per plane
    raan = []
    arglat = []
    for j in range(P):                  # plane j
        for k in range(S):              # satellite k in plane j
            raan.append(2 * np.pi * j / P)
            arglat.append(2 * np.pi * k / S + 2 * np.pi * F * j / T)
    return np.array(raan), np.mod(np.array(arglat), 2 * np.pi)


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def circular_state(a, inc, raan, arglat):
    """Position and velocity of a circular orbit in ECI (Eqs. 10 and 11)."""
    speed = np.sqrt(MU / a)
    r_orb = a * np.array([np.cos(arglat), np.sin(arglat), 0.0])
    v_orb = speed * np.array([-np.sin(arglat), np.cos(arglat), 0.0])
    R = rot_z(raan) @ rot_x(inc)
    return np.concatenate((R @ r_orb, R @ v_orb))


def naive_constellation(inc, T, P, F, a=A):
    """Every satellite on the same circular orbit at its Walker angles.
    Simple, but under J2 the satellites slowly drift apart (see Step 8)."""
    raan, arglat = walker_angles(T, P, F)
    states = np.zeros((T, 6))
    for s in range(T):
        states[s] = circular_state(a, inc, raan[s], arglat[s])
    return states


def raan_and_arglat(u):
    """RAAN and argument of latitude (rad) from states u with shape (..., 6)."""
    r = u[..., 0:3]
    v = u[..., 3:6]
    h = np.cross(r, v)                                   # angular momentum
    raan = np.arctan2(h[..., 0], -h[..., 1])
    node = np.stack((np.cos(raan), np.sin(raan), np.zeros_like(raan)), axis=-1)
    h_hat = h / np.linalg.norm(h, axis=-1, keepdims=True)
    in_plane = np.cross(h_hat, node)                     # 90 deg ahead of the node
    arglat = np.arctan2(np.sum(r * in_plane, axis=-1), np.sum(r * node, axis=-1))
    return raan, arglat


def symmetric_constellation(inc, T, P, F, dt=30.0, n_orbits=6, a=A):
    """Every satellite is a rotated, time-shifted copy of ONE reference orbit (Eq. 13).

    Steps:
      1. propagate a reference satellite (raan = 0, arglat = 0) for a few orbits
      2. measure its mean rates u_dot and raan_dot with a straight-line fit
      3. satellite (j, k): tau = u_jk / u_dot, take the reference state at tau,
         and rotate it about z by (raan_j - raan_dot * tau)
    Returns the (T, 6) initial states and the two measured rates.
    """
    x0 = circular_state(a, inc, 0.0, 0.0)
    period = 2 * np.pi * np.sqrt(a**3 / MU)
    u_ref, t_ref = rk4(f, x0[None, :], n_orbits * period, dt)
    raan_t, arglat_t = raan_and_arglat(u_ref[:, 0, :])

    u_dot = np.polyfit(t_ref, np.unwrap(arglat_t), 1)[0]
    raan_dot = np.polyfit(t_ref, np.unwrap(raan_t), 1)[0]

    raan_w, arglat_w = walker_angles(T, P, F)
    states = np.zeros((T, 6))
    for s in range(T):
        tau = arglat_w[s] / u_dot
        if tau > 0:
            x_tau = rk4(f, x0[None, :], tau, dt)[0][-1, 0, :]
        else:
            x_tau = x0
        R = rot_z(raan_w[s] - raan_dot * tau)
        states[s, 0:3] = R @ x_tau[0:3]
        states[s, 3:6] = R @ x_tau[3:6]
    return states, u_dot, raan_dot
