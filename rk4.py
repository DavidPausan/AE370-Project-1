# rk4.py
# Classical 4-stage Runge-Kutta time marcher (Lecture 10).
# Same convention as the homework:  u, times = rk4(f, u_start, t_end, dt)

import numpy as np


def rk4(f, u_start, t_end, dt, t_start=0.0):
    """March du/dt = f(u, t) from t_start to t_end with step dt.

    u_start can be any shape (for us: (number of satellites, 6)).
    Returns
        u     : array with shape (number of steps + 1, *u_start.shape)
        times : array of the times for each row of u
    If (t_end - t_start) is not a whole number of steps, dt is shrunk
    slightly so the last step lands exactly on t_end.
    """
    n_steps = int(np.ceil((t_end - t_start) / dt - 1e-9))
    n_steps = max(n_steps, 1)
    dt = (t_end - t_start) / n_steps

    u = np.zeros((n_steps + 1,) + np.shape(u_start))
    times = t_start + dt * np.arange(n_steps + 1)
    u[0] = u_start

    for k in range(n_steps):
        t = times[k]
        k1 = f(u[k], t)
        k2 = f(u[k] + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(u[k] + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(u[k] + dt * k3, t + dt)
        u[k + 1] = u[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return u, times
