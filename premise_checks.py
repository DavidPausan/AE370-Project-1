# This py contains all the numbers and checks that we use and show up in the premise.
# AI use. We used AI to help us organise the code in a concise manner, as well as help create a repository to help pull in functions from other files that we use.
# AI used to help us translate some expressions from the report over into python in a manner that would give us a result when tested.


import numpy as np
from constants import MU, R_E, J2, A, ALT, DAY
from dynamics import accel_point_mass, accel_j2
from walker import naive_constellation, symmetric_constellation
from coverage import footprint_half_angle
from rk4 import rk4
from dynamics import f

# Step 4: size of the forces at 1000 km
r_test = np.array([[A, 0.0, 0.0]])
a_grav = np.linalg.norm(accel_point_mass(r_test)) * 1000      # need to convert km/s^2 -> m/s^2
r_test_45 = np.array([[A * np.cos(np.pi / 4), 0.0, A * np.sin(np.pi / 4)]])
a_j2 = np.linalg.norm(accel_j2(r_test_45)) * 1000
print("Step 4: point-mass gravity = %.2f m/s^2" % a_grav)
print("Step 4: J2 (at 45 deg latitude) = %.4f m/s^2, ratio = 1/%.0f" % (a_j2, a_grav / a_j2))

#Step 4: J2 nodal regression (secular theory, Vallado)
inc = np.deg2rad(50.0)
n = np.sqrt(MU / A**3)
raan_dot = -1.5 * n * J2 * (R_E / A)**2 * np.cos(inc)
print("Step 4: J2 node drift at i = 45 deg: %.2f deg/day" % np.rad2deg(raan_dot * DAY))

# Step 10: footprint
lam = footprint_half_angle(ALT)
print("Step 10: lambda = %.2f deg" % np.rad2deg(lam))
print("Step 10: footprint radius = %.0f km" % (R_E * lam))
print("Step 10: fraction of Earth = %.1f %%" % (100 * (1 - np.cos(lam)) / 2))
print("Step 10: longest pass = %.1f min" % (2 * lam / n / 60))

# Step 8: naive vs symmetric initialization
# Baseline placeholder is to be replaced with the design of our sweep later down the project line.
INC, T, P, F = np.deg2rad(50.0), 55, 5, 1

X_naive = naive_constellation(INC, T, P, F)
X_sym, u_dot, raan_dot_meas = symmetric_constellation(INC, T, P, F)


def mean_sma(X0, u_dot, n_periods=5):
    """Average each satellite's osculating semi-major axis over whole periods."""
    period = 2 * np.pi / u_dot
    u, _ = rk4(f, X0, n_periods * period, period / 400)
    r = np.linalg.norm(u[:-1, :, 0:3], axis=2)
    v2 = np.sum(u[:-1, :, 3:6]**2, axis=2)
    sma = 1.0 / (2.0 / r - v2 / MU)                   
    return sma.mean(axis=0)


for name, X0 in [("naive", X_naive), ("symmetric", X_sym)]:
    abar = mean_sma(X0, u_dot)
    spread = abar.max() - abar.min()
    drift = np.rad2deg(1.5 * n * spread / A) * DAY
    print("Step 8 (%s): mean-a range = %.3f m, std = %.3f m, max drift = %.4f deg/day"
          % (name, spread * 1000, abar.std() * 1000, drift))
