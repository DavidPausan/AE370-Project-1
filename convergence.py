"""AE 370 Project 1: RK4 convergence and RK4-vs-AB4 work study.

Run: python convergence.py
Requires: numpy and matplotlib.  No outside ODE/time integrators are used.

IMPORTANT: This is a reproducible standalone implementation of the model in the
report, not a test of a separate group's dynamics.py/rk4.py. Before citing the
numbers as validation of the group's main solver, cross-check the equations and
initial conditions and (preferably) make this script import their solver.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU = 398600.4418  # km^3/s^2
R_E = 6378.137  # km
J2 = 1.08263e-3
ALTITUDE = 1000.0  # km
INCLINATION_DEG = 50.0
DURATION = 86400.0  # 1 day, s
DT_REFERENCE = 0.5  # s
STEPS = np.array([3.75, 7.5, 15.0, 30.0, 60.0, 120.0])
HERE = Path(__file__).resolve().parent


def initial_state():
    """Circular osculating orbit, 50 deg inclination, u=RAAN=0 (km, km/s)."""
    radius = R_E + ALTITUDE
    speed = np.sqrt(MU / radius)
    inc = np.deg2rad(INCLINATION_DEG)
    return np.array([radius, 0, 0, 0, speed * np.cos(inc), speed * np.sin(inc)], float)


def rhs(t, y):
    """Point-mass Earth gravity + J2, ECI; state [r(km), v(km/s)]."""
    x, yy, z, vx, vy, vz = y
    r2 = x*x + yy*yy + z*z
    r = np.sqrt(r2)
    grav = -MU / (r2 * r)
    j2_fac = -1.5 * MU * J2 * R_E**2 / (r2 * r2 * r)
    z2overr2 = z*z / r2
    ax = grav*x + j2_fac*x*(1 - 5*z2overr2)
    ay = grav*yy + j2_fac*yy*(1 - 5*z2overr2)
    az = grav*z + j2_fac*z*(3 - 5*z2overr2)
    return np.array([vx, vy, vz, ax, ay, az])


def rk4_step(t, y, h):
    k1 = rhs(t, y)
    k2 = rhs(t + h/2, y + (h/2)*k1)
    k3 = rhs(t + h/2, y + (h/2)*k2)
    k4 = rhs(t + h, y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def rk4_integrate(y0, duration, h):
    steps = int(round(duration/h))
    assert abs(steps*h - duration) < 1e-7
    y = y0.copy()
    for k in range(steps):
        y = rk4_step(k*h, y, h)
    return y, 4*steps


def ab4_integrate(y0, duration, h):
    """Fourth-order Adams-Bashforth, started with 3 RK4 steps.

    f is evaluated once at each stored node (including final), so counts are
    slightly conservative. Unlike RK4, AB4 needs four previous derivatives.
    """
    steps = int(round(duration/h))
    assert abs(steps*h - duration) < 1e-7 and steps >= 4
    y = y0.copy()
    fs = [rhs(0.0, y)]
    calls = 1
    for k in range(3):
        y = rk4_step(k*h, y, h)
        calls += 4
        fs.append(rhs((k+1)*h, y))
        calls += 1
    # fs = [f(t0), f(t1), f(t2), f(t3)], current y at t3
    for k in range(3, steps):
        f_n, f_nm1, f_nm2, f_nm3 = fs[-1], fs[-2], fs[-3], fs[-4]
        y = y + (h/24) * (55*f_n - 59*f_nm1 + 37*f_nm2 - 9*f_nm3)
        fs.append(rhs((k+1)*h, y))
        fs.pop(0)
        calls += 1
    return y, calls


def main():
    y0 = initial_state()
    print('Computing full-model RK4 reference at dt = 0.5 s ...', flush=True)
    yref, _ = rk4_integrate(y0, DURATION, DT_REFERENCE)
    rows = []
    for h in STEPS:
        rk, nrk = rk4_integrate(y0, DURATION, float(h))
        ab, nab = ab4_integrate(y0, DURATION, float(h))
        err_rk = np.linalg.norm(rk[:3] - yref[:3])  # km
        err_ab = np.linalg.norm(ab[:3] - yref[:3])  # km
        rows.append((h, err_rk, err_ab, nrk, nab))
        print(f'dt={h:6g}s: RK4 err={err_rk:.8e} km  AB4 err={err_ab:.8e} km   RHS calls RK4={nrk:7d}, AB4={nab:7d}', flush=True)
    a = np.asarray(rows)
    h, rkerr, aberr, nrk, nab = a.T
    # Exclude the two smallest step sizes from order fit if roundoff dominates
    mask = h <= 15  # small-step regime is more relevant to asymptotic order
    order_rk = np.polyfit(np.log(h[mask]), np.log(rkerr[mask]), 1)[0]
    order_ab = np.polyfit(np.log(h[mask]), np.log(aberr[mask]), 1)[0]
    print(f'Observed small-step slopes (dt = 3.75, 7.5, 15 s): RK4 {order_rk:.3f}, AB4 {order_ab:.3f}')
    
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(h, rkerr*1e3, 'o-', label=f'RK4 (observed slope {order_rk:.2f} at small steps)')
    ax.loglog(h, (rkerr[0]*1e3)*(h/h[0])**4, ':', label='Fourth-order reference ($\\Delta t^4$)')
    ax.set_xlabel('Time step $\\Delta t$ (s)')
    ax.set_ylabel('Position error after one day (m)')
    ax.set_title('RK4 convergence under point-mass gravity + $J_2$')
    ax.grid(True, which='both', alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / 'fig_convergence.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(nrk, rkerr*1e3, 'o-', label='RK4')
    ax.loglog(nab, aberr*1e3, 's--', label='AB4 (RK4-started)')
    for j in [0, 2, 4, 5]:
        ax.annotate(f'{h[j]:g}s', (nrk[j], rkerr[j]*1e3), textcoords='offset points', xytext=(4,4), fontsize=8)
    ax.set_xlabel('Number of right-hand-side evaluations')
    ax.set_ylabel('Position error after one day (m)')
    ax.set_title('Accuracy versus model evaluation cost')
    ax.grid(True, which='both', alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / 'fig_cost.pdf')
    plt.close(fig)

    hdr='dt_s,rk4_error_km,ab4_error_km,rk4_rhs_calls,ab4_rhs_calls'
    np.savetxt(HERE / 'convergence_results.csv', a, delimiter=',', header=hdr, comments='', fmt=['%.5f','%.12e','%.12e','%.0f','%.0f'])
    (HERE / 'convergence_summary.txt').write_text(
        f'Parameters: mu={MU} km^3/s^2, Re={R_E} km, J2={J2}, altitude={ALTITUDE} km, '
        f'inclination={INCLINATION_DEG} deg, duration={DURATION} s, '
        f'reference RK4 step={DT_REFERENCE} s.\n'
        f'RK4 log-log order fitted on dt=3.75,7.5,15 s: {order_rk:.6f}\n'
        f'AB4 log-log order fitted on dt=3.75,7.5,15 s: {order_ab:.6f}\n'
        'Error = Euclidean satellite position difference at t=86400 s in km.\n'
        'RHS call count includes all AB4 RK4-startup evaluations.\n'
        'These tests use the standalone solver in convergence.py, not a separately provided group solver.\n'
    )
    print('Saved fig_convergence.pdf, fig_cost.pdf, convergence_results.csv, convergence_summary.txt')

if __name__ == '__main__':
    main()
