"""AE370 Project 1: independent orbital-dynamics validation.

Run: python validation.py
Outputs: validation_results.csv and validation_three_panel.png
Units: km, seconds, radians. Uses the team's rk4.py, dynamics.py, walker.py and constants.py.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from constants import MU, R_E as RE, J2, A, DAY
from rk4 import rk4
from dynamics import f, f_two_body
from walker import raan_and_arglat

INCLINATION = np.deg2rad(50.0)
DAYS = 10
DURATION = DAYS * 86400.0
STEPS = (120.0, 60.0, 30.0, 15.0)


def solve_kepler(M, e, tol=1e-13, max_iter=30):
    """Newton solve M=E-e*sin(E), supporting scalar or array mean anomalies."""
    M = np.asarray(M, dtype=float)
    if not 0 <= e < 1:
        raise ValueError('Kepler solver requires 0 <= e < 1')
    E = M.copy()
    for _ in range(max_iter):
        delta = (E - e*np.sin(E) - M) / (1-e*np.cos(E))
        E -= delta
        if np.max(np.abs(delta)) < tol:
            return E
    raise RuntimeError('Kepler solver did not converge')


def kepler_state(t, a=A, e=0.1, mu=MU):
    """Exact perifocal state for periapsis at t=0."""
    E = solve_kepler(np.sqrt(mu/a**3)*np.asarray(t), e)
    c, s = np.cos(E), np.sin(E)
    denom = 1-e*c
    pos = np.stack((a*(c-e), a*np.sqrt(1-e**2)*s, np.zeros_like(c)), axis=-1)
    speed = np.sqrt(mu/a)/denom
    vel = np.stack((-speed*s, speed*np.sqrt(1-e**2)*c, np.zeros_like(c)), axis=-1)
    return np.concatenate((pos, vel), axis=-1)


def rotate_x_state(state, inc=INCLINATION):
    s = np.asarray(state).copy()
    c, si = np.cos(inc), np.sin(inc)
    for j in (0, 3):
        y, z = np.array(s[..., j+1], copy=True), np.array(s[..., j+2], copy=True)
        s[..., j+1] = c*y-si*z
        s[..., j+2] = si*y+c*z
    return s


def integrate(initial, dt, duration=DURATION, j2=J2, sample_every=1):
    """Use the project's RK4 and dynamics, returning a single satellite's states."""
    if j2 == 0:
        rhs = f_two_body
    elif j2 == J2:
        rhs = f
    else:
        raise ValueError('This validation uses J2=0 or the project J2 constant')
    states, times = rk4(rhs, np.asarray(initial)[None, :], duration, dt)
    return times[::sample_every], states[::sample_every, 0, :]


def specific_energy(states, mu=MU, j2=J2, re=RE):
    states = np.asarray(states)
    r, v = states[..., :3], states[..., 3:]
    radius = np.linalg.norm(r, axis=-1)
    z = r[..., 2]
    return (0.5*np.sum(v*v, axis=-1)-mu/radius
            +mu*j2*re**2*(3*z*z/radius**2-1)/(2*radius**3))


def angular_momentum_z(states):
    s = np.asarray(states)
    return s[..., 0]*s[..., 4]-s[..., 1]*s[..., 3]


def raan(states):
    r, v = states[..., :3], states[..., 3:]
    h = np.cross(r, v)
    return np.arctan2(h[..., 0], -h[..., 1])  # node vector = (-hy, hx, 0)


def raan_rate_theory(a=A, inc=INCLINATION, e=0):
    n = np.sqrt(MU/a**3)
    return -1.5*n*J2*(RE/a)**2*np.cos(inc)/(1-e**2)**2


def main():
    rows = []
    kepler_errors, energy_errors, hz_errors = [], [], []
    initial_ecc = kepler_state(0.0, e=0.1)
    initial_j2 = rotate_x_state(kepler_state(0.0, e=0.0))
    for dt in STEPS:
        # Compare at common times over one orbit, including its end.
        period = 2*np.pi*np.sqrt(A**3/MU)
        duration = np.floor(period/dt)*dt
        times, states = integrate(initial_ecc, dt, duration=duration, j2=0)
        exact = kepler_state(times, e=0.1)
        err = np.max(np.linalg.norm(states[:, :3]-exact[:, :3], axis=1))
        kepler_errors.append(err)
        # Evaluate full 10-day conservation on the J2 orbit.
        t, s = integrate(initial_j2, dt, j2=J2, sample_every=max(1, int(60/dt)))
        energy = specific_energy(s)
        hz = angular_momentum_z(s)
        de = np.max(np.abs((energy-energy[0])/energy[0]))
        dh = np.max(np.abs((hz-hz[0])/hz[0]))
        energy_errors.append(de)
        hz_errors.append(dh)
        rows.append({'test':'Kepler maximum position error','dt_s':dt,'expected':'fourth-order global convergence','computed':f'{err:.9g} km'})
        rows.append({'test':'J2 energy conservation','dt_s':dt,'expected':'constant specific energy','computed':f'{de:.9g} max relative deviation'})
        rows.append({'test':'J2 h_z conservation','dt_s':dt,'expected':'constant h_z','computed':f'{dh:.9g} max relative deviation'})
        if dt == 30.0:
            time30, state30 = t, s
            energy30, hz30 = energy, hz
    times, states = time30, state30
    angles = np.unwrap(raan_and_arglat(states)[0])
    slope, intercept = np.polyfit(times, angles, 1)
    expected = raan_rate_theory()
    conv = 86400*180/np.pi
    percent = abs((slope-expected)/expected)*100
    rows.append({'test':'10-day J2 RAAN drift','dt_s':30,'expected':f'{expected*conv:.8f} deg/day',
                 'computed':f'{slope*conv:.8f} deg/day ({percent:.5f}% difference)'})
    with open('validation_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['test','dt_s','expected','computed'])
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.loglog(STEPS, kepler_errors, 'o-', label='RK4 vs Kepler')
    ref = kepler_errors[0]*(np.asarray(STEPS)/STEPS[0])**4
    ax.loglog(STEPS, ref, '--', label='Fourth-order reference')
    ax.set(xlabel='Time step (s)', ylabel='Maximum position error (km)', title='(a) Two-body convergence')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=.3)
    ax = axes[1]
    ax.semilogy(time30/86400, np.maximum(np.abs((energy30-energy30[0])/energy30[0]), 1e-17), label='Energy')
    ax.semilogy(time30/86400, np.maximum(np.abs((hz30-hz30[0])/hz30[0]), 1e-17), label=r'$h_z$')
    ax.set(xlabel='Time (days)', ylabel='Relative deviation', title='(b) J2 invariants (30 s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    ax = axes[2]
    ax.plot(times/86400, np.rad2deg(angles-angles[0]), label='Numerical RAAN')
    ax.plot(times/86400, expected*times*180/np.pi, '--', label='First-order J2 theory')
    ax.set(xlabel='Time (days)', ylabel='Change in RAAN (deg)', title='(c) J2 node precession')
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    fig.savefig('validation_three_panel.png', dpi=200)
    plt.close(fig)
    print('dt(s)  Kepler max error(km)  J2 energy drift  J2 hz drift')
    for vals in zip(STEPS, kepler_errors, energy_errors, hz_errors):
        print('%5.0f  %20.8g  %15.7g  %11.7g' % vals)
    print(f'RAAN theory: {expected*conv:.8f} deg/day; computed: {slope*conv:.8f} deg/day; difference: {percent:.5f}%')
    for name, errors in [('Kepler',kepler_errors),('Energy',energy_errors),('h_z',hz_errors)]:
        print(f'{name} fitted order: {np.polyfit(np.log(STEPS), np.log(np.maximum(errors, 1e-17)), 1)[0]:.3f}')

if __name__ == '__main__':
    main()
