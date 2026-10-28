"""
stability_region.py  --  RK4 absolute-stability region and Delta t * lambda points.

Produces fig01_stability.(png|pdf) for Section 2.2 and prints the checks used in
the text:
  (1) R(w) from applying RK4 to u' = lambda*u (symbolic, via sympy)
  (2) |R(iy)|^2 = 1 - y^6/72 + y^8/576 (symbolic) and its stability bound 2*sqrt(2)
  (3) the Delta t * lambda values plotted, using the eigenvalues from Section 2.1
  (4) the per-step amplitude / energy loss estimate
  (5) agreement of the repo's rk4.py with |R(iy)|^steps on a 10-day rotation

Run from the repo root (next to constants.py):  python stability_region.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
from constants import MU, A     # same constants as the rest of the repo (A = R_E + 1000 km)

N = np.sqrt(MU / A**3)          # mean motion, 1/s  (9.962e-4)


def two_body_eigenvalues():
    """Eigenvalues of the two-body Jacobian J = [[0, I], [G, 0]] at (A, 0, 0),
    built exactly as in eigenvalue_check.py (Section 2.1)."""
    r_hat = np.array([1.0, 0.0, 0.0])
    G = MU / A**3 * (3 * np.outer(r_hat, r_hat) - np.eye(3))
    J = np.zeros((6, 6))
    J[0:3, 3:6] = np.eye(3)
    J[3:6, 0:3] = G
    return np.linalg.eigvals(J)


def label(lam):
    """Classify an eigenvalue (in units of n) for the legend."""
    x = lam / N
    if abs(x.imag) < 1e-6:
        return r"$+\sqrt{2}\,n$" if x.real > 0 else r"$-\sqrt{2}\,n$"
    return r"$+i\,n$" if x.imag > 0 else r"$-i\,n$"


# one entry per distinct eigenvalue (+-i n is a double eigenvalue)
EIGS = {}
for _lam in two_body_eigenvalues():
    EIGS.setdefault(label(_lam), _lam)
DTS = [30.0, 120.0, 600.0]   # s


def R(w):
    """RK4 amplification factor."""
    return 1 + w + w**2 / 2 + w**3 / 6 + w**4 / 24


# ---------------------------------------------------------------- symbolic work
def symbolic_checks():
    try:
        import sympy as sp
    except ImportError:
        print("sympy not installed; skipping symbolic checks (pip install sympy)")
        return
    lam, u, dt, w, y = sp.symbols("lambda u dt w y", real=True)
    f = lambda v: lam * v
    k1 = f(u)
    k2 = f(u + dt / 2 * k1)
    k3 = f(u + dt / 2 * k2)
    k4 = f(u + dt * k3)
    unew = sp.expand(u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    Rsym = sp.expand((unew / u).subs(dt, w / lam))
    print("R(w) from the four stages      :", Rsym)
    print("k1+2k2+2k3+k4 (in units lam*u) :",
          sp.expand((k1 + 2 * k2 + 2 * k3 + k4) / (lam * u)).subs(dt, w / lam))

    Riy = sp.expand(Rsym.subs(w, sp.I * y))
    mod2 = sp.expand(sp.re(Riy) ** 2 + sp.im(Riy) ** 2)
    print("Re R(iy)                       :", sp.re(Riy))
    print("Im R(iy)                       :", sp.im(Riy))
    print("|R(iy)|^2                      :", mod2)
    print("|R(iy)|^2 - 1 factored         :", sp.factor(mod2 - 1))
    print("roots of y^2 - 8               :", sp.solve(y**2 - 8, y))


# ---------------------------------------------------------------- numeric checks
def numeric_checks():
    print(f"\nn = {N:.4e} 1/s   (period {2*np.pi/N/60:.1f} min)")
    print(f"{'dt [s]':>7} {'eigenvalue':>14} {'dt*lambda':>22} {'|R|':>14}")
    for dt in DTS:
        for name, lam in EIGS.items():
            w = dt * lam
            print(f"{dt:7.0f} {name.replace('$',''):>14} {w.real:+10.4f}{w.imag:+10.4f}j "
                  f"{abs(R(w)):14.10f}")
    print(f"\nmax |dt*lambda| at dt=30 s: {30*np.sqrt(2)*N:.4f}")

    # boundary crossings of |R|=1
    y = np.linspace(0, 4, 400001)
    bad = y[np.abs(R(1j * y)) > 1 + 1e-12]
    print(f"imaginary-axis crossing: first |R|>1 at y = {bad[0]:.5f}  (2*sqrt2 = {2*np.sqrt(2):.5f})")
    x = np.linspace(-4, 0, 400001)
    bad = x[np.abs(R(x)) > 1]
    print(f"negative-real-axis crossing: x = {bad[-1]:.4f}")

    # per-step damping for the orbit mode y = n*dt
    print("\nper-step amplitude loss on the +-i n mode (y = n*dt):")
    for dt in (30.0, 60.0):
        yy = N * dt
        loss_amp = 1 - abs(R(1j * yy))
        loss_en = 1 - abs(R(1j * yy)) ** 2
        steps10 = 10 * 86400 / dt
        steps60 = 60 * 86400 / dt
        print(f"  dt={dt:4.0f} s  y={yy:.5f}  1-|R|={loss_amp:.3e} (y^6/144={yy**6/144:.3e})  "
              f"1-|R|^2={loss_en:.3e}  steps/10d={steps10:.0f}  "
              f"energy-ish loss 10 d={loss_en*steps10:.2e}  60 d={loss_en*steps60:.2e}")
    r = (abs(R(1j * N * 60)) ** 2 - 1) / (abs(R(1j * N * 30)) ** 2 - 1)
    print(f"  ratio of per-step loss, 60 s / 30 s = {r:.1f}  (2^6 = 64; per unit time 2^5 = 32)")


# ---------------------------------------------------------------- check against rk4.py
def check_against_rk4py():
    """Run the repo's own rk4() on u' = i n u (a 2x2 rotation) for 10 days and compare
    the final amplitude with |R(i n dt)|^steps from the formula."""
    from rk4 import rk4
    f = lambda u, t: N * np.array([-u[1], u[0]])
    print("\nrk4.py vs R(iy)^steps on the rotation u' = i n u, 10 days:")
    for dt in (30.0, 60.0):
        u, t = rk4(f, np.array([1.0, 0.0]), 10 * 86400.0, dt)
        amp = np.hypot(u[-1, 0], u[-1, 1])
        steps = len(t) - 1
        pred = abs(R(1j * N * dt)) ** steps
        print(f"  dt={dt:4.0f} s  steps={steps}  rk4.py={amp:.10f}  formula={pred:.10f}")


# ---------------------------------------------------------------- figure
def make_figure(outdir):
    xs = np.linspace(-4.0, 1.5, 1200)
    ys = np.linspace(-3.6, 3.6, 1200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.abs(R(X + 1j * Y))

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.8),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    for a in (ax, axz):
        a.contourf(X, Y, Z, levels=[0, 1], colors=["#cfe3f5"])
        a.contour(X, Y, Z, levels=[1], colors=["#1f4e79"], linewidths=1.5)
        a.axhline(0, color="0.6", lw=0.6)
        a.axvline(0, color="0.6", lw=0.6)
        a.set_xlabel(r"Re$(\Delta t\,\lambda)$")
        a.set_ylabel(r"Im$(\Delta t\,\lambda)$")

    ax.set_aspect("equal")
    ax.set_xlim(-4.0, 1.5)
    ax.set_ylim(-3.6, 3.6)
    ax.set_title(r"RK4 stability region $|R(w)|\leq 1$")
    ax.text(-2.4, 0.4, r"stable", color="#1f4e79", fontsize=11)
    ax.annotate(r"$2\sqrt{2}$", xy=(0, 2 * np.sqrt(2)), xytext=(0.35, 2.9),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)

    colors = {30.0: "#2ca02c", 120.0: "#ff7f0e", 600.0: "#d62728"}
    markers = {r"$+\sqrt{2}\,n$": "o", r"$-\sqrt{2}\,n$": "s",
               r"$+i\,n$": "^", r"$-i\,n$": "v"}
    for dt in DTS:
        for name, lam in EIGS.items():
            w = dt * lam
            for a in (ax, axz):
                a.plot(w.real, w.imag, markers[name], color=colors[dt],
                       mfc="none" if name in (r"$+\sqrt{2}\,n$",) else colors[dt],
                       ms=6, mew=1.4)

    # zoom panel
    axz.set_xlim(-1.0, 1.0)
    axz.set_ylim(-0.8, 0.8)
    axz.set_aspect("equal")
    axz.set_title(r"Zoom near the origin")
    for dt in DTS:
        axz.annotate(rf"{dt:.0f} s", xy=(0, N * dt), xytext=(0.07, N * dt + 0.02),
                     color=colors[dt], fontsize=9)

    # legend (dt by color, eigenvalue by marker)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=colors[d], marker="o", ls="", label=rf"$\Delta t={d:.0f}$ s")
         for d in DTS]
    h += [Line2D([], [], color="k", marker=markers[k], ls="", label=k,
                 mfc="none" if k == r"$+\sqrt{2}\,n$" else "k") for k in EIGS]
    axz.legend(handles=h, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"fig01_stability.{ext}")
        fig.savefig(p, dpi=200)
        print("saved", p)


if __name__ == "__main__":
    symbolic_checks()
    numeric_checks()
    check_against_rk4py()
    make_figure(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
