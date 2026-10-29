"""
fig_initialization.py -- mean semi-major axis vs initial argument of latitude,
naive vs symmetric Walker initial conditions (Section 3C, fig:init).

Reuses mean_sma and the two initial-condition sets from premise_checks.py
(importing it re-runs its checks, so its printout is silenced here).
Run from the repo root:  python fig_initialization.py
"""
import os, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with contextlib.redirect_stdout(io.StringIO()):
    from premise_checks import mean_sma, X_naive, X_sym, u_dot, T, P, F
from constants import A
from walker import walker_angles

raan, arglat = walker_angles(T, P, F)
u0 = np.rad2deg(arglat)
abar_naive = mean_sma(X_naive, u_dot)
abar_sym = mean_sma(X_sym, u_dot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
plane = np.rint(np.rad2deg(raan) / (360.0 / P)).astype(int)
ax1.scatter(u0, abar_naive - A, c="tab:red", s=22, label="naive (same osculating circle)")
ax1.scatter(u0, abar_sym - A, c="tab:blue", s=22, label="symmetric (Eq. 13)")
ax1.set_xlabel(r"initial argument of latitude $u_{jk}$ [deg]")
ax1.set_ylabel(r"mean semi-major axis $-\,(R_E+h)$ [km]")
ax1.set_title(f"range: naive {np.ptp(abar_naive):.1f} km, symmetric {np.ptp(abar_sym)*1e3:.2f} m")
ax1.legend(fontsize=8, loc="best")

ax2.scatter(u0, (abar_sym - abar_sym.mean()) * 1e3, c="tab:blue", s=22)
ax2.set_xlabel(r"initial argument of latitude $u_{jk}$ [deg]")
ax2.set_ylabel("mean semi-major axis minus its average [m]")
ax2.set_title(f"symmetric only, std = {abar_sym.std()*1e3:.3f} m")

for a in (ax1, ax2):
    a.grid(alpha=0.3)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_initialization.{ext}", dpi=200)
print(f"naive: range {np.ptp(abar_naive):.4f} km, std {abar_naive.std():.4f} km")
print(f"symmetric: range {np.ptp(abar_sym)*1e3:.3f} m, std {abar_sym.std()*1e3:.3f} m")
print("saved figures/fig_initialization.(pdf|png)")
