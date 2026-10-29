import time
import numpy as np
import matplotlib.pyplot as plt
from constants import MU, A, DAY
from rk4 import rk4
from dynamics import f
from walker import symmetric_constellation
from coverage import region_grid, min_in_view, stereo_availability

#baseline design:
INC, T, P, F = np.deg2rad(50.0), 55, 5, 1
T_END = 10 * DAY
DT_COV = 60.0 #coverage sampling interval
DTS = [7.5, 15.0, 30.0, 60.0] 
V_ORB = np.sqrt(MU/A) # Orbital speed. Use again when turning position error into timing error

#Initial conditions and ground grid
X0, _, _ = symmetric_constellation(INC, T, P, F) #holds starting shape
grid = region_grid() #88 ground points of Korea.

#Run each step
results = {}
for dt in DTS:
    tic = time.time()
    u, times = rk4(f,X0, T_END, dt)
    run_time = time.time() - tic

every = int(round(DT_COV/dt)) # the amount of steps that make up 60seconsd
t_cov, n_min = min_in_view(u, times, grid, sample_every = every)

results[dt] = {"final": u[-1].copy(), "n_min": n_min,
               "Avail": stereo_availability(n_min), "run_time": run_time}
del u #free the full history.

print("dt = %5.1f s: run time %.1f s, stereo availability %.6f, min N = %d"
        % (dt, run_time, results[dt]["avail"], n_min.min()))

ref = results[DTS[0]]
rows = []
for dt in DTS[1:]:
    err = np.linalg.norm(results[dt]["final"][:, 0:3] - ref["final"][:, 0:3], axis=1)
    max_err = err.max()
    timing = max_err / V_ORB
    mismatch = np.mean((results[dt]["n_min"] >= 2) != (ref["n_min"] >= 2))
    rows.append((dt, max_err, timing, results[dt]["avail"], mismatch))

print("dt = %5.1f s: max position error %.3f km, timing error %.3f s, "
        "availability %.6f, samples that differ %.4f %%"
        % (dt, max_err, timing, results[dt]["avail"], 100 * mismatch))
    
dt_arr = np.array([r[0] for r in rows])
timing_arr = np.array([r[2] for r in rows])
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.loglog(dt_arr, timing_arr, "o-", label="timing error at day 10")
ax.loglog(dt_arr, timing_arr[0] * (dt_arr / dt_arr[0])**5, "k--", lw=1,
          label=r"$\propto \Delta t^5$")
ax.axhline(DT_COV, color="r", ls="--", lw=1, label="coverage sampling interval (60 s)")
ax.axvline(30, color="gray", lw=0.8) 

ax.set_xticks(dt_arr)
ax.set_xticklabels(["%g" % d for d in dt_arr])
ax.minorticks_off()
ax.set_xlabel(r"$\Delta t$ [s]")
ax.set_ylabel("position error / orbital speed [s]")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_timestep.pdf")