import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from constants import DAY
from rk4 import rk4
from dynamics import f
from walker import symmetric_constellation
from coverage import region_grid, min_in_view

INCLINATIONS = [40.0 , 45.0, 50.0, 55.0]
PLANES = [4,5,6,7,8]
T_MIN, T_MAX = 40, 66
SWEEP_DAYS = 3
DT = 30.0
SAMPLE_EVERY = 2
OUT_FILE = "sweep_results.csv"
designs = []

for P in PLANES:
    for S in range(1,T_MAX+1):
        T=P*S
        if T_MIN <= T <= T_MAX:
            for F in range(P):
                designs.append((T,P,F)) #should print 148 designgs per inclination, 592 total.
print(len(designs), "designs per inclination,", len(designs) * len(INCLINATIONS), "in total")

grid = region_grid()

with open(OUT_FILE, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["inc_deg", "T", "P", "F", "single_avail", "stereo_avail", "min_N"])

#Used AI to give me code that would write to a csv file.

    for inc_deg in INCLINATIONS:
        tic = time.time()
        for (T, P, F) in designs:
            X0, _, _ = symmetric_constellation(np.deg2rad(inc_deg), T, P, F, dt=DT)
            u, times = rk4(f, X0, SWEEP_DAYS * DAY, DT)
            _, n_min = min_in_view(u, times, grid, sample_every=SAMPLE_EVERY)

            single = np.mean(n_min >= 1)
            stereo = np.mean(n_min >= 2)
            writer.writerow([inc_deg, T, P, F, single, stereo, n_min.min()])
            fh.flush()

        print("inclination %.0f deg done in %.0f s" % (inc_deg, time.time() - tic))

rows = []
with open(OUT_FILE) as fh:
    for r in csv.DictReader(fh):
        rows.append({k: float(v) for k, v in r.items()})

best = {}
for r in rows:
    key = (r["inc_deg"], int(r["T"]))
    if key not in best or r["stereo_avail"] > best[key]["stereo_avail"]:
        best[key] = r

print("\nSmallest designs with 100%% stereo coverage over %d days:" % SWEEP_DAYS)
passing = sorted([b for b in best.values() if b["stereo_avail"] == 1.0],
                 key=lambda b: (b["T"], b["inc_deg"]))
for b in passing[:6]:
    print("  %2.0f deg : %d/%d/%d" % (b["inc_deg"], b["T"], b["P"], b["F"]))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
for inc_deg in INCLINATIONS:
    pts = sorted([b for b in best.values() if b["inc_deg"] == inc_deg], key=lambda b: b["T"])
    Ts = [b["T"] for b in pts]
    axs[0].plot(Ts, [100 * b["single_avail"] for b in pts], "o-", ms=3.5,
                label=r"$i$ = %.0f$^\circ$" % inc_deg)
    axs[1].plot(Ts, [100 * b["stereo_avail"] for b in pts], "o-", ms=3.5,
                label=r"$i$ = %.0f$^\circ$" % inc_deg)
    
for ax, title in zip(axs, [r"single coverage ($N \geq 1$)", r"stereo coverage ($N \geq 2$)"]):
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("satellites in constellation, $T$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("time every point is covered [%]")
axs[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("fig_design_sweep.pdf")
print("\nwrote %s and fig_design_sweep.pdf" % OUT_FILE)

#USED AI TO HELP FORMULATE LOGIC FOR ALL OF THIS CODE AND WRITE THE CODE GIVEN THE CONTEXT OF THE PROJECT