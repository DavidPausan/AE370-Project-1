"""
ten_day_check.py -- certify the smallest passing designs over 10 days (Section 4.3).

The 3-day sweep (design_sweep.py) keeps the best phasing F for every inclination and
size T. Ground tracks do not repeat exactly, so a design that covers 3 days can still
have a short gap later. This script re-reads sweep_results.csv, takes the smallest
designs that passed the 3-day sweep (same selection rule as design_sweep.py) and
propagates each one for 10 days at dt = 30 s, sampling coverage every 60 s.

Outputs
  ten_day_results.csv      one row per design: availability, worst N, gaps, margin
  ten_day_answer.npz       per-sample N(t) and per-point counts of the answer design
  fig_ten_day_map.pdf      map of the region for the answer design

Run from the repo root:  python ten_day_check.py      (about 3-6 minutes)
"""
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from constants import DAY, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from rk4 import rk4
from dynamics import f
from walker import symmetric_constellation
from coverage import region_grid, sats_in_view

SWEEP_FILE = "sweep_results.csv"
N_DESIGNS = 6            # how many of the smallest 3-day passers to re-run
DAYS = 10
DT = 30.0
SAMPLE_EVERY = 2         # 2 steps of 30 s = the 60 s coverage interval
DT_COV = DT * SAMPLE_EVERY
OUT_CSV = "ten_day_results.csv"
OUT_NPZ = "ten_day_answer.npz"
OUT_FIG = "fig_ten_day_map.pdf"


def passing_designs(n):
    """Smallest designs that reached 100% stereo in the 3-day sweep.
    Same rule as design_sweep.py: best F per (inclination, T), sorted by (T, inclination)."""
    rows = []
    with open(SWEEP_FILE) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in r.items()})
    best = {}
    for r in rows:
        key = (r["inc_deg"], int(r["T"]))
        if key not in best or r["stereo_avail"] > best[key]["stereo_avail"]:
            best[key] = r
    passing = sorted([b for b in best.values() if b["stereo_avail"] == 1.0],
                     key=lambda b: (b["T"], b["inc_deg"]))
    return [(b["inc_deg"], int(b["T"]), int(b["P"]), int(b["F"])) for b in passing[:n]]


def longest_run(mask):
    """Length of the longest run of True values in a boolean array."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def run_design(inc_deg, T, P, F, grid):
    """Propagate one design for DAYS days; return counts[sample, grid point]."""
    X0, _, _ = symmetric_constellation(np.deg2rad(inc_deg), T, P, F, dt=DT)
    u, times = rk4(f, X0, DAYS * DAY, DT)
    idx = range(0, len(times), SAMPLE_EVERY)
    counts = np.array([sats_in_view(u[k, :, 0:3], times[k], grid) for k in idx])
    return counts, times[::SAMPLE_EVERY]


if __name__ == "__main__":
    grid = region_grid()
    designs = passing_designs(N_DESIGNS)
    print("Re-running the %d smallest 3-day passers for %d days at dt = %g s:" %
          (len(designs), DAYS, DT))
    for d in designs:
        print("   %.0f deg : %d/%d/%d" % d)

    results, answer = [], None
    for (inc_deg, T, P, F) in designs:
        tic = time.time()
        counts, t_s = run_design(inc_deg, T, P, F, grid)
        n_min = counts.min(axis=1)                 # N(t): worst grid point at each sample
        fail = n_min < 2
        row = {
            "inc_deg": inc_deg, "T": T, "P": P, "F": F,
            "stereo_avail": float(np.mean(~fail)),
            "min_N": int(n_min.min()), "max_N": int(n_min.max()),
            "frac_exactly2": float(np.mean(n_min == 2)),
            "n_fail_samples": int(fail.sum()),
            "fail_minutes": float(fail.sum() * DT_COV / 60.0),
            "longest_gap_min": float(longest_run(fail) * DT_COV / 60.0),
            "first_fail_day": float(t_s[fail][0] / DAY) if fail.any() else float("nan"),
            "passes": bool(not fail.any()),
        }
        results.append(row)
        print("%2.0f deg %d/%d/%d: stereo %.5f%%, min N = %d, fail %g min (longest %g min), "
              "worst point has exactly 2 for %.1f%% of the time  [%.0f s]" %
              (inc_deg, T, P, F, 100 * row["stereo_avail"], row["min_N"], row["fail_minutes"],
               row["longest_gap_min"], 100 * row["frac_exactly2"], time.time() - tic))
        if answer is None and row["passes"]:
            answer = (row, counts, n_min, t_s)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    if answer is None:
        print("\nNo design in the list holds for all %d days; raise N_DESIGNS." % DAYS)
    else:
        row, counts, n_min, t_s = answer
        print("\nSmallest design that holds for all %d days: %.0f deg : %d/%d/%d" %
              (DAYS, row["inc_deg"], row["T"], row["P"], row["F"]))
        print("  worst grid point has exactly 2 satellites for %.1f%% of the samples, never more than %d"
              % (100 * row["frac_exactly2"], row["max_N"]))
        point_min = counts.min(axis=0)             # worst case over 10 days at each point
        frac2 = np.mean(counts == 2, axis=0)       # fraction of time each point has exactly 2
        np.savez(OUT_NPZ, n_min=n_min, t=t_s, counts_point_min=point_min, frac_exactly2=frac2,
                 design=np.array([row["inc_deg"], row["T"], row["P"], row["F"]]))

        n_lat = int(round((LAT_MAX - LAT_MIN))) + 1
        n_lon = int(round((LON_MAX - LON_MIN))) + 1
        extent = [LON_MIN - 0.5, LON_MAX + 0.5, LAT_MIN - 0.5, LAT_MAX + 0.5]
        n_at_floor = int(np.sum(point_min == 2))
        print("  grid points whose worst case over %d days is exactly 2: %d of %d"
              % (DAYS, n_at_floor, point_min.size))
        print("  per-point time at exactly 2: %.1f%% to %.1f%% (mean %.1f%%)"
              % (100 * frac2.min(), 100 * frac2.max(), 100 * frac2.mean()))

        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
        im = axs[0].imshow(100 * frac2.reshape(n_lat, n_lon), origin="lower", extent=extent,
                           cmap="magma_r", aspect="equal")
        cb = fig.colorbar(im, ax=axs[0], fraction=0.046)
        cb.set_label("time with exactly 2 satellites in view [%]")
        axs[0].set_title("(a) answer design, per grid point\n(worst case is 2 at %d of %d points)"
                         % (n_at_floor, point_min.size), fontsize=10)
        axs[0].set_xlabel("longitude [deg E]")
        axs[0].set_ylabel("latitude [deg N]")

        labels = ["%.0f$^\\circ$:%d/%d/%d" % (r["inc_deg"], r["T"], r["P"], r["F"]) for r in results]
        vals = [100 * r["frac_exactly2"] for r in results]
        colors = ["tab:blue" if r["passes"] else "tab:red" for r in results]
        bars = axs[1].bar(range(len(results)), vals, color=colors)
        for k, r in enumerate(results):
            if not r["passes"]:
                axs[1].text(k, vals[k] + 1.5, "fails:\n%g min at N=1" % r["fail_minutes"],
                            ha="center", fontsize=8, color="tab:red")
        axs[1].set_xticks(range(len(results)))
        axs[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axs[1].set_ylim(0, 110)
        axs[1].set_ylabel("time the worst point has exactly 2 [%]")
        axs[1].set_title("(b) margin of each design over %d days" % DAYS, fontsize=10)
        axs[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG)
        print("wrote %s, %s and %s" % (OUT_CSV, OUT_NPZ, OUT_FIG))
