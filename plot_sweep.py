import csv
import matplotlib.pyplot as plt

INCLINATIONS = [40.0, 45.0, 50.0, 55.0]

rows = []
with open("sweep_results.csv") as fh:
    for r in csv.DictReader(fh):
        rows.append({k: float(v) for k, v in r.items()})

# for each (inclination, T), keep the best design separately for each metric
best_single, best_stereo = {}, {}
for r in rows:
    key = (r["inc_deg"], int(r["T"]))
    if key not in best_single or r["single_avail"] > best_single[key]["single_avail"]:
        best_single[key] = r
    if key not in best_stereo or r["stereo_avail"] > best_stereo[key]["stereo_avail"]:
        best_stereo[key] = r

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
for inc_deg in INCLINATIONS:
    Ts = sorted(T for (i, T) in best_stereo if i == inc_deg)
    label = r"$i$ = %.0f$^\circ$" % inc_deg
    axs[0].plot(Ts, [100 * best_single[(inc_deg, T)]["single_avail"] for T in Ts],
                "o-", ms=3.5, label=label)
    axs[1].plot(Ts, [100 * best_stereo[(inc_deg, T)]["stereo_avail"] for T in Ts],
                "o-", ms=3.5, label=label)

for ax, title in zip(axs, [r"best single coverage ($N \geq 1$)",
                           r"best stereo coverage ($N \geq 2$)"]):
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("satellites in constellation, $T$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("time every point is covered [%]")
axs[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("fig_design_sweep.pdf")
print("wrote fig_design_sweep.pdf")