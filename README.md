# AE370-Project-1

# Missile-Warning Constellation Coverage over the Korean Peninsula

**AE 370 (Aerospace Numerical Methods), Fall 2026, Group Project 1, UIUC**

David Pausan · Axel · Francesco · Joel

## The question

What is the smallest Walker-delta constellation at 1000 km altitude that keeps every
point of the Korean peninsula (33–43° N, 124–131° E) in view of at least two
satellites at all times? Two satellites are needed to track a missile launch in 3D
(stereo), not just detect it.

## The approach

Each satellite's motion is an initial value problem: point-mass gravity plus Earth's
J2 oblateness, written in the Earth-centered inertial frame. We solve it with a
fourth-order Runge–Kutta (RK4) method written from scratch; no library integrator is
used. At every 60 s we check which satellites are more than 10° above the horizon
at each point of a 1° grid over the region.

**Result:** the smallest design that keeps continuous stereo coverage over a 3-day
sweep is a Walker 50°:55/5/1 constellation: 55 satellites in 5 orbital planes,
inclined 50°. See the report for the 10-day check.

## Requirements

- Python 3.9 or newer
- NumPy and Matplotlib

```bash
pip install numpy matplotlib
```

## Files

### Model and numerical method

| File | What it does |
|---|---|
| `constants.py` | Earth constants (μ, R_E, J2, ω_E), altitude, elevation mask, region of interest |
| `dynamics.py` | Right-hand side f(u) of the IVP: point-mass gravity + J2 |
| `rk4.py` | The RK4 time marcher: `u, times = rk4(f, u_start, t_end, dt)` |
| `walker.py` | Walker-delta initial conditions, including the J2-corrected (symmetric) setup |
| `coverage.py` | Ground grid, elevation-angle visibility test, N(t) and stereo availability |

### Studies (one script per report section)

| Script | Report section | Output | Run time |
|---|---|---|---|
| `premise_checks.py` | 1 Premise | printed numbers (force sizes, J2 drift, footprint) | ~2 s |
| `eigenvalue_check.py` | 2.1 Choosing the method | printed eigenvalues and stiffness ratio | <1 s |
| `stability_region.py` | 2.3 Absolute stability | `fig01_stability.pdf` | seconds |
| `convergence.py` | 3.1 Convergence | convergence figure | ~1 min |
| `validation.py` | 3.2 Validation | validation figure and table values | ~1 min |
| `fig_initialization.py` | 3.3 Initial conditions | `fig_initialization.pdf` | seconds |
| `timestep_study.py` | 3.4 Choosing the time step | `fig_timestep.pdf` | ~2 min |
| `design_sweep.py` | 4.1 Design sweep | `sweep_results.csv`, `fig_design_sweep.pdf` | ~40 min |

## How to run

Run any script from the repository folder:

```bash
python premise_checks.py
python timestep_study.py
python design_sweep.py
```

`design_sweep.py` takes about 40 minutes (592 designs). On a Mac, keep the laptop
awake with:

```bash
caffeinate -i python design_sweep.py
```

The sweep's results are already saved in `sweep_results.csv`, so you don't need to
rerun it to reproduce the figure or the tables.

## Conventions

- Units: km, s and rad unless a name says otherwise (e.g. `EPS_MIN_DEG`).
- State of one satellite: `[x, y, z, vx, vy, vz]` in the ECI frame.
- A constellation's state has shape `(number of satellites, 6)`, so `f` and `rk4`
  advance every satellite in one call.
- Time step: Δt = 30 s for all coverage runs (justified in Section 3.4 of the report).

## Use of generative AI

We used AI tools (Claude and ChatGPT) throughout this project: to brainstorm the
premise, to help derive and check equations, and to help write and debug the code.
We reviewed, ran and verified all code ourselves, and every number in the report
comes from running these scripts. The report's Acknowledgments section gives
details for each part.
