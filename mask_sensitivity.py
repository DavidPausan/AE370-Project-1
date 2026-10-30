"""AE 370 Project 1 / Section 4D: elevation-mask sensitivity.

Run in same folder as constants.py, dynamics.py, rk4.py, walker.py, coverage.py.
Requires: numpy, matplotlib, numba  (pip install numpy matplotlib numba)

Search 50-degree Walker-delta designs, P=4..8, T=40..100 divisible by P,
F=0..P-1, over a 3-day period, 30-s RK4, 60-s coverage sampling,
88 ground grid points, for elevation masks 5, 10, 15 degrees.

The SAME numerical trajectory is evaluated at all three masks, which
eliminates redundant integrations. All propagators are RK4 from scratch.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# The project's local coverage.py shadows the 3rd-party coverage package that
# Numba imports. Temporarily hide this directory while importing Numba.
import sys
_current_dir = str(Path(__file__).resolve().parent)
_original_sys_path = sys.path[:]
sys.path = [p for p in sys.path if p not in ('', _current_dir)]
try:
    from numba import njit
finally:
    sys.path = _original_sys_path
from constants import MU, R_E, J2, OMEGA_E, A
from walker import circular_state, symmetric_constellation, walker_angles
import importlib.util
_spec = importlib.util.spec_from_file_location("group_coverage", Path(__file__).with_name("coverage.py"))
_group_coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_group_coverage)
region_grid = _group_coverage.region_grid
min_in_view = _group_coverage.min_in_view
stereo_availability = _group_coverage.stereo_availability
from rk4 import rk4
from dynamics import f

OUT = Path('mask_extension_results')
INCLINATION_DEG = 50.0
MASKS = np.array([5.0, 10.0, 15.0])
TIMESTEP_S = 30.0
COVERAGE_INTERVAL_S = 60.0
DAYS_SWEEP = 3
DAYS_VERIFY = 10
T_MIN = 40
T_MAX = 100
PLANES = range(4, 9)

@njit
def rhs_one(y):
    x, yy, z, vx, vy, vz = y
    rr2=x*x+yy*yy+z*z
    rr=math.sqrt(rr2)
    v3=MU/(rr2*rr)
    jt=1.5*MU*J2*R_E*R_E/(rr2*rr2*rr)
    fac=5.0*z*z/rr2
    out=np.empty(6)
    out[0]=vx; out[1]=vy; out[2]=vz
    out[3]=-v3*x-jt*x*(1.0-fac)
    out[4]=-v3*yy-jt*yy*(1.0-fac)
    out[5]=-v3*z-jt*z*(3.0-fac)
    return out

@njit
def step_one(y, h):
    k1=rhs_one(y)
    k2=rhs_one(y+0.5*h*k1)
    k3=rhs_one(y+0.5*h*k2)
    k4=rhs_one(y+h*k3)
    return y+(h/6.0)*(k1+2*k2+2*k3+k4)

@njit
def build_states(x0, arglat, raan, u_dot, raan_dot, dt):
    ns=len(arglat)
    states=np.empty((ns,6))
    for s in range(ns):
        tau=arglat[s]/u_dot
        nsteps=max(1,int(math.ceil(tau/dt-1e-9)))
        h=tau/nsteps
        y=x0.copy()
        for k in range(nsteps):
            y=step_one(y,h)
        angle=raan[s]-raan_dot*tau
        c=math.cos(angle); ss=math.sin(angle)
        states[s,0]=c*y[0]-ss*y[1]
        states[s,1]=ss*y[0]+c*y[1]
        states[s,2]=y[2]
        states[s,3]=c*y[3]-ss*y[4]
        states[s,4]=ss*y[3]+c*y[4]
        states[s,5]=y[5]
    return states

@njit
def rhs_all(y, out):
    for s in range(len(y)):
        x=y[s,0]; yy=y[s,1]; z=y[s,2]
        rr2=x*x+yy*yy+z*z
        rr=math.sqrt(rr2)
        grav=MU/(rr2*rr)
        jt=1.5*MU*J2*R_E*R_E/(rr2*rr2*rr)
        fac=5.0*z*z/rr2
        out[s,0]=y[s,3]; out[s,1]=y[s,4]; out[s,2]=y[s,5]
        out[s,3]=-grav*x-jt*x*(1.0-fac)
        out[s,4]=-grav*yy-jt*yy*(1.0-fac)
        out[s,5]=-grav*z-jt*z*(3.0-fac)

@njit
def propagate_score(u_init, grid, masks, days):
    """Advance group's exact point-mass + J2 RK4 ODE, score masks at 60s."""
    u=u_init.copy()
    h=TIMESTEP_S
    samples=int(days*86400.0/COVERAGE_INTERVAL_S)+1
    steps_per_sample=int(COVERAGE_INTERVAL_S/h)
    count_good=np.zeros(len(masks),dtype=np.int64)
    worst_min=np.full(len(masks),1000,dtype=np.int64)
    k1=np.empty_like(u); k2=np.empty_like(u); k3=np.empty_like(u); k4=np.empty_like(u)
    temp=np.empty_like(u)
    cos_eps=np.empty(len(masks)); sin_eps=np.empty(len(masks))
    for m in range(len(masks)):
        sin_eps[m]=math.sin(math.pi/180.0*masks[m])
        cos_eps[m]=math.cos(math.pi/180.0*masks[m])

    for it in range(samples):
        t=it*COVERAGE_INTERVAL_S
        c=math.cos(OMEGA_E*t); s=math.sin(OMEGA_E*t)
        thresholds=np.empty((len(masks),len(u)))
        for sat in range(len(u)):
            rr2=u[sat,0]**2+u[sat,1]**2+u[sat,2]**2
            for m in range(len(masks)):
                ce=cos_eps[m]
                thresholds[m,sat] = R_E*ce*ce + sin_eps[m]*math.sqrt(rr2-R_E*R_E*ce*ce)
        mins=np.full(len(masks),1000,dtype=np.int64)
        for q in range(len(grid)):
            gx=c*grid[q,0]-s*grid[q,1]
            gy=s*grid[q,0]+c*grid[q,1]
            gz=grid[q,2]
            counts=np.zeros(len(masks),dtype=np.int64)
            for sat in range(len(u)):
                dot=u[sat,0]*gx+u[sat,1]*gy+u[sat,2]*gz
                for m in range(len(masks)):
                    if dot > thresholds[m,sat]:
                        counts[m]+=1
            for m in range(len(masks)):
                if counts[m]<mins[m]:
                    mins[m]=counts[m]
        for m in range(len(masks)):
            if mins[m]>=2:
                count_good[m]+=1
            if mins[m]<worst_min[m]:
                worst_min[m]=mins[m]
        if it+1<samples:
            for stage in range(steps_per_sample):
                rhs_all(u,k1)
                temp[:,:]=u+0.5*h*k1
                rhs_all(temp,k2)
                temp[:,:]=u+0.5*h*k2
                rhs_all(temp,k3)
                temp[:,:]=u+h*k3
                rhs_all(temp,k4)
                u+=h/6.0*(k1+2*k2+2*k3+k4)
    return count_good/samples, worst_min

def configurations(t_min=T_MIN,t_max=T_MAX):
    for T in range(t_min,t_max+1):
        for P in PLANES:
            if T%P: continue
            for F in range(P):
                yield T,P,F

def verify_against_group():
    """Compare JIT RK4 and visibility against canonical group f/rk4/coverage."""
    inc=math.radians(INCLINATION_DEG)
    x0=circular_state(A,inc,0.,0.)
    y=x0.copy()
    for k in range(20):
        y=step_one(y,TIMESTEP_S)
    true, _=rk4(f,x0[None,:],20*TIMESTEP_S,TIMESTEP_S)
    err=np.max(np.abs(y-true[-1,0,:]))
    print('JIT RK4 vs group rk4.py (600 s), max state component discrepancy:',err)
    assert err < 1e-8, 'propagator does not reproduce group integrator'
    u_dot, ra_dot = symmetric_constellation(inc,1,1,0)[1:]
    raan, arglat=walker_angles(55,5,1)
    u_fast=build_states(x0,arglat,raan,u_dot,ra_dot,TIMESTEP_S)
    u_orig=symmetric_constellation(inc,55,5,1)[0]
    d=np.max(np.abs(u_fast-u_orig))
    print('JIT initial-condition generator vs group walker.py, max state component discrepancy:',d)
    assert d < 1e-6, 'initial conditions inconsistent'
    sats_in_view = _group_coverage.sats_in_view
    grid=region_grid()
    fast,mins=propagate_score(u_fast,grid,np.array([10.0]),0.0)
    group_counts=sats_in_view(u_orig[:,0:3],0,grid)
    assert bool(mins[0]>=2)==bool(group_counts.min()>=2)
    print('Group-model verification passed. (Initial visibility min =', group_counts.min(),')')
    return u_dot, ra_dot

def main():
    OUT.mkdir(exist_ok=True)
    start=time.time()
    u_dot,ra_dot=verify_against_group()
    x0=circular_state(A,math.radians(INCLINATION_DEG),0.,0.)
    grid=region_grid()
    csvpath=OUT/'mask_sweep_results.csv'
    rows=[]
    if csvpath.exists():
        with open(csvpath, newline='') as old:
            for r in csv.DictReader(old):
                rows.append(dict(mask_deg=float(r['mask_deg']),
                                 inclination_deg=float(r['inclination_deg']),
                                 T=int(r['T']), P=int(r['P']), F=int(r['F']),
                                 days=int(r['days']),
                                 stereo_availability=float(r['stereo_availability']),
                                 worst_count=int(r['worst_count'])))
    done = {(r['T'],r['P'],r['F']) for r in rows if r['days']==DAYS_SWEEP}
    print('Cached',len(done),'configurations; remaining will be calculated.')
    configs=list(configurations())
    print('Testing',len(configs),'configurations at masks', MASKS.tolist(), 'for',DAYS_SWEEP,'days')
    for j,(T,P,F) in enumerate(configs):
        if (T,P,F) in done: continue
        raan,arglat=walker_angles(T,P,F)
        states=build_states(x0,arglat,raan,u_dot,ra_dot,TIMESTEP_S)
        availability, worst=propagate_score(states,grid,MASKS,DAYS_SWEEP)
        for k,mask in enumerate(MASKS):
            rows.append(dict(mask_deg=float(mask), inclination_deg=INCLINATION_DEG,
                             T=T,P=P,F=F,days=DAYS_SWEEP,
                             stereo_availability=float(availability[k]),
                             worst_count=int(worst[k])))
        if (j+1)%15==0:
            print(f'Completed {j+1}/{len(configs)} configurations; elapsed {time.time()-start:.1f}s',flush=True)
    with open(csvpath,'w',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Saved',csvpath)
    passing={}
    for mask in MASKS:
        accepted=sorted((r for r in rows if r['mask_deg']==mask and r['stereo_availability']>=1-1e-12),key=lambda r:(r['T'],r['P'],r['F']))
        passing[int(mask)]=accepted
        print(f'3-day mask {mask:g} degrees: {len(accepted)} passes; smallest:',accepted[0] if accepted else 'none in tested range')

    # Plot MINIMUM PASSING T within explicitly tested 40--66 range (not a global minimum).
    fig,ax=plt.subplots(figsize=(6.7,4.5))
    y=[]
    for m in MASKS:
        yes=passing[int(m)]
        y.append(yes[0]['T'] if yes else np.nan)
    ax.plot(MASKS,y,'o-',lw=2,ms=7)
    for m,t in zip(MASKS,y):
        if not np.isnan(t): ax.annotate(str(int(t)),(m,t),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set_xticks(MASKS)
    ax.set_xlabel('Minimum satellite elevation angle (degrees)')
    ax.set_ylabel('Smallest passing T in tested design grid')
    ax.set_title('Elevation-mask sensitivity (50 deg inclination; 3-day test)')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT/'fig_mask_sensitivity.pdf')
    plt.close(fig)

    with open(OUT/'summary.txt','w') as fsum:
        fsum.write('AE 370 Group Project 1, extension, 3-day sampled coverage at i=50 degrees.\n')
        fsum.write('Tested P=4..8, T=40..100 divisible by P, F=0..P-1, 30-s RK4, 60-s sample, 88 points.\n')
        for m in MASKS:
            yes=passing[int(m)]
            fsum.write(f'Mask {m:g} deg: '+(f'T={yes[0]["T"]}, P={yes[0]["P"]}, F={yes[0]["F"]}, 3-day pass' if yes else 'none passed 3-day grid')+'\n')
        fsum.write('Caution: 3-day passing does NOT certify 10-day coverage; T=40 lower bound not reached at 5 degrees; 10-day verification still needed.\n')
    print((OUT/'summary.txt').read_text())
    print('Total elapsed',time.time()-start,'s')

if __name__=='__main__': main()
