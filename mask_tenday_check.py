"""Follow-up certification for Section 4D: smallest 3-day pass may fail at day 10.

Run mask_sensitivity.py first; then run this script in same directory.
For each mask, examine every 3-day passing design in ascending T until a design
holds at all sampled ground points/times for 10 days. Thus this certifies the
smallest 10-day passing T *within the evaluated inclination and design grid*.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mask_sensitivity import (OUT, INCLINATION_DEG, MASKS, DAYS_VERIFY,
  TIMESTEP_S, A, build_states, propagate_score, circular_state,
  verify_against_group, walker_angles, region_grid)
import math

def main():
    OUT.mkdir(exist_ok=True)
    source=OUT/'mask_sweep_results.csv'
    if not source.exists(): raise FileNotFoundError('Run mask_sensitivity.py first')
    rows=list(csv.DictReader(open(source,newline='')))
    u_dot, ra_dot=verify_against_group()
    x0=circular_state(A,math.radians(INCLINATION_DEG),0.,0.)
    grid=region_grid()
    target=OUT/'mask_tenday_results.csv'
    checked=[]
    if target.exists(): checked=list(csv.DictReader(open(target,newline='')))
    done={(float(r['mask_deg']), int(r['T']), int(r['P']), int(r['F'])) for r in checked}
    findings=[]
    for mask in MASKS:
        mask=float(mask)
        candidates=sorted((r for r in rows if float(r['mask_deg'])==mask
                         and float(r['stereo_availability'])>=1.-1e-12),
                         key=lambda r:(int(r['T']),int(r['P']),int(r['F'])))
        print(f'Mask {mask:g}: {len(candidates)} 3-day passing designs')
        winner=None
        for r in candidates:
            T,P,F=(int(r[k]) for k in ('T','P','F'))
            key=(mask,T,P,F)
            prev=next((v for v in checked if (float(v['mask_deg']),int(v['T']),int(v['P']),int(v['F']))==key),None)
            if prev is None:
                ra, u=walker_angles(T,P,F)
                states=build_states(x0,u,ra,u_dot,ra_dot,TIMESTEP_S)
                avail,worst=propagate_score(states,grid,np.array([mask],dtype=np.float64),DAYS_VERIFY)
                prev=dict(mask_deg=mask,T=T,P=P,F=F,days=DAYS_VERIFY,
                          stereo_availability=float(avail[0]),worst_count=int(worst[0]))
                checked.append(prev)
                with open(target,'w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['mask_deg','T','P','F','days','stereo_availability','worst_count']);w.writeheader();w.writerows(checked)
                print(f'   {T}/{P}/{F}: ten-day availability {avail[0]:.9f}; min seen {worst[0]}',flush=True)
            if float(prev['stereo_availability'])>=1-1e-12:
                winner=prev
                break
        findings.append(winner)
        print('10-day winner:',winner)
    with open(OUT/'ten_day_summary.txt','w') as f:
        for m,w in zip(MASKS,findings):
            f.write(f'{m:g} deg: ' + (f'{w["T"]}/{w["P"]}/{w["F"]}, 10-day sampled coverage passed' if w else 'NO PASS among 3-day candidates')+'\n')
    fig, ax=plt.subplots(figsize=(6.8,4.5))
    ys=[int(w['T']) if w else float('nan') for w in findings]
    ax.plot(MASKS,ys,'o-',lw=2,ms=7)
    for m,t in zip(MASKS,ys):
        if not np.isnan(t):ax.annotate(str(int(t)),(m,t),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set_xticks(MASKS)
    ax.set_xlabel('Minimum satellite elevation angle (degrees)')
    ax.set_ylabel('Smallest T passing 10-day sampled-coverage test')
    ax.set_title('Elevation-mask sensitivity at 50-degree inclination')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT/'fig_mask_sensitivity_tenday.pdf')
    plt.close(fig)
    print((OUT/'ten_day_summary.txt').read_text())

if __name__=='__main__': main()
