#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redesign Figure 5 for the validated final 10-repetition S7 campaign.

This is NOT a two-colour heatmap. It produces:
  (a) a minimal per-run execution timeline: thin light-grey edge baseline,
      dark copper line segments for actual cloud selections, and diamonds for fallback;
  (b) measured per-phase cloud-selection fractions across the ten repetitions.

Reads only the real first-stage analysis master CSV. No synthesis, interpolation,
resampling or inference-event filtering. Strictly checks 10 x 289 S7 decisions.

Usage in WSL:
  python3 ClusterComputing_Fig05_Research_Redesign.py
  python3 ClusterComputing_Fig05_Research_Redesign.py --replace-existing
  python3 ClusterComputing_Fig05_Research_Redesign.py --campaign /path/to/03_final_matched_campaign

Requirements: numpy, pandas, matplotlib
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

DEFAULT = Path(
    '/mnt/d/ESDE/Publications/ClusterComputing/Newimpimantation/evidence/'
    'S5_S6_S7_LOAD_MATCHED_V1_20261008/03_final_matched_campaign'
)
PHASES = [
    ('L0_BASELINE','L0',1,40,0),
    ('L1_STABLE','L1',41,80,4),
    ('L2_ONSET','L2',81,120,6),
    ('L3_SATURATION','L3',121,170,8),
    ('L4_RECOVERY','L4',171,210,4),
    ('L5_REOVERLOAD','L5',211,250,8),
    ('L6_FINAL_RECOVERY','L6',251,289,0),
]
COLOR_EDGE = '#bec6cc'       # quiet grey, not a big filled background
COLOR_CLOUD = '#a85a2a'      # muted copper, strong in print
COLOR_INK = '#242b33'
COLOR_PHASE = '#b5bfc7'


def is_true(value):
    return str(value).strip().lower() in {'1','true','yes','y'}


def read_measured_site_matrix(csv_path):
    df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(s).strip() for s in df.columns]
    required = ['strategy','repetition','decision_number','selected_site',
                'load_phase','load_target_workers','fallback_used']
    missing = [s for s in required if s not in df.columns]
    if missing:
        raise ValueError('Missing columns: '+str(missing))
    s7 = df[df['strategy'].astype(str).str.upper().str.strip() == 'S7'].copy()
    if len(s7) != 2890:
        raise ValueError(f'Expected 2,890 S7 decisions, found {len(s7)}; refusing to plot.')
    s7['repetition'] = pd.to_numeric(s7['repetition'],errors='raise').astype(int)
    s7['decision_number'] = pd.to_numeric(s7['decision_number'],errors='raise').astype(int)
    s7['load_target_workers'] = pd.to_numeric(s7['load_target_workers'],errors='raise').astype(int)
    s7['selected_site'] = s7['selected_site'].astype(str).str.strip().str.lower()
    if not set(s7['selected_site']).issubset({'edge','cloud'}):
        raise ValueError('Unexpected site labels in S7 selected_site: '+str(set(s7['selected_site'])))
    if sorted(s7['repetition'].unique()) != list(range(1,11)):
        raise ValueError('Expected repetitions 1 through 10, exactly.')
    if s7.duplicated(['repetition','decision_number']).any():
        raise ValueError('Duplicate (repetition, decision_number) rows.')
    expected = [i for i in range(1,290)]
    cloud = np.zeros((10,289),dtype=bool)
    failures = []
    for r in range(1,11):
        part = s7.loc[s7['repetition'].eq(r)].sort_values('decision_number')
        if part['decision_number'].tolist() != expected:
            raise ValueError(f'Repetition {r} has missing/unsorted decision numbers.')
        for phase, short, start, end, workers in PHASES:
            segment=part.loc[part['decision_number'].between(start,end)]
            if len(segment) != end-start+1 or not segment['load_phase'].eq(phase).all():
                raise ValueError(f'Repetition {r}: schedule mismatch for {phase}')
            if not segment['load_target_workers'].eq(workers).all():
                raise ValueError(f'Repetition {r}: load-worker count mismatch in {phase}')
        cloud[r-1] = part['selected_site'].eq('cloud').to_numpy()
        flags=part['fallback_used'].map(is_true)
        failures += [(int(x),r) for x in part.loc[flags,'decision_number']]
    return cloud, failures


def consecutive_spans(mask):
    """Return contiguous cloud intervals, with 1-based inclusive decision endpoints."""
    diff = np.diff(np.r_[False, mask, False].astype(int))
    starts=np.where(diff==1)[0]+1
    ends=np.where(diff==-1)[0]
    return list(zip(starts,ends))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=DEFAULT)
    parser.add_argument('--replace-existing',action='store_true',
                        help='Overwrite the legacy Fig05 filename. Run this AFTER the updater.')
    args=parser.parse_args()
    campaign=args.campaign.expanduser().resolve()
    out=campaign/'04_publication_figures'
    src=out/'MASTER_S5_S6_S7_30_RUN_MATCHED.csv'
    if not src.is_file():
        sys.exit('Master CSV not found: '+str(src)+'\nRun the validated four-figure script first.')
    cloud, fallback=read_measured_site_matrix(src)
    phase_stats=[]
    for key,label,start,end,workers in PHASES:
        window=cloud[:,start-1:end]
        phase_stats.append({'phase':label,'target_cpu_workers':workers,
                            'start_decision':start,'end_decision':end,
                            'cloud_decisions':int(window.sum()),
                            'total_decisions':int(window.size),
                            'cloud_selection_pct':round(100*window.mean(),5)})

    plt.rcParams.update({
        'font.family':'DejaVu Sans',
        'font.size':9,
        'axes.labelsize':10,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
        'svg.fonttype':'none',  # retain editable text in SVG
        'axes.linewidth':.7,
        'savefig.facecolor':'white',
    })
    fig=plt.figure(figsize=(10.4,6.0),facecolor='white')
    gs=fig.add_gridspec(2,1,height_ratios=[2.55,1.1],hspace=.40,
                        left=.094,right=.97,bottom=.105,top=.81)
    ax=fig.add_subplot(gs[0])
    ax2=fig.add_subplot(gs[1])

    # Panel (a): cloud interruptions in ten otherwise edge-executed timelines.
    # No imshow, no blue-filled cells, and no artificial smoothing.
    for rep in range(1,11):
        ax.hlines(rep, .5,289.5, color=COLOR_EDGE, linewidth=5.0,zorder=1)
        for start, end in consecutive_spans(cloud[rep-1]):
            ax.hlines(rep,start-.5,end+.5,color=COLOR_CLOUD,linewidth=5.0,zorder=2)

    for b in [40,80,120,170,210,250]:
        ax.axvline(b+.5,color=COLOR_PHASE,linestyle=':',linewidth=.75,zorder=0)
    if fallback:
        ax.scatter([d for d,r in fallback],[r for d,r in fallback],
                   marker='D',s=22,c=COLOR_INK,edgecolors='white',linewidths=.5,
                   zorder=5)
    ax.set_xlim(.5,289.5);ax.set_ylim(.4,10.6)
    ax.set_yticks(np.arange(1,11));ax.set_ylabel('S7 repetition')
    ax.set_xticks([1,40,80,120,170,210,250,289]);ax.set_xlabel('Logical inference decision')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3)
    ax.text(-.073,1.03,'(a)',transform=ax.transAxes,fontweight='bold',fontsize=10)
    phase_axis=ax.secondary_xaxis('top')
    phase_axis.set_xticks([(start+end)/2 for _,_,start,end,_ in PHASES],
                          [f'{label}\n{workers} workers' for _,label,_,_,workers in PHASES])
    phase_axis.tick_params(length=0,pad=6,labelsize=8)
    phase_axis.spines[['top']].set_visible(False)

    legend=[Line2D([0],[0],color=COLOR_EDGE,lw=5,label='Edge selected'),
            Line2D([0],[0],color=COLOR_CLOUD,lw=5,label='Cloud selected')]
    if fallback:
        legend.append(Line2D([0],[0],linestyle='None',marker='D',markersize=5,
                             markerfacecolor=COLOR_INK,markeredgecolor='white',
                             label='Fallback'))
    fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.97,.99),
               ncol=len(legend),frameon=False,handlelength=2.4,columnspacing=1.6,
               fontsize=9)

    # Panel (b): independent aggregate statistic showing adaptation by phase.
    xs=np.arange(7)
    fractions=np.array([row['cloud_selection_pct'] for row in phase_stats])
    ax2.bar(xs,fractions,color=COLOR_CLOUD,alpha=.88,width=.53,zorder=3)
    ax2.set_xticks(xs,[f"{row['phase']} ({row['target_cpu_workers']}w)" for row in phase_stats])
    ax2.set_ylabel('Cloud selection (%)')
    ax2.set_ylim(0, max(6, min(100, np.ceil(float(fractions.max())/10)*10+8)))
    ax2.grid(axis='y',linewidth=.65,color='#e6e9eb',zorder=0)
    ax2.spines[['top','right']].set_visible(False)
    ax2.tick_params(direction='out',length=3)
    ax2.text(0.0,1.13,'(b)',transform=ax2.transAxes,fontweight='bold',fontsize=10)
    for x,v in zip(xs,fractions):
        if v>0:
            ax2.annotate(f'{v:.1f}%',(x,v),xytext=(0,4),textcoords='offset points',
                         ha='center',va='bottom',fontsize=8,color=COLOR_INK)

    stem=('Fig05_Final_Matched_S7_Placement' if args.replace_existing
          else 'Fig05_Final_Matched_S7_Placement_Research')
    # Existing paths are not modified in the default mode.
    out.mkdir(parents=True,exist_ok=True)
    for extension,kwargs in [('png',{'dpi':600}),('svg',{})]:
        dest=out/f'{stem}.{extension}'
        fig.savefig(dest,format=extension,bbox_inches='tight',pad_inches=.14,**kwargs)
        print('Created:',dest)
    plt.close(fig)
    pd.DataFrame(phase_stats).to_csv(out/'Fig05_phase_cloud_selection_statistics.csv',index=False)
    src_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    info={'source_master_csv':str(src),'source_sha256':src_hash,
          'independent_repetitions':10,'decisions_per_repetition':289,
          'cloud_selections_total':int(cloud.sum()),
          'edge_selections_total':int((~cloud).sum()),
          'fallback_events':len(fallback),'statistics':phase_stats}
    (out/'Fig05_research_plot_manifest.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    print('\nScientific audit:')
    print('S7 decisions:',cloud.size,'| Cloud selected:',int(cloud.sum()),
          '| Edge selected:',int((~cloud).sum()),'| Fallback events:',len(fallback))
    print('Recorded phase breakdown:')
    print(pd.DataFrame(phase_stats)[['phase','target_cpu_workers','cloud_decisions','total_decisions','cloud_selection_pct']].to_string(index=False))
    print('\nNo inference observations were generated or modified.')
    print('\nLaTeX:')
    print(r'\includegraphics[width=\linewidth]{figures/'+stem+'.png}')


if __name__=='__main__':
    main()
