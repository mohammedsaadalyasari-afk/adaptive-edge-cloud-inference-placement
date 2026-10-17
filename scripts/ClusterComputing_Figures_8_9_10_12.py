#!/usr/bin/env python3
"""Cluster Computing | FINAL load-matched campaign | Figs 8, 9, 10, 12.

READ-ONLY analysis of the frozen, validated S5/S6/S7 campaign. Does not launch
Node-RED, alter experimental data, or synthesize missing measurements.

Requirements: numpy, pandas, matplotlib (Python 3.9+).
Usage:
 python3 ClusterComputing_Figures_8_9_10_12.py
 python3 ClusterComputing_Figures_8_9_10_12.py --campaign /path/to/03_final_matched_campaign

Only draws figures if all 10 x 3 original run logs pass consistency checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as e:
    sys.exit('Missing dependency: %s. Install with: python3 -m pip install numpy pandas matplotlib' % e)

PROJECT = Path('/mnt/d/ESDE/Publications/ClusterComputing/Newimpimantation')
CAMPAIGN_DEFAULT = PROJECT / 'evidence/S5_S6_S7_LOAD_MATCHED_V1_20261008/03_final_matched_campaign'
STRATEGIES = {'S5':'S5_edge_only', 'S6':'S6_cloud_only', 'S7':'S7_adaptive'}
LABELS = {'S5':'S5  Edge-only', 'S6':'S6  Cloud-only', 'S7':'S7  Adaptive'}
STYLES = {'S5':('C0','o'), 'S6':('C1','s'), 'S7':('C2','^')}
PHASES = ['L0_BASELINE','L1_STABLE','L2_ONSET','L3_SATURATION',
          'L4_RECOVERY','L5_REOVERLOAD','L6_FINAL_RECOVERY']
EXPECTED_PHASE_COUNTS = [40,40,40,50,40,40,39]
REQUIRED = ['decision_number','load_phase','load_target_workers','total_latency_ms',
            'service_inference_ms','final_success']
S7_REQUIRED = ['edge_cost_ms','cloud_cost_ms','edge_CPU_EWMA_percent',
               'selected_site','final_execution_site','switch_reason','fallback_used']
SOURCE_CONTRACT_SHA = '1343dcac788544f83ba943c850aa285b0094316623941f44d6911a32b492a80f'


def fail(message):
    raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def true_value(x):
    if pd.isna(x): return False
    return str(x).strip().lower() in ('1','true','yes','y','ok','success','passed')


def clean_site(x):
    s = str(x).strip().lower()
    return s if s in ('edge','cloud') else ''


def find_experiment_csv(run_dir, strategy):
    exact = [run_dir / f'{strategy.lower()}_experiment_log.csv',
             run_dir / 'experiment_log.csv']
    for p in exact:
        if p.is_file(): return p
    matches = sorted(p for p in run_dir.glob('*experiment*log*.csv')
                     if 'probe' not in p.name.lower())
    if len(matches)!=1:
        fail(f'{run_dir}: expected one experiment log, found {len(matches)}. '
             'Do not use publish_log.csv or S7_probe_log.csv as decision results.')
    return matches[0]


def read_one(csv_path,strategy,repetition):
    df = pd.read_csv(csv_path,encoding='utf-8-sig',low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if strategy=='S7': missing += [c for c in S7_REQUIRED if c not in df.columns]
    if missing: fail(f'{csv_path}: required columns missing: {missing}; have {list(df.columns)}')
    if len(df)!=289: fail(f'{csv_path}: expected 289 inference decisions, got {len(df)}.')
    df['decision_number'] = pd.to_numeric(df['decision_number'],errors='coerce')
    if df['decision_number'].isna().any() or sorted(df['decision_number'].tolist())!=list(range(1,290)):
        fail(f'{csv_path}: decision_number must be unique 1..289.')
    df = df.sort_values('decision_number').reset_index(drop=True)
    if df['load_phase'].astype(str).tolist() != [p for p,n in zip(PHASES,EXPECTED_PHASE_COUNTS) for _ in range(n)]:
        fail(f'{csv_path}: incorrect frozen L0..L6 load-phase schedule.')
    df['load_target_workers'] = pd.to_numeric(df['load_target_workers'],errors='coerce')
    if df['load_target_workers'].isna().any(): fail(f'{csv_path}: missing CPU worker-load schedule.')
    if dict(df.groupby('load_target_workers',dropna=False).size().astype(int)) != {0:79,4:80,6:40,8:90}:
        fail(f'{csv_path}: unexpected load-worker counts.')
    for col in ('total_latency_ms','service_inference_ms'):
        df[col] = pd.to_numeric(df[col],errors='coerce')
    if not np.isfinite(df['total_latency_ms'].to_numpy()).all() or (df['total_latency_ms']<0).any():
        fail(f'{csv_path}: missing/invalid total latency; cannot use unmeasured request times.')
    df['success'] = df['final_success'].map(true_value)
    if df.loc[df['success'],'service_inference_ms'].isna().any():
        fail(f'{csv_path}: successful decisions without service_inference_ms.')
    if (df.loc[df['success'],'service_inference_ms']<0).any():
        fail(f'{csv_path}: negative service duration.')
    # Do not create negative residuals through clipping or imputation.
    bad = df['success'] & (df['total_latency_ms'] + 0.01 < df['service_inference_ms'])
    if bad.any(): fail(f'{csv_path}: service timing exceeds total latency for {sum(bad)} decisions.')
    df['non_service_ms'] = df['total_latency_ms'] - df['service_inference_ms']
    df.loc[~df['success'],'non_service_ms'] = np.nan
    df.loc[df['non_service_ms'].between(-0.01,0),'non_service_ms'] = 0
    if 'selected_site' in df:
        sites = set(df['selected_site'].map(clean_site))
        expected = {'edge'} if strategy=='S5' else {'cloud'} if strategy=='S6' else {'edge','cloud'}
        if not sites.issubset(expected) or '' in sites:
            fail(f'{csv_path}: invalid selected site(s) for {strategy}: {sites}')
    if 'model_sha256' in df:
        values = set(df.loc[df['success'],'model_sha256'].dropna().astype(str).str.strip())-set([''])
        if len(values)!=1: fail(f'{csv_path}: model identity missing/inconsistent: {values}')
    df['strategy'] = strategy
    df['repetition'] = repetition
    df['experiment_file'] = str(csv_path)
    if strategy == 'S7':
        for c in ('edge_cost_ms','cloud_cost_ms','edge_CPU_EWMA_percent'):
            df[c]=pd.to_numeric(df[c],errors='coerce')
        if not np.isfinite(df['edge_cost_ms'].to_numpy()).any() or not np.isfinite(df['cloud_cost_ms'].to_numpy()).any():
            fail(f'{csv_path}: S7 cost traces absent; Fig. 9 must use genuine controller traces.')
        df['selected_site']=df['selected_site'].map(clean_site)
        df['final_execution_site']=df['final_execution_site'].map(clean_site)
        df['fallback_used']=df['fallback_used'].map(true_value)
    return df


def load_campaign(root):
    frames=[]
    audit=[]
    for s,dirname in STRATEGIES.items():
        for rep in range(1,11):
            run_dir = root/dirname/f'rep{rep:02d}'
            if not run_dir.is_dir(): fail(f'Missing validated run directory: {run_dir}')
            p=find_experiment_csv(run_dir,s)
            frame=read_one(p,s,rep)
            # The original complete marker should exist in frozen final runs.
            marker = run_dir/'RUN_COMPLETE.txt'
            if not marker.exists():
                fail(f'{run_dir}: RUN_COMPLETE.txt is missing. Analysis requires final validated runs.')
            # Never infer failure of a logical request from a failed first attempt alone.
            probe = sorted(run_dir.glob('*probe*log*.csv')) if s=='S7' else []
            if s=='S7':
                if len(probe)!=1: fail(f'{run_dir}: expected one separate S7 probe log.')
                pcount=len(pd.read_csv(probe[0],encoding='utf-8-sig'))
                if pcount!=28: fail(f'{run_dir}: expected 28 non-performance probe rows, got {pcount}.')
            frames.append(frame)
            audit.append({'strategy':s,'repetition':rep,'source_file':str(p),
                          'sha256':sha256_file(p),'count':len(frame),
                          'success_count':int(frame['success'].sum()),
                          'failed_count':int((~frame['success']).sum()),
                          'probe_rows_separate':28 if s=='S7' else 0})
    df=pd.concat(frames,ignore_index=True)
    # The same source events and workload schedule must be aligned by decision index.
    for c in ('load_phase','load_target_workers','replay_seq','source_line','source_epoch_ms','raw_slots_bits'):
        if c not in df.columns: continue
        p=df.pivot_table(index='decision_number',columns=['strategy','repetition'],values=c,aggfunc='first')
        # Missing identifiers may be intentionally unavailable in a subset of strategies.
        values=p.to_numpy(dtype=object)
        for row in values:
            nonnull=[str(x) for x in row if pd.notna(x) and str(x).strip()]
            if nonnull and len(set(nonnull))>1:
                fail(f'Frozen trace mismatch in field {c}: strategies/repetitions are not paired.')
    if 'model_sha256' in df.columns:
        hashes=set(df.loc[df['success'],'model_sha256'].dropna().astype(str).str.strip())-set([''])
        if len(hashes)!=1: fail(f'Model identity mismatch across strategies/runs: {hashes}')
    # Not necessary to access the frozen schedule to plot, but verify frozen S4 trace when present.
    trace=root/'frozen_inputs'/'common_S4_hw_secure_trace.csv'
    if trace.exists() and sha256_file(trace)!=SOURCE_CONTRACT_SHA:
        fail(f'Frozen replay source mismatch: {trace}')
    return df,pd.DataFrame(audit)


def format_axes(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.20,linewidth=.7)
    ax.set_axisbelow(True)


def save(fig,out,stem):
    fig.savefig(out/f'{stem}.png',dpi=350,bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'{stem}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('Created:', out/f'{stem}.png', '+ SVG')


def plot_fig8(df,out):
    per=[]; pooled=[]
    for (s,rep,phase),part in df.groupby(['strategy','repetition','load_phase'],sort=False):
        per.append({'strategy':s,'rep':rep,'phase':phase,
                    'median':float(part.total_latency_ms.median()),
                    'p95':float(np.percentile(part.total_latency_ms,95)),
                    'n':len(part)})
    for (s,phase),part in df.groupby(['strategy','load_phase'],sort=False):
        pooled.append({'strategy':s,'phase':phase,'n':len(part),'workers':int(part.load_target_workers.iloc[0]),
                       'median_ms':float(part.total_latency_ms.median()),
                       'p95_ms':float(np.percentile(part.total_latency_ms,95))})
    pd.DataFrame(pooled).to_csv(out/'Fig08_load_phase_metrics.csv',index=False)
    repdf=pd.DataFrame(per)
    fig,axes=plt.subplots(2,1,figsize=(11.8,7.2),sharex=True,layout='constrained')
    workers=[int(df.loc[df.load_phase.eq(ph),'load_target_workers'].iloc[0]) for ph in PHASES]
    for idx,(key,ytitle) in enumerate((('median','Median total latency (ms)'),('p95','P95 total latency (ms)'))):
        ax=axes[idx]
        for s in STRATEGIES:
            subset=repdf[repdf.strategy.eq(s)]
            stats=subset.groupby('phase')[key].agg(lambda x: np.median(x)).reindex(PHASES)
            lo=subset.groupby('phase')[key].quantile(.25).reindex(PHASES)
            hi=subset.groupby('phase')[key].quantile(.75).reindex(PHASES)
            color,marker=STYLES[s]
            x=np.arange(len(PHASES))
            ax.plot(x,stats.to_numpy(),color=color,marker=marker,lw=2,label=LABELS[s])
            ax.fill_between(x,lo.to_numpy(),hi.to_numpy(),color=color,alpha=.12)
        format_axes(ax);ax.set_ylabel(ytitle)
        ax.text(0,.98,'(a)' if idx==0 else '(b)',transform=ax.transAxes,va='top',weight='bold')
    axes[0].legend(loc='upper left',bbox_to_anchor=(.05,1.0),ncol=3,fontsize=9)
    axes[-1].set_xticks(np.arange(7),[f'L{i}\n({w} workers)' for i,w in enumerate(workers)])
    axes[-1].set_xlabel('Controlled load phase (chronological order)')
    fig.suptitle('Fig. 8  CPU-load-aware total inference latency\nLine: median of run-level statistics; shade: interquartile range across 10 runs',fontsize=12)
    save(fig,out,'Fig08_CPU_Load_Aware_Latency')


def choose_s7_repetition(df):
    choices=[]
    for rep,g in df[df.strategy.eq('S7')].groupby('repetition'):
        reasons=g['switch_reason'].fillna('').astype(str).str.lower()
        normal=int(reasons.str.contains('cost_hysteresis').sum())
        reverse=int(reasons.str.contains('cost_hysteresis_cloud_to_edge').sum())
        fallback=int(g['fallback_used'].sum())
        choices.append((reverse,normal,fallback,-rep,rep))
    choices.sort(reverse=True)
    return int(choices[0][-1])


def plot_fig9(df,out,rep=None):
    if rep is None: rep=choose_s7_repetition(df)
    g=df[(df.strategy=='S7') & (df.repetition==rep)].sort_values('decision_number')
    if len(g)!=289: fail('Cannot create Fig. 9 without complete S7 state trace.')
    x=g.decision_number.to_numpy()
    ec=g.edge_cost_ms.to_numpy(dtype=float)
    cc=g.cloud_cost_ms.to_numpy(dtype=float)
    workers=g.load_target_workers.to_numpy(dtype=float)
    cpu=g.edge_CPU_EWMA_percent.to_numpy(dtype=float)
    sites=g.selected_site.map({'edge':0,'cloud':1}).to_numpy(dtype=float)
    final=g.final_execution_site.map({'edge':0,'cloud':1}).to_numpy(dtype=float)
    reasons=g.switch_reason.fillna('').astype(str).str.lower()
    normal=reasons.str.contains('cost_hysteresis').to_numpy()
    fallback=g.fallback_used.to_numpy(dtype=bool)
    fig,(a,b,c)=plt.subplots(3,1,figsize=(12,9),sharex=True,
                             gridspec_kw={'height_ratios':[2,1.3,1.25]},layout='constrained')
    a.plot(x,ec,lw=1.7,label='Estimated edge cost $J_e$',color='C0')
    a.plot(x,cc,lw=1.7,label='Estimated cloud cost $J_c$',color='C1')
    if np.any(normal):
        for k in x[normal]: a.axvline(k,c='0.35',lw=.8,ls='--',alpha=.42)
    for k in x[fallback]: a.axvline(k,c='C3',lw=1.4,ls=':',alpha=.9)
    a.set_ylabel('Controller estimated cost (ms)')
    nonzero=np.r_[ec[np.isfinite(ec)&(ec>0)],cc[np.isfinite(cc)&(cc>0)]]
    if len(nonzero)>2 and nonzero.max()/max(np.median(nonzero),1e-8)>20: a.set_yscale('symlog',linthresh=80)
    format_axes(a);a.legend(loc='upper right',ncol=2,fontsize=9)
    a.text(0,.97,'(a) Decision-time estimated costs',transform=a.transAxes,va='top',weight='bold')
    b.plot(x,cpu,color='C4',lw=1.3,label='CPU EWMA (%)')
    b.set_ylabel('Edge CPU EWMA (%)')
    b.set_ylim(bottom=0)
    d=b.twinx();d.step(x,workers,where='post',color='0.35',lw=1,ls='--',label='Load workers')
    d.set_ylabel('Active load workers')
    b.text(0,.96,'(b) Observed edge utilization and injected load',transform=b.transAxes,va='top',weight='bold')
    format_axes(b);d.spines['top'].set_visible(False)
    b.legend(loc='upper right',fontsize=8)
    d.legend(loc='center right',fontsize=8)
    c.step(x,sites,where='post',lw=1.7,color='C2',label='Selected site')
    change=np.flatnonzero(~np.isclose(sites,final,equal_nan=True))
    if len(change): c.scatter(x[change],final[change],marker='x',s=90,color='C3',zorder=5,label='Final site differs')
    if fallback.any(): c.scatter(x[fallback],np.full(sum(fallback),.5),marker='X',s=65,color='C3',zorder=7,label='Fallback event')
    if normal.any(): c.scatter(x[normal],sites[normal],s=44,marker='D',color='0.15',zorder=6,label='Cost-triggered switch')
    c.set_yticks([0,1],['Edge','Cloud']);c.set_ylim(-.25,1.30)
    c.set_ylabel('Execution site'); c.set_xlabel('Logical inference decision (within repetition)')
    c.text(0,.96,'(c) Selection, normal switching, and fallback',transform=c.transAxes,va='top',weight='bold')
    c.legend(loc='upper right',ncol=2,fontsize=8);format_axes(c)
    fig.suptitle(f'Fig. 9  Recorded adaptive controller state, S7 repetition {rep:02d}',fontsize=12)
    save(fig,out,'Fig09_Adaptive_Decision_Cost_Evolution')
    g[['decision_number','edge_cost_ms','cloud_cost_ms','edge_CPU_EWMA_percent',
       'load_target_workers','selected_site','final_execution_site','switch_reason','fallback_used']].to_csv(
       out/'Fig09_recorded_controller_trace.csv',index=False)
    return rep


def plot_fig10(df,out):
    # Mean components add; P95 components do NOT add and are shown as grouped bars.
    rows=[]
    for s in STRATEGIES:
        all_=df[df.strategy.eq(s)]
        ok=all_[all_.success]
        if len(ok)==0: fail(f'No successes for {s}; cannot decompose latency.')
        rows.append({'strategy':s,'n_total':len(all_),'n_success':len(ok),'n_failed':len(all_)-len(ok),
                     'mean_total_ms':ok.total_latency_ms.mean(),
                     'mean_service_ms':ok.service_inference_ms.mean(),
                     'mean_non_service_ms':ok.non_service_ms.mean(),
                     'p95_total_ms':ok.total_latency_ms.quantile(.95),
                     'p95_service_ms':ok.service_inference_ms.quantile(.95),
                     'p95_non_service_ms':ok.non_service_ms.quantile(.95)})
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'Fig10_latency_decomposition_metrics.csv',index=False)
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    xx=np.arange(3)
    a.bar(xx,summary.mean_service_ms,label='Mean service-side inference',color='C0')
    a.bar(xx,summary.mean_non_service_ms,bottom=summary.mean_service_ms,
          label='Mean non-service overhead',color='C1')
    a.set_ylabel('Mean latency (ms)')
    a.set_xticks(xx,['S5','S6','S7']);format_axes(a)
    a.legend(loc='upper left',fontsize=8)
    a.set_title('(a) Additive decomposition of mean latency',fontsize=10)
    width=.25
    for j,(col,label,color) in enumerate((('p95_total_ms','Total','C2'),
                                         ('p95_service_ms','Service','C0'),
                                         ('p95_non_service_ms','Non-service','C1'))):
        b.bar(xx+(j-1)*width,summary[col],width,color=color,label=f'P95 {label}')
    b.set_xticks(xx,['S5','S6','S7']);b.set_ylabel('Independent P95 (ms)');format_axes(b)
    b.legend(loc='upper left',fontsize=8)
    b.set_title('(b) Separate component P95s (not additive)',fontsize=10)
    b.text(.5,-.18,'Successful logical decisions only; failure counts in accompanying CSV',
           transform=b.transAxes,ha='center',fontsize=8)
    fig.suptitle('Fig. 10  Service execution versus non-service path overhead',fontsize=12)
    save(fig,out,'Fig10_Inference_Latency_Decomposition')


def latency_stats(vals):
    return np.array([np.mean(vals),np.median(vals),np.percentile(vals,95),np.percentile(vals,99)])


def plot_fig12(df,out,bootstrap=5000,block=10,seed=519):
    if bootstrap<100 or block<1: fail('Use >=100 bootstrap resamples and block length >=1.')
    pairs=[]
    for rep in range(1,11):
        sub6=df[(df.strategy=='S6') & (df.repetition==rep)].sort_values('decision_number')
        sub7=df[(df.strategy=='S7') & (df.repetition==rep)].sort_values('decision_number')
        if len(sub6)!=289 or len(sub7)!=289: fail('Incomplete paired S6/S7 dataset.')
        pairs.append((sub6.total_latency_ms.to_numpy(),sub7.total_latency_ms.to_numpy()))
    a=np.concatenate([p[0] for p in pairs]); b=np.concatenate([p[1] for p in pairs])
    orig6=latency_stats(a); orig7=latency_stats(b)
    point=100*(orig6-orig7)/orig6
    rng=np.random.default_rng(seed)
    reps=len(pairs); n=len(pairs[0][0]);k=int(np.ceil(n/block))
    boot=np.empty((bootstrap,4))
    for z in range(bootstrap):
        run_ids=rng.integers(0,reps,size=reps)
        x6=[];x7=[]
        for ri in run_ids:
            starts=rng.integers(0,n,size=k)
            ind=(starts[:,None]+np.arange(block)[None,:])%n
            ind=ind.ravel()[:n]
            x6.append(pairs[ri][0][ind]);x7.append(pairs[ri][1][ind])
        m6=latency_stats(np.concatenate(x6));m7=latency_stats(np.concatenate(x7))
        boot[z]=100*(m6-m7)/m6
    lo=np.percentile(boot,2.5,axis=0);hi=np.percentile(boot,97.5,axis=0)
    labels=['Mean','Median','P95','P99']
    summary=pd.DataFrame({'metric':labels,'S6_ms':orig6,'S7_ms':orig7,
                          'relative_reduction_pct':point,'CI95_lower_pct':lo,
                          'CI95_upper_pct':hi,'resamples':bootstrap,
                          'block_length_decisions':block,'paired_run_count':reps})
    summary.to_csv(out/'Fig12_paired_hierarchical_block_bootstrap_CI.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    y=np.arange(4)[::-1]
    low=point-lo;up=hi-point
    ax.errorbar(point,y,xerr=np.vstack([low,up]),fmt='o',color='C0',
                capsize=4,markersize=7,elinewidth=1.7)
    ax.axvline(0,color='0.3',lw=1,ls='--')
    ax.set_yticks(y,labels);ax.set_xlabel('Relative S7 latency reduction versus S6 (%)')
    ax.set_title('Fig. 12  Paired hierarchical moving-block bootstrap\n'
                 f'{reps} matched repetitions; {bootstrap:,} resamples; {block}-decision blocks',fontsize=12)
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    for j,row in summary.iterrows():
        ax.annotate(f"{row.relative_reduction_pct:.1f}%  [{row.CI95_lower_pct:.1f}, {row.CI95_upper_pct:.1f}]",
                    xy=(row.CI95_upper_pct,y[j]),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
    xmax=max(hi)*1.10 + max(5,abs(max(hi))*.05)
    xmin=min(0,min(lo)) - max(5,abs(min(lo))*.08)
    ax.set_xlim(xmin,max(xmax,100))
    save(fig,out,'Fig12_Bootstrap_Confidence_Intervals')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign',type=Path,default=CAMPAIGN_DEFAULT,
                        help='Root of final 30-run load-matched campaign')
    parser.add_argument('--output',type=Path,default=None,
                        help='Output folder (default: campaign/04_publication_figures)')
    parser.add_argument('--s7-rep',type=int,choices=range(1,11),metavar='1..10',default=None,
                        help='Representative S7 repetition for Figure 9 (default: auto)')
    parser.add_argument('--resamples',type=int,default=5000)
    parser.add_argument('--block',type=int,default=10)
    parser.add_argument('--seed',type=int,default=519)
    args=parser.parse_args()
    root=args.campaign.expanduser().resolve()
    if not root.is_dir():fail(f'Campaign folder does not exist: {root}')
    out=(args.output or root/'04_publication_figures').expanduser().resolve()
    print('Loading ORIGINAL frozen per-run measurements:',root)
    df,audit=load_campaign(root)
    print('Validated:',len(audit),'independent run logs,',len(df),'logical decisions')
    print(audit.groupby('strategy')[['count','success_count','failed_count']].sum().to_string())
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                        'legend.fontsize':9,'svg.fonttype':'none','savefig.transparent':False})
    df.to_csv(out/'MASTER_S5_S6_S7_30_RUN_MATCHED.csv',index=False)
    audit.to_csv(out/'source_integrity_manifest.csv',index=False)
    plot_fig8(df,out)
    rep=plot_fig9(df,out,args.s7_rep)
    plot_fig10(df,out)
    summary=plot_fig12(df,out,args.resamples,args.block,args.seed)
    manifest={'status':'PASS; generated ONLY from final validated original CSV logs',
              'campaign':str(root),'generated_utc':datetime.now(timezone.utc).isoformat(),
              'strategies':{s:10 for s in STRATEGIES},'logical_decisions':len(df),
              'decisions_per_run':289,'s7_representative_repetition':rep,
              'bootstrap':{'method':'paired hierarchical circular moving-block bootstrap',
                           'run_resampling':'10 paired repetitions sampled with replacement',
                           'block_length':args.block,'resamples':args.resamples,'seed':args.seed},
              'important':('Fig 10: means are additive; P95 components are separately computed, '
                           'not additive. Fig 12 uses all measured logical-decision total '
                           'latencies, including measured failed requests. Fig 10 conditional '
                           'on successful requests. No probes enter main results.'),
              'figure12_results':summary.to_dict(orient='records')}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nALL FIGURES COMPLETE and provenance recorded:',out)


if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,KeyError,FileNotFoundError) as exc:
        sys.stderr.write('\nVALIDATION FAILED: '+str(exc)+'\n'
                         'The source campaign was not modified. Fix the missing/invalid data before plotting.\n')
        sys.exit(2)
