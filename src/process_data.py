"""Strict JSONL validation, non-destructive IQR flags and cluster bootstrap.

python -m src.process_data --raw data/raw/test_run.jsonl --out data/processed
Latency CI: resample whole process/split units, then average unit percentiles.
This is NOT a request-level IID bootstrap and NOT evidence of deployment quality.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['data_origin', 'timing_origin', 'model', 'n_features', 'precision']
REQUIRED = KEYS + ['schema_version', 'run_id', 'unit_id', 'split', 'phase',
                   'sequence', 'timestamp_utc', 'latency_ms', 'status', 'batch_size']

def load_raw(path):
    records = []
    with Path(path).open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
                if not isinstance(r, dict) or set(REQUIRED) - r.keys():
                    raise ValueError('missing required fields')
                if r['schema_version'] != 1 or r['phase'] not in ('warmup','benchmark'):
                    raise ValueError('invalid schema or phase')
                if r['status'] not in ('ok','error') or r['batch_size'] != 1:
                    raise ValueError('invalid status or batch size')
                if r['data_origin'] not in ('synthetic_smoke','real_ciciot2023','unverified'):
                    raise ValueError('unrecognized data provenance')
                if r['timing_origin'] != 'measured':
                    raise ValueError('fabricated timings not accepted by this pipeline')
                for field in ('split','sequence','n_features'):
                    if not isinstance(r[field], int) or r[field] < 0:
                        raise ValueError('invalid integer: '+field)
                if not 1 <= r['n_features'] <= 46:
                    raise ValueError('feature count out of range')
                if not isinstance(r['latency_ms'], (float,int)) or not np.isfinite(r['latency_ms']) or r['latency_ms'] < 0:
                    raise ValueError('invalid latency')
                pd.to_datetime(r['timestamp_utc'], utc=True, errors='raise')
                records.append(r)
            except Exception as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
    if not records:
        raise ValueError('Empty raw log')
    df = pd.DataFrame(records)
    if df.duplicated(KEYS+['unit_id','phase','sequence']).any():
        raise ValueError('Duplicate measurement identity')
    if df['data_origin'].nunique() != 1:
        raise ValueError('Never mix real and synthetic datasets in one analysis')
    return df

def bootstrap_mean(values, seed=42, resamples=4000):
    """Percentile interval for mean of run-level metrics; NaN if <2 units."""
    a=np.asarray(values,dtype=float)
    if len(a)<2:
        return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed)
    means=rng.choice(a,(resamples,len(a)),replace=True).mean(axis=1)
    return tuple(np.quantile(means,[.025,.975]))

def summarize_quality(path, out, seed=42, resamples=4000):
    """Separate configuration-level quality, throughput and resource statistics."""
    q=pd.read_csv(path)
    rows=[]
    for key,g in q[q.model.ne('dummy')].groupby(['model','n_features','precision']):
        row=dict(zip(['model','n_features','precision'],key))
        row.update(n_configurations=len(g),n_failed=int(g.status.ne('ok').sum()))
        for metric in ('macro_f1','benign_fpr','throughput_rps','model_size_kb','ram_peak_mb'):
            a=g.loc[g.status.eq('ok'),metric].dropna().to_numpy(dtype=float)
            row[metric+'_mean']=a.mean() if len(a) else np.nan
            row[metric+'_sd']=a.std(ddof=1) if len(a)>1 else np.nan
            for p in (50,95,99):
                row[metric+f'_p{p}']=np.quantile(a,p/100) if len(a) else np.nan
            lo,hi=bootstrap_mean(a,seed,resamples)
            row[metric+'_ci_low']=lo; row[metric+'_ci_high']=hi
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(out,index=False)
    return result

def process(raw='data/raw/test_run.jsonl', out='data/processed', seed=42, resamples=4000):
    df=load_raw(raw)
    target=Path(out); target.mkdir(parents=True,exist_ok=True)
    bench=df.loc[df.phase.eq('benchmark')].copy()
    if bench.empty:
        raise ValueError('No benchmark rows')
    bench['iqr_outlier']=False
    rows=[]
    for key,g in bench.groupby(KEYS+['unit_id'],sort=True):
        ok=g[g.status.eq('ok')]
        vals=ok.latency_ms
        q1,q3=vals.quantile([.25,.75]) if len(vals) else (np.nan,np.nan)
        flags=(vals < q1-1.5*(q3-q1)) | (vals > q3+1.5*(q3-q1))
        bench.loc[ok.index,'iqr_outlier']=flags
        row=dict(zip(KEYS+['unit_id'],key))
        row.update(n_attempts=len(g),n_success=len(ok),n_errors=len(g)-len(ok),
                   error_rate=(len(g)-len(ok))/len(g),n_outliers=int(flags.sum()),
                   mean_ms=vals.mean(),sd_ms=vals.std(ddof=1))
        for p in (25,50,75,95,99):
            row[f'p{p}_ms']=vals.quantile(p/100)
        rows.append(row)
    runs=pd.DataFrame(rows)
    summaries=[]
    for key,g in runs.groupby(KEYS,sort=True):
        row=dict(zip(KEYS,key)); row['n_units']=len(g)
        row['n_attempts']=int(g.n_attempts.sum()); row['n_errors']=int(g.n_errors.sum())
        row['error_rate']=row['n_errors']/row['n_attempts']
        row['n_outliers']=int(g.n_outliers.sum())
        for metric in ('mean_ms','p50_ms','p95_ms','p99_ms'):
            a=g[metric].dropna().to_numpy()
            row[metric]=np.mean(a) if len(a) else np.nan
            row[metric+'_sd']=np.std(a,ddof=1) if len(a)>1 else np.nan
            lo,hi=bootstrap_mean(a,seed,resamples)
            row[metric+'_ci_low']=lo; row[metric+'_ci_high']=hi
        row['p50_median_ms']=g.p50_ms.median()
        row['p50_q25_ms']=g.p50_ms.quantile(.25); row['p50_q75_ms']=g.p50_ms.quantile(.75)
        row['ci_scope']='mean of process/split percentiles; descriptive only'
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    runs.to_csv(target/'run_summary.csv',index=False)
    summary.to_csv(target/'metrics_summary.csv',index=False)
    bench[KEYS+['unit_id','sequence','status','latency_ms','iqr_outlier']].to_csv(target/'request_flags.csv',index=False)
    audit={'raw_rows':len(df),'warmup_rows':int(df.phase.eq('warmup').sum()),
           'benchmark_attempts':len(bench),'benchmark_errors':int(bench.status.ne('ok').sum()),
           'iqr_flagged_not_removed':int(bench.iqr_outlier.sum()),'seed':seed,
           'bootstrap_resamples':resamples,'origin':df.data_origin.unique().tolist(),
           'warning':'CI is descriptive: folds overlap and few process units give weak uncertainty estimates.'}
    (target/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if (target/'quality_metrics.csv').exists():
        summarize_quality(target/'quality_metrics.csv',target/'quality_summary.csv',seed,resamples)
    return df,bench,runs,summary,audit

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw',default='data/raw/test_run.jsonl')
    p.add_argument('--out',default='data/processed'); a=p.parse_args()
    print(json.dumps(process(a.raw,a.out)[-1],indent=2))
