import json
from pathlib import Path
import numpy as np
import pytest
from src.process_data import load_raw, process, bootstrap_mean

def record(i=0, phase='benchmark', status='ok', latency=1.0):
    return dict(schema_version=1,data_origin='synthetic_smoke',timing_origin='measured',
                model='random_forest',n_features=46,precision='fp32',run_id='test',
                unit_id='test:0',split=0,sequence=i,phase=phase,status=status,
                latency_ms=latency,batch_size=1,timestamp_utc='2026-11-05T00:00:00+00:00')

def save(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

def test_warmup_errors_and_spikes_are_accounted(tmp_path):
    rows=[record(0,'warmup',latency=999),*[record(i) for i in range(10)],
          record(10,latency=100),record(11,status='error',latency=50)]
    p=tmp_path/'raw.jsonl'; save(p,rows)
    _,b,_,s,a=process(p,tmp_path/'out')
    assert a['warmup_rows']==1 and len(b)==12
    assert a['benchmark_errors']==1 and s.error_rate.iloc[0]==pytest.approx(1/12)
    assert a['iqr_flagged_not_removed']==1
    assert b.latency_ms.max()==100
    assert np.isnan(s.p50_ms_ci_low.iloc[0])  # no invented CI with one unit

def test_duplicate_rejected(tmp_path):
    p=tmp_path/'x'; save(p,[record(),record()])
    with pytest.raises(ValueError,match='Duplicate'): load_raw(p)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_invalid_latency_rejected(tmp_path,bad):
    p=tmp_path/'x'; save(p,[record(latency=bad)])
    with pytest.raises(ValueError,match='latency'): load_raw(p)

def test_real_and_synthetic_cannot_mix(tmp_path):
    a=record(); b=record(1); b['data_origin']='real_ciciot2023'
    p=tmp_path/'x'; save(p,[a,b])
    with pytest.raises(ValueError,match='mix'): load_raw(p)

def test_bootstrap_seed_and_constant_interval():
    assert bootstrap_mean([2,2,2])==(2,2)
    assert bootstrap_mean([1,2,9])==bootstrap_mean([1,2,9])

def test_missing_guardrail_fails():
    from src.benchmark import check_guardrails
    ok,why=check_guardrails({'status':'ok'},{'max_latency_p50_ms':1})
    assert not ok and why==['latency_p50_ms:missing']

def test_quality_summary_keeps_failed_denominator(tmp_path):
    import pandas as pd
    from src.process_data import summarize_quality
    rows=[]
    for status,value in [('ok',10),('ok',20),('failed',999)]:
        row=dict(model='random_forest',n_features=46,precision='fp32',status=status)
        row.update({m:value for m in ('macro_f1','benign_fpr','throughput_rps','model_size_kb','ram_peak_mb')})
        rows.append(row)
    p=tmp_path/'q.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    result=summarize_quality(p,tmp_path/'summary.csv')
    assert result.n_configurations.iloc[0]==3 and result.n_failed.iloc[0]==1
    assert result.throughput_rps_mean.iloc[0]==15
