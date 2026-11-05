"""Task 3: collect smoke measurements, or reproduce analysis of saved logs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from .utils import REPO_ROOT, load_config, capture_env
from .process_data import process
from .visualize import figures

def main():
    p=argparse.ArgumentParser(); p.add_argument('--collect',action='store_true')
    p.add_argument('--config',default='configs/task3_smoke.yaml'); args=p.parse_args()
    import os
    os.chdir(REPO_ROOT)
    if args.collect:
        from .benchmark import run
        cfg=load_config(args.config)
        if not cfg['data'].get('smoke_augment'):
            raise ValueError('This convenience command only packages smoke data; use benchmark for real runs.')
        folder=run(cfg)
        dest=Path('data/raw/test_run.jsonl')
        with dest.open('w',encoding='utf-8') as target:
            for source in sorted((folder/'raw').glob('*.jsonl')):
                target.write(source.read_text(encoding='utf-8'))
        shutil.copy(folder/'metrics.csv','data/processed/quality_metrics.csv')
        import pandas as pd
        q=pd.read_csv(folder/'metrics.csv')
        q.loc[q.status.ne('ok')].to_csv('data/processed/failures.csv',index=False)
        meta={'data_origin':'synthetic_smoke','timing_origin':'measured',
              'hardware_scope':'local Windows host, unconstrained; NOT Raspberry Pi or 1-vCPU Docker',
              'source_run':folder.name,'raw_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
              'sample_sha256':hashlib.sha256(Path(cfg['data']['path']).read_bytes()).hexdigest(),
              'config':{k:v for k,v in cfg.items() if not k.startswith('_')},'environment':capture_env(),
              'failed_configurations':int(q.status.ne('ok').sum()),
              'expected_configurations':28,'expected_splits':2,
              'hypotheses':'NOT TESTED - synthetic smoke data'}
        Path('data/raw/provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    raw,bench,runs,summary,audit=process()
    figures(summary,runs)
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
