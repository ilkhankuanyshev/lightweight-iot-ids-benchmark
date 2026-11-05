"""Publication plots: explicit units, bootstrap CIs, accessible styles."""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'iot-ids-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def figures(summary, runs, out='reports/figures'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style='whitegrid',font_scale=1.05,palette='colorblind')
    banner='Measured local timings | SYNTHETIC smoke input | not edge validation'
    paths=[]
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight')
        paths.append(out/(name+'.png')); plt.close(fig)
    # Feature count is the actual manipulated factor. No request rate is invented.
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    rf=summary[(summary.model=='random_forest') & (summary.precision=='fp32')].sort_values('n_features')
    for p,mark,style in [(50,'o','-'),(95,'s','--'),(99,'^',':')]:
        m=f'p{p}_ms'; x=rf.n_features.to_numpy(); y=rf[m].to_numpy()
        ax.plot(x,y,marker=mark,linestyle=style,label=f'p{p}: mean across process/split units')
        ax.fill_between(x,rf[m+'_ci_low'],rf[m+'_ci_high'],alpha=.15)
    ax.set(xlabel='Selected input features (count)',ylabel='Single-record inference latency (ms)',
           title='Random Forest: latency vs feature workload\n'+banner,ylim=(0,None))
    ax.set_xticks(sorted(rf.n_features.unique())); ax.legend(fontsize=9)
    fig.subplots_adjust(bottom=.18,top=.85)
    fig.text(.12,.02,'Bands: 95% process-unit bootstrap CI; only two overlapping CV splits.',fontsize=9)
    save(fig,'fig1_latency_vs_load')
    # Distribution is across unit p50s, not pseudoreplicated timed calls.
    chosen=runs[(runs.n_features==46)&(runs.precision=='fp32')].copy()
    order=['decision_tree','random_forest','lightgbm','mlp','cnn1d']
    order=[m for m in order if m in set(chosen.model)]
    fig,ax=plt.subplots(figsize=(8.6,5))
    sns.boxplot(data=chosen,x='model',y='p50_ms',order=order,ax=ax,color='#b4cfe3',showfliers=True)
    sns.stripplot(data=chosen,x='model',y='p50_ms',order=order,ax=ax,color='black',jitter=False,size=5)
    sub=summary[(summary.n_features==46)&(summary.precision=='fp32')].set_index('model')
    for i,m in enumerate(order):
        a=sub.loc[m]; y=a.p50_ms
        ax.errorbar(i,y,yerr=[[max(0,y-a.p50_ms_ci_low)],[max(0,a.p50_ms_ci_high-y)]],fmt='D',color='#D55E00',capsize=5)
    ax.set(xlabel='Model (46 features, FP32)',ylabel='Process-unit p50 latency (ms)',
           title='Distribution across two process/split units\n'+banner,ylim=(0,None))
    ax.set_xticks(range(len(order)),[m.replace('_','\n') for m in order])
    fig.subplots_adjust(bottom=.24,top=.85)
    fig.text(.12,.02,'Black dots: units. Orange diamonds/caps: mean and 95% bootstrap CI. Tiny n.',fontsize=9)
    save(fig,'fig2_distribution_boxplot')
    return paths

if __name__=='__main__':
    figures(pd.read_csv('data/processed/metrics_summary.csv'),pd.read_csv('data/processed/run_summary.csv'))
