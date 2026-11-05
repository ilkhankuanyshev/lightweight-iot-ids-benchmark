"""Generate Task 3 PDF/Markdown and README from the actual saved smoke outputs."""
from pathlib import Path
import json
import os
import html
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
summary=pd.read_csv('data/processed/metrics_summary.csv')
quality=pd.read_csv('data/processed/quality_metrics.csv')
audit=json.loads(Path('data/processed/audit.json').read_text())
meta=json.loads(Path('data/raw/provenance.json').read_text())
q=quality[quality.model!='dummy']

def formatted_table():
    rows=[]
    for k,label in [(46,'Baseline RF, 46, FP32'),(15,'Proposed RF, 15, FP32')]:
        s=summary[(summary.model=='random_forest')&(summary.n_features==k)&(summary.precision=='fp32')].iloc[0]
        g=q[(q.model=='random_forest')&(q.n_features==k)&(q.precision=='fp32')]
        rows.append({'Configuration':label,
            'p50 ms, mean +/- SD':f'{s.p50_ms:.5f} +/- {s.p50_ms_sd:.5f}',
            '95% CI (ms)':f'[{s.p50_ms_ci_low:.5f}, {s.p50_ms_ci_high:.5f}]',
            'Synthetic F1':f'{g.macro_f1.mean():.4f}',
            'Size (KiB)':f'{g.model_size_kb.mean():.1f}'})
    return pd.DataFrame(rows)

table=formatted_table()
base=summary[(summary.model=='random_forest')&(summary.n_features==46)&(summary.precision=='fp32')].iloc[0]
prop=summary[(summary.model=='random_forest')&(summary.n_features==15)&(summary.precision=='fp32')].iloc[0]
reduction=100*(1-prop.p50_ms/base.p50_ms)
sections=[
('1. Research aim and evidence status',[
 'This individual project continues the accuracy-cost trade-off study of lightweight machine-learning models for IoT intrusion detection. The full research design uses CICIoT2023, 46 features, eight categories and a constrained edge inference platform. Author: Ilkhan Kuanyshev.',
 'This submission delivers the Task 3 analysis implementation and an executed technical demonstration. The repository contained a ten-row synthetic schema fixture, not a real CICIoT2023 sample. The fixture was augmented by the inherited smoke configuration. No real-data detection result is reported.',
 'The latency values are actual clock measurements from ONNX inference on this local Windows computer. They are not invented latency samples. However, they describe models trained on synthetic smoke input and cannot establish operational IDS quality, Raspberry Pi performance or the project hypotheses.',
 'Evidence classes: (1) verified execution artifacts and local timings; (2) demonstration-only quality scores on synthetic data; (3) planned, unperformed real-data and target-device experiments. H1.1, H1.2 and H1.3 are NOT TESTED.'
 ]),
('2. Continuity with Task 2 and implementation',[
 'The source project is preserved, including dataset mapping, train-only mutual-information selection, five model families, ONNX export, neural INT8 quantization, train/measure separation and the original test suite. Upstream revision: 51759722daee772e40310fccc0ce72e9b03dea8c.',
 'Task 3 adds strict raw-log ingestion, warm-up filtering, request-level IQR flags, process-unit summaries, seeded bootstrap confidence intervals, figures, an executed Jupyter notebook and a formal domain model. The measurement worker now writes phase-labelled JSONL after the timed loop. Missing guardrail values fail rather than silently passing.',
 'The delivered smoke run uses two stratified folds, feature counts 46 and 15, five model families and FP32/INT8 variants for MLP and CNN. Random Forest uses 100 trees to match the Task 2 baseline. Each model process executes 100 warm-up and 1,000 measured single-record predictions; throughput uses batch 256. Smoke training budgets remain intentionally small.',
 f"Run: {meta['source_run']}. Python {meta['environment']['python']}; {meta['environment']['platform']}. ONNX Runtime {meta['environment']['packages']['onnxruntime']}. No CPU quota or 1 GiB memory limit was imposed. The snapshot and full package lock document this deviation from Task 2's Python 3.12 container."
 ]),
('3. Data processing and statistical estimands',[
 f"The raw log contains {audit['raw_rows']:,} rows: {audit['warmup_rows']:,} warm-up rows and {audit['benchmark_attempts']:,} measured benchmark attempts. There are {audit['benchmark_errors']} request errors and {meta['failed_configurations']} failed configuration rows. IQR flags identify {audit['iqr_flagged_not_removed']:,} unusual benchmark durations; none are deleted.",
 'Validation rejects invalid schemas, missing fields, duplicate identities, invalid phases, negative/non-finite timings and mixed real/synthetic provenance. SHA-256 binds the notebook to the packaged raw log. Successful-call latency and all-attempt error rate are separate; the configuration manifest covers workers that fail before emitting request logs.',
 'Within each configuration/process unit, calculate mean, sample SD and p25/p50/p75/p95/p99 of successful latency. Across units, report the mean of each percentile and sample SD. Resample entire unit metrics 4,000 times with seed 42; use the 2.5th and 97.5th percentiles of bootstrap means for the displayed 95% interval.',
 'Only two overlapping CV splits are present. The interval is descriptive, not a reliable population confidence statement. Training sets and synthetic ancestors overlap; request-level IID resampling would exaggerate precision. At n=1 the implementation returns no interval rather than manufacturing uncertainty.',
 'Throughput is taken from timed batch execution, not the reciprocal of median latency. Model size uses 1,024 bytes per KiB; RAM uses 1,048,576 bytes per MiB. Windows peak working set is reported for this run; Linux uses VmHWM. Neither is model-only memory.'
 ]),
('4. Results and discussion: smoke evidence only',[
 'The table contrasts the prespecified baseline (Random Forest, 46 features, FP32) with the proposed feature-reduction configuration (Random Forest, 15 features, FP32). Latency is mean +/- sample SD of process-unit p50 values. F1 is a synthetic-input demonstration score, not a real CICIoT2023 result.',
 f'The observed change in mean unit p50 from 46 to 15 features is {reduction:.1f}% reduction (a negative value means an increase). This is descriptive for this run only. It neither confirms nor refutes the 30% reduction hypothesis: two synthetic-data folds on an unconstrained host are not the planned experiment.',
 'The size and F1 columns show that the analysis can join quality and cost artifacts. A configuration passing thresholds in this smoke run is not a deployment recommendation. INT8 is genuinely exported and run, but its benefit on real data remains untested. All configuration summaries, including guardrail failures, are retained.',
 'Scientific figure rules: explicit units, zero latency baseline, colorblind-safe palette, distinct percentile markers/line styles and confidence bands. Distribution points and error caps remain distinguishable in grayscale. Both figures are exported as vector PDF and 300-DPI PNG.',
 'Figure 1 uses feature count as the manipulated workload. The supplied rubric filename is retained for compatibility; no request-arrival-rate experiment or latency-versus-RPS claim is made. Figure 2 shows process-unit p50 distributions and CIs at 46 features/FP32, rather than presenting repeated calls as independent folds.'
 ]),
('5. Formal domain model and research validity',[
 'The full formal artifact is docs/domain_model.md. It defines the dataset, eight-class labels, split indices, train-only feature selector, model family, precision, probabilities, timing quantiles, model size, RAM and non-inferiority margin. It includes LaTeX equations, a Mermaid measurement flow and a component-level STRIDE matrix.',
 'Neural loss: L = -(1/|T|) sum over training records and classes of 1[y=c] log p(c|x). Class F1 = 2TP/(2TP+FP+FN); macro-F1 is the arithmetic mean over all eight classes. Benign FPR is the fraction of benign evaluation rows predicted as attacks. Undefined denominators remain explicit.',
 'The constrained objective maximizes macro-F1 with p50 <= 1 ms, ONNX size <= 1,024 KiB, peak process memory <= 512 MiB, benign FPR <= 0.02 and no failed or missing splits. Missing metrics fail closed. No optimum or Pareto winner is selected from smoke data.',
 'Task 2 specifies Friedman/Nemenyi comparisons and paired Wilcoxon tests with Holm corrections and a one-percentage-point non-inferiority margin. Those routines are retained, but hypothesis testing is disabled for synthetic smoke runs. Full-run interpretation must review dependence between repeated-CV folds and correction across the two RQ1 endpoints.',
 'Additional validity threats: jittered samples are not independent traffic records; class capping changes prevalence; background processes perturb timings; two process units yield weak uncertainty estimates; x86 CPU limits do not emulate ARM; feature extraction and network queuing are outside the clock boundary. These limits prevent deployment claims.'
 ]),
('6. Requirements, reproducibility and remaining work',[
 'Task 3 mapping: statistics and error audit -> src/process_data.py and data/processed/; scientific plots -> src/visualize.py and reports/figures/; executed notebook -> notebooks/01_data_analysis.ipynb; formal model -> docs/domain_model.md; structure and Git hygiene -> README.md and .gitignore.',
 'SS6 is implemented through repository structure, a recorded environment, pinned dependencies, hashes and reproduction instructions. SS7 is implemented through isolated timing, warm-up phases, structured JSONL, error accounting and environment capture. SS8 is implemented through non-destructive outlier flags, percentile summaries, bootstrap uncertainty and dual-format figures.',
 'To reproduce the saved analysis: install requirements-task3-lock.txt, run python -m src.task3, then python scripts/execute_notebook.py. To collect a new demonstration, add --collect --config configs/task3_smoke.yaml. Re-running collection changes wall-clock values; re-analysis of the saved log is seeded and deterministic.',
 'Before scientific completion: obtain and verify real CICIoT2023 records; execute the full 5x3 design with all four feature counts; measure under the specified target limits; retain raw failures and complete split pairing; inspect leakage and dependence; replace smoke tables and figures; then review and publish the GitHub release. Repository: https://github.com/ilkhankuanyshev/lightweight-iot-ids-benchmark (release v3.0).',
 'Source materials: Task1_Report_Kuanyshev.pdf (research aim and hypotheses); English.pdf (Task 2 protocol); ss6.docx.pdf (reproducibility); ss7.docx.pdf (clean measurement); ss8.docx (1).pdf (processing and graphics); Task 3.docx.pdf (deliverables and rubric). The attached documents are project requirements, not empirical measurement evidence. The Task 1 literature claims were not independently revalidated for this implementation.'
 ])]

md=['# Task 3: Lightweight IoT IDS benchmark','',
    '**Ilkhan Kuanyshev | 5 November 2026**','',
    '> MEASURED LOCAL SMOKE TIMINGS / SYNTHETIC INPUT / NO CONFIRMATORY RESULTS','']
for title,paras in sections:
    md.extend(['## '+title,'','\n\n'.join(paras),''])
    if title.startswith('4.'):
        md.extend([table.to_markdown(index=False),'',
                   '![Latency](figures/fig1_latency_vs_load.png)','',
                   '![Distribution](figures/fig2_distribution_boxplot.png)',''])
Path('reports/Task3_Report.md').write_text('\n'.join(md),encoding='utf-8')

readme=f'''# Lightweight IoT IDS benchmark - Task 3

**Author: Ilkhan Kuanyshev | Status: executed smoke analysis | License: inherited MIT**

> Timings are measured on this local computer. Input data are SYNTHETIC.
> Real CICIoT2023 detection results, target-device results and H1.1-H1.3 conclusions are unavailable.

## Project overview

Study the accuracy-cost trade-off across Decision Tree, Random Forest, LightGBM,
MLP and 1D-CNN, with train-only feature selection and neural INT8 quantization.
The scientific goal is maximum macro-F1 within p50 <= 1 ms and model size <= 1 MiB,
plus RAM, false-positive and stability guardrails. This package continues the
existing Task 2 implementation at revision 51759722daee772e40310fccc0ce72e9b03dea8c.

## Quickstart

Python 3.13 / Windows is the tested Task 3 environment. CPU only; no GPU required.
The test fixture is small. Full dataset training may require substantially more RAM.

```bash
python -m pip install -r requirements-task3-lock.txt
python -m src.task3
python scripts/execute_notebook.py
```

This regenerates analysis from the bundled raw log; it does not generate invented timings.
To recollect actual smoke timings: `python -m src.task3 --collect`.
Tests: `python -m pytest -q`. PDF/Markdown rebuild: `python scripts/build_report.py`.

## Results & Discussion

**Measured local execution / synthetic input. Two process/split units per configuration.**

{table.to_markdown(index=False)}

p50 is mean +/- sample SD across unit p50s; CI is a 4,000-resample whole-unit bootstrap.
With only two overlapping folds, these CIs are descriptive and weak. F1 has no
real-world interpretation. The observed p50 reduction is {reduction:.1f}%, with no
hypothesis decision or selected deployment winner.

{audit['warmup_rows']:,} warm-up records excluded; {audit['benchmark_attempts']:,} benchmark
calls retained; {audit['benchmark_errors']} request errors; {meta['failed_configurations']}
configuration failures; {audit['iqr_flagged_not_removed']:,} outliers flagged without deletion.

![Latency Plot](reports/figures/fig1_latency_vs_load.png)

Feature count is the actual workload factor. No offered RPS was measured.

![Distribution Plot](reports/figures/fig2_distribution_boxplot.png)

Dots show process-unit p50s; orange caps show bootstrap CI. Both figures also have vector PDFs.

## Repository structure

- `src/`: original training/ONNX code plus raw logging, processing and visualization.
- `configs/`: full Task 2 design and explicitly synthetic smoke configurations.
- `data/raw/test_run.jsonl`: phase-labelled, actually timed smoke calls.
- `data/raw/provenance.json`: origin, input hashes, environment and configuration.
- `data/processed/`: summaries, quality outputs, error manifest and outlier flags.
- `notebooks/01_data_analysis.ipynb`: executed notebook with visible outputs.
- `docs/domain_model.md`: formal ML model, measurement diagram and STRIDE matrix.
- `reports/`: PDF/Markdown report, figures and validation record.
- `tests/`: inherited tests plus Task 3 integrity/aggregation tests.

## Domain model and reproducibility

[Formal domain model](docs/domain_model.md) | [Reproduction and full-run instructions](docs/reproduction.md)

[PDF report](reports/Task3_Report.pdf) | [Markdown report](reports/Task3_Report.md)

The original requirements.txt/Dockerfile remain for Task 2. The Task 3 lock records
the actual Windows environment and is not a platform-independent Docker lock.
No commit, remote push or release tag has been published by this work.
Do not label task3-smoke-v1 a completed scientific benchmark.

## Citation & license

Original code license is retained in LICENSE; dataset licensing remains with its publisher.

```bibtex
@software{{kuanyshev_task3_2026,
  author = {{Kuanyshev, Ilkhan}},
  title = {{Lightweight IoT IDS Benchmark: Task 3 Smoke Analysis}},
  year = {{2026}},
  note = {{Measured local smoke timings; synthetic input; unpublished Task 3 package}}
}}
```
'''
Path('README.md').write_text(readme,encoding='utf-8')

styles=getSampleStyleSheet()
ink=colors.HexColor('#151515')
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10.5,leading=14.2,spaceAfter=7,textColor=ink))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.3,leading=11,spaceAfter=0,textColor=ink))
styles.add(ParagraphStyle(name='TableHead',parent=styles['SmallCustom'],fontName='Helvetica-Bold'))
for name in ('Heading1','Heading2','Title'):
    styles[name].textColor=ink
    styles[name].fontName='Helvetica-Bold'
styles['Heading1'].fontSize=11.5; styles['Heading1'].leading=15
styles['Heading1'].spaceBefore=11; styles['Heading1'].spaceAfter=6
styles['Heading1'].keepWithNext=True
styles['Heading2'].fontSize=11; styles['Heading2'].leading=15
styles['Heading2'].spaceAfter=7
styles['Title'].fontSize=20; styles['Title'].leading=24
styles['Title'].alignment=TA_LEFT; styles['Title'].spaceAfter=5
story=[]
def para(t,style='BodyCustom'): return Paragraph(html.escape(t),styles[style])
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#b5b5b5')); canvas.line(42,39,A4[0]-42,39)
    canvas.setFont('Helvetica',8); canvas.setFillColor(ink)
    canvas.drawString(42,26,'TASK 3 | Kuanyshev | Synthetic-input smoke study')
    canvas.drawRightString(A4[0]-42,26,str(doc.page))
story += [para('Lightweight IoT IDS benchmark','Title'),
          para('Task 3: Results, Analysis and Scientific Visualization','Heading2'),
          para('Ilkhan Kuanyshev | 5 November 2026','SmallCustom'),Spacer(1,9),
          para('Measured local timings. Synthetic input. No confirmatory real-data results.','TableHead'),Spacer(1,4)]
for idx,(title,paras) in enumerate(sections):
    story.append(para(title,'Heading1'))
    for t in paras: story.append(para(t))
    if idx==3:
        vals=[[para(str(v),'TableHead') for v in table.columns]]+[[para(str(v),'SmallCustom') for v in row] for row in table.values]
        t=Table(vals,colWidths=[118,118,112,73,74],repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,ink),('LINEBELOW',(0,-1),(-1,-1),.4,ink),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([Spacer(1,3),t,Spacer(1,5)])
story.extend([PageBreak(),para('Figures. Measured smoke execution','Heading1')])
for f in ['fig1_latency_vs_load','fig2_distribution_boxplot']:
    from PIL import Image as PILImage
    path=Path('reports/figures')/(f+'.png')
    w,h=PILImage.open(path).size
    story.append(Image(str(path),width=495,height=495*h/w)); story.append(Spacer(1,12))
SimpleDocTemplate('reports/Task3_Report.pdf',pagesize=A4,rightMargin=42,leftMargin=42,
                  topMargin=36,bottomMargin=52,title='Task 3 - Lightweight IoT IDS Benchmark',
                  author='Ilkhan Kuanyshev').build(story,onFirstPage=footer,onLaterPages=footer)
print('Report and README generated from saved measurements.')
