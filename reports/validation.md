# Validation record

- Full test suite: **16 passed**, 104 warnings, 25.77 seconds.
- Warnings: 100 inherited ONNX/protobuf boolean-to-int deprecations; four MLP non-convergence warnings from deliberately short smoke training. These are not hidden or interpreted as convergence.
- Measured collection: 28 configuration/split executions; 28,000 benchmark calls; 2,800 warm-up calls; zero runtime/configuration errors.
- Scientific guardrails: some configurations fail benign FPR. Zero runtime errors does not mean all guardrails pass.
- Notebook: six code cells executed sequentially [1, 2, 3, 4, 5, 6], all have outputs, no error outputs.
- Environment: pip check reports no broken requirements.
- PDF: four pages, rendered and visually inspected. Plot labels/captions do not overlap.
- Figures: two vector PDFs and two 300-DPI PNGs.
- Data hash: notebook input matches the recorded raw SHA-256.
- Hypotheses H1.1-H1.3: NOT TESTED, because the input dataset is synthetic.
- Remote publication: not performed; release instructions are provided.

The full training tests also verify the 34-label mapping, CNN numerical gradients,
NumPy/ONNX equivalence and both one-phase and two-phase execution. Task 3 tests
verify bad-data rejection, error denominators, warm-up filtering, non-destructive
outlier handling, bootstrap reproducibility and missing-value guardrails.
