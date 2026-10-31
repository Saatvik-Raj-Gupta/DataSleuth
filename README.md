# DataSleuth

DataSleuth is a modular auditor for tabular dataset quality and machine-learning
leakage risks. The project is being implemented one roadmap phase at a time.

## Development setup

The project is tested with Python 3.13 in the `dataseluth` Conda environment:

```bash
conda activate dataseluth
python -m pip install -e '.[dev]'
```

Run the test and lint checks with:

```bash
pytest
ruff check .
```

## Phase 1 and 2 usage

CSV ingestion is available through the package API:

```python
from datasleuth.ingestion import load_csv

data = load_csv("path/to/data.csv")
print(data.shape)
print(data.columns.tolist())
print(data.dtypes)
```

Generate a non-mutating profile:

```python
from datasleuth.profiling import profile_dataframe

profile = profile_dataframe(data)
print(profile.row_count, profile.column_count)
print(profile.to_dict())
```

Datetime inference is disabled by default. Pass `infer_datetime=True` only when
string columns should be classified as datetime, and only fully parseable
non-null columns are inferred.

Only CSV files are accepted. Empty, oversized, undecodable, and malformed
inputs raise `IngestionError`; uploaded data is not silently cleaned or
mutated. `Finding` and `AuditContext` are the shared typed contracts for later
audit checks.

Independent data-quality checks are available through `datasleuth.checks`.
Checks return typed findings and never mutate the input DataFrame:

```python
from datasleuth.checks import MissingValueCheck, OutlierCheck
from datasleuth.engine import AuditContext

context = AuditContext(data)
findings = MissingValueCheck(threshold=0.2).run(context)
findings += OutlierCheck(method="iqr").run(context)
```

Outlier findings identify candidates for review; they do not assert that a
value is erroneous. Allowed-value rules are explicit and must be supplied by
the caller.

Leakage screening is available through the same check interface. Exact target
copies and train/test row overlap are critical findings. Correlation, mutual
information, categorical association, and ID-like features are warnings or
informational heuristics and are never labelled confirmed leakage by themselves.

Distribution checks compare `train_data` (the reference) with `test_data`:

```python
from datasleuth.checks import DistributionShiftCheck, ImbalanceCheck

imbalance = ImbalanceCheck().run(context)
shift = DistributionShiftCheck(psi_threshold=0.2).run(context)
```

PSI uses bins fitted from the reference data only. KS p-values and PSI are
heuristics whose sensitivity depends on sample size. Unseen categories are
reported explicitly. Plotly helpers are available from `datasleuth.reporting`.

Checks can be orchestrated with explicit statuses and isolated failures:

```python
from datasleuth.engine import AuditOrchestrator, CheckRegistry
from datasleuth.reporting import export_html, export_json

result = AuditOrchestrator(CheckRegistry(checks)).run(context)
json_report = export_json(result)
html_report = export_html(result)
```

Each check is reported as `passed`, `failed`, or `skipped`. A failed check
records its error without erasing findings from successful checks. The JSON
structure is canonical; HTML is a readable summary export. The Streamlit UI is
available from the repository root:

```bash
streamlit run app.py
```

The UI owns upload/configuration and presentation only. Profiling, checks,
orchestration, and report generation remain in the package modules. It shows
overview/profile data, quality/leakage/distribution results, filtered findings,
check statuses, and JSON/HTML downloads.

## Benchmark and hardening

Phase 8 provides deterministic benchmark utilities:

```python
from datasleuth.benchmark import generate_synthetic_dataset, run_benchmark

dataset = generate_synthetic_dataset(seed=7, rows=100)
result = run_benchmark(dataset)
print(result.to_dict())
```

The generator stores injected defects separately from the DataFrame as ground
truth. The benchmark reports precision, recall, F1, finding count, and measured
runtime. Clean datasets can be generated with `clean=True` to evaluate false
positives separately. Metrics and runtime are measured outputs and must not be
treated as universal performance guarantees.

The current implementation is tested on Python 3.13 with the `dataseluth`
environment. Run the full validation commands with:

```bash
pytest
ruff check .
python -m compileall -q src tests app.py
```

Known limitations include heuristic thresholds, dataset-size sensitivity,
limited CSV-only ingestion, and no automatic data cleaning. Findings require
domain review; benchmark results do not prove production readiness.
