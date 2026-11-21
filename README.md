# OSA ECG development

Research source for ECG apnea detection, forecasting, reference-AHI estimation, representation learning and residual analyses. This is a code-only companion; the manuscript and research results are distributed separately.

## Contents

- `src/`: signal handling, features, classical/neural models, representations, residuals, grouped evaluation and runner.
- `configs/`: fixed experiment settings with generic data paths.
- `tests/`: synthetic software tests.
- `reports/`: scientific protocol and correction history needed by source locking.
- `scripts/fetch_primary.py`: optional public-data preparation; never runs automatically.

## Software checks

Python 3.12 and the recorded dependency versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-public.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests use artificial fixtures and may perform small artificial checkpoint/optimization checks. They do not train on patient data. Dependency pins describe a validated environment, not guaranteed availability on every package index.

## Data and execution

The primary dataset is public Apnea-ECG: https://physionet.org/content/apnea-ecg/1.0.0/ . MIMIC and EchoNext are not inputs to this study. Data, reference rows, weights, individual predictions and caches are excluded. The public c05/c06 overlap rule remains explicitly encoded in source.

For a separate fresh workspace, the optional source-data helper is:

```bash
python scripts/fetch_primary.py --raw-dir /NEW/RAW/PATH --workspace /NEW/STUDY/PATH
```

Verify source files/reference provenance and set source/copy paths in configs/study.json before any clinical run. From a prepared new workspace, the original bounded runner supports:

```bash
python src/runner.py status
python src/runner.py resume --hours 8
python src/runner.py stop
```

Clinical execution needs compatible local data and private preparation. Public tests are not a full result reproduction. No aggregate plotting or manuscript generation is advertised by this minimal release.

## Scientific limits

Historically explored recordings and conservative proxy groups do not establish known independent patient identities. Detection, state forecasting, new transitions and reference AHI are distinct tasks. External generalization, new physiological phenotypes, causal injury and clinical utility are not established. The original computational algorithms are retained; earlier reporting-template cleanup changed display text only.

No raw data, patient identifiers/rosters, predictions, embeddings, trained weights, credential files, manuscript or saved research results are included. No software license grant is specified; data-access terms are separate from code rights.
