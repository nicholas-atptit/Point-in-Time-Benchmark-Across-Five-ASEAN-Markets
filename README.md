# Five-market point-in-time equity forecasting benchmark
Supporting code and audited aggregate artifacts for **Do Current-Constituent Universes Distort Equity Forecasting Evidence? A Point-in-Time Benchmark Across Five ASEAN Markets**.

This is a partial public reproducibility release for the frozen V8.1.1 benchmark. It contains six manuscript figures (PNG/PDF plus exact source CSVs), eight main tables, twenty supplementary tables, original pipeline code, recovered imported helpers, configurations and a fresh audit. Restricted security-level inputs and predictions remain local. See [publication audit](audit/PUBLICATION_AUDIT_REPORT.md) for verification boundaries and failures.

The five indices are IDX30, SET50, VN30, PSEI and FBM KLCI. Historical PIT, terminal-current-applied-backward and separately trained intersection arms remain distinct. Four models are registered: historical class probability, logistic regression, random forest and LightGBM. The primary five-market block-20 Brier comparison supports no difference for any model; this does not establish equivalence. Current-arm operational coverage is 3.4463 percentage points lower.

## Install and run
Use Python 3.13 in an isolated environment:
```sh
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/reproduce_public.py
python -m pytest tests -q
```
Activate the environment using the command appropriate to your shell before installing. The tested package versions are pinned; the historical whole-machine snapshot is archival evidence, not an install recipe.

`reproduce_public.py` uses the original figure/table functions and public aggregate inputs. It writes to ignored `reproduced/`, compares all eight tables and six figure-source datasets, and exits nonzero on any discrepancy. It does not rerun models or simulate missing empirical inputs. The public tests validate release integrity, inference-family labels and genuine software behavior.

## Layout
- `scripts/point_in_time/`: original construction, training, evaluation, inference and validation code.
- `papers/`, `shared/`: only imported model/feature dependencies, with no unrelated paper results.
- `research_data/point_in_time/config/`: original protocol/environment metadata.
- `results/`: selected aggregate inputs used by figure/table generation.
- `figures/`, `tables/`: exact original manuscript assets; supplementary tables also exported to CSV.
- `audit/`: mapping, numerical verification, data schemas, exclusions and checksum manifests.
- `archive_tests/`: historical tests requiring private evidence, with their known failure documented.
- `docs/`, `data/`: methods, reproduction instructions, limitations and data-access policy.

See [reproduction guide](docs/REPRODUCIBILITY.md), [methods](docs/METHODOLOGY.md), [limitations](docs/LIMITATIONS.md), and [data access](data/README.md). No DOI, publication venue or acceptance status is assigned. Cite the manuscript title and this repository with the actual commit used; `CITATION.cff` describes this software/artifact release. No public reuse license has yet been assigned; see [licensing decision](docs/LICENSING_DECISION_PENDING.md).
