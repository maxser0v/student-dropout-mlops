# PAN 2013 passage reuse results

The real-data Docker run trained XGBoost on candidates retrieved by a frozen MiniLM encoder. MLflow registered `passage-reuse` version 1 after the validation gate passed. On the custom held-out split, the registered model produced 65 false alerts at 82.8% span recall. MiniLM cosine alone produced 4,055 false alerts at 84.3% span recall. The combined model cut false alerts by 98.4% with a 1.5 percentage-point drop in recall. [Training run](http://127.0.0.1:5001/#/experiments/1/runs/2f1ee43689c349dd93d8a70222fc75d9) · [Held-out run](http://127.0.0.1:5001/#/experiments/3/runs/d5f51ef602e640cfa4749af5d1127b2a)

## Dataset and split

The PAN 2013 Text Alignment download matched the published archive MD5 `fec7fca7a82193a97fbb97cde7781b63`. The importer preserved the original line endings, validated XML span offsets against the text, and collapsed 5,185 repeated comparison rows in the training archive while keeping the distinct aligned spans. The first run used the published training and test partitions. Exact-text overlap across the two archives left two positive training pairs, zero negative training pairs, and zero validation pairs after the leakage check, so Metaflow stopped before fitting. [Data preparation guide](PLAGIARISM_DATA.md) · [Initial data check](http://127.0.0.1:5001/#/experiments/1/runs/b6cf288550c6406f983c9d188327c5f2)

With approval, the completed run used only PAN’s official **training** archive. The importer formed connected document groups from exact text matches and annotated reuse links, shuffled the groups with seed `903`, and allocated approximately 60% of groups to train, 20% to validation, and 20% to test. It excluded 1,075 negative pairs whose endpoints fell in different splits. Two imports produced identical manifests. The split audit verified that no text hash, reuse group, or positive annotation crossed partitions. DVC verified the manifest before training. [DVC pointer](../plagiarism_data.dvc)

| Partition | Documents | Positive pairs | Negative pairs |
|---|---:|---:|---:|
| Train | 2,707 | 1,538 | 716 |
| Validation | 1,113 | 769 | 69 |
| Held-out test | 1,237 | 931 | 87 |

## Training and release workflow

1. **Stage the encoder.** The ARM64 Docker image used the isolated, locked runtime. The staging script downloaded the `all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, selected its safetensors weights, and recorded per-file hashes. The offline smoke test verified the hashes and produced a normalized 384-dimensional CPU embedding. [Runtime lock](../plagiarism-runtime/requirements.lock.txt) · [Runbook](PLAGIARISM_RUNBOOK.md)
2. **Retrieve candidate passages.** Metaflow chunked text into 180-token windows with 40-token overlap and a 256-token cap, preserving character offsets. MiniLM indexed 17,678 development source chunks and retrieved up to 10 candidates per query chunk. During benchmark evaluation, retrieval was restricted to the source documents paired with each submission by PAN. The run produced 94,858 training candidates and 30,356 validation candidates. [Retrieval code](../src/plagiarism/candidates.py) · [Configuration](../config/plagiarism.yaml)
3. **Fit the XGBoost ranker.** The candidate classifier used MiniLM cosine similarity, TF-IDF cosine similarity, shared word n-gram overlap, rare-term overlap, and passage length ratio. The workflow fit preprocessing on training text, labeled candidates against the PAN span annotations, and excluded 40,457 ambiguous training candidates with partial overlap. It trained CPU histogram trees, saved the XGBoost model and feature transforms in MLflow, and verified that the reloaded model reproduced its original predictions. [Training flow](../flows/train_plagiarism_flow.py)
4. **Select thresholds on validation and register.** The flow compared TF-IDF cosine, MiniLM cosine, and MiniLM plus XGBoost around a common 80% recall target. The gate required at least 75% shared recall, XGBoost recall within five percentage points of MiniLM cosine, and at least one fewer false alert. XGBoost passed with 93 false alerts versus 2,536 for MiniLM cosine at the same 81.0% span recall. MLflow registered version 1, and the evaluation script froze that concrete version and the three validation thresholds before scoring the held-out partition. [Gate report](../artifacts/plagiarism/reports/1795360483306022/gate_report.json) · [Freeze record](../reports/plagiarism/frozen_final_model.json)

## Held-out results

The held-out partition contained 405 submissions, 832 source documents, 1,018 judged document pairs, and 1,204 annotated reuse spans. MiniLM retrieved a qualifying passage pair for 1,091 spans, giving 90.6% retrieval recall at 10 before XGBoost scoring. [Full metrics](../reports/plagiarism/final_test_metrics.json)

| Method | Frozen threshold | Alert precision | Span recall | False alerts | False alerts per 100 submissions |
|---|---:|---:|---:|---:|---:|
| TF-IDF cosine | 0.240866 | 52.4% | 85.5% | 919 | 226.9 |
| MiniLM cosine | 0.467471 | 19.5% | 84.3% | 4,055 | 1,001.2 |
| MiniLM + XGBoost | 0.922696 | 93.8% | 82.8% | 65 | 16.0 |

Alert precision is the share of merged alerts that match an annotated span. Span recall is the share of annotated spans recovered by at least one alert. One submission can generate multiple alerts. The combined model recovered 997 spans, compared with 1,015 for MiniLM cosine and 1,030 for TF-IDF cosine. This measures passage alignment on the custom holdout conditional on PAN’s supplied document pairs. An open-corpus search or live university deployment would require a separate evaluation. [Scoring and evaluation code](../src/plagiarism/evaluation.py)

## XGBoost tuning findings

**The 50-tree model recovered the same 728 of 899 validation spans (81.0% recall) as the 80-tree model, with one additional false alert.** Both models used a maximum depth of 4. The 50-tree model used a learning rate of 0.14 and produced 94 false alerts. The 80-tree model used a learning rate of 0.08 and produced 93. This demonstrates a trade-off between the number of trees and false alerts at the same validation recall. [50-tree trial](http://127.0.0.1:5001/#/experiments/4/runs/a0b883aeba9b4b5a8d3d83e775bd0696) · [80-tree trial](http://127.0.0.1:5001/#/experiments/4/runs/4a3b193b95e44d0ba1b5f9307a3e40ea)

The sweep compared 20 configurations on the same MiniLM candidates and selected each model’s threshold on validation. The 80-tree configuration ranked first on false alerts at or above version 1’s span recall. The held-out results above apply to version 1. [Full MLflow sweep](http://127.0.0.1:5001/#/experiments/4/runs/cbe82b67a5fb46ba9dcaf922d69063ef)

## Verification and evidence

- All 92 repository tests passed, including a regression test for deterministic group splitting. All 22 plagiarism tests also passed in the isolated runtime. [Test suite](../tests/test_plagiarism_data.py)
- The synthetic fixture rehearsed training, deliberate gate rejection, promotion, scoring, and rollback. It verified that rollback restored identical scoring output. [Fixture report](../reports/plagiarism/fixture-workflow/workflow-summary.json)
- The production scoring command loaded registered version 1 and flagged the expected source for a known training passage. Its review score was 0.986. The JSONL includes both excerpts and character offsets for review. [Scoring output](../reports/plagiarism/pan-custom-review.jsonl)
- The completion record and MLflow copy matched the saved held-out metrics. The report SHA-256 is `9a4a6e46a212fb56597a9d2a28a122e91cd6fcb63f3b5d3056ffc0dc16c8a3f7`. [Attempt record](../reports/plagiarism/final_test_attempt.json) · [Held-out run](http://127.0.0.1:5001/#/experiments/3/runs/d5f51ef602e640cfa4749af5d1127b2a)

## Reproduce the development run

Use the archive checksum and training-only import command in the data guide, followed by the isolated runtime, offline smoke check, and training commands in the runbook. This run used Compose project `plagiarism-pan`, MLflow at `127.0.0.1:5001`, and Metaflow run `1795360483306022`. The local PAN manifest SHA-256 is `48b2e868255100eedc1f94eb314ece1ec92ddb07c07a8522be08d4bb8ef2615e`. The held-out partition has been consumed by its one-time evaluation. Use development data for subsequent tuning and reserve a fresh holdout for another final assessment. Run artifacts and the benchmark corpus are stored locally. [Data guide](PLAGIARISM_DATA.md) · [Runbook](PLAGIARISM_RUNBOOK.md) · [Training summary](../reports/plagiarism/training-result.json)
