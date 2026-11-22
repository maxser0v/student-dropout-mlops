# Passage-level plagiarism review implementation plan

## Primary goal

A professor can run a small batch of English submissions and review flagged passages alongside their matching sources. The assignment demonstrates how to reproduce training, measure false alerts, approve a model version, and roll back a release.

The experiment asks whether XGBoost reduces false alerts on similar passages while preserving the ability to find annotated text reuse. MiniLM retrieves candidate passages and stays frozen. XGBoost is trained on features from those candidates. The demo uses a fixed source collection and returns evidence for human review.

## Phase 0 — Verify dependencies and corpus access

**How this serves the primary goal.** Confirm that the proposed data and models can run in the project's Python 3.11 and ARM64 Docker environment.

- Use the current Metaflow graph and MLflow service as integration patterns. Add a separate experiment and registered model called `passage-reuse`, along with a separate configuration and DVC pointer. See [training flow](../flows/train_flow.py#L279), [Compose services](../compose.yaml), and [environment guide](ENVIRONMENT.md).
- Fetch and inspect the 28.7 MB PAN 2013 Text Alignment release `pan13-text-alignment-test-and-training.zip` (published MD5 `fec7fca7a82193a97fbb97cde7781b63`). It contains separate training and test ZIPs. Record the directory layout, encoding, annotation format, and redistribution terms before bundling any corpus files. Confirm there are enough disjoint document groups and usable reuse spans. The main PAN 2014 evaluation reused PAN 2013, so group those releases together if using any overlap and verify any alternative corpus before substituting it. [PAN 2013 release](https://zenodo.org/records/3715980) · [PAN 2013 format](https://pan.webis.de/clef13/pan13-web/text-alignment.html) · [PAN 2014 overview, §3.3](https://downloads.webis.de/pan/publications/papers/potthast_2014a.pdf)
- Resolve CPU-compatible Sentence Transformers, PyTorch, XGBoost, Metaflow, and PyYAML in a separate Python 3.11 runtime target with a pytest-only CI group. Preserve the explicit PyTorch CPU package index in pip hash exports (`uv export --emit-index-url`) and verify a clean `linux/arm64` build and imports. Pin the exact `sentence-transformers/all-MiniLM-L6-v2` revision and stage only required tokenizer/configuration files plus one PyTorch weight file; verify required-file and SHA-256 manifests before offline loading. Keep a deterministic authored fixture for pull-request checks with no benchmark download, plus a real-encoder smoke check with staged weights. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)

**Done when** the corpus is readable, one annotated span maps to the expected text, and MiniLM encodes a short passage in the target container with network access disabled.

## Phase 1 — Version and validate labeled examples

**How this serves the primary goal.** A training run can show where every label came from and reproduce the same evaluation split.

- Create `src/plagiarism/data.py` to import PAN text and XML. Decode with the verified corpus encoding and preserve original line endings. Record document IDs, content hashes, role, source links, and character offsets. Check that each annotation points to existing documents and bounded spans. Keep the original character positions through chunking so reported excerpts can be recovered exactly.
- Create a dedicated DVC-tracked `plagiarism_data/` directory and split manifest. Group linked source and submission documents, including exact duplicates by content hash, before assigning splits. Hold out the supplied test set, check for cross-release duplicates and linked groups, and exclude any conflicting groups from development. Split the remainder into train and validation with a fixed seed and verify zero document-group overlap. Use a small authored fixture to exercise ingestion and scoring independently of the benchmark. Follow the existing [DVC verification pattern](../src/dropout/lineage.py#L33).
- Label pairs positive when both chunks meet the configured overlap rule for the same annotated reuse link. Exclude ambiguous boundary cases. Sample high-similarity nonmatches for training from annotated document pairs and review a sample for missed labels. For the PAN benchmark, restrict each query's search to the source documents paired with it in PAN, since other pairings lack judgments. Keep validation and test candidates at their natural retrieval mix. Use a reviewed same-topic set to check false alerts separately. Sheffield short answers are an option if accessible. With answer-level labels, report the share of independently written answers receiving any flag. [Sheffield corpus study](https://eprints.whiterose.ac.uk/id/eprint/42922/)

**Done when** a DVC restore reproduces the hashes, a known annotation yields the expected positive pair, and invalid offsets or group leakage stop the run.

## Phase 2 — Retrieve candidates and train XGBoost

**How this serves the primary goal.** Create a repeatable training run that learns to prioritize the matches MiniLM finds.

- Add `src/plagiarism/retrieval.py`, `features.py`, and `training.py`. Chunk text into overlapping windows with a starting target of 180 tokenizer tokens and 40 tokens of overlap. Verify the actual token count fits MiniLM's 256-token limit, including special tokens. Encode in batches on CPU using `SentenceTransformer(..., device="cpu", revision=..., local_files_only=True)` and `encode(..., normalize_embeddings=True)`. Use exact top-10 cosine retrieval with a stable tie-break. Cache source vectors by source manifest hash, encoder revision, and chunking configuration. [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html) · [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Fit TF-IDF and rare-term statistics on training text. Use an ordered feature list containing MiniLM cosine, TF-IDF cosine, shared word n-gram overlap, rare-term overlap, and length ratio. Train `XGBClassifier` using a fixed seed, a small tree budget, and CPU histogram training. Use `predict_proba` to obtain the positive-class score. Share feature generation between training and scoring. [TF-IDF API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) · [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- Add `flows/train_plagiarism_flow.py` and `config/plagiarism.yaml`. Run `start → validate → prepare → train → evaluate → gate → register → end`. Prepare and train from the training split, then construct validation candidates for the evaluation step. Use Metaflow artifacts for handoffs and one MLflow run to record data, source revision, encoder revision, split, feature, and benchmark information. Save timings for cold indexing and cached batch scoring. Include any new root build and lock files in the source provenance manifest and omit the corpus and caches from the Docker build context. [Metaflow linear flow and artifacts](https://docs.metaflow.org/metaflow/basics) · [Existing training flow](../flows/train_flow.py#L279) · [Source provenance](../scripts/demo_provenance.py)

**Done when** two fixture runs produce the same candidate IDs, feature order, and scores within tolerance, and changes to the corpus or encoder invalidate the corresponding cache.

## Phase 3 — Compare baselines and gate releases

**How this serves the primary goal.** Show whether the trained classifier improves the shortlist of passages a professor would review.

- Compare TF-IDF plus cosine retrieval, MiniLM plus cosine retrieval, and MiniLM plus XGBoost. Use the same top-10 MiniLM candidates for its cosine and XGBoost variants. Record retrieval Recall@10 as the share of annotated reuse links reached before scoring. For review metrics, merge overlapping flags for the same submission and source, match merged alerts to annotated links once, and count missed links from retrieval against end-to-end recall.
- Record precision, recall, and false alerts per 100 submissions. Select thresholds on validation only, using a shared recall target that is feasible for the corpus. Freeze the thresholds before testing. Use a configurable release rule requiring a strict reduction in false-alert rate against MiniLM cosine, at most a five-percentage-point drop in recall, and successful data and artifact checks. Treat five points as an assignment starting value and record the final rule before evaluating the candidate. If there is no valid operating point, record a failed gate.
- Log the model with `mlflow.xgboost.log_model` and load it with `mlflow.xgboost.load_model`. Register passing candidates from the logged model URI and keep failed candidates as tracked runs with a gate report. Hold the final test for one evaluation of the selected, frozen specification. Report the raw sample counts and identify the evaluation as conditional on the document pairs supplied by PAN. [MLflow XGBoost API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.xgboost.html) · [Existing gate pattern](../src/dropout/gates.py)

**Done when** MLflow shows all three results, the validation thresholds, the retrieval misses, and a candidate that failed the gate without creating a registered version.

## Phase 4 — Score in Docker and demonstrate rollback

**How this serves the primary goal.** Run the exact approved version on a batch and trace each flag back to its source and release.

- Add `python -m plagiarism.score` and corresponding Compose jobs. Resolve `champion` once per batch or accept a concrete version. Verify that the model, threshold, feature schema, fitted text transforms, encoder snapshot, and source index match the versioned bundle. Search the fixed demo collection using only the submission text and the verified bundle. Return submission and source IDs, offsets and excerpts for both passages, review score, model version, and source index version in JSON Lines. Use the current non-root, read-only input mount patterns. [Current Compose services](../compose.yaml)
- Follow the existing release scripts' explicit promotion and rollback pattern. Use a separate release history for `passage-reuse`. Run the demo in an isolated Compose project and tracking volume. Demonstrate rollback with two passing versions on a deterministic fixture if fewer than two benchmark candidates pass. Log aggregate operational metrics such as chunk count, embedding time, score distribution, and flag rate. Keep submission excerpts in the review output. [Promotion script](../scripts/promote_model.py) · [Rollback script](../scripts/rollback_model.py) · [MLflow registry aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

**Done when** a batch scored before promotion matches the batch scored after rollback, and the review record identifies the exact model and source index for every returned flag.

## Phase 5 — Verify and rehearse the demo

**How this serves the primary goal.** Provide a walkthrough another person can reproduce from setup to scored evidence.

- Run the existing dropout checks and add focused coverage for split leakage, character offsets, training/scoring feature parity, threshold and gate behavior, cache invalidation, and pinned-version loading. Add an offline fixture flow to CI and run a separate real MiniLM smoke test in the target container.
- Document corpus setup, training, the held-out evaluation, scoring, promotion, and rollback in the [plagiarism runbook](PLAGIARISM_RUNBOOK.md). CI runs the fixture flow twice using actual XGBoost, forces one validation-gate failure, promotes passing concrete versions, scores a fixture query, rolls back, and asserts byte-identical JSONL after restoring the earlier version. The authored fixture is synthetic software-check data and must not be reported as benchmark performance. Present PAN false-alert and recall counts only after the real corpus and one-time held-out evaluation are available.

**Done when** a fresh run can restore the corpus, reproduce the reported evaluation within tolerance, and demonstrate the full release workflow with versioned artifacts.
