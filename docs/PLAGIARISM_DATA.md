# Plagiarism corpus data

## PAN 2013 access and provenance

The benchmark source is PAN 2013 Text Alignment, DOI [10.5281/zenodo.3715980](https://zenodo.org/records/3715980). The Zenodo record identifies `pan13-text-alignment-test-and-training.zip` (28.7 MB) and publishes MD5 `fec7fca7a82193a97fbb97cde7781b63`. PAN's [task specification](https://pan.webis.de/clef13/pan13-web/text-alignment.html) documents the `pairs` comparison list and XML character-offset fields.

Download the archive locally. Keep the benchmark text outside Git and confirm the record’s terms before redistribution. The authored fixture is for offline software checks.

## Prepare a local PAN manifest

Download the release from Zenodo, then verify its published checksum before importing it:

```sh
md5 pan13-text-alignment-test-and-training.zip
```

The output should be `fec7fca7a82193a97fbb97cde7781b63`. Import and assign the held-out test plus seeded development splits:

```sh
python scripts/prepare_plagiarism_data.py /path/to/pan13-text-alignment-test-and-training.zip \
  --output plagiarism_data/pan13.json --seed 903 --validation-fraction 0.2
```

The importer preserves decoded text line endings, hashes every document, validates each XML span against both documents, and restricts candidate pairs to PAN's supplied `pairs` files. Test archives are assigned to the held-out split. Positive links and exact-content duplicates stay grouped; negative links do not join groups, and cross-split negative comparisons are excluded and counted in lineage.

PAN 2013’s official training and test archives share identical documents. Applying the repository’s group leakage check with the full official test set leaves too few eligible development comparisons for training and validation. The training-archive-only protocol creates a custom 60/20/20 split of document groups with seed 903:

```sh
python scripts/prepare_plagiarism_data.py /path/to/pan13-text-alignment-test-and-training.zip \
  --output plagiarism_data/pan13.json --training-archive-only --seed 903 \
  --validation-fraction 0.2 --test-fraction 0.2
```

Metrics from this protocol apply to the custom held-out split. The manifest records the selected archive, source checksum, seed, and split fractions.

After inspecting the manifest and confirming the corpus terms, track the local derived corpus with DVC:

```sh
dvc add plagiarism_data
```

This creates the optional `plagiarism_data.dvc` pointer mounted read-only by the real-data Compose training job. In restricted environments where DVC cannot write its default site cache, set it to a workspace path before DVC commands:

```sh
mkdir -p .dvc/tmp/plagiarism-site-cache
export DVC_SITE_CACHE_DIR="$PWD/.dvc/tmp/plagiarism-site-cache"
dvc add plagiarism_data
dvc push
```

To restore it later, keep the same `DVC_SITE_CACHE_DIR` and run `dvc pull plagiarism_data.dvc`. The pointer is optional for the authored fixture job and should not be mounted for it.

Keep the benchmark corpus in DVC storage rather than Git. A training entry point should read only `train` and `validation` via `load_corpus(path, include_splits=("train", "validation"))`; hold out `test` until the frozen evaluation step.

## Authored offline fixture

`fixtures/plagiarism/fixture.json` is included in the workspace as a small, reproducible same-topic stress fixture. Regenerate it with:

```sh
python scripts/create_plagiarism_fixture.py
```

It includes one positive and one negative candidate for each submission in train, validation, and test. Each negative is a deterministic bag-of-words stress control with the same token multiset as its query and no shared word trigrams. These controls intentionally rank above the positive in the deterministic lexical fixture encoder; they are synthetic demonstration labels, not independently written human answers and not an evaluation benchmark.
