# SIGIR-2026-Submission-488
Anonymous code repository for peer review


This repository contains a **implementation** of the **ConRAC** used in our anonymized submission.

## 1) Scope (What is included)
- Core Python code for:
  - loading the training dataset
  - chunked document embedding (long-document handling)
  - similarity-based triplet mining
  - exporting generated triplets to CSV
- `requirements.txt`
- a CLI script for running the triplet-generation pipeline

This is an **anonymized review repository** and some paths / environment-specific details are intentionally omitted or generalized.

## 2) Repository structure (minimal)

~~~text
repo/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_io.py
│   ├── chunked_embedding.py
│   ├── triplet_mining.py
│   └── triplet_pipeline_v2.py
└── scripts/
    └── generate_triplets_v2.py
~~~

---

## 3) Paper-context summary (for reviewers)

This code corresponds to the **triplet-generation / retrieval-backbone support stage** in a 3-class security-level classification setting:

- `U` = Unclassified
- `C` = Confidential
- `S` = Secret

In the paper, the downstream system is evaluated on a WikiLeaks diplomatic cable dataset with:

- **9,005 documents total**
- **6,033 training documents** (after preprocessing)
- **2,972 test documents**
- Class proportions approximately:
  - **Unclassified:** 58.9%
  - **Confidential:** 33.8%
  - **Secret:** 7.3%

The paper’s full ConRAC-SE pipeline (not included here) uses a contrastive retrieval backbone plus a deterministic dual-gate selective escalation controller. For reference, the paper reports a retrieval/reranking setup with `K=20` retrieval candidates, `M=5` reranked candidates, and an escalation confidence threshold `τ_g = 0.95`. This minimal repository supports the **triplet-generation stage** used before those downstream components.

---

## 4) Environment and installation

### Recommended environment
- Python 3.10+ (Python 3.x)
- Linux recommended
- GPU A100

### Install dependencies
~~~bash
pip install -r requirements.txt
~~~

### Notes
- `faiss-cpu` is included in `requirements.txt` for portability.
- If your environment uses GPU FAISS, you may replace it accordingly.
- Embedding model weights (e.g., `BAAI/bge-m3`) may be downloaded at runtime depending on your environment/network settings.

---

## 5) Expected input dataset format (IMPORTANT)

This repository expects a **training CSV** containing at least the following columns.

### Required columns
- `Content` : document text (string)
- `label` : class label (string)

### Optional columns
- `source` : source tag for filtering (e.g., `original`, `generated`)
  - If present, you may use:
    - `--source_col source`
    - `--source_value original`

### Accepted label forms (normalized internally)
The loader normalizes common label names as follows:
- `Unclassified` → `U`
- `Confidential` → `C`
- `Secret` → `S`

If your dataset uses different label names, update the normalization map in `src/data_io.py`.

---

## 6) Input data schema details (recommended conventions)

### CSV encoding
- UTF-8 recommended

### Text preprocessing expectations
The pipeline assumes:
- `Content` contains the document text used for embedding
- empty/null texts are removed during dataset preparation
- labels exist for rows used in triplet generation

### Duplicate rows
This minimal implementation does **not** aggressively deduplicate documents.
If your dataset contains duplicates, perform deduplication before running this script (recommended for cleaner triplets).

### Dataset split policy
This repository **does not create train/test splits**.
It generates triplets from **the CSV file you provide** (typically the **training split**).

---

## 7) Minimal input example (illustrative only)

~~~csv
Content,label,source
"Example cable text A ...",U,original
"Example cable text B ...",C,original
"Example cable text C ...",S,original
~~~

---

## 8) What the pipeline does (high level)

The V2 triplet-generation pipeline performs:

1. **Load and validate training data**
2. **Normalize labels** into `U/C/S` where applicable
3. **Optionally filter rows** by `source_col/source_value`
4. **Compute long-document embeddings**
   - chunk long texts
   - encode chunks
   - pool chunk embeddings into a document-level embedding
5. **Build pairwise similarity matrix**
6. **Mine triplets** `(anchor, positive, negative)`
   - positive: same label as anchor
   - negative: different label from anchor
   - negative sampling strategy is configurable (`semi_hard`, `hard`, `random`, `class_aware`)
7. **Export triplets to CSV** for downstream contrastive retriever training

---

## 9) Run the triplet generation script (example)

~~~bash
python scripts/generate_triplets_v2.py \
  --train_csv /path/to/train.csv \
  --out_csv /path/to/wikileaks_triplets_token_based_final.csv \
  --text_col Content \
  --label_col label \
  --model_name BAAI/bge-m3 \
  --n_triplets 5000 \
  --negative_strategy semi_hard \
  --cache_embeddings_path /path/to/wikileaks_embeddings_chunked.npy \
  --batch_size 8 \
  --max_tokens 7500 \
  --overlap_tokens 500 \
  --seed 42 \
  --force_class_balance
~~~

### Optional source filtering example
~~~bash
python scripts/generate_triplets_v2.py \
  --train_csv /path/to/train.csv \
  --out_csv /path/to/triplets.csv \
  --source_col source \
  --source_value original \
  --n_triplets 5000
~~~

---

## 10) CLI arguments (reference)

### Required arguments
- `--train_csv` : path to input training CSV
- `--out_csv` : path to output triplet CSV
- `--n_triplets` : number of triplets to generate

### Common optional arguments
- `--text_col` (default: `Content`)
- `--label_col` (default: `label`)
- `--source_col` / `--source_value` (optional filtering)
- `--model_name` (default: `BAAI/bge-m3`)
- `--batch_size` (default: `8`)
- `--max_tokens` (default: `7500`)
- `--overlap_tokens` (default: `500`)
- `--negative_strategy` (`random`, `hard`, `semi_hard`, `class_aware`)
- `--cache_embeddings_path` (optional `.npy` cache path)
- `--disable_cache` (ignore cache and recompute embeddings)
- `--seed` (default: `42`)
- `--force_class_balance` (enable class-balanced anchor sampling)

---

## 11) Output format (triplet CSV schema)

The generated CSV (`--out_csv`) contains triplet rows with fields such as:

### Index fields
- `anchor_idx`
- `positive_idx`
- `negative_idx`

### Text fields
- `anchor_text`
- `positive_text`
- `negative_text`

### Label fields
- `anchor_label`
- `positive_label`
- `negative_label`

### Similarity diagnostics (if exported)
- `sim_anchor_positive`
- `sim_anchor_negative`

These triplets are intended for downstream contrastive retriever training (e.g., triplet-loss-based fine-tuning).

---

## 12) Reproducibility notes (important)

### Randomness
Triplet sampling includes randomness.  
Use `--seed` to improve reproducibility.

### Embedding cache
Use `--cache_embeddings_path` to store/load `.npy` embeddings and speed up repeated runs.  
Use `--disable_cache` to force a fresh recomputation.

### Runtime and memory
- Pairwise similarity over many documents may require substantial memory.
- GPU improves embedding speed but is not strictly required.
- For quick validation, test with a smaller CSV first.

### Minimal-scope disclaimer
This repository intentionally exposes only the **triplet-generation core** for anonymized review and should not be interpreted as the complete research pipeline.

---

## 13) Data usage / redistribution / privacy notice

- This repository does **not** include raw datasets.
- Users are responsible for obtaining and using data in compliance with the corresponding license/terms.
- Do not place sensitive private documents into this pipeline without proper authorization and governance controls.

---

## 14) Troubleshooting (quick)

### Issue: model download/auth/network problems
- Ensure your environment can access model hosting endpoints.
- If needed, pre-download model weights in your environment.

### Issue: out-of-memory during embedding or similarity computation
- Reduce `--batch_size`
- Run a smaller subset first for validation
- Use `--cache_embeddings_path` for repeated runs
- Use a GPU environment for faster embedding generation (optional)

### Issue: zero or too few triplets generated
- Check label distribution in the input CSV
- Verify `--text_col` and `--label_col`
- Try a different `--negative_strategy`
- Check optional filtering (`--source_col`, `--source_value`) is not too restrictive

---

## 15) Anonymity notice

This repository is anonymized for peer review.  
Identifying metadata, environment-specific paths, credentials, and experiment logs have been removed.

A fuller repository (including additional training/inference components) may be released in a camera-ready or post-review version, subject to data/license constraints.
