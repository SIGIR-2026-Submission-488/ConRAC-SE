# SIGIR-2026-Submission-488
Anonymous code repository for peer review


This repository contains a **implementation** of the **ConRAC-SE** used in our anonymized submission.

## 1) Scope (What is included)
- Core Python code for:
  - loading the training dataset
  - chunked document embedding (long-document handling)
  - similarity-based triplet mining
  - exporting generated triplets to CSV
- `requirements.txt`
- a CLI script for running the triplet-generation pipeline

This is an **anonymized review repository** and some paths / environment-specific details are intentionally omitted or generalized.

## 2) Repository structure

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

## 3) Paper-context summary 

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

## 5) Expected input dataset format

This repository expects a **training CSV** containing at least the following columns.

### Required columns
- `Content` : document text (string)
- `label` : class label (string)

### Optional columns
- `source` : source tag for filtering (e.g., `original`, `generated`)
  - If present, you may use:
    - `--source_col source`
    - `--source_value original`

### Accepted label forms
The loader normalizes common label names as follows:
- `Unclassified` → `U`
- `Confidential` → `C`
- `Secret` → `S`

If your dataset uses different label names, update the normalization map in `src/data_io.py`.

---

## 6) Reproducibility notes 

### Randomness
Triplet sampling includes randomness.  
Use `--seed` to improve reproducibility.

### Embedding cache
Use `--cache_embeddings_path` to store/load `.npy` embeddings and speed up repeated runs.  
Use `--disable_cache` to force a fresh recomputation.

### Minimal-scope disclaimer
This repository intentionally exposes only the **triplet-generation core** for anonymized review and should not be interpreted as the complete research pipeline.

-

## Anonymity notice

This repository is anonymized for peer review.  
Identifying metadata, environment-specific paths, credentials, and experiment logs have been removed.

A fuller repository (including additional training/inference components) may be released in a camera-ready or post-review version, subject to data/license constraints.
