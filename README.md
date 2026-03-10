# SIGIR-2026-Submission-488
Anonymous code repository for peer review


This repository contains a **implementation** of the **ConRAC-SE** used in our anonymized submission.

## 1) Scope (What is included)
- Core Python code for:
  - loading the dataset (with consistent indexing)
  - long-document chunked embedding
  - Stage-2 bi-encoder retriever fine-tuning (FT retriever)
  - cross-encoder reranker fine-tuning (FT reranker)
  - ConRAC inference (Top-K retrieval → Top-M reranking)
  - ConRAC-SE Hybrid evaluation (gated selective escalation with Qwen)
- `requirements.txt`
- runnable scripts for triplet generation, training, and evaluation

This is an **anonymized review repository** and some paths / environment-specific details are intentionally omitted or generalized.

## 2) Repository structure

~~~text
repo/
├── README.md
├── requirements.txt
├── dataset/
│   ├── Base_Dataset/
│   │   ├── Copy of Combined_OG_New_Gen.csv
│   │   └── test.csv
│   ├── Embeddings/
│   │   └── wikileaks_embeddings_chunked.npy
│   ├── Retriever/
│   │   └── stage2_triplets_filtered.parquet
│   ├── Re-Ranker/
│   │   └── wikileaks_triplets_token_based_final.csv
├── src/
│   ├── __init__.py
│   ├── data_io.py
│   ├── chunked_embedding.py
│   ├── triplets.py
│   ├── conrac_backbone.py
│   └── conrac_se_hybrid.py
└── scripts/
    ├── 01_generate_triplets.py
    ├── 02_train_reranker.py
    └── 03_run_hybrid_eval.py
~~~



## 3) Dataset
### Dataset download and path configuration (IMPORTANT)

This repository follows a **minimal-modification policy** to preserve the original notebook logic while making the code runnable in a standard repo setting.  
As a result, **all dataset files are referenced by fixed relative paths** under the repo root (i.e., no Google Drive/Colab paths are used in code).

### 1. Download the dataset
You can download the dataset package from the following Google Drive folder:

```text
https://drive.google.com/drive/folders/1wExY6YOtq9qxOq2HBgYhcrz5bdR2WA-t?usp=sharing
```

After downloading, place the dataset/ directory at the same level as src/ (repo root), following the structure below.

### 2. Required dataset folder structure and paths
The code expects the following paths to exist:

~~~text
dataset/
├── Base_Dataset/
│   ├── Copy of Combined_OG_New_Gen.csv
│   └── test.csv
├── Embeddings/
│   └── wikileaks_embeddings_chunked.npy
├── Retriever/
│   └── stage2_triplets_filtered.parquet
└── Re-Ranker/
    └── wikileaks_triplets_token_based_final.csv
~~~

> Note: Some files (e.g., `.npy` cache) may be created automatically when you run the scripts.  
> Other files must be present beforehand (e.g., train/test CSV splits and pre-mined triplets).

### 3. How paths are resolved in code

All paths are resolved relative to the repository root, not absolute machine paths and not Drive/Colab paths.



## 4) Paper-context summary 

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

For reference, the paper reports a retrieval/reranking setup with `K=20` retrieval candidates, `M=5` reranked candidates, and a gating threshold `τ_g = 0.95`.




## 5) Environment and installation

### Recommended environment
- Python 3.10+ (Python 3.x)
- Linux recommended
- GPU recommended (A100-class is ideal for speed, not required)

### Install dependencies
~~~bash
pip install -r requirements.txt
~~~

### Notes
- `faiss-cpu` is included in `requirements.txt` for portability.
- If your environment uses GPU FAISS, you may replace it accordingly.
- Embedding model weights (e.g., `BAAI/bge-m3``BAAI/bge-reranker-v2-m3`, `Qwen/Qwen2.5-7B-Instruct`) may be downloaded at runtime.



## 6) Expected input dataset format

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

### Index alignment requirement (critical)
The Stage-2 retriever triplet file:

- `dataset/Retriever/stage2_triplets_filtered.parquet`

contains `anchor_idx / positive_idx / negative_idx` indices that are assumed to be based on:

1) `train_df = train_df[train_df["source"] == "original"]` (if `source` exists), then  
2) `train_df = train_df.reset_index(drop=True)`

This filtering + reset is applied consistently in the provided scripts before training/evaluation.



## 7) How to run (3 steps)

### Step 1: Generate V2 triplets (for reranker fine-tuning)
This creates/overwrites:
- `dataset/Re-Ranker/wikileaks_triplets_token_based_final.csv`
and uses/creates:
- `dataset/Embeddings/wikileaks_embeddings_chunked.npy`

~~~bash
python scripts/01_generate_triplets_v2.py
~~~

### Step 2: Train ConRAC backbone (FT retriever + FT reranker)
This trains:
- Stage-2 bi-encoder retriever from `dataset/Retriever/stage2_triplets_filtered.parquet`
  - output: `models/bge-m3-rac-stage2-only/`
- cross-encoder reranker from `dataset/Re-Ranker/wikileaks_triplets_token_based_final.csv`
  - output: `outputs/models/bge-reranker-finetuned/`

~~~bash
python scripts/02_train_retriever_reranker.py
~~~

### Step 3: Evaluate (ConRAC baseline + ConRAC-SE Hybrid)
This runs:
- ConRAC decoding baseline (always-on LLM decoding)
- ConRAC-SE Hybrid (gated selective escalation)

Outputs:
- `outputs/results/rac_qwen_finetuned_results.csv`
- `outputs/results/hybrid_qwen_results.csv`

~~~bash
python scripts/03_run_hybrid_eval_qwen.py
~~~



## 8) Reproducibility notes

### Randomness
- Triplet sampling includes randomness.
- Use `--seed` to improve reproducibility.

### Embedding cache
- Use `--cache_embeddings_path` to store/load `.npy` embeddings and speed up repeated runs.  
- Use `--disable_cache` to force a fresh recomputation.

### Minimal-scope disclaimer
- This repository intentionally exposes only the **triplet-generation core** for anonymized review and should not be interpreted as the complete research pipeline.


## Anonymity notice

This repository is anonymized for peer review.  
Identifying metadata, environment-specific paths, credentials, and experiment logs have been removed.

A fuller repository (including additional components) may be released post-review, subject to data/license constraints.

A fuller repository (including additional training/inference components) may be released in a camera-ready or post-review version, subject to data/license constraints.
