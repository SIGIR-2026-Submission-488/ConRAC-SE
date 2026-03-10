from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer, losses
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.chunked_embedding import ChunkedEmbeddingHandler
from src.data_io import load_triplets_csv, select_hard_triplets


# User requested: dataset path fixed to ../dataset/Retriever/stage2_triplets_filtered.parquet
# Important: Resolve relative to THIS FILE location.
TRIPLETS_PARQUET_DEFAULT = (
    Path(__file__).resolve().parent / "../dataset/Retriever/stage2_triplets_filtered.parquet"
).resolve()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_mem_savers(model: SentenceTransformer) -> SentenceTransformer:
    """
    Minimal: enable gradient checkpointing + disable use_cache when available.
    """
    try:
        first = model._first_module()
        auto_model = getattr(first, "auto_model", None)
        if auto_model is not None:
            try:
                auto_model.gradient_checkpointing_enable()
            except Exception:
                pass
            try:
                auto_model.config.use_cache = False
            except Exception:
                pass
    except Exception:
        pass
    return model


def _load_triplets_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """
    Loads stage2_triplets_filtered.parquet.
    Requires: pyarrow (recommended) or fastparquet.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Stage-2 triplets parquet not found: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet: {parquet_path}\n"
            f"Install pyarrow: pip install pyarrow\n"
            f"Original error: {e}"
        )
    return df


def _build_triplet_examples_from_indices(
    train_df: pd.DataFrame,
    stage2_triplets_df: pd.DataFrame,
    text_col: str = "Content",
    anchor_col: str = "anchor_idx",
    pos_col: str = "positive_idx",
    neg_col: str = "negative_idx",
    max_rows: Optional[int] = None,
) -> List[InputExample]:
    """
    IMPORTANT: train_df must be source=='original' filtered + reset_index(drop=True),
               consistent with the parquet indices.
    """
    for c in [anchor_col, pos_col, neg_col]:
        if c not in stage2_triplets_df.columns:
            raise ValueError(
                f"Parquet missing required column '{c}'. "
                f"Available columns: {list(stage2_triplets_df.columns)}"
            )

    n = len(stage2_triplets_df) if max_rows is None else min(max_rows, len(stage2_triplets_df))
    examples: List[InputExample] = []

    for i in range(n):
        a = int(stage2_triplets_df.iloc[i][anchor_col])
        p = int(stage2_triplets_df.iloc[i][pos_col])
        nidx = int(stage2_triplets_df.iloc[i][neg_col])

        # safety checks
        if a < 0 or p < 0 or nidx < 0:
            continue
        if a >= len(train_df) or p >= len(train_df) or nidx >= len(train_df):
            # If this happens, index alignment is broken.
            raise IndexError(
                "Index out of range while building stage2 examples.\n"
                f"Got (a,p,n)=({a},{p},{nidx}) but train_df has len={len(train_df)}.\n"
                "This indicates train_df filtering/order is NOT aligned with parquet indices."
            )

        anchor_text = str(train_df.iloc[a][text_col])
        pos_text = str(train_df.iloc[p][text_col])
        neg_text = str(train_df.iloc[nidx][text_col])

        examples.append(InputExample(texts=[anchor_text, pos_text, neg_text]))

    return examples


def train_biencoder_from_parquet(
    train_df: pd.DataFrame,
    output_dir: str | Path,
    parquet_path: str | Path = TRIPLETS_PARQUET_DEFAULT,
    base_model_name: str = "BAAI/bge-m3",
    text_col: str = "Content",
    batch_size: int = 8,
    epochs: int = 2,
    lr: float = 1e-5,
    margin: float = 0.2,
    seed: int = 42,
    max_rows: Optional[int] = None,
) -> Path:

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load stage2 triplets parquet
    stage2_df = _load_triplets_parquet(parquet_path)
    print(f"[Stage2-Train] Loaded parquet: {parquet_path} (rows={len(stage2_df)})")

    # 2) Build InputExample triplets from indices
    examples = _build_triplet_examples_from_indices(
        train_df=train_df,
        stage2_triplets_df=stage2_df,
        text_col=text_col,
        max_rows=max_rows,
    )
    print(f"[Stage2-Train] Built {len(examples)} triplet examples")

    if len(examples) == 0:
        raise RuntimeError("No stage2 triplet examples constructed. Check parquet contents.")

    # 3) Init base model
    print(f"[Stage2-Train] Loading base bi-encoder: {base_model_name}")
    model = SentenceTransformer(base_model_name)
    model.max_seq_length = 8192  # notebook uses 8192
    model = _enable_mem_savers(model)

    # 4) Loss + dataloader + fit
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    # TripletLoss (margin=0.2 from notebook)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE_DISTANCE,
        triplet_margin=margin,
    )

    warmup_steps = math.ceil(len(train_loader) * 0.1)

    print(
        f"[Stage2-Train] Start training | epochs={epochs}, bs={batch_size}, lr={lr}, "
        f"margin={margin}, warmup_steps={warmup_steps}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        use_amp=torch.cuda.is_available(),
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    # 5) Verify reload
    _ = SentenceTransformer(str(output_dir))
    print(f"[Stage2-Train] ✓ Saved and reloaded stage2-only bi-encoder -> {output_dir}")
    return output_dir


# ===========================================================
#  ConRAC backbone (reranker FT + retrieval + RAC baseline)
# ===========================================================

def build_reranker_training_pairs(
    hard_triplets: pd.DataFrame,
    max_anchor_chars: int = 2000,
    max_doc_chars: int = 20000,
) -> List[InputExample]:
    """
    (From Stage_2_Only...QWEN.ipynb Cell 10)
    Convert triplets to Cross-Encoder pairs:
      (anchor, positive) -> label 1
      (anchor, negative) -> label 0
    """
    train_examples: List[InputExample] = []
    for _, row in hard_triplets.iterrows():
        anchor = str(row["anchor_text"])[:max_anchor_chars]
        pos_doc = str(row["positive_text"])[:max_doc_chars]
        neg_doc = str(row["negative_text"])[:max_doc_chars]

        train_examples.append(InputExample(texts=[anchor, pos_doc], label=1.0))
        train_examples.append(InputExample(texts=[anchor, neg_doc], label=0.0))
    return train_examples


def finetune_reranker(
    train_examples: List[InputExample],
    output_path: str | Path,
    batch_size: int = 2,
    epochs: int = 2,
    lr: float = 1e-5,
    seed: int = 42,
) -> Path:
    """
    Fine-tune BAAI/bge-reranker-v2-m3 and save artifacts.
    """
    set_seed(seed)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading Cross-Encoder: BAAI/bge-reranker-v2-m3 ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        num_labels=1,
        max_length=8192,
        device=device,
    )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    eval_size = min(200, len(train_examples) // 10) if len(train_examples) else 0
    eval_examples = train_examples[:eval_size] if eval_size > 0 else train_examples[:10]
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_examples, name="dev_eval")

    print(f"\nStarting fine-tuning for {epochs} epochs...")
    warmup_steps = math.ceil(len(train_dataloader) * 0.1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reranker.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        save_best_model=True,
        optimizer_params={"lr": lr},
        use_amp=torch.cuda.is_available(),
    )

    print(f"Forcing save to: {output_path}...")
    reranker.save(str(output_path))
    reranker.model.save_pretrained(str(output_path))
    reranker.tokenizer.save_pretrained(str(output_path))

    files = os.listdir(output_path)
    if "config.json" not in files:
        raise RuntimeError(f"config.json missing after save. Files: {files}")
    print("✓ Model artifacts saved successfully.")
    return output_path


def build_faiss_index(train_embeddings: np.ndarray) -> faiss.Index:
    d = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_embeddings.astype("float32"))
    return index


def encode_train_test_with_biencoder(
    bi_encoder: SentenceTransformer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "Content",
) -> Tuple[np.ndarray, np.ndarray]:

    tokenizer = getattr(bi_encoder, "tokenizer", None)
    if tokenizer is None:
        # fallback to a reasonable default; this keeps execution from breaking
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    handler = ChunkedEmbeddingHandler(bi_encoder, tokenizer)
    train_texts = train_df[text_col].fillna("").astype(str).tolist()
    test_texts = test_df[text_col].fillna("").astype(str).tolist()
    E_train = handler.encode_batch(train_texts)
    E_test = handler.encode_batch(test_texts)
    return E_train, E_test


class GenerativeRAC:
    """
    Retrieval (FAISS) + Cross-Encoder rerank + LLM decoding (always-on).
    """

    def __init__(self, bi_encoder_path, reranker_path, train_df, train_embeddings, index):
        self.train_df = train_df
        self.train_embeddings = train_embeddings
        self.index = index

        print("Loading Bi-Encoder...")
        self.bi_encoder = SentenceTransformer(bi_encoder_path)
        self.bi_encoder.max_seq_length = 8192

        print("Loading Fine-Tuned Reranker...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(reranker_path, device=device)

        self.RERANKER_MAX_CHARS = 24000
        self.LLM_MAX_DOC_CHARS = 100000

    def retrieve_and_rerank(self, query_text, query_emb, top_k=20, top_n=3):
        dists, indices = self.index.search(query_emb.reshape(1, -1).astype("float32"), top_k)
        candidate_indices = indices[0]

        pairs = []
        valid_indices = []
        query_for_rerank = query_text[: self.RERANKER_MAX_CHARS]

        for idx in candidate_indices:
            if idx == -1:
                continue
            full_doc_text = str(self.train_df.iloc[idx]["Content"])
            doc_for_rerank = full_doc_text[: self.RERANKER_MAX_CHARS]
            pairs.append([query_for_rerank, doc_for_rerank])
            valid_indices.append(idx)

        if not pairs:
            return []

        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        sorted_args = np.argsort(-scores)[:top_n]

        best_context = []
        for rank_idx in sorted_args:
            original_idx = valid_indices[rank_idx]
            score = scores[rank_idx]
            best_context.append(
                {
                    "text": str(self.train_df.iloc[original_idx]["Content"]),
                    "label": self.train_df.iloc[original_idx]["label"],
                    "score": float(score),
                }
            )
        return best_context

    def construct_prompt(self, query_text, context_examples):
        system_msg = (
            "You are an expert intelligence analyst specializing in diplomatic cables. "
            "Classify the input into Confidential, Secret, Unclassified, and return the answer as the corresponding label.\n"
            "Consider the following criteria for classification:\n"
            "1. 'Confidential': the text contains sensitive information that should be restricted to certain individuals or groups. \n"
            "2. 'Secret': the text involves highly sensitive or proprietary information that must be encrypted or handled with strict security measures.\n"
            "3. 'Unclassified': the text contains information that is publicly accessible or non-sensitive in nature."
        )

        examples_str = ""
        for i, ex in enumerate(context_examples):
            full_text = ex["text"][: self.LLM_MAX_DOC_CHARS]
            examples_str += f"--- Example {i+1} (Label: {ex['label']}) ---\n{full_text}\n\n"

        user_msg = (
            "Here are similar historical cables to guide you. Use these to understand the classification pattern:\n\n"
            f"{examples_str}"
            "========================================\n"
            "--- TARGET DOCUMENT TO CLASSIFY ---\n"
            f"{query_text[:self.LLM_MAX_DOC_CHARS]}\n\n"
            "========================================\n"
            "Based on the examples above, classify the Target Document.\n"
            "Provide NO reasoning. Output ONLY the label in this format: 'Final Answer: [Label]'"
        )

        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def load_model(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model.to("cpu")

    print("✓ Qwen loaded successfully!")
    return model, tokenizer


def run_inference(model, tokenizer, messages, max_new_tokens: int = 20) -> str:
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)

    terminators = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id not in [None, -1]:
            terminators.append(eot_id)
    except Exception:
        pass

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True)
    return response


def run_full_evaluation(rac_system: GenerativeRAC, test_df: pd.DataFrame, test_embeddings: np.ndarray, model, tokenizer):
    import re

    results = []
    print("Starting RAC Evaluation using qwen...")

    for idx in tqdm(range(len(test_df))):
        query_text = str(test_df.iloc[idx]["Content"])
        true_label = test_df.iloc[idx]["label"]
        query_emb = test_embeddings[idx]

        context = rac_system.retrieve_and_rerank(query_text, query_emb, top_k=20, top_n=3)
        messages = rac_system.construct_prompt(query_text, context)

        try:
            raw_response = run_qwen_inference(model, tokenizer, messages)

            pred_label = "Unknown"
            clean_resp = raw_response.strip()
            match = re.search(r"Final Answer:\s*(.*)", clean_resp, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip(".'\" ")
                for valid in ["Secret", "Confidential", "Unclassified"]:
                    if valid.lower() in extracted.lower():
                        pred_label = valid
                        break

            results.append({"idx": idx, "true_label": true_label, "pred_label": pred_label, "raw_response": raw_response})
        except Exception as e:
            results.append({"idx": idx, "true_label": true_label, "pred_label": "Error", "raw_response": str(e)})

    return pd.DataFrame(results)