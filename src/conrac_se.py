from __future__ import annotations

import numpy as np
import pandas as pd
import re
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

# Hybrid class 
class HybridGenerativeRAC:
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

        self.RERANKER_MAX_CHARS = 24000 # Safe limit for Reranker (model constraint)
        self.LLM_MAX_DOC_CHARS = 100000 # High limit for LLM (128k context)
        self.MAX_CTX_CHARS = 100000     # Full context for LLM

    # ---------------------------------
    # Retrieval + Reranking
    # ---------------------------------
    def retrieve_and_rerank(self, query_text, query_emb, top_k=30, top_n=3, force_diversity=True):
        """
        1) Uses FAISS to retrieve top_k candidates.
        2) Uses CrossEncoder (reranker) to score each (query, doc) pair.
        3) Optionally enforces diversity by forcing at least one 'Secret' example.
        4) Returns:
            - best_context: list of dicts {text, label, score}
            - scores: raw logits from reranker for all candidates
            - debug_labels: labels for the selected examples in order
        """

        # A. Retrieval
        dists, indices = self.index.search(query_emb.reshape(1, -1).astype("float32"), top_k)
        candidate_indices = indices[0]

        # B. Prepare Pairs for CrossEncoder
        pairs = []
        valid_indices = []
        q_trunc = query_text[: self.RERANKER_MAX_CHARS]

        for idx in candidate_indices:
            if idx == -1:
                continue
            doc_text = str(self.train_df.iloc[idx]["Content"])[: self.RERANKER_MAX_CHARS]
            pairs.append([q_trunc, doc_text])
            valid_indices.append(idx)

        if not pairs:               # No valid docs
            return [], None, []

        # C. Rerank (raw logits)
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        sorted_args = np.argsort(-scores)

        # D. Selection Strategy (Top-N with optional forced diversity)
        selected_indices = sorted_args[:top_n].tolist()

        if force_diversity:
            top_labels = [self.train_df.iloc[valid_indices[i]]["label"] for i in selected_indices]
            if "Secret" not in top_labels:
                for i in sorted_args[top_n:]:
                    candidate_label = self.train_df.iloc[valid_indices[i]]["label"]
                    if candidate_label == "Secret":
                        selected_indices[-1] = i
                        break

        best_context = []
        debug_labels = []
        for rank_idx in selected_indices:
            original_idx = valid_indices[rank_idx]
            lbl = self.train_df.iloc[original_idx]["label"]
            best_context.append({"text": str(self.train_df.iloc[original_idx]["Content"]), "label": lbl, "score": float(scores[rank_idx])})
            debug_labels.append(lbl)

        return best_context, scores, debug_labels

    # ---------------------------------
    # Prompt Construction
    # ---------------------------------
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
            full_text = ex["text"][: self.LLM_MAX_DOC_CHARS]    # ~25k tokens per example max
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

    # ---------------------------------
    # Hybrid Prediction (Reranker + LLM)
    # ---------------------------------
    def predict_hybrid(self, query_text, query_emb, model_llm, tokenizer_llm, debug=False):
        """
        Hybrid decision pipeline:
        - Step 1: Retrieve + rerank to get a small set of labeled examples.
        - Gate 1: If top-2 labels agree → return that label (Consensus).
        - Gate 2: If top-1 reranker score is extremely confident → return that label.
        - Gate 3: Otherwise, call the LLM with in-context examples to resolve ambiguity.
        """

        # 1. Get context and scores from reranker
        context, all_scores, context_labels = self.retrieve_and_rerank(query_text, query_emb, top_n=3, force_diversity=True)

        if not context:
            return "Unclassified", "Fallback", 0.0, {}

        top1_label = context[0]["label"]
        top1_logit = context[0]["score"]
        top1_abs_prob = 1.0 / (1.0 + np.exp(-top1_logit))

        debug_info = {
            "context_labels": context_labels,
            "top1_logit": top1_logit,
            "top1_prob": top1_abs_prob,
            "consensus": False,
            "raw_llm_response": "",
        }

        # Gate 1: Consensus
        # If Top-2 retrieved examples agree on label, trust that label.
        top_2_labels = [c["label"] for c in context[:2]]
        if len(top_2_labels) == 2 and len(set(top_2_labels)) == 1:
            debug_info["consensus"] = True
            if debug:
                print(f"[Consensus] Labels: {top_2_labels} -> Trusting {top1_label}")
            return top1_label, "Consensus", 1.0, debug_info

        # Gate 2: High confidence
        # If top-1 probability is very high (e.g. > 95%), trust the reranker
        if top1_abs_prob > 0.95:
            if debug:
                print(f"[High Conf] Prob: {top1_abs_prob:.4f} -> Trusting {top1_label}")
            return top1_label, "Reranker_HighConf", top1_abs_prob, debug_info

        # Gate 3: LLM FALLBACK
        if debug:
            print(f"[Ambiguous] Labels: {context_labels}, Top1 Prob: {top1_abs_prob:.2f} -> Calling LLM")

        messages = self.construct_prompt(query_text, context)
        input_ids = tokenizer_llm.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model_llm.device)
        attention_mask = torch.ones_like(input_ids)

        terminators = [tokenizer_llm.eos_token_id]
        try:
            eot_id = tokenizer_llm.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id not in [None, -1]:
                terminators.append(eot_id)
        except Exception:
            pass

        with torch.no_grad():
            outputs = model_llm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=15,
                eos_token_id=terminators,
                do_sample=False,
                pad_token_id=tokenizer_llm.eos_token_id,
            )

        # Decode only generated portion
        response = tokenizer_llm.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True)
        debug_info["raw_llm_response"] = response

        # Parse "Final Answer: [Label]"
        pred_label = "Unknown"
        match = re.search(r"Final Answer:\s*(.*)", response, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip(".'\" ")
            for valid in ["Secret", "Confidential", "Unclassified"]:
                if valid.lower() in extracted.lower():
                    pred_label = valid
                    break

        # If LLM didn't give a valid label, fall back to top-1 reranker label
        if pred_label == "Unknown":
            pred_label = top1_label

        return pred_label, "LLM", 0.0, debug_info


def run_hybrid_evaluation(rac_system: HybridGenerativeRAC, test_df: pd.DataFrame, test_embeddings: np.ndarray, model_llm, tokenizer_llm):
    results = []
    print("Starting Hybrid RAC Evaluation...")

    reranker_calls = 0      # Consensus + Reranker_HighConf
    llm_calls = 0           # LLM
    fallback_calls = 0      # Fallback (no context)

    for idx in tqdm(range(len(test_df))):
        query_text = str(test_df.iloc[idx]["Content"])
        true_label = test_df.iloc[idx]["label"]
        query_emb = test_embeddings[idx]

        # Enable debug print for first N samples
        show_debug = idx < 25

        # Hybrid Predict (note 4 returned values)
        pred_label, source, confidence, debug = rac_system.predict_hybrid(query_text, query_emb, model_llm, tokenizer_llm, debug=show_debug)

        # Count how often each path is used
        if source in ("Consensus", "Reranker_HighConf"):
            reranker_calls += 1
        elif source == "LLM":
            llm_calls += 1
        else:
            fallback_calls += 1

        results.append(
            {
                "idx": idx,
                "true_label": true_label,
                "pred_label": pred_label,
                "source": source,
                "confidence": confidence,
                "top1_prob": debug.get("top1_prob", None),
                "top1_logit": debug.get("top1_logit", None),
                "context_labels": debug.get("context_labels", None),
            }
        )

    print("\nEvaluation Complete.")
    print(f"Reranker used (Consensus + HighConf): {reranker_calls} times")
    print(f"LLM used: {llm_calls} times")
    print(f"Fallback used: {fallback_calls} times")

    return pd.DataFrame(results)