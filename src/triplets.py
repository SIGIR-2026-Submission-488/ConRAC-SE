from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from src.chunked_embedding import ChunkedEmbeddingHandler

warnings.filterwarnings("ignore")


def initialize_bge_m3(model_name: str = "BAAI/bge-m3") -> SentenceTransformer:
    """
    Initialize BGE-M3 with proper settings
    """
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    model.max_seq_length = 8192
    print("✓ Model loaded")
    print(f"✓ Max sequence length: {model.max_seq_length}")
    print(f"✓ Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


class WikiLeaksTripletGeneratorAdvanced:
    """
    Complete triplet generator with chunking support.
    """

    def __init__(self, df, model, tokenizer, text_col="Content", label_col="label"):
        self.df = df.reset_index(drop=True)
        self.model = model
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col

        # Initialize chunked encoder
        self.chunked_encoder = ChunkedEmbeddingHandler(model, tokenizer)

        # Label information
        self.labels = df[label_col].unique()
        self.label_counts = df[label_col].value_counts().to_dict()

        # Create label to indices mapping
        self.label_to_indices = {}
        for label in self.labels:
            self.label_to_indices[label] = df[df[label_col] == label].index.tolist()

        # Class weights for balancing
        total = len(df)
        self.class_weights = {}
        for label, count in self.label_counts.items():
            self.class_weights[label] = total / (len(self.labels) * count)

        print("=" * 60)
        print("TRIPLET GENERATOR INITIALIZED")
        print("=" * 60)
        print(f"Classes: {list(self.labels)}")
        print("\nClass distribution:")
        for label, count in self.label_counts.items():
            pct = count / total * 100
            weight = self.class_weights[label]
            print(f"  {label}: {count} samples ({pct:.1f}%) - weight: {weight:.2f}")

        self.embeddings = None

    def compute_embeddings(self, cache_path: str | Path, use_cache: bool = True):
        """
        Compute embeddings with chunking for long documents.
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if use_cache and cache_path.exists():
            print("Loading cached embeddings...")
            self.embeddings = np.load(cache_path)
            print(f"✓ Loaded embeddings: {self.embeddings.shape}")
        else:
            print("\nComputing embeddings with chunking strategy...")
            print("This will handle long documents automatically")

            texts = [str(text) if text else "" for text in self.df[self.text_col].values]   # Get all texts
            self.embeddings = self.chunked_encoder.encode_batch(texts, batch_size=8)        # Encode with chunking

            np.save(cache_path, self.embeddings)                                            # Save cache
            print(f"✓ Embeddings computed and cached: {self.embeddings.shape}")

        return self.embeddings

    def compute_similarities(self, anchor_idx, candidate_indices):
        if self.embeddings is None:
            raise ValueError("Compute embeddings first!")

        anchor_emb = self.embeddings[anchor_idx : anchor_idx + 1]
        candidate_embs = self.embeddings[candidate_indices]
        similarities = cosine_similarity(anchor_emb, candidate_embs)[0]
        return similarities

    def select_semihard_negative(self, anchor_idx, anchor_label, positive_idx, margin=0.2):
        # Get positive similarity
        pos_sim = self.compute_similarities(anchor_idx, [positive_idx])[0]

        # Get all negative candidates with priority
        priority_map = {
            "Secret": ["Confidential"],         # Most confusing
            "Confidential": ["Secret"],         # Most confusing
            "Unclassified": ["Confidential"],
        }

        # Collect negatives with priority
        priority_labels = priority_map.get(anchor_label, [])
        priority_candidates = []
        other_candidates = []

        for label in self.labels:
            if label != anchor_label:
                indices = self.label_to_indices[label]
                if label in priority_labels:
                    priority_candidates.extend(indices)
                else:
                    other_candidates.extend(indices)

        all_candidates = priority_candidates + other_candidates

        if not all_candidates:
            return None

        # Compute similarities
        similarities = self.compute_similarities(anchor_idx, all_candidates)

        # Semi-hard criteria
        lower = max(0, pos_sim - margin)
        upper = min(1, pos_sim + margin)

        # Find semi-hard negatives
        semihard_mask = (similarities > lower) & (similarities < upper)

        # Priority for confusing pairs
        n_priority = len(priority_candidates)

        if semihard_mask[:n_priority].any():            # Semi-hard from priority classes
            priority_semihard = np.where(semihard_mask[:n_priority])[0]
            selected_idx = np.random.choice(priority_semihard)
            return all_candidates[selected_idx]
        elif semihard_mask.any():                       # Any semi-hard
            semihard_indices = np.where(semihard_mask)[0]
            selected_idx = np.random.choice(semihard_indices)
            return all_candidates[selected_idx]
        else:
            safe_mask = similarities < 0.85             # No semi-hard found - select hardest safe negative
            if safe_mask.any():
                hardest_safe = np.argmax(similarities[safe_mask])
                return all_candidates[np.where(safe_mask)[0][hardest_safe]]
            else:
                return np.random.choice(all_candidates)

    def generate_triplets(self, strategy="semi-hard", triplets_per_anchor=3):
        if strategy in ["semi-hard", "hard"] and self.embeddings is None:
            raise ValueError("Embeddings are required for semi-hard/hard triplets. Call compute_embeddings first.")

        triplets = []

        # Triplets per class (balanced)
        triplet_multipliers = {
            "Secret": 3,            # 3x for minority
            "Confidential": 2,      # 2x for medium
            "Unclassified": 1,      # 1x for majority
        }

        print("\n" + "=" * 60)
        print(f"GENERATING {strategy.upper()} TRIPLETS")
        print("=" * 60)

        for label in self.labels:
            indices = self.label_to_indices[label]
            multiplier = triplet_multipliers[label]
            n_triplets = triplets_per_anchor * multiplier

            # Sample anchors
            if label == "Secret":
                anchor_indices = indices
            else:
                max_anchors = min(len(indices), len(self.label_to_indices.get("Secret", indices)) * 2)
                anchor_indices = random.sample(indices, min(max_anchors, len(indices)))

            print(f"\n{label}:")
            print(f"  Anchors: {len(anchor_indices)}")
            print(f"  Triplets per anchor: {n_triplets}")
            print(f"  Expected triplets: {len(anchor_indices) * n_triplets}")

            for anchor_idx in tqdm(anchor_indices, desc=f"Processing {label}"):
                # Get positive candidates
                pos_candidates = [i for i in self.label_to_indices[label] if i != anchor_idx]
                if not pos_candidates:
                    continue

                for _ in range(n_triplets):
                    positive_idx = random.choice(pos_candidates)

                    if strategy == "semi-hard":
                        negative_idx = self.select_semihard_negative(anchor_idx, label, positive_idx, margin=0.2)
                    elif strategy == "hard":
                        negative_idx = self.select_semihard_negative(anchor_idx, label, positive_idx, margin=0.05)
                    else:  # random
                        neg_labels = [l for l in self.labels if l != label]
                        neg_label = random.choice(neg_labels)
                        negative_idx = random.choice(self.label_to_indices[neg_label])

                    if negative_idx is not None:
                        triplets.append(
                            {
                                "anchor_idx": anchor_idx,
                                "positive_idx": positive_idx,
                                "negative_idx": negative_idx,
                                "anchor_label": label,
                                "negative_label": self.df.iloc[negative_idx][self.label_col],
                                "strategy": strategy,
                            }
                        )

        print(f"\n✓ Total triplets generated: {len(triplets)}")
        return triplets


def analyze_triplet_quality(triplets, generator: WikiLeaksTripletGeneratorAdvanced):
    """
    Analyze the quality of generated triplets
    """
    df = pd.DataFrame(triplets)

    print("=" * 60)
    print("TRIPLET QUALITY ANALYSIS")
    print("=" * 60)

    # Distribution by anchor class
    print("\n1. Anchor class distribution:")
    anchor_dist = df["anchor_label"].value_counts()
    for label, count in anchor_dist.items():
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")

    # Negative selection patterns
    print("\n2. Negative selection patterns:")
    confusion = pd.crosstab(df["anchor_label"], df["negative_label"], normalize="index") * 100
    print(confusion.round(1))

    # Sample similarity analysis
    print("\n3. Similarity statistics (sample of 500):")
    sample = df.sample(min(500, len(df)))

    pos_sims = []
    neg_sims = []
    for _, row in sample.iterrows():
        a_emb = generator.embeddings[row["anchor_idx"]]
        p_emb = generator.embeddings[row["positive_idx"]]
        n_emb = generator.embeddings[row["negative_idx"]]

        pos_sim = cosine_similarity([a_emb], [p_emb])[0][0]
        neg_sim = cosine_similarity([a_emb], [n_emb])[0][0]

        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)

    print(f"   Anchor-Positive similarity: {np.mean(pos_sims):.3f} (±{np.std(pos_sims):.3f})")
    print(f"   Anchor-Negative similarity: {np.mean(neg_sims):.3f} (±{np.std(neg_sims):.3f})")
    print(f"   Average margin: {np.mean(pos_sims) - np.mean(neg_sims):.3f}")

    # Check for valid triplets
    valid = np.array(pos_sims) > np.array(neg_sims)
    print(f"\n4. Valid triplets (pos_sim > neg_sim): {valid.sum()}/{len(valid)} ({valid.mean()*100:.1f}%)")

    return df


def save_triplets_to_csv(triplets, generator: WikiLeaksTripletGeneratorAdvanced, output_path: str | Path):
    """
    Save triplets with text (truncated for CSV)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    print("Preparing triplet data for saving...")
    for triplet in tqdm(triplets):
        anchor_row = generator.df.iloc[triplet["anchor_idx"]]
        pos_row = generator.df.iloc[triplet["positive_idx"]]
        neg_row = generator.df.iloc[triplet["negative_idx"]]

        data.append(
            {
                # Indices
                "anchor_idx": triplet["anchor_idx"],
                "positive_idx": triplet["positive_idx"],
                "negative_idx": triplet["negative_idx"],

                # Labels
                "anchor_label": triplet["anchor_label"],
                "negative_label": triplet["negative_label"],

                # Text (truncated for CSV - first 3000 chars) -- keeping full
                # 'anchor_text': str(anchor_row['Content'])[:3000],
                # 'positive_text': str(pos_row['Content'])[:3000],
                # 'negative_text': str(neg_row['Content'])[:3000],
                "anchor_text": str(anchor_row["Content"]),
                "positive_text": str(pos_row["Content"]),
                "negative_text": str(neg_row["Content"]),

                # Strategy
                "strategy": triplet["strategy"],
                
                # Metadata
                "anchor_word_len": anchor_row.get("word_len", 0),
                "positive_word_len": pos_row.get("word_len", 0),
                "negative_word_len": neg_row.get("word_len", 0),
            }
        )

    final_df = pd.DataFrame(data)
    final_df.to_csv(output_path, index=False)

    print(f"\nSaved {len(final_df)} triplets to {output_path}")
    try:
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    except OSError:
        pass

    return final_df