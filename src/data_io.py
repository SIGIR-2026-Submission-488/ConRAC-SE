from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer


@dataclass(frozen=True)
class DatasetPaths:
    """
    Assumed folder layout (repo root):
      dataset/
        Base_Dataset/
          Combined_OG_New_Gen.csv           # train
          test.csv                          # test 
        Embeddings/
          wikileaks_embeddings_chunked.npy
        Retriever/
          triplets_filtered.parquet
        Re-Ranker/
          wikileaks_triplets_token_based.csv
    """
    repo_root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.repo_root / "dataset"

    @property
    def train_csv(self) -> Path:
        return self.dataset_dir / "Base_Dataset" / "Combined_OG_New_Gen.csv"

    @property
    def test_csv(self) -> Path:
        return self.dataset_dir / "Base_Dataset" / "test.csv"

    @property
    def embeddings_cache(self) -> Path:
        return self.dataset_dir / "Embeddings" / "wikileaks_embeddings_chunked.npy"

    @property
    def triplets_csv(self) -> Path:
        return self.dataset_dir / "Re-Ranker" / "wikileaks_triplets_token_based.csv"

    @property
    def retriever_triplets_parquet(self) -> Path:
        return self.dataset_dir / "Retriever" / "triplets_filtered.parquet"


def resolve_repo_root() -> Path:
    # repo/src/data_io.py -> parents[1] is repo root
    return Path(__file__).resolve().parents[1]


def load_and_analyze_data(
    train_path: str | Path,
    test_path: str | Path,
    tokenizer_name: str = "BAAI/bge-m3",
    filter_source_original: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, "AutoTokenizer"]:
    """
    Loads train/test CSV and prints long-document statistics.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {test_path}\n"
            "Expected location: dataset/Base_Dataset/test.csv (or pass your own path)."
        )

    train_df = pd.read_csv(train_path)
    if filter_source_original and "source" in train_df.columns:
        train_df = train_df[train_df["source"] == "original"].reset_index(drop=True)

    test_df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    if "label" in train_df.columns:
        print("\nTraining label distribution:")
        print(train_df["label"].value_counts())

    print("\n" + "=" * 60)
    print("LONG DOCUMENT ANALYSIS (>8192 tokens)")
    print("=" * 60)

    for df_name, df in [("Train", train_df), ("Test", test_df)]:
        long_docs = []
        for idx, row in df.iterrows():
            text = str(row.get("Content", "")) if row.get("Content", "") else ""
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            if len(tokens) > 8192:
                long_docs.append({"idx": idx, "label": row.get("label", ""), "tokens": len(tokens)})

        if long_docs:
            long_df = pd.DataFrame(long_docs)
            print(f"\n{df_name} set:")
            print(f"  Total long docs: {len(long_df)}")
            if "label" in long_df.columns and "label" in df.columns:
                print("  By class:")
                for label in long_df["label"].value_counts().index:
                    count = len(long_df[long_df["label"] == label])
                    total = len(df[df["label"] == label])
                    print(f"    {label}: {count}/{total} ({count/total*100:.1f}%)")

    return train_df, test_df, tokenizer


def load_triplets_csv(triplets_path: str | Path) -> pd.DataFrame:
    triplets_path = Path(triplets_path)
    if not triplets_path.exists():
        raise FileNotFoundError(f"Triplets CSV not found: {triplets_path}")
    return pd.read_csv(triplets_path)


def select_hard_triplets(triplets_df: pd.DataFrame, min_hard: int = 1000) -> pd.DataFrame:
    """
    - Take strategy == 'hard'
    - If too few, append 'semi-hard'
    """
    if "strategy" not in triplets_df.columns:
        raise ValueError("Triplets CSV must contain a 'strategy' column.")

    hard_triplets = triplets_df[triplets_df["strategy"] == "hard"].reset_index(drop=True)
    if len(hard_triplets) < min_hard:
        print("Warning: Low count of hard triplets. Adding semi-hard triplets...")
        semi_hard = triplets_df[triplets_df["strategy"] == "semi-hard"]
        hard_triplets = pd.concat([hard_triplets, semi_hard]).reset_index(drop=True)
    return hard_triplets