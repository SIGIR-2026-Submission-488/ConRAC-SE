from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.data_io import DatasetPaths, load_and_analyze_data, resolve_repo_root
from src.triplets import (
    WikiLeaksTripletGeneratorAdvanced,
    analyze_triplet_quality,
    initialize_bge_m3,
    save_triplets_to_csv,
)

def main():
    repo_root = REPO_ROOT
    paths = DatasetPaths(repo_root=repo_root)

    # Default paths (edit here if your filenames differ)
    TRAIN_CSV = paths.train_csv
    TEST_CSV = paths.test_csv
    EMB_CACHE = paths.embeddings_cache
    OUT_TRIPLETS = paths.triplets_csv

    train_df, test_df, tokenizer = load_and_analyze_data(TRAIN_CSV, TEST_CSV)

    embedding_model = initialize_bge_m3()

    generator = WikiLeaksTripletGeneratorAdvanced(
        df=train_df,
        model=embedding_model,
        tokenizer=tokenizer,
        text_col="Content",
        label_col="label",
    )

    embeddings = generator.compute_embeddings(cache_path=EMB_CACHE, use_cache=True)

    all_triplets = []
    semihard_triplets = generator.generate_triplets(strategy="semi-hard", triplets_per_anchor=2)
    all_triplets.extend(semihard_triplets)

    hard_triplets = generator.generate_triplets(strategy="hard", triplets_per_anchor=1)
    all_triplets.extend(hard_triplets)

    random_triplets = generator.generate_triplets(strategy="random", triplets_per_anchor=1)
    all_triplets.extend(random_triplets)

    print("\n" + "=" * 60)
    print("TRIPLET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total triplets: {len(all_triplets)}")
    print("By strategy:")
    import pandas as pd
    strategy_counts = pd.DataFrame(all_triplets)["strategy"].value_counts()
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")

    triplet_df = analyze_triplet_quality(all_triplets, generator)
    saved_triplets = save_triplets_to_csv(all_triplets, generator, output_path=OUT_TRIPLETS)
    print("\nTriplet generation and saving complete!")
    print(f"Saved triplets -> {OUT_TRIPLETS}")

if __name__ == "__main__":
    main()
