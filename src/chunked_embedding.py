from __future__ import annotations

import numpy as np
from tqdm import tqdm


class ChunkedEmbeddingHandler:
    """
    Handle long documents through chunking and weighted pooling.
    """

    def __init__(self, model, tokenizer, chunk_size: int = 7500, overlap: int = 500):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def smart_chunk_text(self, text: str, max_chunks: int = 3):
        """
        Create overlapping chunks with focus on beginning and end.
        """
        tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)

        if len(tokens) <= self.chunk_size:
            return [text], ["full"]

        chunks = []
        chunk_types = []

        # First chunk - beginning (most important for classification)
        chunk1_tokens = tokens[: self.chunk_size]
        chunk1_text = self.tokenizer.decode(chunk1_tokens, skip_special_tokens=True)
        chunks.append(chunk1_text)
        chunk_types.append("beginning")

        # If document is very long, add middle chunk
        if len(tokens) > self.chunk_size * 2:
            mid_start = (len(tokens) - self.chunk_size) // 2
            chunk2_tokens = tokens[mid_start : mid_start + self.chunk_size]
            chunk2_text = self.tokenizer.decode(chunk2_tokens, skip_special_tokens=True)
            chunks.append(chunk2_text)
            chunk_types.append("middle")

        # Last chunk - end (conclusions, summaries)
        if len(tokens) > self.chunk_size:
            chunk3_tokens = tokens[-self.chunk_size :]
            chunk3_text = self.tokenizer.decode(chunk3_tokens, skip_special_tokens=True)
            chunks.append(chunk3_text)
            chunk_types.append("end")

        return chunks[:max_chunks], chunk_types[:max_chunks]

    def get_weighted_embedding(self, text: str):
        """
        Get weighted pooled embedding for text.
        """
        chunks, chunk_types = self.smart_chunk_text(text)

        if len(chunks) == 1:
            # Short document - regular encoding
            embedding = self.model.encode(chunks[0], show_progress_bar=False)
        else:
            # Long document - weighted pooling
            chunk_embeddings = self.model.encode(chunks, show_progress_bar=False)
            
            # Weights based on chunk importance for classification
            weight_map = {
                "beginning": 0.5,    # Most important - classification signals
                "middle": 0.2,       # Context
                "end": 0.3,          # Conclusions
            }
            weights = np.array([weight_map.get(t, 0.3) for t in chunk_types])
            weights = weights / weights.sum()

            embedding = np.average(chunk_embeddings, axis=0, weights=weights)

        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def encode_batch(self, texts, batch_size: int = 8):
        """
        Encode a batch of texts with progress bar.
        Note: batch_size kept for signature compatibility, but notebook encodes one-by-one.
        """
        embeddings = []
        for text in tqdm(texts, desc="Encoding documents"):
            embedding = self.get_weighted_embedding(str(text) if text else "")
            embeddings.append(embedding)
        return np.array(embeddings)