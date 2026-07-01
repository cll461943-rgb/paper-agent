#!/usr/bin/env python3
"""Build FAISS vector index from pasa_local_fts.sqlite (568K papers).

Optimized for speed: fp16, max_length=256, batch_size=128.
Measured throughput: ~10 papers/s on RTX 3060 (BGE-M3 568M params).
ETA: ~16h for full 568K corpus. Supports checkpoint/resume.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

DB_PATH = "data/pasa_local_fts.sqlite"
INDEX_DIR = "data/cache/vector_index"


def read_papers_from_db(db_path: str, limit: int | None = None):
    """Read all papers (arxiv_id, title, abstract) from the FTS database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    sql = "SELECT c0, c1, c2 FROM papers_fts_content WHERE c1 IS NOT NULL AND c1 != ''"
    if limit:
        sql += f" LIMIT {limit}"

    cur.execute(sql)
    papers = []
    for row in cur:
        arxiv_id, title, abstract = row
        if not title:
            continue
        text = f"{title} {abstract}" if abstract else title
        papers.append((arxiv_id, text))

    conn.close()
    return papers


CHECKPOINT_INTERVAL = 50_000  # save checkpoint every 50K papers


def _save_checkpoint(index, arxiv_ids_done, offset, index_dir):
    """Save intermediate checkpoint for resume."""
    ckpt_dir = os.path.join(index_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(ckpt_dir, "faiss.index"))
    np.save(os.path.join(ckpt_dir, "arxiv_ids.npy"), np.array(arxiv_ids_done, dtype=object))
    with open(os.path.join(ckpt_dir, "offset.txt"), "w") as f:
        f.write(str(offset))
    LOGGER.info("Checkpoint saved: %d papers at offset %d", len(arxiv_ids_done), offset)


def _load_checkpoint(index_dir):
    """Load checkpoint if exists. Returns (index, arxiv_ids_done, offset) or None."""
    ckpt_dir = os.path.join(index_dir, "checkpoint")
    offset_file = os.path.join(ckpt_dir, "offset.txt")
    if not os.path.exists(offset_file):
        return None
    with open(offset_file) as f:
        offset = int(f.read().strip())
    index = faiss.read_index(os.path.join(ckpt_dir, "faiss.index"))
    arxiv_ids_done = np.load(os.path.join(ckpt_dir, "arxiv_ids.npy"), allow_pickle=True).tolist()
    LOGGER.info("Resumed from checkpoint: %d papers at offset %d", len(arxiv_ids_done), offset)
    return index, arxiv_ids_done, offset


def build_index(
    batch_size: int = 128,
    max_length: int = 256,
    device: str = "cuda",
    limit: int | None = None,
):
    """Build FAISS index from all papers in the database. Supports checkpoint/resume."""
    LOGGER.info("Reading papers from %s", DB_PATH)
    papers = read_papers_from_db(DB_PATH, limit=limit)
    n_papers = len(papers)
    LOGGER.info("Read %d papers with titles", n_papers)

    arxiv_ids = [p[0] for p in papers]
    texts = [p[1] for p in papers]

    # Load model directly (bypass EmbeddingService for speed)
    LOGGER.info("Loading BGE-M3 model (fp16)...")
    import torch
    from transformers import AutoTokenizer, AutoModel

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    dim = model.config.hidden_size
    LOGGER.info("Model loaded: %s, dim=%d", type(model).__name__, dim)

    # Try to resume from checkpoint
    os.makedirs(INDEX_DIR, exist_ok=True)
    checkpoint = _load_checkpoint(INDEX_DIR)
    if checkpoint is not None:
        index, arxiv_ids_done, start_offset = checkpoint
        LOGGER.info("Resuming from checkpoint: offset=%d, %d papers already indexed",
                     start_offset, len(arxiv_ids_done))
    else:
        index = faiss.IndexFlatIP(dim)
        arxiv_ids_done = []
        start_offset = 0

    # Process in batches
    t0 = time.time()
    remaining = n_papers - start_offset
    total_batches = (remaining + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_idx in range(total_batches):
            start = start_offset + batch_idx * batch_size
            end = min(start + batch_size, n_papers)
            batch_texts = texts[start:end]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Forward pass
            outputs = model(**encoded)

            # Mean pooling (attention-mask weighted)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state.float() * mask).sum(1) / mask.sum(1)

            # Convert to numpy float32
            vecs = embeddings.cpu().numpy().astype(np.float32)

            # L2 normalize for cosine similarity
            faiss.normalize_L2(vecs)

            # Add to FAISS index
            index.add(vecs)
            arxiv_ids_done.extend(arxiv_ids[start:end])

            # Progress log every 100 batches
            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - t0
                done = end - start_offset
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_papers - end) / rate if rate > 0 else 0
                pct = end / n_papers * 100
                LOGGER.info(
                    "Batch %d/%d: %d/%d papers (%.1f%%, %.1f/s, ETA %.1fh)",
                    batch_idx + 1,
                    total_batches,
                    end,
                    n_papers,
                    pct,
                    rate,
                    eta / 3600,
                )

            # Checkpoint every 50K papers
            if end % CHECKPOINT_INTERVAL < batch_size and end < n_papers:
                _save_checkpoint(index, arxiv_ids_done, end, INDEX_DIR)

    elapsed = time.time() - t0
    LOGGER.info(
        "Indexing complete: %d papers in %.1fs (%.1f papers/s)",
        index.ntotal,
        elapsed,
        index.ntotal / elapsed if elapsed > 0 else 0,
    )

    # Save final index
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    np.save(f"{INDEX_DIR}/arxiv_ids.npy", np.array(arxiv_ids_done, dtype=object))
    LOGGER.info("Saved FAISS index to %s/faiss.index (%d vectors)", INDEX_DIR, index.ntotal)
    LOGGER.info("Saved arxiv_id mapping to %s/arxiv_ids.npy", INDEX_DIR)

    # Cleanup checkpoint dir
    import shutil
    ckpt_dir = os.path.join(INDEX_DIR, "checkpoint")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        LOGGER.info("Cleaned up checkpoint directory")

    # Sanity check: search for known gold paper
    test_query = "Extracting training data from large language models"
    encoded = tokenizer([test_query], return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    q_vec = (outputs.last_hidden_state.float() * mask).sum(1) / mask.sum(1)
    q_vec = q_vec.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, 5)
    LOGGER.info("Sanity check: query='%s'", test_query)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        aid = arxiv_ids_done[idx]
        LOGGER.info("  [%d] score=%.4f arxiv=%s", rank, score, aid)

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    build_index(
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
