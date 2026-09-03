import time

import numpy as np


def benchmark_pipeline(eval_dataset: list[dict], pipeline_fn, top_k: int = 5):
    """
    eval_dataset format: [{"query": "...", "relevant_ids": ["c1", "c4"]}]
    """
    recalls, mrrs, latencies = [], [], []

    for entry in eval_dataset:
        query = entry["query"]
        ground_truth = set(entry["relevant_ids"])

        start_time = time.perf_counter()
        retrieved = pipeline_fn(query, top_n=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        retrieved_ids = [r["id"] for r in retrieved]

        # Recall@K calculation
        hits = len(ground_truth.intersection(set(retrieved_ids)))
        recall = hits / len(ground_truth) if ground_truth else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in ground_truth:
                mrr = 1.0 / rank
                break

        recalls.append(recall)
        mrrs.append(mrr)
        latencies.append(elapsed_ms)

    print(f"Recall@{top_k}: {np.mean(recalls):.4f}")
    print(f"MRR:        {np.mean(mrrs):.4f}")
    print(f"Avg Latency:{np.mean(latencies):.2f} ms")
