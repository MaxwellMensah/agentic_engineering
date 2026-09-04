def reciprocal_rank_fusion(
    bm25_hits: list[dict], vector_hits: list[dict], k: int = 60
) -> list[dict]:
    fused_scores = {}
    chunk_texts = {}

    for item in bm25_hits:
        cid = item["id"]
        chunk_texts[cid] = item["text"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + item["rank_bm25"]))

    for item in vector_hits:
        cid = item["id"]
        chunk_texts[cid] = item["text"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (
            1.0 / (k + item["rank_vector"])
        )

    sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": cid, "text": chunk_texts[cid], "rrf_score": score}
        for cid, score in sorted_chunks 
    ]  