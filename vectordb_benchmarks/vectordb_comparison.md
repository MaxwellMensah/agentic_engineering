# Vector DB Benchmark: ChromaDB vs Pinecone (same embedding model)

## Setup
- Data: `fraud_detection_dataset_V4.jsonl` (first 100 records)
- Embedding model: `all-MiniLM-L6-v2` (384 dims) – generated locally in both cases
- Pinecone region: aws/us-east-1 (serverless)

## Results

| Operation | ChromaDB (local) | Pinecone (cloud) |
|-----------|------------------|------------------|
| Ingest (100 vectors) | 0.09 sec | 9.03 sec |
| Query (top‑5) | 0.0150 ms | 0.2722 ms |

## Conclusion
- ChromaDB is faster for local development (no network).
- Pinecone adds cloud latency but scales infinitely.
- Result quality identical because embeddings are the same.
