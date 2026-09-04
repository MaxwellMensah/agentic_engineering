import json

import chromadb
from chromadb.utils import embedding_functions

# Import your modules
from evaluation import benchmark_pipeline
from fusion import reciprocal_rank_fusion
from parallel_retriever import HybridRetriever
from reranker import ResilientCrossEncoderReranker

# Load your generated JSON datasets
with open("fraud_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open("fraud_eval_dataset.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Initialize ChromaDB and index the corpus
chroma_client = chromadb.Client()
chroma_coll = chroma_client.create_collection(
    name="fraud_bench",
    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
)

chroma_coll.add(ids=[c["id"] for c in corpus], documents=[c["text"] for c in corpus])


# Instantiate retrievers
retriever = HybridRetriever(corpus_chunks=corpus, chroma_collection=chroma_coll)
reranker = ResilientCrossEncoderReranker()


# Define the pipeline callbacks
def bm25_only_pipeline(query: str, top_n: int = 5) -> list[dict]:
    return retriever.get_bm25_candidates(query, top_k=top_n)


def vector_only_pipeline(query: str, top_n: int = 5) -> list[dict]:
    return retriever.get_vector_candidates(query, top_k=top_n)


def hybrid_rerank_pipeline(query: str, top_n: int = 5) -> list[dict]:
    bm25_hits = retriever.get_bm25_candidates(query, top_k=20)
    vector_hits = retriever.get_vector_candidates(query, top_k=20)
    fused = reciprocal_rank_fusion(bm25_hits, vector_hits, k=60)
    return reranker.rerank(query, fused, top_n=top_n)


# Run benchmarks
if __name__ == "__main__":
    print("--- BM25 Baseline ---")
    benchmark_pipeline(eval_dataset=eval_data, pipeline_fn=bm25_only_pipeline, top_k=5)

    print("\n--- Vector Baseline ---")
    benchmark_pipeline(
        eval_dataset=eval_data, pipeline_fn=vector_only_pipeline, top_k=5
    )

    print("\n--- Hybrid + Resilient Reranker ---")
    benchmark_pipeline(
        eval_dataset=eval_data, pipeline_fn=hybrid_rerank_pipeline, top_k=5
    )
