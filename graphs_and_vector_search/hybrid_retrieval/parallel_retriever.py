import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, corpus_chunks: list[dict], chroma_collection):
        """
        corpus_chunks: list of dicts [{'id': 'c1', 'text': '...'}]
        """
        self.chunks = corpus_chunks
        self.chroma = chroma_collection

        # Index BM25
        corpus_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        self.chunk_map = {c["id"]: c["text"] for c in self.chunks}

    def get_bm25_candidates(self, query: str, top_k: int = 20) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.chunks[idx]["id"],
                "text": self.chunks[idx]["text"],
                "rank_bm25": rank,
            }
            for rank, idx in enumerate(top_indices, start=1)
        ]

    def get_vector_candidates(self, query: str, top_k: int = 20) -> list[dict]:
        res = self.chroma.query(query_texts=[query], n_results=top_k)

        return [
            {"id": doc_id, "text": text, "rank_vector": rank}
            for rank, (doc_id, text) in enumerate(
                zip(res["ids"][0], res["documents"][0]), start=1
            )
        ]
