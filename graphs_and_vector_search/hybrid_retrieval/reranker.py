import logging
import os

import cohere
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ResilientCrossEncoderReranker:
    def __init__(
        self,
        cohere_api_key: str | None = None,
        local_model_name: str = "BAAI/bge-reranker-base",
        force_local: bool = False,
    ):
        """
        Cross-encoder reranker with cloud API primary execution and local model fallback:
        1. Attempts Cohere Rerank API (Cloud)
        2. Falls back to BGE CrossEncoder (Local Transformer) on failure or missing key
        """
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.local_model_name = local_model_name
        self.force_local = force_local
        self._local_model: CrossEncoder | None = None

    def _get_local_model(self) -> CrossEncoder:
        """Lazy loads the HuggingFace CrossEncoder model only when required."""
        if self._local_model is None:
            logger.info(f"Loading local CrossEncoder: {self.local_model_name}")
            self._local_model = CrossEncoder(self.local_model_name)
        return self._local_model

    def rerank(
        self, query: str, candidate_chunks: list[dict], top_n: int = 5
    ) -> list[dict]:
        if not candidate_chunks:
            return []

        # Attempt Cloud Cross-Encoder API
        if self.cohere_api_key and not self.force_local:
            try:
                co_client = cohere.ClientV2(api_key=self.cohere_api_key)
                docs = [c["text"] for c in candidate_chunks]

                res = co_client.rerank(
                    model="rerank-v3.5", query=query, documents=docs, top_n=top_n
                )

                results = []
                for hit in res.results:
                    chunk = candidate_chunks[hit.index].copy()
                    chunk["rerank_score"] = float(hit.relevance_score)
                    chunk["provider"] = "cohere-api"
                    results.append(chunk)

                return results

            except Exception as e:                                                                                                         #noqa
                logger.warning(
                    f"Cohere API failed ({type(e).__name__}: {e}). "
                    "Falling back to local CrossEncoder model."
                )

        # Fallback to Local Cross-Encoder Transformer
        return self._rerank_local(query, candidate_chunks, top_n)

    def _rerank_local(
        self, query: str, candidate_chunks: list[dict], top_n: int = 5
    ) -> list[dict]:
        model = self._get_local_model()
        pairs = [[query, c["text"]] for c in candidate_chunks]
        scores = model.predict(pairs)

        ranked = []
        for chunk, score in zip(candidate_chunks, scores):
            c_copy = chunk.copy()
            c_copy["rerank_score"] = float(score)
            c_copy["provider"] = "bge-local"
            ranked.append(c_copy)

        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]