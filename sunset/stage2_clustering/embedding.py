"""
Embedding Client for Event Similarity

Uses the embedding model to compute event embeddings for cosine similarity.
Supports both API-based and local (sentence-transformers) embeddings.
"""

import asyncio
import numpy as np
from typing import List, Optional, Union
import httpx
from ..config import EmbeddingConfig
from ..utils import StageLogger, MetricsCollector


class LocalEmbeddingClient:
    """
    Local embedding client using sentence-transformers.

    Uses GTE-Modernbert-Base as specified in the paper.
    """

    def __init__(self, model_name: str = "Alibaba-NLP/gte-modernbert-base"):
        self.logger = StageLogger("stage2", "local_embedding")
        self.metrics = MetricsCollector("stage2_local_embedding")
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            self.logger.info(f"Loading local embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
                self.logger.info("Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        model = self._load_model()

        with self.metrics.timer("embed_batch"):
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(texts, show_progress_bar=False)
            )

        self.metrics.count("texts_embedded", len(texts))
        return np.array(embeddings)

    async def close(self):
        """Release model resources."""
        self._model = None

    def get_metrics_report(self) -> dict:
        """Get metrics report."""
        return self.metrics.get_report()


class EmbeddingClient:
    """
    Client for computing text embeddings.

    Supports OpenAI-compatible embedding APIs.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = StageLogger("stage2", "embedding")
        self.metrics = MetricsCollector("stage2_embedding")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with self.metrics.timer("embed_batch"):
                embeddings = await self._embed_batch_request(batch)
                all_embeddings.extend(embeddings)

        self.metrics.count("texts_embedded", len(texts))

        return np.array(all_embeddings)

    async def _embed_batch_request(self, texts: List[str]) -> List[List[float]]:
        """Make embedding API request for a batch."""
        client = await self._get_client()

        url = f"{self.config.url}/embeddings"
        payload = {
            "model": self.config.model,
            "input": texts,
        }

        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract embeddings in order
            embeddings = [None] * len(texts)
            for item in data["data"]:
                embeddings[item["index"]] = item["embedding"]

            return embeddings

        except httpx.HTTPError as e:
            self.logger.error(f"Embedding request failed: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def get_metrics_report(self) -> dict:
        """Get metrics report."""
        return self.metrics.get_report()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array of shape (n, dim)

    Returns:
        Similarity matrix of shape (n, n)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute similarity matrix
    return normalized @ normalized.T
