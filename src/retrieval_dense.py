from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from src.dataset import Passage
from src.retrieval_dense_colbert import ColBERTRetriever
from src.utils import RetrievedPassage


class DenseRetriever:
    def __init__(
        self,
        corpus: Sequence[Passage],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        allow_tfidf_fallback: bool = False,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Install dependencies from environment.yml."
            ) from exc
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.corpus = list(corpus)
        self.model_name = model_name
        self.batch_size = batch_size
        self.allow_tfidf_fallback = allow_tfidf_fallback
        self.local_files_only = local_files_only
        self.corpus_texts = [self._format_text(passage) for passage in self.corpus]
        self.use_tfidf_fallback = False

        try:
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=self.local_files_only)
            self.corpus_embeddings = self.model.encode(
                self.corpus_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            if not self.allow_tfidf_fallback:
                raise RuntimeError(
                    f"Could not load dense encoder '{model_name}'. "
                    "For research runs, install/cache the encoder explicitly or set "
                    "`dense_allow_tfidf_fallback: true` in the config if you intentionally "
                    "want lexical fallback."
                ) from exc
            warnings.warn(
                f"Could not load dense encoder '{model_name}'. Falling back to TF-IDF retrieval for this run."
            )
            self.use_tfidf_fallback = True
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.corpus_embeddings = self.vectorizer.fit_transform(self.corpus_texts)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        if self.use_tfidf_fallback:
            query_embedding = self.vectorizer.transform([query])
            scores = (self.corpus_embeddings @ query_embedding.T).toarray().ravel()
        else:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            scores = np.dot(self.corpus_embeddings, query_embedding)
        top_indices = self._top_k_indices(scores, top_k)

        results: list[RetrievedPassage] = []
        for index in top_indices:
            passage = self.corpus[index]
            results.append(
                RetrievedPassage(
                    id=passage.id,
                    text=passage.text,
                    title=passage.title,
                    source_doc_id=passage.source_doc_id,
                    score=float(scores[index]),
                )
            )
        return results

    def _format_text(self, passage: Passage) -> str:
        return passage.text

    def _top_k_indices(self, scores: np.ndarray, top_k: int) -> list[int]:
        effective_k = min(top_k, len(scores))
        if effective_k == 0:
            return []
        candidate_indices = np.argpartition(-scores, effective_k - 1)[:effective_k]
        ordered = sorted(candidate_indices.tolist(), key=lambda idx: (-float(scores[idx]), self.corpus[idx].id))
        return ordered


def build_dense_retriever(
    corpus: Sequence[Passage],
    config: dict,
    project_root: Path,
):
    backend = str(config.get("dense_backend", "sentence_transformers")).lower()
    if backend in {"sentence_transformers", "sentence-transformers", "minilm", "st"}:
        return DenseRetriever(
            corpus=corpus,
            model_name=str(config.get("dense_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            batch_size=int(config.get("dense_batch_size", 32)),
            allow_tfidf_fallback=bool(config.get("dense_allow_tfidf_fallback", False)),
            local_files_only=bool(config.get("dense_local_files_only", False)),
        )

    if backend in {"colbert", "colbertv2"}:
        root = Path(str(config.get("colbert_root", "colbert_cache/default")))
        if not root.is_absolute():
            root = project_root / root
        return ColBERTRetriever(
            corpus=corpus,
            root=str(root),
            experiment_name=str(config.get("colbert_experiment_name", "pilot0_colbert")),
            index_name=str(config.get("colbert_index_name", "pilot0.nbits=2")),
            checkpoint=str(config.get("colbert_checkpoint", "colbert-ir/colbertv2.0")),
            nbits=int(config.get("colbert_nbits", 2)),
            partitions=int(config["colbert_partitions"]) if config.get("colbert_partitions") is not None else None,
            doc_maxlen=int(config.get("colbert_doc_maxlen", 220)),
            query_maxlen=int(config.get("colbert_query_maxlen", 64)),
            kmeans_niters=int(config.get("colbert_kmeans_niters", 4)),
            rebuild_index=bool(config.get("colbert_rebuild_index", False)),
            trust_existing_index=bool(config.get("colbert_trust_existing_index", False)),
        )

    raise ValueError(f"Unsupported dense_backend: {backend}")
