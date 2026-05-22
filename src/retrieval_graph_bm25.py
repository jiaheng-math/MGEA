from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.dataset import Passage
from src.utils import RetrievedPassage


class BM25GraphRetriever:
    def __init__(self, corpus: Sequence[Passage]) -> None:
        self.corpus = list(corpus)
        self.passage_lookup = {passage.id: passage for passage in self.corpus}
        self.corpus_texts = [self._format_text(passage) for passage in self.corpus]
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
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
        if passage.title:
            return f"{passage.title}. {passage.text}"
        return passage.text

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _top_k_indices(self, scores: np.ndarray, top_k: int) -> list[int]:
        effective_k = min(top_k, len(scores))
        if effective_k == 0:
            return []
        candidate_indices = np.argpartition(-scores, effective_k - 1)[:effective_k]
        ordered = sorted(candidate_indices.tolist(), key=lambda idx: (-float(scores[idx]), self.corpus[idx].id))
        return ordered
