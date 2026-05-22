from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.dataset import Passage
from src.utils import RetrievedPassage


class HippoRAGGraphRetriever:
    def __init__(
        self,
        corpus: Sequence[Passage],
        save_dir: str,
        llm_model_name: str,
        embedding_model_name: str,
        llm_base_url: str | None = None,
        embedding_base_url: str | None = None,
        rebuild_index: bool = False,
    ) -> None:
        try:
            from hipporag import HippoRAG
        except ImportError as exc:
            raise RuntimeError(
                "HippoRAG backend requires `hipporag`. Install it with `pip install hipporag`."
            ) from exc

        api_key = os.getenv("HIPPORAG_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

        self._HippoRAG = HippoRAG
        self.corpus = list(corpus)
        self.save_dir = Path(save_dir)
        self.manifest_path = self.save_dir / "pilot0_manifest.json"
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.llm_base_url = llm_base_url
        self.embedding_base_url = embedding_base_url
        self.rebuild_index = rebuild_index

        if self.rebuild_index and self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.passage_lookup = {passage.id: passage for passage in self.corpus}
        self.serialized_docs = [self._serialize_passage(passage) for passage in self.corpus]
        self.doc_to_id = {doc: passage.id for doc, passage in zip(self.serialized_docs, self.corpus)}

        init_kwargs: dict[str, Any] = {
            "save_dir": str(self.save_dir),
            "llm_model_name": self.llm_model_name,
            "embedding_model_name": self.embedding_model_name,
        }
        if self.llm_base_url:
            init_kwargs["llm_base_url"] = self.llm_base_url
        if self.embedding_base_url:
            init_kwargs["embedding_base_url"] = self.embedding_base_url

        self.hipporag = self._HippoRAG(**init_kwargs)
        self._ensure_index()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        raw = self.hipporag.retrieve(queries=[query], num_to_retrieve=top_k)
        ranked_entries = self._extract_ranked_entries(raw)

        results: list[RetrievedPassage] = []
        for rank, (passage_id, score) in enumerate(ranked_entries[:top_k]):
            passage = self.passage_lookup.get(passage_id)
            if passage is None:
                continue
            results.append(
                RetrievedPassage(
                    id=passage.id,
                    text=passage.text,
                    title=passage.title,
                    source_doc_id=passage.source_doc_id,
                    score=float(score if score is not None else max(0, top_k - rank)),
                )
            )
        return results

    def _ensure_index(self) -> None:
        current_fingerprint = self._corpus_fingerprint()
        manifest = self._load_manifest()
        if manifest and manifest.get("corpus_fingerprint") == current_fingerprint:
            return

        self.hipporag.index(docs=self.serialized_docs)
        payload = {
            "corpus_fingerprint": current_fingerprint,
            "document_count": len(self.serialized_docs),
            "llm_model_name": self.llm_model_name,
            "embedding_model_name": self.embedding_model_name,
            "llm_base_url": self.llm_base_url,
            "embedding_base_url": self.embedding_base_url,
        }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _corpus_fingerprint(self) -> str:
        payload = [
            {
                "id": passage.id,
                "title": passage.title,
                "text": passage.text,
            }
            for passage in self.corpus
        ]
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _serialize_passage(self, passage: Passage) -> str:
        title = passage.title or ""
        return "\n".join(
            [
                f"PASSAGE_ID::{passage.id}",
                f"TITLE::{title}",
                "TEXT::",
                passage.text,
            ]
        )

    def _extract_ranked_entries(self, raw: Any) -> list[tuple[str, float | None]]:
        ranked_entries: list[tuple[str, float | None]] = []
        seen: set[str] = set()

        for passage_value, score in self._flatten_ranked(raw):
            passage_id = self._resolve_item_to_passage_id(passage_value)
            if passage_id and passage_id not in seen:
                ranked_entries.append((passage_id, score))
                seen.add(passage_id)
        return ranked_entries

    def _flatten_ranked(self, value: Any) -> list[tuple[Any, float | None]]:
        if value is None:
            return []

        docs_attr = getattr(value, "docs", None)
        if isinstance(docs_attr, list):
            scores_attr = getattr(value, "doc_scores", None)
            normalized_scores = self._normalize_scores(scores_attr, len(docs_attr))
            return list(zip(docs_attr, normalized_scores))

        if isinstance(value, dict):
            if isinstance(value.get("docs"), list):
                docs = value["docs"]
                normalized_scores = self._normalize_scores(value.get("doc_scores"), len(docs))
                return list(zip(docs, normalized_scores))
            return [(value, None)]

        if isinstance(value, (list, tuple)):
            output: list[tuple[Any, float | None]] = []
            for item in value:
                output.extend(self._flatten_ranked(item))
            return output
        return [(value, None)]

    def _normalize_scores(self, value: Any, expected_len: int) -> list[float | None]:
        if value is None:
            return [None] * expected_len
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            scores = [float(item) for item in value[:expected_len]]
            if len(scores) < expected_len:
                scores.extend([None] * (expected_len - len(scores)))
            return scores
        return [None] * expected_len

    def _resolve_item_to_passage_id(self, item: Any) -> str | None:
        if isinstance(item, str):
            return self._match_string_to_passage_id(item)

        if isinstance(item, dict):
            for key in ("doc_id", "idx", "id", "passage_id"):
                if key in item:
                    passage_id = self._match_identifier(item[key])
                    if passage_id:
                        return passage_id

            for key in ("doc", "document", "text", "content", "passage"):
                if key in item:
                    passage_id = self._match_string_to_passage_id(item[key])
                    if passage_id:
                        return passage_id
        return None

    def _match_identifier(self, value: Any) -> str | None:
        if isinstance(value, str):
            if value in self.passage_lookup:
                return value
            return self._match_string_to_passage_id(value)
        if isinstance(value, int):
            if 0 <= value < len(self.corpus):
                return self.corpus[value].id
        return None

    def _match_string_to_passage_id(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        if value in self.doc_to_id:
            return self.doc_to_id[value]

        if "PASSAGE_ID::" in value:
            marker = value.split("PASSAGE_ID::", 1)[1].splitlines()[0].strip()
            if marker in self.passage_lookup:
                return marker

        normalized = self._normalize_text(value)
        if not normalized:
            return None

        for doc, passage_id in self.doc_to_id.items():
            if normalized == self._normalize_text(doc):
                return passage_id

        for passage in self.corpus:
            title = self._normalize_text(passage.title or "")
            text = self._normalize_text(passage.text)
            if title and title in normalized:
                return passage.id
            if text and text[:160] and text[:160] in normalized:
                return passage.id
        return None

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())
