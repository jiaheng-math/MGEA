from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.dataset import Passage
from src.utils import RetrievedPassage


class ColBERTRetriever:
    def __init__(
        self,
        corpus: Sequence[Passage],
        root: str,
        experiment_name: str,
        index_name: str,
        checkpoint: str = "colbert-ir/colbertv2.0",
        nbits: int = 2,
        partitions: int | None = None,
        doc_maxlen: int = 220,
        query_maxlen: int = 64,
        kmeans_niters: int = 4,
        rebuild_index: bool = False,
        trust_existing_index: bool = False,
    ) -> None:
        try:
            from colbert import Indexer, Searcher
            from colbert.infra import ColBERTConfig, Run, RunConfig
        except ImportError as exc:
            raise RuntimeError(
                "ColBERTv2 backend requires `colbert-ai`. Install dependencies from environment.yml."
            ) from exc

        self._Indexer = Indexer
        self._Searcher = Searcher
        self._ColBERTConfig = ColBERTConfig
        self._Run = Run
        self._RunConfig = RunConfig

        self.corpus = list(corpus)
        self.root = Path(root)
        self.experiment_name = experiment_name
        self.index_name = index_name
        self.checkpoint = checkpoint
        self.nbits = nbits
        self.partitions = partitions
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.kmeans_niters = kmeans_niters
        self.rebuild_index = rebuild_index
        self.trust_existing_index = trust_existing_index

        self.run_root = self.root / self.experiment_name
        self.collection_path = self.run_root / "collection.tsv"
        self.manifest_path = self.run_root / "pilot0_colbert_manifest.json"
        self.pid_mapping_path = self.run_root / "pid_to_passage_id.json"

        if self.rebuild_index and self.run_root.exists():
            shutil.rmtree(self.run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.passage_lookup = {passage.id: passage for passage in self.corpus}
        self.pid_to_passage_id = {str(pid): passage.id for pid, passage in enumerate(self.corpus)}

        self._ensure_index()
        self.searcher = self._build_searcher()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        raw = self.searcher.search(query, k=top_k)
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
        current_manifest = self._build_manifest_payload()
        existing_manifest = self._load_manifest()
        if existing_manifest == current_manifest:
            return
        if self._can_trust_existing_index(existing_manifest, current_manifest):
            return

        self._write_collection()
        with self._Run().context(
            self._RunConfig(
                nranks=1,
                experiment=self.experiment_name,
                root=str(self.root),
            )
        ):
            config = self._ColBERTConfig(**self._build_config_kwargs(include_indexing=True))
            indexer = self._Indexer(checkpoint=self.checkpoint, config=config)
            indexer.index(name=self.index_name, collection=str(self.collection_path), overwrite=True)

        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(current_manifest, handle, indent=2, ensure_ascii=False)
        with self.pid_mapping_path.open("w", encoding="utf-8") as handle:
            json.dump(self.pid_to_passage_id, handle, indent=2, ensure_ascii=False)

    def _build_searcher(self):
        with self._Run().context(
            self._RunConfig(
                nranks=1,
                experiment=self.experiment_name,
                root=str(self.root),
            )
        ):
            config = self._ColBERTConfig(**self._build_config_kwargs(include_indexing=False))
            return self._Searcher(index=self.index_name, config=config)

    def _write_collection(self) -> None:
        with self.collection_path.open("w", encoding="utf-8") as handle:
            for pid, passage in enumerate(self.corpus):
                text = self._format_passage_text(passage)
                handle.write(f"{pid}\t{text}\n")

    def _format_passage_text(self, passage: Passage) -> str:
        if passage.title:
            text = f"{passage.title}. {passage.text}"
        else:
            text = passage.text
        return " ".join(text.replace("\t", " ").splitlines())

    def _build_manifest_payload(self) -> dict[str, Any]:
        return {
            "corpus_fingerprint": self._corpus_fingerprint(),
            "document_count": len(self.corpus),
            "checkpoint": self.checkpoint,
            "nbits": self.nbits,
            "partitions": self.partitions,
            "doc_maxlen": self.doc_maxlen,
            "query_maxlen": self.query_maxlen,
            "kmeans_niters": self.kmeans_niters,
            "experiment_name": self.experiment_name,
            "index_name": self.index_name,
        }

    def _build_config_kwargs(self, include_indexing: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "root": str(self.root),
            "query_maxlen": self.query_maxlen,
        }
        if include_indexing:
            kwargs.update(
                {
                    "nbits": self.nbits,
                    "doc_maxlen": self.doc_maxlen,
                    "kmeans_niters": self.kmeans_niters,
                }
            )

        if self.partitions is not None:
            signature = inspect.signature(self._ColBERTConfig.__init__)
            if "partitions" in signature.parameters:
                kwargs["partitions"] = self.partitions

        return kwargs

    def _load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _can_trust_existing_index(
        self,
        existing_manifest: dict[str, Any] | None,
        current_manifest: dict[str, Any],
    ) -> bool:
        if self.rebuild_index or not self.trust_existing_index or not existing_manifest:
            return False
        if existing_manifest.get("corpus_fingerprint") != current_manifest.get("corpus_fingerprint"):
            return False
        index_path = self.run_root / "indexes" / self.index_name
        return index_path.exists() and self.collection_path.exists() and self.pid_mapping_path.exists()

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

    def _extract_ranked_entries(self, raw: Any) -> list[tuple[str, float | None]]:
        seen: set[str] = set()
        ranked: list[tuple[str, float | None]] = []

        for value, score in self._flatten_ranked(raw):
            passage_id = self._resolve_to_passage_id(value)
            if not passage_id or passage_id in seen:
                continue
            ranked.append((passage_id, score))
            seen.add(passage_id)
        return ranked

    def _flatten_ranked(self, value: Any) -> list[tuple[Any, float | None]]:
        if value is None:
            return []

        for doc_key in ("docs", "docids", "pids", "passage_ids", "ids"):
            doc_values = getattr(value, doc_key, None)
            if hasattr(doc_values, "tolist"):
                doc_values = doc_values.tolist()
            if isinstance(doc_values, (list, tuple)):
                score_values = getattr(value, "scores", None)
                if score_values is None:
                    score_values = getattr(value, "doc_scores", None)
                normalized_scores = self._normalize_scores(score_values, len(doc_values))
                return list(zip(doc_values, normalized_scores))

        if isinstance(value, dict):
            for doc_key in ("docs", "docids", "pids", "passage_ids", "ids"):
                doc_values = value.get(doc_key)
                if hasattr(doc_values, "tolist"):
                    doc_values = doc_values.tolist()
                if isinstance(doc_values, (list, tuple)):
                    normalized_scores = self._normalize_scores(
                        value.get("scores", value.get("doc_scores")),
                        len(doc_values),
                    )
                    return list(zip(doc_values, normalized_scores))
            return [(value, None)]

        if isinstance(value, tuple) and len(value) >= 3:
            doc_values = value[0]
            if hasattr(doc_values, "tolist"):
                doc_values = doc_values.tolist()
            score_values = value[2]
            if isinstance(doc_values, (list, tuple)):
                normalized_scores = self._normalize_scores(score_values, len(doc_values))
                return list(zip(doc_values, normalized_scores))

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

    def _resolve_to_passage_id(self, value: Any) -> str | None:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, int):
            return self.pid_to_passage_id.get(str(value))
        if isinstance(value, str):
            if value in self.passage_lookup:
                return value
            if value in self.pid_to_passage_id:
                return self.pid_to_passage_id[value]
            normalized = self._normalize_text(value)
            for passage in self.corpus:
                if normalized == self._normalize_text(self._format_passage_text(passage)):
                    return passage.id
            return None
        if isinstance(value, dict):
            for key in ("pid", "docid", "passage_id", "id"):
                if key in value:
                    return self._resolve_to_passage_id(value[key])
            for key in ("doc", "text", "content", "passage"):
                if key in value:
                    return self._resolve_to_passage_id(value[key])
        return None

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())
