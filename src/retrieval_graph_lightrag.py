from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.dataset import Passage
from src.utils import RetrievedPassage


class LightRAGGraphRetriever:
    def __init__(
        self,
        corpus: Sequence[Passage],
        working_dir: str,
        llm_model_name: str,
        embedding_model_name: str,
        llm_base_url: str | None = None,
        embedding_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_api_key: str | None = None,
        embedding_dim: int | None = None,
        query_mode: str = "hybrid",
        rebuild_index: bool = False,
    ) -> None:
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import openai_embed
            from lightrag.utils import EmbeddingFunc
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LightRAG backend requires `lightrag-hku` and `openai`. "
                "Install them with `pip install lightrag-hku openai`."
            ) from exc

        self._LightRAG = LightRAG
        self._QueryParam = QueryParam
        self._openai_embed = openai_embed
        self._EmbeddingFunc = EmbeddingFunc
        self._AsyncOpenAI = AsyncOpenAI

        self.corpus = list(corpus)
        self.working_dir = Path(working_dir)
        self.manifest_path = self.working_dir / "pilot0_manifest.json"
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.llm_base_url = llm_base_url
        self.embedding_base_url = embedding_base_url
        self.llm_api_key = llm_api_key
        self.embedding_api_key = embedding_api_key or llm_api_key
        self.embedding_dim = embedding_dim or infer_embedding_dim(embedding_model_name)
        self.query_mode = query_mode
        self.rebuild_index = rebuild_index

        if self.rebuild_index and self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.passage_lookup = {passage.id: passage for passage in self.corpus}
        self.serialized_docs = [self._serialize_passage(passage) for passage in self.corpus]
        self.doc_to_id = {doc: passage.id for doc, passage in zip(self.serialized_docs, self.corpus)}

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.rag = self._build_rag()
        self._run_async(self._ainitialize_and_index())

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        ranked_entries = self._run_async(self._aretrieve(query=query, top_k=top_k))
        results: list[RetrievedPassage] = []
        for rank, passage_id in enumerate(ranked_entries[:top_k]):
            passage = self.passage_lookup.get(passage_id)
            if passage is None:
                continue
            results.append(
                RetrievedPassage(
                    id=passage.id,
                    text=passage.text,
                    title=passage.title,
                    source_doc_id=passage.source_doc_id,
                    score=float(max(0, top_k - rank)),
                )
            )
        return results

    def _run_async(self, coroutine):
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coroutine)

    def _build_rag(self):
        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            **kwargs,
        ):
            return await self._chat_complete_text(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                kwargs=kwargs,
            )

        embedding_func = self._EmbeddingFunc(
            embedding_dim=int(self.embedding_dim),
            func=lambda texts: self._openai_embed(
                texts,
                model=self.embedding_model_name,
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
            ),
        )

        return self._LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_model_func,
            llm_model_name=self.llm_model_name,
            embedding_func=embedding_func,
        )

    async def _chat_complete_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        history_messages: list[dict[str, str]],
        kwargs: dict[str, Any],
    ) -> str:
        client = self._AsyncOpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._normalize_chat_messages(history_messages))
        messages.append({"role": "user", "content": prompt})

        request_kwargs = self._filter_chat_kwargs(kwargs)
        response = await client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages,
            **request_kwargs,
        )
        return self._extract_chat_content(response)

    def _normalize_chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            else:
                text = json.dumps(content, ensure_ascii=False)
            normalized.append({"role": role, "content": text})
        return normalized

    def _filter_chat_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "frequency_penalty",
            "max_completion_tokens",
            "max_tokens",
            "presence_penalty",
            "seed",
            "stop",
            "temperature",
            "timeout",
            "top_p",
        }
        output = {key: value for key, value in kwargs.items() if key in allowed and value is not None}

        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict):
            output["response_format"] = response_format
        elif response_format is not None or bool(kwargs.get("keyword_extraction", False)):
            output["response_format"] = {"type": "json_object"}
        return output

    def _extract_chat_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if content is not None:
                return self._strip_markdown_json_fence(str(content))
        if isinstance(response, dict):
            try:
                return self._strip_markdown_json_fence(str(response["choices"][0]["message"]["content"]))
            except Exception:
                pass
        return str(response)

    def _strip_markdown_json_fence(self, content: str) -> str:
        text = content.strip()
        if not text.startswith("```"):
            return content
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return content

    async def _ainitialize_and_index(self) -> None:
        await self.rag.initialize_storages()
        current_fingerprint = self._corpus_fingerprint()
        manifest = self._load_manifest()
        if manifest and manifest.get("corpus_fingerprint") == current_fingerprint:
            return

        await self.rag.ainsert(self.serialized_docs)
        payload = {
            "corpus_fingerprint": current_fingerprint,
            "document_count": len(self.serialized_docs),
            "llm_model_name": self.llm_model_name,
            "embedding_model_name": self.embedding_model_name,
            "llm_base_url": self.llm_base_url,
            "embedding_base_url": self.embedding_base_url,
            "embedding_dim": self.embedding_dim,
            "query_mode": self.query_mode,
        }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    async def _aretrieve(self, query: str, top_k: int) -> list[str]:
        response = await self.rag.aquery(
            query,
            param=self._QueryParam(
                mode=self.query_mode,
                only_need_context=True,
                top_k=max(40, top_k),
                chunk_top_k=max(20,top_k),
                enable_rerank=False,
            ),
        )
        return self._extract_ranked_entries(response)

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

    def _extract_ranked_entries(self, raw: Any) -> list[str]:
        seen: set[str] = set()
        ranked_entries: list[str] = []

        for value in self._flatten_context(raw):
            for chunk in self._split_context_chunks(value):
                passage_id = self._match_string_to_passage_id(chunk)
                if passage_id and passage_id not in seen:
                    ranked_entries.append(passage_id)
                    seen.add(passage_id)
        return ranked_entries

    def _flatten_context(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            output: list[str] = []
            for key in ("context", "contexts", "text", "content", "result", "response", "answer", "data"):
                if key in value:
                    output.extend(self._flatten_context(value[key]))
            return output
        if isinstance(value, (list, tuple)):
            output: list[str] = []
            for item in value:
                output.extend(self._flatten_context(item))
            return output
        return [str(value)]

    def _split_context_chunks(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        if "PASSAGE_ID::" not in text:
            return [text]
        parts = text.split("PASSAGE_ID::")
        output: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            output.append("PASSAGE_ID::" + part)
        return output

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


def infer_embedding_dim(model_name: str) -> int:
    lowered = model_name.lower()
    if "text-embedding-3-large" in lowered:
        return 3072
    if "text-embedding-3-small" in lowered:
        return 1536
    if "bge-m3" in lowered:
        return 1024
    if "bge-large" in lowered:
        return 1024
    return 1536
