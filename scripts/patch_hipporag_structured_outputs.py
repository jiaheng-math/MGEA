from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


OPENIE_OPENAI_TEMPLATE = '''import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


NER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ner_extraction",
        "description": "Extract a deduplicated list of named entities from a passage.",
        "strict": True,
        "schema": {
            "type": "object",
            "description": "Named-entity extraction result.",
            "additionalProperties": False,
            "properties": {
                "named_entities": {
                    "type": "array",
                    "description": "Unique named entities explicitly grounded in the passage.",
                    "items": {"$ref": "#/$defs/non_empty_clean_string"},
                    "minItems": 0,
                    "maxItems": 64,
                }
            },
            "required": ["named_entities"],
            "$defs": {
                "non_empty_clean_string": {
                    "type": "string",
                    "description": "A non-empty string without control characters.",
                    "pattern": "^(?=.*\\\\S)[^\\\\u0000-\\\\u001F]+$",
                }
            },
        },
    },
}


TRIPLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "triple_extraction",
        "description": "Extract grounded subject-predicate-object triples from a passage.",
        "strict": True,
        "schema": {
            "type": "object",
            "description": "OpenIE triple extraction result.",
            "additionalProperties": False,
            "properties": {
                "triples": {
                    "type": "array",
                    "description": "Grounded relation triples extracted from the passage.",
                    "items": {"$ref": "#/$defs/triple"},
                    "minItems": 0,
                    "maxItems": 64,
                }
            },
            "required": ["triples"],
            "$defs": {
                "non_empty_clean_string": {
                    "type": "string",
                    "description": "A non-empty string without control characters.",
                    "pattern": "^(?=.*\\\\S)[^\\\\u0000-\\\\u001F]+$",
                },
                "triple": {
                    "type": "object",
                    "description": "A single subject-predicate-object triple.",
                    "additionalProperties": False,
                    "properties": {
                        "subject": {"$ref": "#/$defs/non_empty_clean_string"},
                        "predicate": {"$ref": "#/$defs/non_empty_clean_string"},
                        "object": {"$ref": "#/$defs/non_empty_clean_string"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
        },
    },
}


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _parse_json_payload(real_response: str):
    text = _repair_truncated_payloads(
        _repair_common_triple_json_damage(_sanitize_json_like_text(real_response.strip()))
    )
    if _looks_like_truncated_empty_payload(text):
        return {}
    candidates = []

    if text:
        candidates.append(text)

    fenced_blocks = re.findall(r"```(?:json)?\\s*(.*?)\\s*```", text, re.DOTALL)
    for block in fenced_blocks:
        block = block.strip()
        if block:
            candidates.append(block)

    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    end_positions = [pos for pos in (text.rfind("}"), text.rfind("]")) if pos != -1]
    if start_positions and end_positions:
        start_pos = min(start_positions)
        end_pos = max(end_positions)
        if start_pos < end_pos:
            snippet = text[start_pos:end_pos + 1].strip()
            if snippet:
                candidates.append(snippet)

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            pass

        try:
            payload = ast.literal_eval(candidate)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            pass

    if text.count("{") < text.count("}") and text.rstrip().endswith("}"):
        trimmed = text.rstrip()
        while trimmed.count("{") < trimmed.count("}"):
            trimmed = trimmed[:-1].rstrip()
        try:
            payload = json.loads(trimmed)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            pass

        try:
            payload = ast.literal_eval(trimmed)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            pass

    if text.lstrip().startswith("["):
        trimmed = text.strip()
        if trimmed.endswith(","):
            trimmed = trimmed[:-1].rstrip()
        while trimmed and not trimmed.endswith("]"):
            last_comma = trimmed.rfind(",")
            if last_comma == -1:
                break
            trimmed = trimmed[:last_comma].rstrip()
            if trimmed.endswith(","):
                trimmed = trimmed[:-1].rstrip()
        if trimmed and not trimmed.endswith("]"):
            trimmed = trimmed + "]"
        try:
            payload = json.loads(trimmed)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            pass

        try:
            payload = ast.literal_eval(trimmed)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from response: {real_response[:500]!r}")


def _sanitize_json_like_text(text: str) -> str:
    # Drop escaped and raw control characters that commonly leak into relay responses.
    text = re.sub(r"(?:\\\\u000[bB])+", " ", text)
    text = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]+", " ", text)
    # Drop a trailing incomplete unicode escape from truncated output.
    text = re.sub(r"\\\\u[0-9a-fA-F]{0,3}$", "", text)
    return text


def _repair_common_triple_json_damage(text: str) -> str:
    # Repair malformed triple objects like:
    # {"subject":"...","predicate":"...","foo"}
    # -> {"subject":"...","predicate":"...","object":"foo"}
    pattern = re.compile(
        r'("predicate"\\s*:\\s*"(?:[^"\\\\]|\\\\.)*")\\s*,\\s*("(?:(?:[^"\\\\]|\\\\.)*)")\\s*([,}])'
    )
    return pattern.sub(r'\\1, "object": \\2\\3', text)


def _repair_truncated_payloads(text: str) -> str:
    text = _repair_truncated_named_entities_payload(text)
    return text


def _repair_truncated_named_entities_payload(text: str) -> str:
    normalized = text.strip()
    prefix = '{"named_entities":['
    if not normalized.startswith(prefix):
        return text

    inner = normalized[len(prefix):]
    matches = re.findall(r'"((?:[^"\\\\]|\\\\.)*)"', inner)
    if not matches:
        return '{"named_entities":[]}'

    repaired_items = ",".join(json.dumps(json.loads(f'"{item}"')) for item in matches)
    return prefix + repaired_items + ']}'


def _looks_like_truncated_empty_payload(text: str) -> bool:
    normalized = text.strip()
    if normalized in {"", "{", "[", '{"', '["', "{'", "['"}:
        return True
    if len(normalized) <= 8 and normalized.startswith("{") and "}" not in normalized:
        return True
    if len(normalized) <= 8 and normalized.startswith("[") and "]" not in normalized:
        return True
    return False


def _extract_ner_from_response(real_response):
    payload = _parse_json_payload(real_response)

    if isinstance(payload, list):
        entities = payload
    elif isinstance(payload, dict):
        entities = payload.get("named_entities", [])
    else:
        raise ValueError(f"Unexpected NER payload type: {type(payload)} | payload={payload}")

    if not isinstance(entities, list):
        raise ValueError(f"named_entities is not a list: {payload}")
    return entities


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
                response_format=NER_RESPONSE_FORMAT,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            extracted_entities = _extract_ner_from_response(real_response)

            unique_entities = []
            seen_entities = set()
            for entity in extracted_entities:
                if isinstance(entity, str):
                    normalized = entity.strip()
                elif isinstance(entity, dict):
                    normalized = (
                        entity.get("entity")
                        or entity.get("name")
                        or entity.get("text")
                        or entity.get("mention")
                        or json.dumps(entity, ensure_ascii=False, sort_keys=True)
                    )
                    normalized = str(normalized).strip()
                else:
                    normalized = str(entity).strip()

                if not normalized or normalized in seen_entities:
                    continue
                seen_entities.add(normalized)
                unique_entities.append(normalized)

        except Exception as e:
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                unique_entities=[],
                metadata=metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            payload = _parse_json_payload(real_response)

            if isinstance(payload, list):
                triples = payload
            elif isinstance(payload, dict):
                triples = payload.get("triples", [])
            else:
                raise ValueError(f"Unexpected triples payload type: {type(payload)} | payload={payload}")

            normalized_triples = []
            for triple in triples:
                if isinstance(triple, dict):
                    subj = triple.get("subject") or triple.get("head") or triple.get("source")
                    pred = triple.get("predicate") or triple.get("relation") or triple.get("rel")
                    obj = triple.get("object") or triple.get("tail") or triple.get("target")
                    if subj and pred and obj:
                        normalized_triples.append([str(subj).strip(), str(pred).strip(), str(obj).strip()])
                elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    normalized_triples.append([str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()])
                elif isinstance(triple, str):
                    parts = [part.strip() for part in re.split(r"\\s*[,|;\\t]\\s*", triple) if part.strip()]
                    if len(parts) >= 3:
                        normalized_triples.append(parts[:3])

            return normalized_triples

        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                response_format=TRIPLE_RESPONSE_FORMAT,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        ner_results = {}
        triple_results = {}

        with ThreadPoolExecutor() as executor:
            future_to_chunk = {
                executor.submit(self.openie, chunk_key, chunk_info["content"]): chunk_key
                for chunk_key, chunk_info in chunks.items()
            }

            for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc="OpenIE"):
                chunk_key = future_to_chunk[future]
                try:
                    result = future.result()
                    ner_results[chunk_key] = result["ner"]
                    triple_results[chunk_key] = result["triplets"]
                except Exception as e:
                    logger.warning(f"Batch OpenIE failed for chunk {chunk_key}: {e}")
                    ner_results[chunk_key] = NerRawOutput(
                        chunk_id=chunk_key,
                        response="",
                        unique_entities=[],
                        metadata={"error": str(e)}
                    )
                    triple_results[chunk_key] = TripleRawOutput(
                        chunk_id=chunk_key,
                        response="",
                        metadata={"error": str(e)},
                        triples=[]
                    )

        return ner_results, triple_results
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch an installed HippoRAG package to use native Structured Outputs."
    )
    parser.add_argument(
        "--site-packages",
        default="",
        help="Optional explicit site-packages directory. If omitted, auto-detect from the installed hipporag package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = resolve_site_packages(args.site_packages)
    openie_path = target_dir / "hipporag" / "information_extraction" / "openie_openai.py"
    if not openie_path.exists():
        raise FileNotFoundError(f"Target file not found: {openie_path}")

    backup_path = openie_path.with_suffix(openie_path.suffix + ".structured_outputs.bak")
    if not backup_path.exists():
        backup_path.write_text(openie_path.read_text(encoding="utf-8"), encoding="utf-8")

    openie_path.write_text(OPENIE_OPENAI_TEMPLATE, encoding="utf-8")
    print(f"Patched {openie_path}")
    print(f"Backup saved to {backup_path}")


def resolve_site_packages(explicit_value: str) -> Path:
    if explicit_value:
        return Path(explicit_value)
    import hipporag  # noqa: PLC0415

    return Path(inspect.getfile(hipporag)).resolve().parent.parent


if __name__ == "__main__":
    main()
