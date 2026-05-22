from __future__ import annotations

import argparse
import inspect
import re
import shutil
from pathlib import Path


OPENAI_INIT_OLD = """            self.client = OpenAI(
                base_url=self.global_config.embedding_base_url
            )"""

OPENAI_INIT_NEW = """            self.client = OpenAI(
                base_url=self.global_config.embedding_base_url,
                timeout=120.0,
                max_retries=5,
            )"""

AZURE_INIT_OLD = """            self.client = AzureOpenAI(api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                                      azure_endpoint=self.global_config.azure_embedding_endpoint)"""

AZURE_INIT_NEW = """            self.client = AzureOpenAI(api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                                      azure_endpoint=self.global_config.azure_embedding_endpoint,
                                      timeout=120.0,
                                      max_retries=5)"""

BATCH_LOOP_OLD = """        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                results.append(self.encode(batch))
            except:
                import ipdb; ipdb.set_trace()
            pbar.update(batch_size)"""

BATCH_LOOP_NEW = """        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            last_exc = None
            for attempt in range(5):
                try:
                    results.append(self.encode(batch))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        f\"Embedding batch {i}:{i + len(batch)} failed on attempt {attempt + 1}/5: {exc}\"
                    )
            if last_exc is not None:
                raise RuntimeError(
                    f\"Embedding batch {i}:{i + len(batch)} failed after 5 attempts: {last_exc}\"
                ) from last_exc
            pbar.update(len(batch))"""

BATCH_SIZE_OLD = """        batch_size = params.pop("batch_size", 16)"""
BATCH_SIZE_NEW = """        batch_size = min(int(params.pop("batch_size", 8) or 8), 8)"""
BAD_RUNTIME_ERROR_OLD = """                raise RuntimeError(f"Embedding batch failed: {exc}") from exc"""
BAD_RUNTIME_ERROR_NEW = """                raise RuntimeError(
                    f\"Embedding batch {i}:{i + len(batch)} failed: {exc}\"
                ) from exc"""

STALE_LAST_EXC_RUNTIME_ERROR_RE = re.compile(
    r"""(?P<indent>\s*)raise RuntimeError\(\n"""
    r"""(?P=indent)\s*f["']Embedding batch \{i\}:\{i \+ len\(batch\)\} failed after 5 attempts: \{last_exc\}["']\n"""
    r"""(?P=indent)\) from last_exc""",
    flags=re.MULTILINE,
)

ONE_SHOT_RUNTIME_ERROR_LOOP_RE = re.compile(
    r"""(?P<indent>\s*)for i in range\(0, len\(texts\), batch_size\):\n"""
    r"""(?P=indent)    batch = texts\[i:i \+ batch_size\]\n"""
    r"""(?P=indent)    try:\n"""
    r"""(?P=indent)        results\.append\(self\.encode\(batch\)\)\n"""
    r"""(?P=indent)    except Exception as exc:\n"""
    r"""(?P=indent)        raise RuntimeError\(\n"""
    r"""(?P=indent)            f["']Embedding batch \{i\}:\{i \+ len\(batch\)\} failed: \{exc\}["']\n"""
    r"""(?P=indent)        \) from exc\n"""
    r"""(?P=indent)    pbar\.update\((?:len\(batch\)|batch_size)\)""",
    flags=re.MULTILINE,
)

BACKUP_SUFFIX = ".pilot0_embedding_openai.bak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the installed HippoRAG OpenAI embedding backend to remove ipdb breakpoints "
            "and add stable timeout/retry behavior for OpenAI-compatible embedding APIs."
        )
    )
    parser.add_argument(
        "--path",
        help="Optional explicit path to hipporag/embedding_model/OpenAI.py. Defaults to the installed module.",
    )
    return parser.parse_args()


def locate_openai_embedding_file(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"HippoRAG OpenAI embedding file not found: {path}")
        return path

    try:
        import hipporag.embedding_model.OpenAI as module
    except ImportError as exc:
        raise RuntimeError(
            "Could not import hipporag.embedding_model.OpenAI. Install hipporag in this environment first."
        ) from exc

    module_path = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if not module_path:
        raise RuntimeError("Could not resolve the installed HippoRAG OpenAI embedding file path.")

    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resolved HippoRAG OpenAI embedding path does not exist: {path}")
    return path


def ensure_backup(path: Path) -> Path:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def patch_text(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    patched = text

    if "timeout=120.0" in patched and "max_retries=5" in patched:
        changes.append("OpenAI client timeout/retry override already present")
    elif OPENAI_INIT_OLD in patched:
        patched = patched.replace(OPENAI_INIT_OLD, OPENAI_INIT_NEW, 1)
        changes.append("added timeout/retry to OpenAI-compatible embedding client")
    else:
        raise RuntimeError("Target OpenAI client init block not found exactly.")

    if "AzureOpenAI" not in patched:
        changes.append("AzureOpenAI branch not present; skipped")
    elif AZURE_INIT_NEW in patched:
        changes.append("AzureOpenAI timeout/retry override already present")
    elif AZURE_INIT_OLD in patched:
        patched = patched.replace(AZURE_INIT_OLD, AZURE_INIT_NEW, 1)
        changes.append("added timeout/retry to AzureOpenAI embedding client")
    else:
        changes.append("AzureOpenAI branch present but did not match expected text; skipped")

    if BATCH_SIZE_NEW in patched:
        changes.append("embedding batch size cap already present")
    else:
        batch_size_patterns = [
            r'^\s*batch_size = params\.pop\("batch_size", 16\)\s*$',
            r'^\s*batch_size = params\.pop\("batch_size", 8\)\s*$',
            r'^\s*batch_size = min\(params\.pop\("batch_size", 16\), 4\)\s*$',
            r'^\s*batch_size = min\(params\.pop\("batch_size", 16\), 8\)\s*$',
            r'^\s*batch_size = min\(int\(params\.pop\("batch_size", 16\) or 16\), 8\)\s*$',
        ]
        for pattern in batch_size_patterns:
            next_patched, count = re.subn(pattern, BATCH_SIZE_NEW, patched, count=1, flags=re.MULTILINE)
            if count:
                patched = next_patched
                changes.append("capped embedding batch size at 8")
                break
        else:
            raise RuntimeError("Target embedding batch size line not found.")

    if "import ipdb; ipdb.set_trace()" not in patched and (
        "failed on attempt" in patched or "failed after 5 attempts" in patched
    ):
        changes.append("batch retry loop already present")
    elif BATCH_LOOP_OLD in patched:
        patched = patched.replace(BATCH_LOOP_OLD, BATCH_LOOP_NEW, 1)
        changes.append("replaced ipdb breakpoint with retrying batch encode loop")
    else:
        patched, count = ONE_SHOT_RUNTIME_ERROR_LOOP_RE.subn(BATCH_LOOP_NEW, patched, count=1)
        if count:
            changes.append("upgraded one-shot embedding RuntimeError loop to retry loop")
        else:
            raise RuntimeError("Target batch_encode loop not found.")

    if BAD_RUNTIME_ERROR_OLD in patched:
        patched = patched.replace(BAD_RUNTIME_ERROR_OLD, BAD_RUNTIME_ERROR_NEW, 1)
        changes.append("fixed stale RuntimeError(exc) NameError bug")

    if "last_exc" in patched and "last_exc =" not in patched:
        patched, count = STALE_LAST_EXC_RUNTIME_ERROR_RE.subn(
            r"""\g<indent>raise RuntimeError(
\g<indent>    f"Embedding batch {i}:{i + len(batch)} failed: {exc}"
\g<indent>) from exc""",
            patched,
            count=1,
        )
        if count:
            changes.append("fixed stale RuntimeError(last_exc) NameError bug")

    return patched, changes


def verify_patches(text: str) -> None:
    if "timeout=120.0" not in text or "max_retries=5" not in text:
        raise RuntimeError("Client timeout/retry verification failed.")
    if BATCH_SIZE_NEW not in text:
        raise RuntimeError("Embedding batch size cap verification failed.")
    if "import ipdb; ipdb.set_trace()" in text:
        raise RuntimeError("ipdb breakpoint removal verification failed.")
    if "failed on attempt" not in text:
        raise RuntimeError("Batch retry loop verification failed.")
    if "last_exc" in text and "last_exc =" not in text:
        raise RuntimeError("Stale last_exc NameError verification failed.")


def main() -> None:
    args = parse_args()
    path = locate_openai_embedding_file(args.path)
    original = path.read_text(encoding="utf-8")
    patched, changes = patch_text(original)

    if patched != original:
        backup_path = ensure_backup(path)
        path.write_text(patched, encoding="utf-8")
        print(f"Patched HippoRAG OpenAI embedding backend: {path}")
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"HippoRAG OpenAI embedding backend already patched: {path}")

    verify_patches(path.read_text(encoding="utf-8"))
    for change in changes:
        print(f"- {change}")


if __name__ == "__main__":
    main()
