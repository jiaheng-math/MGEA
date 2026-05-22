from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path


PARTITION_OLD = "        self.num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))"
PARTITION_NEW = "        self.num_partitions = min(256, int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est)))))"

KMEANS_OLD = (
    "    use_gpu = torch.cuda.is_available()\n"
    "    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=123)"
)
KMEANS_NEW = (
    "    use_gpu = False\n"
    "    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=123)"
)

BACKUP_SUFFIX = ".pilot0_colbert_stability.bak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the installed ColBERT collection_indexer.py to cap auto partitions at 256 "
            "and force FAISS k-means to CPU for stable indexing."
        )
    )
    parser.add_argument(
        "--path",
        help="Optional explicit path to colbert/indexing/collection_indexer.py. Defaults to the installed module.",
    )
    return parser.parse_args()


def locate_collection_indexer(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Collection indexer not found: {path}")
        return path

    try:
        import colbert.indexing.collection_indexer as module
    except ImportError as exc:
        raise RuntimeError(
            "Could not import colbert.indexing.collection_indexer. Install colbert-ai in this environment first."
        ) from exc

    module_path = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if not module_path:
        raise RuntimeError("Could not resolve the installed collection_indexer.py path.")

    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resolved collection indexer path does not exist: {path}")
    return path


def ensure_backup(path: Path) -> Path:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def patch_text(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    patched = text

    if PARTITION_NEW in patched:
        changes.append("partition cap already present")
    elif PARTITION_OLD in patched:
        patched = patched.replace(PARTITION_OLD, PARTITION_NEW, 1)
        changes.append("capped auto partitions at 256")
    else:
        raise RuntimeError("Target partition line not found exactly in collection_indexer.py.")

    if KMEANS_NEW in patched:
        changes.append("CPU k-means override already present")
    elif KMEANS_OLD in patched:
        patched = patched.replace(KMEANS_OLD, KMEANS_NEW, 1)
        changes.append("forced FAISS k-means to CPU")
    else:
        raise RuntimeError(
            "Target k-means block not found exactly in collection_indexer.py. "
            "Inspect the installed ColBERT version before patching."
        )

    return patched, changes


def verify_patches(text: str) -> None:
    if PARTITION_NEW not in text:
        raise RuntimeError("Partition cap verification failed.")
    if KMEANS_NEW not in text:
        raise RuntimeError("CPU k-means verification failed.")


def main() -> None:
    args = parse_args()
    path = locate_collection_indexer(args.path)
    original = path.read_text(encoding="utf-8")
    patched, changes = patch_text(original)

    if patched != original:
        backup_path = ensure_backup(path)
        path.write_text(patched, encoding="utf-8")
        print(f"Patched ColBERT stability settings: {path}")
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"ColBERT stability settings already patched: {path}")

    verify_patches(path.read_text(encoding="utf-8"))
    for change in changes:
        print(f"- {change}")


if __name__ == "__main__":
    main()
