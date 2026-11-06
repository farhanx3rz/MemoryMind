"""Create a sanitized distributable ZIP archive of MemoryMind / RECALL.

Excludes:
- Secret environment files (.env, backend/.env)
- Local vector database (ChromaDB) and SQLite FTS indexes
- Python cache files and bytecode
- Editor / OS metadata (.DS_Store, .vscode, .idea)
- Test caches (.pytest_cache)
"""

from __future__ import annotations

import argparse
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
import zipfile

EXCLUDE_PATTERNS = [
    "*.env",
    "*.env.*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".vscode",
    ".idea",
    ".DS_Store",
    "Thumbs.db",
    "data/chroma",
    "data/*.sqlite3*",
    "data/jobs",
    "backend/data/chroma",
    "backend/data/*.sqlite3*",
    "backend/data/jobs",
    "dist",
    "build",
    "*.egg-info",
    "extension/dist",
    "extension/*.crx",
    "*.log",
    ".coverage",
    "htmlcov",
]


def should_exclude(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    # Never exclude template / example files (e.g. .env.example)
    if Path(normalized).name.endswith(".example"):
        return False
    for pattern in EXCLUDE_PATTERNS:
        # Match file name
        if fnmatch.fnmatch(Path(normalized).name, pattern):
            return True
        # Match relative path or any parent component
        if fnmatch.fnmatch(normalized, pattern):
            return True
        if f"/{pattern}/" in f"/{normalized}/" or normalized.startswith(f"{pattern}/"):
            return True
    return False


def build_dist_archive(
    root_dir: Path,
    output_zip: Path | None = None,
    include_model_weights: bool = True,
) -> Path:
    if output_zip is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_zip = root_dir / "dist" / f"memorymind_recall_{timestamp}.zip"

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    included_count = 0
    excluded_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root_dir.rglob("*"):
            if not p.is_file():
                continue

            rel = p.relative_to(root_dir)
            rel_str = str(rel).replace("\\", "/")

            if should_exclude(rel_str):
                excluded_count += 1
                continue

            if not include_model_weights and (rel_str.endswith(".safetensors") or rel_str.endswith(".bin")):
                excluded_count += 1
                continue

            # Absolute safeguard against leaking .env files
            if p.name.startswith(".env") and not p.name.endswith(".example"):
                print(f"[SECURITY GUARD] Skipping secret file: {rel_str}")
                excluded_count += 1
                continue

            zf.write(p, arcname=rel_str)
            included_count += 1

    print(f"Sanitized archive created: {output_zip}")
    print(f"  Files included: {included_count}")
    print(f"  Files excluded: {excluded_count}")
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Package MemoryMind into a sanitized distributable ZIP.")
    parser.add_argument("--output", default=None, help="Output ZIP file path.")
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Exclude large neural model weights (.safetensors / .bin).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out = Path(args.output) if args.output else None
    build_dist_archive(project_root, out, include_model_weights=not args.no_weights)


if __name__ == "__main__":
    main()
