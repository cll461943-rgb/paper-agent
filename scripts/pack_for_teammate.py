#!/usr/bin/env python3
"""Pack paper-agent for teammate distribution.

Generates a zip archive containing:
- Source code (src/scholar_agent/)
- Configs (configs/)
- Tests (tests/)
- Build scripts (scripts/, pyproject.toml)
- Documentation (docs/, .env.example, .gitignore)
- Core evaluation scripts (evaluate.py, eval_*.py, etc.)
- Required data files (data/benchmarks/, data/id2paper.json, data/pasa_local_fts.sqlite)

Excludes:
- Caches (__pycache__, data/cache/, .pytest_cache, .venv)
- Secrets (.env, .env.bak_*)
- Issue tracker state (.dolt/, .beads/)
- Runtime outputs (outputs/, logs/)
- Large corpus (data/cs_paper_2nd/, data/cs_paper_2nd.zip)
- Misc artifacts (uv.lock, export-state.json, pool_recall_*.json)

Usage:
    python scripts/pack_for_teammate.py [--output paper-agent-dist.zip] [--no-sqlite]
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from script location to find pyproject.toml."""
    here = Path(__file__).resolve().parent
    current = here
    for _ in range(5):
        if (current / "pyproject.toml").is_file():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError("Cannot find project root (pyproject.toml)")


# ── Exclusion rules ──────────────────────────────────────────────────────────

EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    ".dolt",
    ".beads",
    ".git",
    "outputs",
    "logs",
    "cache",  # data/cache/
    "cs_paper_2nd",  # data/cs_paper_2nd/
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".egg-info",
    ".log",
    ".pid",
    ".darc",
}

EXCLUDED_FILES = {
    ".env",
    "uv.lock",
    "export-state.json",
    "push-state.json",
    "last-touched",
    ".local_version",
    "metadata.json",
    "evaluation_report.json",
    "cs_paper_2nd.zip",
    "LOCK",
    ".lock",
    ".dolt-gc-last",
    ".dolt-gc.lock",
}

EXCLUDED_PREFIXES = {
    ".env.bak",
    "pool_recall_",
    "dev1000_",
    "test_llm_titles",
    "pipeline_v4_output",
}


def should_exclude(rel_path: Path) -> bool:
    """Check if a file should be excluded from the zip."""
    parts = rel_path.parts

    # Exclude by directory name
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True

    name = rel_path.name

    # Exclude by suffix
    for suffix in EXCLUDED_SUFFIXES:
        if name.endswith(suffix):
            return True

    # Exclude exact filenames
    if name in EXCLUDED_FILES:
        return True

    # Exclude by prefix
    for prefix in EXCLUDED_PREFIXES:
        if name.startswith(prefix):
            return True

    # Exclude *.egg-info directories
    if ".egg-info" in name:
        return True

    return False


# ── Include rules ────────────────────────────────────────────────────────────

# Directories to include entirely
INCLUDE_DIRS = [
    "src/scholar_agent",
    "configs",
    "tests",
    "scripts",
    "docs",
    "data/benchmarks",
]

# Root-level files to include
INCLUDE_ROOT_FILES = [
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "README.md",
    "evaluate.py",
    "evaluate_pasa.py",
    "eval_pool_recall_temp.py",
    "eval_ablation.py",
    "run_ablation.py",
    "full_pipeline_eval.py",
    "analyze_recall.py",
    "build_vector_index.py",
    "simulate_k_approaches.py",
    "test_hyde_recall.py",
]

# Specific data files to include (bypass exclusion rules)
INCLUDE_DATA_FILES = [
    "data/id2paper.json",
    "data/cache/vector_index/faiss.index",
    "data/cache/vector_index/arxiv_ids.npy",
]


def collect_files(root: Path, include_sqlite: bool = True) -> list[tuple[Path, str]]:
    """Collect all files to include in the zip.

    Returns list of (absolute_path, archive_path) tuples.
    """
    files: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def add_file(abs_path: Path, archive_path: str, *, bypass_exclude: bool = False) -> None:
        if archive_path in seen:
            return
        if not abs_path.is_file():
            return
        if not bypass_exclude and should_exclude(abs_path.relative_to(root)):
            return
        seen.add(archive_path)
        files.append((abs_path, archive_path))

    # 1. Include directories
    for dir_rel in INCLUDE_DIRS:
        dir_abs = root / dir_rel
        if not dir_abs.is_dir():
            print(f"  [SKIP] Directory not found: {dir_rel}")
            continue
        for file_path in dir_abs.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(root)
                archive_path = f"paper-agent/{rel}"
                add_file(file_path, archive_path)

    # 2. Include root-level files
    for filename in INCLUDE_ROOT_FILES:
        file_abs = root / filename
        if file_abs.is_file():
            add_file(file_abs, f"paper-agent/{filename}")
        else:
            print(f"  [SKIP] File not found: {filename}")

    # 3. Include specific data files (bypass exclusion for vector index etc.)
    for data_file in INCLUDE_DATA_FILES:
        file_abs = root / data_file
        if file_abs.is_file():
            add_file(file_abs, f"paper-agent/{data_file}", bypass_exclude=True)
        else:
            print(f"  [SKIP] Data file not found: {data_file}")

    # 4. Include SQLite (optional, 3.2GB)
    if include_sqlite:
        sqlite_path = root / "data" / "pasa_local_fts.sqlite"
        if sqlite_path.is_file():
            add_file(sqlite_path, "paper-agent/data/pasa_local_fts.sqlite")
            print(f"  [OK] Including pasa_local_fts.sqlite ({sqlite_path.stat().st_size / 1024 / 1024:.0f} MB)")
        else:
            print("  [SKIP] pasa_local_fts.sqlite not found")

    return files


def create_zip(
    files: list[tuple[Path, str]],
    output_path: Path,
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    """Create the zip archive."""
    print(f"\nCreating zip: {output_path}")
    print(f"Total files: {len(files)}")

    total_size = sum(f.stat().st_size for f, _ in files)
    print(f"Total uncompressed: {total_size / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(output_path, "w", compression=compression) as zf:
        for i, (abs_path, archive_path) in enumerate(files, 1):
            file_size = abs_path.stat().st_size
            # Show progress for large files
            if file_size > 10 * 1024 * 1024:
                print(f"  [{i}/{len(files)}] Compressing {archive_path} ({file_size / 1024 / 1024:.0f} MB)...")
            zf.write(abs_path, archive_path)

    zip_size = output_path.stat().st_size
    print(f"\nDone! Zip size: {zip_size / 1024 / 1024:.1f} MB")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack paper-agent for teammate distribution")
    parser.add_argument(
        "--output", "-o",
        default="paper-agent-dist.zip",
        help="Output zip path (default: paper-agent-dist.zip in project root)",
    )
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Exclude pasa_local_fts.sqlite (3.2GB) from the zip",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list files that would be included, don't create zip",
    )
    args = parser.parse_args()

    root = find_project_root()
    print(f"Project root: {root}")

    include_sqlite = not args.no_sqlite
    files = collect_files(root, include_sqlite=include_sqlite)

    if not files:
        print("ERROR: No files collected!")
        sys.exit(1)

    # Sort by archive path for consistent ordering
    files.sort(key=lambda x: x[1])

    if args.list_only:
        print(f"\nFiles to include ({len(files)}):")
        for _, archive_path in files:
            print(f"  {archive_path}")
        total = sum(f.stat().st_size for f, _ in files)
        print(f"\nTotal uncompressed: {total / 1024 / 1024:.1f} MB")
        return

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    create_zip(files, output)


if __name__ == "__main__":
    main()
