"""Materialize the runnable SpreadsheetBench (Verified 400) split used by
configs/spreadsheetbench/default.yaml, using the manifest already checked
into data/spreadsheetbench_id_split/.

Mirrors the structure of scripts/materialize_searchqa.py.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = ("id", "instruction", "spreadsheet_path", "instruction_type", "answer_position")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "spreadsheetbench_id_split")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "spreadsheetbench_split")
    p.add_argument("--data-root-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "spreadsheetbench_verified_400")
    p.add_argument("--repo-id", default="KAKA22/SpreadsheetBench")
    p.add_argument("--filename", default="spreadsheetbench_verified_400.tar.gz")
    return p.parse_args()


def load_manifest_ids(manifest_dir: Path) -> dict[str, list[str]]:
    split_ids = {}
    for split in SPLITS:
        path = manifest_dir / split / "items.json"
        with path.open(encoding="utf-8") as f:
            items = json.load(f)
        split_ids[split] = [str(it["id"]) for it in items]
    return split_ids


def download_and_extract(repo_id: str, filename: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {dest} already populated")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    print(f"[extract] {archive_path} -> {dest}")
    with tarfile.open(archive_path) as tf:
        tf.extractall(dest)


def flatten_if_nested(extract_dir: Path) -> None:
    """The tarball may unpack flat, or under one extra top-level folder.

    configs/spreadsheetbench/default.yaml hardcodes
    data_root: data/spreadsheetbench_verified_400, so make sure `spreadsheet/`
    ends up directly under extract_dir.
    """
    if (extract_dir / "spreadsheet").is_dir():
        return
    candidates = [p for p in extract_dir.iterdir()
                  if p.is_dir() and (p / "spreadsheet").is_dir()]
    if len(candidates) == 1:
        nested = candidates[0]
        for child in nested.iterdir():
            child.rename(extract_dir / child.name)
        nested.rmdir()
    elif not candidates:
        raise FileNotFoundError(f"Could not find a 'spreadsheet/' folder under {extract_dir}")
    else:
        raise RuntimeError(f"Multiple nested folders with 'spreadsheet/' under {extract_dir}: {candidates}")


def find_metadata(root: Path) -> list[dict]:
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, list) and data and all(f in data[0] for f in REQUIRED_FIELDS):
            print(f"[metadata] using {path} ({len(data)} items)")
            return data
    raise FileNotFoundError(
        f"No metadata JSON with fields {REQUIRED_FIELDS} found under {root}. "
        "Inspect the extracted archive and adjust find_metadata()."
    )


def main() -> None:
    args = parse_args()
    download_and_extract(args.repo_id, args.filename, args.data_root_dir)
    flatten_if_nested(args.data_root_dir)
    metadata = find_metadata(args.data_root_dir)
    by_id = {str(it["id"]): it for it in metadata}

    split_ids = load_manifest_ids(args.manifest_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        items, missing = [], []
        for item_id in ids:
            if item_id in by_id:
                items.append(by_id[item_id])
            else:
                missing.append(item_id)
        if missing:
            print(f"[warn] {split}: {len(missing)}/{len(ids)} ids missing from metadata "
                  f"(e.g. {missing[:5]})")
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "items.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[write] {split}: {len(items)} items -> {split_dir / 'items.json'}")

    print(f"\nsplit_dir = {args.output_dir}")
    print(f"data_root = {args.data_root_dir}")


if __name__ == "__main__":
    main()
