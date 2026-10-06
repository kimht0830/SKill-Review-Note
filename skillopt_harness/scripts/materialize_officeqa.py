"""Materialize the runnable OfficeQA split used by configs/officeqa/default.yaml
(split_dir: data/officeqa_split, data_dirs: data/officeqa_docs_official) from the
manifest in data/officeqa_id_split/.

databricks/officeqa is gated: accept the terms on HF and export HF_TOKEN first.
Only the Treasury Bulletins referenced by the manifest are downloaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, RevisionNotFoundError

from skillopt.envs.officeqa.dataloader import _parse_list_field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
MANIFEST_KEYS = ("id", "uid", "category", "source_files", "source_docs", "source_split")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "officeqa_id_split")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "officeqa_split")
    p.add_argument("--docs-dir", type=Path, default=PROJECT_ROOT / "data" / "officeqa_docs_official")
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "_officeqa_hf")
    p.add_argument("--revision", default="", help="HF revision (default: manifest source_revision)")
    return p.parse_args()


def with_revision_fallback(fn, revision: str, **kw):
    try:
        return fn(revision=revision or None, **kw)
    except GatedRepoError as e:
        raise SystemExit(
            "databricks/officeqa is gated: accept the dataset terms on Hugging Face and set HF_TOKEN.\n" + str(e)
        )
    except RevisionNotFoundError:
        print(f"[warn] revision {revision!r} not found -> falling back to main")
        return fn(revision=None, **kw)


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    meta = json.loads((args.manifest_dir / "split_manifest.json").read_text(encoding="utf-8"))
    repo_id = meta["source_repo"]
    revision = args.revision or meta.get("source_revision", "")

    # 1) QA payload
    csv_path = with_revision_fallback(
        hf_hub_download, revision, repo_id=repo_id, repo_type="dataset",
        filename=meta.get("source_file", "officeqa_full.csv"), token=token,
    )
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    id_col = next((c for c in ("uid", "UID", "id") if rows and c in rows[0]), None)
    if id_col is None:
        raise SystemExit(f"No uid column in {csv_path}: columns={list(rows[0]) if rows else []}")
    by_uid = {str(r[id_col]).strip(): r for r in rows}
    print(f"[csv] {len(rows)} rows, columns={list(rows[0])}")

    # 2) split items = CSV row (question/answer/...) + manifest fields (paper split metadata)
    needed_files: set[str] = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        manifest = json.loads((args.manifest_dir / split / "items.json").read_text(encoding="utf-8"))
        items, missing = [], []
        for m in manifest:
            uid = str(m.get("uid") or m["id"]).strip()
            row = by_uid.get(uid)
            if row is None:
                missing.append(uid)
                continue
            item = dict(row)
            item.update({k: m[k] for k in MANIFEST_KEYS if k in m and m[k] not in (None, "")})
            if not (item.get("answer") or item.get("ground_truth")):
                raise SystemExit(f"{uid}: no answer/ground_truth column in CSV row: {list(row)}")
            items.append(item)
            needed_files.update(_parse_list_field(item.get("source_files")))
        if missing:
            print(f"[warn] {split}: {len(missing)}/{len(manifest)} uids missing from CSV (e.g. {missing[:5]})")
        out = args.output_dir / split
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "items.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[write] {split}: {len(items)} items -> {out / 'items.json'}")

    # 3) only the referenced documents: transformed txt (read by tools) + parsed json (oracle pages)
    stems = sorted({Path(n).stem for n in needed_files})
    patterns = [f"treasury_bulletins_parsed/transformed/{s}.txt" for s in stems]
    patterns += [f"treasury_bulletins_parsed/jsons/{s}.json" for s in stems]
    print(f"[docs] downloading {len(stems)} bulletins (txt + json)")
    with_revision_fallback(
        snapshot_download, revision, repo_id=repo_id, repo_type="dataset",
        allow_patterns=patterns, local_dir=str(args.cache_dir), token=token,
    )
    parsed_root = args.cache_dir / "treasury_bulletins_parsed"
    n_txt = sum(1 for s in stems if (parsed_root / "transformed" / f"{s}.txt").is_file())
    n_json = sum(1 for s in stems if (parsed_root / "jsons" / f"{s}.json").is_file())
    print(f"[docs] txt {n_txt}/{len(stems)}, json {n_json}/{len(stems)}")

    # config의 data_dirs(data/officeqa_docs_official)가 treasury_bulletins_parsed를 가리키게 함.
    # tool_runtime은 <root>/transformed 를 문서 루트로, 그 부모의 jsons/ 를 oracle page 소스로 찾습니다.
    if args.docs_dir.is_symlink() or args.docs_dir.exists():
        print(f"[skip] {args.docs_dir} already exists")
    else:
        args.docs_dir.symlink_to(parsed_root.resolve(), target_is_directory=True)
        print(f"[link] {args.docs_dir} -> {parsed_root.resolve()}")


if __name__ == "__main__":
    main()
