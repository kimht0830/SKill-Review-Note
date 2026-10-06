"""Materialize the runnable LiveMathematicianBench split used by
configs/livemathematicianbench/default.yaml (split_dir: data/livemathematicianbench_split)
from the manifest in data/livemathematicianbench_id_split/.

split_dir mode loads items as-is, so items are normalized here with the repo's
own _normalize_item (same schema as ratio mode produces).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RevisionNotFoundError

from skillopt.envs.livemathematicianbench.dataloader import _normalize_item

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "livemathematicianbench_id_split")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "livemathematicianbench_split")
    p.add_argument("--revision", default="", help="HF revision (default: manifest source_revision)")
    return p.parse_args()


def download(repo_id: str, filename: str, revision: str, token: str | None) -> str:
    kw = dict(repo_id=repo_id, repo_type="dataset", filename=filename, token=token)
    try:
        return hf_hub_download(revision=revision or None, **kw)
    except GatedRepoError as e:
        raise SystemExit(f"{repo_id} is gated: accept the terms on Hugging Face and set HF_TOKEN.\n{e}")
    except RevisionNotFoundError:
        print(f"[warn] revision {revision!r} not found -> falling back to main")
        return hf_hub_download(revision=None, **kw)


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    meta = json.loads((args.manifest_dir / "split_manifest.json").read_text(encoding="utf-8"))
    repo_id = meta["source_repo"]
    revision = args.revision or meta.get("source_revision", "")

    # 1) 월별 원본을 받아 id(<month>:<no>) -> 정규화된 item
    by_id: dict[str, dict] = {}
    for rel in meta["source_files"]:
        local = download(repo_id, rel, revision, token)
        month = Path(rel).parent.name            # data/202602/qa_202602_final.json -> 202602
        raw = json.loads(Path(local).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit(f"Expected a JSON array in {rel}, got {type(raw).__name__}")
        for row_idx, item in enumerate(raw):
            item = dict(item)
            item.setdefault("month", month)      # 원본에 month가 없으면 파일 경로의 월을 사용
            norm = _normalize_item(item, row_idx=row_idx, source_path=rel)
            if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
                by_id[norm["id"]] = norm
        print(f"[source] {rel}: {len(raw)} rows")

    # 2) manifest 순서대로 split 작성
    for split in SPLITS:
        manifest = json.loads((args.manifest_dir / split / "items.json").read_text(encoding="utf-8"))
        items = [by_id[str(m["id"])] for m in manifest if str(m["id"]) in by_id]
        missing = [str(m["id"]) for m in manifest if str(m["id"]) not in by_id]
        if missing:
            print(f"[warn] {split}: {len(missing)}/{len(manifest)} ids missing (e.g. {missing[:5]})")
        out = args.output_dir / split
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "items.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[write] {split}: {len(items)} items -> {out / 'items.json'}")


if __name__ == "__main__":
    main()
