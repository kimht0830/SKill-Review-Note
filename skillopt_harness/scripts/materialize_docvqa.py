"""Materialize the runnable DocVQA split used by configs/docvqa/default.yaml
(split_dir: data/docvqa/splits) from the manifest in data/docvqa_id_split/.

Mirrors scripts/materialize_searchqa.py / materialize_spreadsheetbench.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
FIELDS = ["id", "questionId", "docId", "question", "answer", "image_path", "topic",
          "ucsf_document_id", "ucsf_document_page_no", "source_split",
          # src_* = HF 원본에서 읽은 값 (검증 셀이 manifest와 대조하는 용도, 없으면 빈 칸)
          "src_docId", "src_ucsf_document_id", "src_ucsf_document_page_no"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "docvqa_id_split")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "docvqa" / "splits")
    p.add_argument("--revision", default="", help="HF revision (default: manifest source_revision)")
    return p.parse_args()


def load_manifest(manifest_dir: Path):
    meta = json.loads((manifest_dir / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {}
    for split in SPLITS:
        splits[split] = json.loads((manifest_dir / split / "items.json").read_text(encoding="utf-8"))
    return meta, splits


def open_stream(meta: dict, revision: str):
    kwargs = dict(split=meta.get("source_split", "validation"), streaming=True)
    repo, config = meta["source_repo"], meta.get("source_config", "DocVQA")
    try:
        ds = load_dataset(repo, config, revision=revision or None, **kwargs)
    except Exception as e:  # revision이 사라졌거나 접근 불가하면 main으로 폴백
        print(f"[warn] revision={revision!r} 로드 실패 ({type(e).__name__}: {e}) -> main으로 재시도")
        ds = load_dataset(repo, config, **kwargs)
    # 이미지 디코딩은 필요한 항목에서만 하도록 bytes 상태로 받음
    return ds.cast_column("image", HFImage(decode=False))


def save_png(image_field: dict, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = image_field.get("bytes")
    img = Image.open(io.BytesIO(raw)) if raw else Image.open(image_field["path"])
    img.save(dest, format="PNG")  # 확장자(.png)와 실제 포맷을 일치시켜 data URI mime이 맞도록


def main() -> None:
    args = parse_args()
    meta, splits = load_manifest(args.manifest_dir)
    revision = args.revision or meta.get("source_revision", "")
    wanted = {str(it["questionId"]): (split, it) for split, items in splits.items() for it in items}
    print(f"[manifest] need {len(wanted)} questions (revision={revision or 'main'})")

    found: dict[str, dict] = {}
    for ex in open_stream(meta, revision):
        qid = str(ex["questionId"])
        if qid not in wanted or qid in found:
            continue
        split, m = wanted[qid]
        img_path = (PROJECT_ROOT / m["image_path"]).resolve()
        save_png(ex["image"], img_path)
        found[qid] = {
            "id": qid,
            "questionId": qid,
            "docId": str(ex.get("docId", m.get("docId", ""))),
            "question": ex["question"],
            "answer": json.dumps(list(ex.get("answers") or []), ensure_ascii=False),
            "image_path": str(img_path),  # 절대경로: eval 실행 cwd와 무관하게 열리도록
            "topic": m.get("topic", ""),
            "ucsf_document_id": m.get("ucsf_document_id", ""),
            "ucsf_document_page_no": m.get("ucsf_document_page_no", ""),
            "source_split": m.get("source_split", "validation"),
            "src_docId": str(ex.get("docId") or ""),
            "src_ucsf_document_id": str(ex.get("ucsf_document_id") or ""),
            "src_ucsf_document_page_no": str(ex.get("ucsf_document_page_no") or ""),
        }
        if len(found) % 100 == 0:
            print(f"  ... {len(found)}/{len(wanted)}")
        if len(found) == len(wanted):
            break

    for split, items in splits.items():
        rows = [found[str(it["questionId"])] for it in items if str(it["questionId"]) in found]
        missing = len(items) - len(rows)
        if missing:
            print(f"[warn] {split}: {missing}/{len(items)} questionIds not found in source")
        out = args.output_dir / split
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "items.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {split}: {len(rows)} items -> {out / 'items.csv'}")


if __name__ == "__main__":
    main()
