"""Build contrastive (failed, successful) trajectory pairs from the no-PI / PI rollouts.

Reads the stage-1 rollouts (scripts/run_rollouts.py) and writes, under --out-dir:
    pairs/<benchmark>.jsonl   one pair per line, with the analyst user message pre-rendered
    images/<file>.png         DocVQA page images referenced by the pairs (path relative to --out-dir)
    pairs/summary.json        pair counts per benchmark / source

A pair exists when the two runs disagree on the same task:
    no-PI fail  + PI success  -> source "hindsight" (success saw the reference answer)
    no-PI success + PI fail   -> source "natural"   (success saw nothing extra)

Usage (notebook layout, one folder per benchmark):
    python scripts/build_pairs.py --nopi-root <drive>/rollouts/nopi --pi-root <drive>/rollouts/pi --out-dir <drive>/pairs
Earlier baseline layout:
    --nopi-root .../qwen3.5-9B/run1 --nopi-fmt "{prefix}_noskill_full" \
    --pi-root .../qwen3.5-9B-PI/run1 --pi-fmt "{prefix}_noskill_pi_full"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "skillopt_harness"

BENCHMARKS = {  # name used in this repo -> SkillOpt env name (folder prefix of the earlier baseline runs)
    "searchqa": "searchqa",
    "docvqa": "docvqa",
    "officeqa": "officeqa",
    "livemath": "livemathematicianbench",
    "spreadsheetbench": "spreadsheetbench",
}

OBS_HEAD, OBS_TAIL = 800, 400       # tool output truncation
TASK_MAX = 24000                    # task prompt truncation (chars)
MIN_STEP_CHARS = 250                # merge text paragraphs shorter than this


def pred_dir(run_dir: Path, task_id: str) -> Path:
    """predictions/<id>/ as written by SkillOpt. On Linux (Colab) the raw id is used (e.g. LiveMath
    "202602:21"); copies made on Windows have ":" and "/" replaced by "_"."""
    raw = run_dir / "predictions" / str(task_id)
    if raw.is_dir():
        return raw
    return run_dir / "predictions" / str(task_id).replace(":", "_").replace("/", "_")


def is_success(r: dict) -> bool:
    return r.get("hard") in (1, True, 1.0, "1")


def load_results(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


def truncate(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 50:
        return text
    return f"{text[:head]}\n... [{len(text) - head - tail} chars omitted] ...\n{text[-tail:]}"


def split_paragraphs(text: str) -> list[str]:
    """Split a long free-text response into steps (paragraphs), merging tiny ones."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    steps, buf = [], ""
    for p in parts:
        buf = f"{buf}\n\n{p}" if buf else p
        if len(buf) >= MIN_STEP_CHARS:
            steps.append(buf)
            buf = ""
    if buf:
        if steps and len(buf) < MIN_STEP_CHARS:
            steps[-1] += "\n\n" + buf
        else:
            steps.append(buf)
    return steps


def mask_spreadsheet_feedback(text: str, mode: str) -> str:
    """mode 'cells': keep which cells are wrong and what was produced, hide expected values."""
    # collapse the per-cell list: keep wrong cells, count the correct ones
    lines = text.splitlines()
    n_ok = sum(1 for l in lines if l.rstrip().endswith("✓"))
    lines = [l for l in lines if not l.rstrip().endswith("✓")]
    if n_ok:
        lines.append(f"  ({n_ok} other checked cells correct)")
    text = "\n".join(lines)
    if mode == "full":
        return text
    text = re.sub(r"expected=.*?(?=\s*[✓✗]|$)", "expected=<hidden>", text, flags=re.M)
    text = re.sub(r"gt=.*? pred=", "pred=", text)
    return text


def trajectory_steps(conv: list[dict], bench: str, ss_feedback: str) -> list[str]:
    steps: list[str] = []
    for m in conv:
        role, typ = m.get("role"), m.get("type")
        if role == "user":
            continue  # task prompt is shown once in the shared section
        if typ == "tool_call":
            obs = "" if m.get("obs") is None else str(m.get("obs"))
            steps.append(f"TOOL CALL: {m.get('cmd', '')}\nTOOL OUTPUT:\n{truncate(obs, OBS_HEAD, OBS_TAIL)}")
            continue
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else json.dumps(content, ensure_ascii=False)
        if not content.strip():
            continue
        if role == "system":
            # QA evaluators print the gold answer here -> drop. Spreadsheet execution feedback is kept.
            if bench == "spreadsheetbench" and "VERIFICATION" in content:
                steps.append("EXECUTION FEEDBACK:\n" + mask_spreadsheet_feedback(content, ss_feedback))
            continue
        steps.extend(split_paragraphs(content))
    return steps


def render_steps(prefix: str, steps: list[str]) -> str:
    return "\n\n".join(f"[{prefix}{i}] {s}" for i, s in enumerate(steps, 1))


def final_answer(r: dict) -> str:
    for k in ("predicted_answer", "predicted_label"):
        if r.get(k) not in (None, ""):
            return str(r[k])
    return "(see trajectory)"


def strip_reference(prompt: str) -> str:
    """Remove the PI header block ('## Reference answer' ... up to the next task heading)."""
    prompt = re.sub(r"(?s)^#+ Reference answer.*?(?=^#+ (?:Instructions?|Question|Instruction)\b)", "", prompt, flags=re.M)
    prompt = re.sub(r"(?s)^#+ Notes on the reference answer.*?(?=^#+ )", "", prompt, flags=re.M)
    return prompt.strip()


def docvqa_image_paths() -> dict[str, str]:
    """question id -> page image, from the harness DocVQA split files (written by materialize_docvqa.py)."""
    out = {}
    for f in (HARNESS / "data" / "docvqa" / "splits").glob("*/items.csv"):
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out[str(row.get("id") or row.get("questionId"))] = row.get("image_path", "")
    return out


def build_user_message(system: str, task: str, fail_steps: list[str], fail_eval: str,
                       succ_steps: list[str], succ_answer: str, source: str) -> str:
    return "\n\n".join([
        "## Current Skill\n(empty)",
        "## Agent instructions (system prompt given to both trajectories)\n" + (system.strip() or "(not recorded)"),
        "## Task (shared input given to both trajectories)\n" + truncate(task, TASK_MAX - 4000, 4000),
        "## FAILED trajectory"
        + (" (was shown the reference answer)" if source == "natural" else "")
        + f"\n{fail_eval}\n\n" + render_steps("F", fail_steps),
        f"## SUCCESSFUL trajectory (source: {source})\nFinal answer: {succ_answer}\nResult: CORRECT\n\n"
        + render_steps("S", succ_steps),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nopi-root", type=Path, required=True, help="no-PI rollouts, one folder per benchmark")
    ap.add_argument("--pi-root", type=Path, required=True, help="PI rollouts, one folder per benchmark")
    ap.add_argument("--nopi-fmt", default="{bench}", help="benchmark folder name ({bench} or {prefix})")
    ap.add_argument("--pi-fmt", default="{bench}")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--ss-feedback", choices=["cells", "full"], default="cells",
                    help="SpreadsheetBench feedback: 'cells' hides expected values, 'full' keeps them")
    args = ap.parse_args()

    out_dir, img_dir = args.out_dir / "pairs", args.out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    doc_images = docvqa_image_paths() if "docvqa" in args.benchmarks else {}

    for bench in args.benchmarks:
        prefix = BENCHMARKS[bench]
        nopi_dir = args.nopi_root / args.nopi_fmt.format(bench=bench, prefix=prefix)
        pi_dir = args.pi_root / args.pi_fmt.format(bench=bench, prefix=prefix)
        if not (nopi_dir / "results.jsonl").exists() or not (pi_dir / "results.jsonl").exists():
            print(f"{bench:18s} skipped: results.jsonl missing under {nopi_dir} or {pi_dir}")
            continue
        a, b = load_results(nopi_dir / "results.jsonl"), load_results(pi_dir / "results.jsonl")
        rows, counts = [], {"hindsight": 0, "natural": 0, "skipped": 0}

        for tid in sorted(set(a) & set(b)):
            sa, sb = is_success(a[tid]), is_success(b[tid])
            if sa == sb:
                continue
            source = "hindsight" if sb else "natural"
            (f_run, f_dir), (s_run, s_dir) = ((a, nopi_dir), (b, pi_dir)) if sb else ((b, pi_dir), (a, nopi_dir))
            f_pred, s_pred = pred_dir(f_dir, tid), pred_dir(s_dir, tid)
            try:
                f_conv = json.loads((f_pred / "conversation.json").read_text(encoding="utf-8"))
                s_conv = json.loads((s_pred / "conversation.json").read_text(encoding="utf-8"))
                task = (pred_dir(nopi_dir, tid) / "target_user_prompt.txt").read_text(encoding="utf-8")
            except FileNotFoundError as e:
                counts["skipped"] += 1
                if counts["skipped"] <= 3:
                    print(f"  [{bench}] skipped {tid}: {e.filename} not found")
                continue
            task = strip_reference(task)
            sys_path = f_pred / "target_system_prompt.txt"
            system = sys_path.read_text(encoding="utf-8") if sys_path.exists() else ""

            f_steps = trajectory_steps(f_conv, bench, args.ss_feedback)
            s_steps = trajectory_steps(s_conv, bench, args.ss_feedback)
            fr = f_run[tid]
            fail_eval = f"Final answer: {final_answer(fr)}\nResult: WRONG"
            if fr.get("fail_reason"):
                reason = str(fr["fail_reason"])
                if bench == "spreadsheetbench":
                    reason = mask_spreadsheet_feedback(reason, args.ss_feedback)
                    fail_eval += f" ({reason[:300]})"

            image = None
            if bench == "docvqa":
                # earlier baselines saved <run>/images/q<task_id>_d<doc_id>.png;
                # new rollouts read the page from the harness split (item image_path)
                found = sorted((nopi_dir / "images").glob(f"q{pred_dir(nopi_dir, tid).name}_*"))
                src = found[0] if found else Path(doc_images.get(tid) or "/nonexistent")
                if src.is_file():
                    shutil.copy2(src, img_dir / src.name)
                    image = f"images/{src.name}"

            rows.append({
                "pair_id": f"{bench}:{tid}",
                "benchmark": bench,
                "task_id": tid,
                "source": source,
                "failed_run": "nopi" if sb else "pi",
                "image": image,
                "n_fail_steps": len(f_steps),
                "n_succ_steps": len(s_steps),
                "user_message": build_user_message(system, task, f_steps, fail_eval, s_steps,
                                                   final_answer(s_run[tid]), source),
            })
            counts[source] += 1

        with open(out_dir / f"{bench}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[bench] = {**counts, "n_nopi": len(a), "nopi_acc": sum(map(is_success, a.values())) / max(len(a), 1),
                          "n_pi": len(b), "pi_acc": sum(map(is_success, b.values())) / max(len(b), 1)}
        print(f"{bench:18s} no-PI {summary[bench]['nopi_acc']:.1%} (n={len(a)})  PI {summary[bench]['pi_acc']:.1%} "
              f"(n={len(b)})  hindsight={counts['hindsight']:4d} natural={counts['natural']:3d} skipped={counts['skipped']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
