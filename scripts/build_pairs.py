"""Build contrastive (failed, successful) trajectory pairs from the no-PI / PI baseline runs.

Run this LOCALLY (the raw results are not in the repo). It writes, inside the repo:
    data/pairs/<benchmark>.jsonl   one pair per line, with the analyst user message pre-rendered
    data/images/<file>.png         DocVQA page images referenced by the pairs
    data/pairs/summary.json        pair counts per benchmark / source

A pair exists when the two runs disagree on the same task:
    no-PI fail  + PI success  -> source "hindsight" (success saw the reference answer)
    no-PI success + PI fail   -> source "natural"   (success saw nothing extra)

Usage:
    python scripts/build_pairs.py --results-root <.../results/no-skill_baseline>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BENCHMARKS = {  # name used in this repo -> folder prefix used by the baseline runs
    "searchqa": "searchqa",
    "docvqa": "docvqa",
    "officeqa": "officeqa",
    "livemath": "livemathematicianbench",
    "spreadsheetbench": "spreadsheetbench",
}

OBS_HEAD, OBS_TAIL = 800, 400       # tool output truncation
TASK_MAX = 24000                    # task prompt truncation (chars)
MIN_STEP_CHARS = 250                # merge text paragraphs shorter than this


def folder_id(task_id: str) -> str:
    return str(task_id).replace(":", "_").replace("/", "_")


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


def build_user_message(task: str, fail_steps: list[str], fail_eval: str,
                       succ_steps: list[str], succ_answer: str, source: str) -> str:
    return "\n\n".join([
        "## Current Skill\n(empty)",
        "## Task (shared input given to both trajectories)\n" + truncate(task, TASK_MAX - 4000, 4000),
        f"## FAILED trajectory\n{fail_eval}\n\n" + render_steps("F", fail_steps),
        f"## SUCCESSFUL trajectory (source: {source})\nFinal answer: {succ_answer}\nResult: CORRECT\n\n"
        + render_steps("S", succ_steps),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True,
                    help="folder that holds qwen3.5-9B/ and qwen3.5-9B-PI/")
    ap.add_argument("--nopi", default="qwen3.5-9B/run1")
    ap.add_argument("--pi", default="qwen3.5-9B-PI/run1")
    ap.add_argument("--ss-feedback", choices=["cells", "full"], default="cells",
                    help="SpreadsheetBench feedback: 'cells' hides expected values, 'full' keeps them")
    args = ap.parse_args()

    out_dir, img_dir = REPO / "data" / "pairs", REPO / "data" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for bench, prefix in BENCHMARKS.items():
        nopi_dir = args.results_root / args.nopi / f"{prefix}_noskill_full"
        pi_dir = args.results_root / args.pi / f"{prefix}_noskill_pi_full"
        a, b = load_results(nopi_dir / "results.jsonl"), load_results(pi_dir / "results.jsonl")
        rows, counts = [], {"hindsight": 0, "natural": 0, "skipped": 0}

        for tid in sorted(set(a) & set(b)):
            sa, sb = is_success(a[tid]), is_success(b[tid])
            if sa == sb:
                continue
            source = "hindsight" if sb else "natural"
            (f_run, f_dir), (s_run, s_dir) = ((a, nopi_dir), (b, pi_dir)) if sb else ((b, pi_dir), (a, nopi_dir))
            fid = folder_id(tid)
            try:
                f_conv = json.loads((f_dir / "predictions" / fid / "conversation.json").read_text(encoding="utf-8"))
                s_conv = json.loads((s_dir / "predictions" / fid / "conversation.json").read_text(encoding="utf-8"))
                task = (nopi_dir / "predictions" / fid / "target_user_prompt.txt").read_text(encoding="utf-8")
            except FileNotFoundError:
                counts["skipped"] += 1
                continue
            task = strip_reference(task)

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
                # images are named q<task_id>_d<doc_id>.png
                found = sorted((nopi_dir / "images").glob(f"q{fid}_*"))
                if found:
                    shutil.copy2(found[0], img_dir / found[0].name)
                    image = f"data/images/{found[0].name}"

            rows.append({
                "pair_id": f"{bench}:{tid}",
                "benchmark": bench,
                "task_id": tid,
                "source": source,
                "failed_run": "nopi" if sb else "pi",
                "image": image,
                "n_fail_steps": len(f_steps),
                "n_succ_steps": len(s_steps),
                "user_message": build_user_message(task, f_steps, fail_eval, s_steps,
                                                   final_answer(s_run[tid]), source),
            })
            counts[source] += 1

        with open(out_dir / f"{bench}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[bench] = counts
        print(f"{bench:18s} hindsight={counts['hindsight']:4d} natural={counts['natural']:3d} skipped={counts['skipped']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
