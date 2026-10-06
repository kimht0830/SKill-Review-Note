"""Stage 1: no-skill rollouts (with or without the reference answer) on a SkillOpt split.

Runs the vendored SkillOpt harness (skillopt_harness/, commit fa4ca18 — the one used for the
earlier no-skill baselines) with an empty skill, once per benchmark:
    --mode nopi  ->  scripts/eval_only.py
    --mode pi    ->  scripts/eval_only_pi.py  (reference answer prepended to the user turn, pi_config.json)

With --skill-dir, <skill-dir>/<benchmark>.md is used as the skill instead (benchmarks without
one are skipped) — used to evaluate a skill built from validated rules on the test split.

Output: <out-root>/<benchmark>/ with results.jsonl, predictions/<id>/, eval_summary.json, run_config.json.
Re-running resumes: finished items are skipped (SpreadsheetBench returns an existing results.jsonl as is).

    python scripts/run_rollouts.py --mode nopi --split train --out-root <drive>/rollouts/nopi
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "skillopt_harness"
BENCHMARKS = ["searchqa", "docvqa", "officeqa", "livemath", "spreadsheetbench"]
SKILLOPT_ENV = {"livemath": "livemathematicianbench"}
_PROGRESS = re.compile(r"^\s*(?:\[rollout\]\s*)?(\d+)/(\d+)\b")


def run_streaming(cmd: list[str], log_path: Path, env: dict, every: int) -> int:
    """Show the harness output in the cell (progress lines thinned to every N) and keep it all in a log."""
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=HARNESS, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                m = _PROGRESS.match(line)
                if m and every > 1 and int(m.group(1)) % every and m.group(1) != m.group(2):
                    continue
                print(line, end="", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nopi", "pi"], required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--split", default="train", help="train / val / test (SkillOpt split names)")
    ap.add_argument("--limit", type=int, default=0, help="first N items per benchmark (0 = whole split)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--skill-dir", type=Path, help="use <skill-dir>/<benchmark>.md as the skill (default: empty)")
    args = ap.parse_args()

    empty = args.out_root / "empty_skill.md"
    args.out_root.mkdir(parents=True, exist_ok=True)
    empty.write_text("", encoding="utf-8")
    script = "scripts/eval_only_pi.py" if args.mode == "pi" else "scripts/eval_only.py"
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(HARNESS), os.environ.get("PYTHONPATH")])))
    split = {"test": "valid_unseen"}.get(args.split, args.split)

    failed = []
    for bench in args.benchmarks:
        env_name = SKILLOPT_ENV.get(bench, bench)
        skill = empty
        if args.skill_dir:
            skill = args.skill_dir / f"{bench}.md"
            if not skill.exists():
                print(f"===== {bench}: no skill file {skill} -> skipped =====", flush=True)
                continue
        out = args.out_root / bench
        out.mkdir(parents=True, exist_ok=True)
        cfg = [f"env.workers={args.workers}", "model.target_qwen_chat_thinking_mode=disabled",
               f"model.target_qwen_chat_temperature={args.temperature}",
               f"model.target_qwen_chat_max_tokens={args.max_tokens}"]
        if args.limit:
            cfg.append(f"evaluation.test_env_num={args.limit}")
        cmd = [sys.executable, script, "--config", f"configs/{env_name}/default.yaml", "--skill", str(skill),
               "--split", split, "--target_backend", "qwen_chat", "--target_model", args.model,
               "--out_root", str(out), "--cfg-options", *cfg]
        (out / "run_config.json").write_text(json.dumps(
            {"mode": args.mode, "split": args.split, "model": args.model, "limit": args.limit,
             "cfg_options": cfg, "skill": str(skill) if args.skill_dir else "none (empty file)", "harness": "skillopt_harness (SkillOpt fa4ca18)"}, indent=2), encoding="utf-8")
        print(f"\n===== {args.mode} / {bench} =====\n$ {' '.join(cmd)}", flush=True)
        if run_streaming(cmd, out / "eval_stdout.log", env, args.print_every) != 0:
            failed.append(bench)
    if failed:
        raise SystemExit(f"failed: {failed} — see <out-root>/<bench>/eval_stdout.log")


if __name__ == "__main__":
    main()
