"""Minimum validity check for analyst rules: does the failed task pass once the rule is added?

For every rule in <analyst-dir>/<bench>.jsonl, the target agent re-solves the pair's task under the
condition its FAILED trajectory was produced in, with the rule added to the system prompt
(SkillOpt puts the skill document in a "## Skill" section of the system prompt; the skill here is
only the rule), --repeats times:
    hindsight pair -> failed run was no-PI  -> re-run no-PI
    natural pair   -> failed run was PI     -> re-run with the reference answer in the user turn
As a control, the same task is re-run under the same condition with the empty skill --repeats
times, so a rule is not credited for a pass that sampling alone would give.

    valid  <=>  rule_pass > control_pass   (and rule_pass >= 1)

Output (resumable — finished jobs are skipped on re-run):
    <out>/<bench>/<nopi|pi>/jobs.json, job_results.jsonl, jobs/<job_id>/   raw SkillOpt rollouts
    <out>/rule_validity.jsonl                                               one line per rule
    <out>/summary.md

Usage (from the Colab notebook):
    python scripts/validate_rules.py --analyst-dir <drive>/analyst_v1 --pairs-dir <drive>/pairs/pairs \
        --out <drive>/validity_v1 --split train --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "skillopt_harness"   # vendored SkillOpt (commit fa4ca18) used by the no-skill rollouts
BENCHMARKS = ["searchqa", "docvqa", "officeqa", "livemath", "spreadsheetbench"]
SKILLOPT_ENV = {"livemath": "livemathematicianbench"}  # repo name -> SkillOpt env/config name


def skill_doc(rule: str) -> str:
    # rendered by SkillOpt as "## Skill\n<this>" inside the agent's system prompt
    return f"- {rule.strip()}\n"


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_rules(analyst_dir: Path, pairs_dir: Path, bench: str, sources: list[str]) -> list[dict]:
    path = analyst_dir / f"{bench}.jsonl"
    if not path.exists():
        return []
    failed_run = {}
    if (pairs_dir / f"{bench}.jsonl").exists():
        failed_run = {p["pair_id"]: p["failed_run"] for p in read_jsonl(pairs_dir / f"{bench}.jsonl")}
    rules = []
    for r in read_jsonl(path):
        p = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
        rule = p.get("rule")
        if r.get("source") in sources and isinstance(rule, str) and rule.strip() and rule.strip() != "None":
            run = failed_run.get(r["pair_id"]) or ("nopi" if r["source"] == "hindsight" else "pi")
            rules.append({"pair_id": r["pair_id"], "task_id": str(r["task_id"]), "source": r["source"],
                          "failed_run": run, "rule": rule.strip(), "recoverable": p.get("recoverable")})
    return rules


def build_jobs(rules: list[dict], repeats: int, control: bool) -> list[dict]:
    jobs = []
    for r in rules:
        for k in range(repeats):
            jobs.append({"job_id": f"{safe(r['pair_id'])}__rule__r{k}", "task_id": r["task_id"],
                         "skill": skill_doc(r["rule"])})
    if control:
        for tid in sorted({r["task_id"] for r in rules}):
            for k in range(repeats):
                jobs.append({"job_id": f"{safe(tid)}__control__r{k}", "task_id": tid, "skill": ""})
    return jobs


def run_jobs(args, bench: str, run: str, jobs: list[dict]) -> None:
    env_name = SKILLOPT_ENV.get(bench, bench)
    out_root = args.out / bench / run
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_path = out_root / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    empty = out_root / "empty_skill.md"
    empty.write_text("", encoding="utf-8")
    split = {"test": "valid_unseen"}.get(args.split, args.split)

    cmd = [sys.executable, str(args.skillopt_dir / "scripts" / "eval_jobs.py"),
           "--config", f"configs/{env_name}/default.yaml", "--skill", str(empty),
           "--split", split, "--target_backend", "qwen_chat", "--target_model", args.model,
           "--out_root", str(out_root),
           "--cfg-options", f"env.workers={args.workers}",
           "model.target_qwen_chat_thinking_mode=disabled",
           f"model.target_qwen_chat_temperature={args.temperature}",
           f"model.target_qwen_chat_max_tokens={args.max_tokens}"]
    env = dict(os.environ, RULECHECK_JOBS=str(jobs_path), RULECHECK_JOB_WORKERS=str(args.job_workers),
               RULECHECK_PI="1" if run == "pi" else "0", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(args.skillopt_dir), os.environ.get("PYTHONPATH")])))
    print(f"\n===== {bench} / failed run = {run}: {len(jobs)} jobs =====", flush=True)
    with open(out_root / "eval_stdout.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=args.skillopt_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            # the full log is kept in eval_stdout.log; show only job progress and errors
            if line.startswith(("[job", "  Results", "  [PI]")) or re.search(r"Error|Traceback", line):
                print(line, end="", flush=True)
        if proc.wait() != 0:
            print(f"[{bench}/{run}] eval_jobs exited with {proc.returncode} — see {out_root / 'eval_stdout.log'}")


def aggregate(args, rules_by_bench: dict[str, list[dict]]) -> str:
    out_rows = []
    for bench, rules in rules_by_bench.items():
        res = {}
        for run in ("nopi", "pi"):
            path = args.out / bench / run / "job_results.jsonl"
            if path.exists():
                res[run] = {r["job_id"]: r for r in read_jsonl(path)}

        def passes(run: str, prefix: str) -> tuple[int, int]:
            rs = [res.get(run, {}).get(f"{prefix}__r{k}") for k in range(args.repeats)]
            rs = [r for r in rs if r is not None]
            return sum(1 for r in rs if r.get("hard") in (1, True, 1.0)), len(rs)

        for r in rules:
            rp, rn = passes(r["failed_run"], f"{safe(r['pair_id'])}__rule")
            cp, cn = passes(r["failed_run"], f"{safe(r['task_id'])}__control") if args.control else (0, 0)
            complete = rn == args.repeats and (cn == args.repeats or not args.control)
            out_rows.append({**r, "benchmark": bench, "rule_pass": rp, "rule_n": rn,
                             "control_pass": cp, "control_n": cn, "complete": complete,
                             "valid": complete and rp >= 1 and rp > cp})
    with open(args.out / "rule_validity.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    k = args.repeats
    lines = ["# Rule validity check", "",
             f"Each rule is re-run {k}x on its failed task, under the failed trajectory's condition "
             "(hindsight: no-PI, natural: PI), with only the rule added to the system prompt"
             + (f"; control = {k}x with the empty skill." if args.control else "."),
             "valid = rule passes > control passes (and at least one pass).", "",
             "| benchmark | source | rules | complete | valid | mean rule pass | mean control pass |",
             "|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in out_rows:
        groups[(r["benchmark"], r["source"])].append(r)
    for (b, s), rs in sorted(groups.items()):
        done = [r for r in rs if r["complete"]]
        mr = sum(r["rule_pass"] for r in done) / (k * len(done)) if done else 0
        mc = sum(r["control_pass"] for r in done) / (k * len(done)) if done else 0
        lines.append(f"| {b} | {s} | {len(rs)} | {len(done)} | {sum(r['valid'] for r in done)} | "
                     f"{mr:.2f} | {mc:.2f} |")
    lines += ["", "## Valid rules", ""]
    lines += [f"- `{r['pair_id']}` ({r['source']}, rule {r['rule_pass']}/{k}, control {r['control_pass']}/{k}): "
              f"{r['rule']}" for r in out_rows if r["valid"]]
    text = "\n".join(lines)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyst-dir", type=Path, required=True, help="folder with <bench>.jsonl analyst outputs")
    ap.add_argument("--pairs-dir", type=Path, required=True, help="folder with <bench>.jsonl pairs (for failed_run)")
    ap.add_argument("--skillopt-dir", type=Path, default=HARNESS,
                    help="SkillOpt harness root with materialized data (default: vendored skillopt_harness/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", help="split the pairs came from (train / val / test)")
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--sources", nargs="+", default=["hindsight", "natural"])
    ap.add_argument("--limit", type=int, default=0, help="max rules per benchmark (0 = all)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-control", dest="control", action="store_false")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4, help="SkillOpt env.workers inside one job")
    ap.add_argument("--job-workers", type=int, default=16, help="jobs run concurrently")
    ap.add_argument("--aggregate-only", action="store_true", help="skip rollouts, rebuild the summary")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rules_by_bench = {}
    for bench in args.benchmarks:
        rules = load_rules(args.analyst_dir, args.pairs_dir, bench, args.sources)
        if args.limit:
            rules = rules[:args.limit]
        if not rules:
            print(f"[{bench}] no rules to check")
            continue
        rules_by_bench[bench] = rules
        if args.aggregate_only:
            continue
        for run in ("nopi", "pi"):
            sub = [r for r in rules if r["failed_run"] == run]
            if sub:
                run_jobs(args, bench, run, build_jobs(sub, args.repeats, args.control))
    print(aggregate(args, rules_by_bench))


if __name__ == "__main__":
    main()
