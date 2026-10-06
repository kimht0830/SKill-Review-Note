"""Run SkillOpt's scripts/eval_only.py on a list of (task, skill) jobs instead of one skill on a whole split.

eval_only.py evaluates ONE skill on every item of a split. Rule validation needs a different
skill per task (and several repeats), so this wrapper imports eval_only, swaps get_adapter for
one whose rollout() runs each job as a one-item rollout with the job's own skill, then calls
eval_only.main() unchanged (same config handling, target backend, and scoring).

Jobs come from the JSON file named by $RULECHECK_JOBS:
    [{"job_id": "...", "task_id": "...", "skill": "<skill document text>"}, ...]
Each job writes to <out_root>/jobs/<job_id>/ and one line to <out_root>/job_results.jsonl.
With RULECHECK_PI=1 the reference answer is also injected into the user turn (eval_only_pi.py),
which is the condition the failed trajectory of a "natural" pair was produced under.
Jobs already in job_results.jsonl are skipped (resume).

Run from the skillopt_harness/ root (vendored SkillOpt), with the usual eval_only.py arguments:
    RULECHECK_JOBS=jobs.json python <this> --config configs/searchqa/default.yaml --skill empty.md \
        --split valid_unseen --target_backend qwen_chat --target_model Qwen/Qwen3.5-9B --out_root <dir>
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SKILLOPT_DIR = os.getcwd()
sys.path[:0] = [SKILLOPT_DIR, os.path.join(SKILLOPT_DIR, "scripts")]

import eval_only  # noqa: E402  (SkillOpt/scripts/eval_only.py)

JOBS = json.load(open(os.environ["RULECHECK_JOBS"], encoding="utf-8"))
JOB_WORKERS = int(os.environ.get("RULECHECK_JOB_WORKERS", "16"))
KEEP = ("hard", "soft", "predicted_answer", "predicted_label", "fail_reason", "error")


class JobAdapter:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def rollout(self, items, skill_content, out_dir, **kwargs):
        by_id = {str(it["id"]): it for it in items}
        res_path = os.path.join(out_dir, "job_results.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        done = {}
        if os.path.exists(res_path):
            for line in open(res_path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    done[r["job_id"]] = r
        missing = sorted({j["task_id"] for j in JOBS if str(j["task_id"]) not in by_id})
        if missing:
            print(f"[jobs] {len(missing)} task ids not in this split, skipped: {missing[:10]}", flush=True)
        pending = [j for j in JOBS if j["job_id"] not in done and str(j["task_id"]) in by_id]
        print(f"[jobs] {len(pending)} to run, {len(done)} already done", flush=True)

        lock = threading.Lock()

        def run(job):
            job_dir = os.path.join(out_dir, "jobs", job["job_id"])
            try:
                rows = self._inner.rollout([by_id[str(job["task_id"])]], job["skill"], job_dir, **kwargs)
                r = rows[0] if rows else {"hard": 0, "soft": 0.0, "error": "empty rollout"}
            except Exception as e:  # noqa: BLE001 — one broken job must not stop the rest
                r = {"hard": 0, "soft": 0.0, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            return {"job_id": job["job_id"], "task_id": str(job["task_id"]),
                    **{k: r.get(k) for k in KEEP if k in r}}

        with ThreadPoolExecutor(JOB_WORKERS) as ex, open(res_path, "a", encoding="utf-8") as f:
            futs = [ex.submit(run, j) for j in pending]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    done[r["job_id"]] = r
                print(f"[job {i}/{len(pending)}] {r['job_id']} hard={r.get('hard')}"
                      + (f" ERROR {r['error']}" if r.get("error") else ""), flush=True)
        return list(done.values())


if os.environ.get("RULECHECK_PI") == "1":
    # natural pairs failed WITH the reference answer: re-create that condition (eval_only_pi.py patches)
    import atexit

    import eval_only_pi as _pi

    _pi._PATCHES[_pi._ENV]()
    atexit.register(_pi._report)
    print(f"  [PI] privileged-information injection enabled for env={_pi._ENV}", flush=True)

_orig_get_adapter = eval_only.get_adapter
eval_only.get_adapter = lambda cfg: JobAdapter(_orig_get_adapter(cfg))

if __name__ == "__main__":
    eval_only.main()
