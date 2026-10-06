"""Run the contrastive analyst prompt over trajectory pairs with a vLLM OpenAI-compatible server.

Start the server first (see notebooks/run_contrastive_analyst.ipynb), then:
    python scripts/run_analyst.py --base-url http://localhost:8000/v1 --benchmarks officeqa livemath --limit 20

Outputs (resumable — finished pair_ids are skipped on re-run):
    outputs/<tag>/<benchmark>.jsonl    one analyst result per pair
    outputs/<tag>/summary.md           parse rate, recoverable rate, sample rules
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
BENCHMARKS = ["searchqa", "docvqa", "officeqa", "livemath", "spreadsheetbench"]

SCHEMA = {
    "type": "object",
    "properties": {
        "divergence_step": {"type": "string"},
        "missed_evidence": {"type": "string"},
        "recoverable": {"type": "boolean"},
        "how_without_answer": {"type": "string"},
        "rule": {"type": "string"},
        "already_in_skill": {"type": "boolean"},
    },
    "required": ["divergence_step", "missed_evidence", "recoverable",
                 "how_without_answer", "rule", "already_in_skill"],
    "additionalProperties": False,
}


def extract_json(text: str) -> dict | None:
    """Fenced block -> bare {...} -> json_repair on a single object (same policy as SkillOpt)."""
    if not text:
        return None
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    for pat in (r"```json\s*(.*?)```", r"\{.*\}"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1) if m.groups() else m.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        from json_repair import repair_json
        obj = repair_json(m.group(0), return_objects=True)
        return obj if isinstance(obj, dict) and obj else None
    except Exception:  # noqa: BLE001
        return None


def load_pairs(benchmarks: list[str], limit: int, sources: list[str], seed: int) -> list[dict]:
    pairs = []
    for b in benchmarks:
        rows = [json.loads(l) for l in (REPO / "data" / "pairs" / f"{b}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r["source"] in sources]
        if limit and len(rows) > limit:
            # keep natural pairs (rare) first, then a fixed random sample of the rest
            nat = [r for r in rows if r["source"] == "natural"]
            rest = [r for r in rows if r["source"] != "natural"]
            random.Random(seed).shuffle(rest)
            rows = (nat + rest)[:limit]
        pairs.extend(rows)
    return pairs


def build_messages(system: str, pair: dict) -> list[dict]:
    if pair.get("image"):
        img = (REPO / pair["image"]).read_bytes()
        mime = "image/png" if pair["image"].endswith(".png") else "image/jpeg"
        user = [
            {"type": "text", "text": "Document image (shared input given to both trajectories):"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(img).decode()}"}},
            {"type": "text", "text": pair["user_message"]},
        ]
    else:
        user = pair["user_message"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call(client: OpenAI, args, system: str, pair: dict) -> dict:
    extra = {"chat_template_kwargs": {"enable_thinking": args.thinking}, "top_k": 20}
    kw = {}
    if args.structured:
        kw["response_format"] = {"type": "json_schema",
                                 "json_schema": {"name": "contrastive_analysis", "schema": SCHEMA}}
    out = {"pair_id": pair["pair_id"], "benchmark": pair["benchmark"], "source": pair["source"],
           "task_id": pair["task_id"]}
    try:
        resp = client.chat.completions.create(
            model=args.model, messages=build_messages(system, pair),
            temperature=args.temperature, top_p=0.8, max_tokens=args.max_tokens,
            extra_body=extra, **kw)
        raw = resp.choices[0].message.content or ""
        parsed = extract_json(raw)
        out.update(raw=raw, parsed=parsed, json_ok=parsed is not None,
                   finish_reason=resp.choices[0].finish_reason,
                   prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
                   completion_tokens=resp.usage.completion_tokens if resp.usage else None)
    except Exception as e:  # noqa: BLE001 — e.g. prompt longer than max-model-len
        out.update(raw=None, parsed=None, json_ok=False, error=f"{type(e).__name__}: {str(e)[:500]}")
    return out


def summarize(out_dir: Path) -> str:
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["source"])].append(r)
    lines = ["# Contrastive analyst results", "",
             "| benchmark | source | n | error | json_ok | recoverable | already_in_skill | median prompt tok |",
             "|---|---|---|---|---|---|---|---|"]
    for (b, s), rs in sorted(groups.items()):
        ok = [r for r in rs if r["json_ok"]]
        rec = sum(1 for r in ok if r["parsed"].get("recoverable") is True)
        ais = sum(1 for r in ok if r["parsed"].get("already_in_skill") is True)
        toks = sorted(r["prompt_tokens"] for r in rs if r.get("prompt_tokens"))
        med = toks[len(toks) // 2] if toks else "-"
        lines.append(f"| {b} | {s} | {len(rs)} | {sum(1 for r in rs if r.get('error'))} | "
                     f"{len(ok)}/{len(rs)} | {rec}/{len(ok)} | {ais}/{len(ok)} | {med} |")
    lines += ["", "## Sample rules (first 5 recoverable per benchmark)", ""]
    for b in BENCHMARKS:
        rs = [r for r in rows if r["benchmark"] == b and r["json_ok"] and r["parsed"].get("recoverable")]
        if rs:
            lines.append(f"### {b}")
            lines += [f"- `{r['pair_id']}` ({r['source']}): {r['parsed'].get('rule', '')}" for r in rs[:5]]
            lines.append("")
    errs = [r for r in rows if r.get("error")]
    if errs:
        lines += ["## Errors (first 5)", ""] + [f"- `{r['pair_id']}`: {r['error'][:200]}" for r in errs[:5]]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--sources", nargs="+", default=["hindsight", "natural"])
    ap.add_argument("--limit", type=int, default=0, help="max pairs per benchmark (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="prompts/contrastive_analyst.md")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking (default off)")
    ap.add_argument("--no-structured", dest="structured", action="store_false",
                    help="disable JSON-schema constrained decoding")
    args = ap.parse_args()

    system = (REPO / args.prompt).read_text(encoding="utf-8")
    out_dir = REPO / "outputs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    done = set()
    for f in out_dir.glob("*.jsonl"):
        done |= {json.loads(l)["pair_id"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    pairs = [p for p in load_pairs(args.benchmarks, args.limit, args.sources, args.seed) if p["pair_id"] not in done]
    print(f"{len(pairs)} pairs to run ({len(done)} already done)", flush=True)

    client = OpenAI(base_url=args.base_url, api_key="dummy", timeout=1800)
    files = {b: open(out_dir / f"{b}.jsonl", "a", encoding="utf-8") for b in args.benchmarks}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(call, client, args, system, p) for p in pairs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            files[r["benchmark"]].write(json.dumps(r, ensure_ascii=False) + "\n")
            files[r["benchmark"]].flush()
            status = "ERR" if r.get("error") else ("ok" if r["json_ok"] else "bad-json")
            print(f"[{i}/{len(pairs)}] {r['pair_id']} {status}", flush=True)
    for f in files.values():
        f.close()
    print(summarize(out_dir))


if __name__ == "__main__":
    sys.exit(main())
