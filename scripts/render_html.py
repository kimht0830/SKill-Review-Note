"""Render analyst outputs (<run_dir>/<bench>.jsonl) into one self-contained HTML viewer.

Each card shows the parsed analyst fields; the analyst input (task + both trajectories)
from --pairs (build_pairs.py output) is attached as a collapsible section when available.

Usage:
    python scripts/render_html.py <run_dir> [--pairs <pairs dir>] [--out <run_dir>/report.html]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIELDS = ["divergence_step", "missed_evidence", "recoverable", "how_without_answer", "rule", "already_in_skill"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--pairs", type=Path, help="folder with <bench>.jsonl pairs, to attach the analyst input")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = []
    for f in sorted(args.run_dir.glob("*.jsonl")):
        inputs = {}
        pf = args.pairs / f.name if args.pairs else None
        if args.pairs and pf.exists():
            inputs = {p["pair_id"]: p["user_message"] for p in load_jsonl(pf)}
        for r in load_jsonl(f):
            p = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
            empty = not any(p.get(k) not in (None, "", "None") for k in ("divergence_step", "rule"))
            rows.append({
                "id": r["pair_id"], "bench": r.get("benchmark", f.stem), "source": r.get("source", ""),
                "ok": bool(r.get("json_ok")) and not empty, "finish": r.get("finish_reason", ""),
                "ptok": r.get("prompt_tokens"), "ctok": r.get("completion_tokens"),
                "fields": {k: p.get(k) for k in FIELDS}, "raw": "" if p and not empty else str(r.get("raw", "")),
                "input": inputs.get(r["pair_id"], ""),
            })

    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    out = args.out or args.run_dir / "report.html"
    out.write_text(TEMPLATE.replace("__TITLE__", args.run_dir.name).replace("__DATA__", data), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} pairs)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analyst Report __TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mute:#6b7280;--line:#e3e6eb;--acc:#2f6fde;--ok:#1a7f4b;--bad:#c0392b;--hi:#fff6d6;--nat:#7b4fd0;--hin:#c26a00}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--card:#1d2027;--fg:#e6e8ec;--mute:#9aa1ad;--line:#2d313a;--acc:#6b9cff;--ok:#4cc38a;--bad:#ff7a6b;--hi:#3a3420;--nat:#b292ff;--hin:#ffa94d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px}
h1{font-size:17px;margin:0 0 8px}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
input[type=search]{flex:1;min-width:180px}.count{color:var(--mute);font-size:13px}
table.sum{border-collapse:collapse;margin:12px 16px;font-size:13px;background:var(--card)}
table.sum td,table.sum th{border:1px solid var(--line);padding:4px 10px;text-align:right}table.sum th:first-child,table.sum td:first-child{text-align:left}
main{padding:0 16px 40px;max-width:1100px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.card.bad{border-left:4px solid var(--bad)}
.top{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}.pid{font-weight:600;font-family:ui-monospace,Consolas,monospace}
.tag{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor}.natural{color:var(--nat)}.hindsight{color:var(--hin)}
.t-ok{color:var(--ok)}.t-bad{color:var(--bad)}.meta{color:var(--mute);font-size:12px;margin-left:auto}
.rule{background:var(--hi);border:1px solid var(--hin);border-radius:6px;padding:8px 10px;margin:6px 0 10px;font-weight:500}.rl{display:inline-block;font-size:11px;font-weight:700;color:var(--hin);margin-right:8px;letter-spacing:.05em}
dl{margin:0;display:grid;grid-template-columns:150px 1fr;gap:4px 12px}dt{color:var(--mute);font-size:12px;padding-top:2px}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
details{margin-top:10px}summary{cursor:pointer;color:var(--acc);font-size:13px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;font-size:12px;max-height:600px;overflow:auto}
@media (max-width:600px){dl{grid-template-columns:1fr}dt{padding-top:6px}}
</style></head><body>
<header><h1>Contrastive analyst — __TITLE__</h1>
<div class="bar"><select id="fb"><option value="">all benchmarks</option></select>
<select id="fs"><option value="">all sources</option><option>hindsight</option><option>natural</option></select>
<select id="fo"><option value="">all outputs</option><option value="ok">parsed OK</option><option value="bad">failed / empty</option><option value="nr">recoverable = false</option></select>
<input type="search" id="q" placeholder="search pair id or text…"><span class="count" id="cnt"></span></div></header>
<table class="sum" id="sum"></table><main id="list"></main>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s),esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const benches=[...new Set(D.map(r=>r.bench))];benches.forEach(b=>$("#fb").insertAdjacentHTML("beforeend",`<option>${b}</option>`));
let h="<tr><th>benchmark</th><th>source</th><th>n</th><th>parsed</th><th>recoverable</th></tr>";
for(const b of benches)for(const s of ["hindsight","natural"]){const g=D.filter(r=>r.bench==b&&r.source==s);if(!g.length)continue;
const ok=g.filter(r=>r.ok);h+=`<tr><td>${b}</td><td>${s}</td><td>${g.length}</td><td>${ok.length}</td><td>${ok.filter(r=>r.fields.recoverable===true).length}</td></tr>`}
$("#sum").innerHTML=h;
const LBL={divergence_step:"divergence",missed_evidence:"missed evidence",recoverable:"recoverable",how_without_answer:"how w/o answer",already_in_skill:"already in skill"};
function card(r){const f=r.fields;let dl="";for(const k in LBL)if(f[k]!==undefined&&f[k]!==null)dl+=`<dt>${LBL[k]}</dt><dd>${esc(f[k])}</dd>`;
return `<div class="card ${r.ok?"":"bad"}"><div class="top"><span class="pid">${esc(r.id)}</span><span class="tag ${r.source}">${r.source}</span>
<span class="${r.ok?"t-ok":"t-bad"}">${r.ok?"parsed":"FAILED ("+esc(r.finish)+")"}</span>${f.recoverable===false?'<span class="t-bad">not recoverable</span>':""}
<span class="meta">prompt ${r.ptok??"?"} / completion ${r.ctok??"?"} tok</span></div>
${r.ok?`<div class="rule"><span class="rl">RULE</span>${f.rule?esc(f.rule):'<i>(empty)</i>'}</div>`:""}${r.ok?`<dl>${dl}</dl>`:""}
${r.raw?`<details open><summary>raw output</summary><pre>${esc(r.raw)}</pre></details>`:""}
${r.input?`<details><summary>analyst input (task + trajectories)</summary><pre data-i="${D.indexOf(r)}"></pre></details>`:""}</div>`}
function render(){const b=$("#fb").value,s=$("#fs").value,o=$("#fo").value,q=$("#q").value.toLowerCase();
const rs=D.filter(r=>(!b||r.bench==b)&&(!s||r.source==s)&&(!o||(o=="ok"?r.ok:o=="bad"?!r.ok:r.ok&&r.fields.recoverable===false))
&&(!q||(r.id+" "+JSON.stringify(r.fields)+r.raw).toLowerCase().includes(q)));
$("#list").innerHTML=rs.map(card).join("");$("#cnt").textContent=`${rs.length} / ${D.length} pairs`}
document.addEventListener("toggle",e=>{const p=e.target.querySelector("pre[data-i]");if(p&&!p.textContent)p.textContent=D[p.dataset.i].input},true);
["#fb","#fs","#fo"].forEach(s=>$(s).onchange=render);$("#q").oninput=render;render();
</script></body></html>"""

if __name__ == "__main__":
    main()
