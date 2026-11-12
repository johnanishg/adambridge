"""Print the README's arena table from an adambridge-lab report, so the README never claims more than was measured.

Usage (from the repository root, with the lab checkout in adambridge-lab/):

    python benchmarks/readme_claims.py adambridge-lab/reports/<report>.json

Every number and verdict comes from the lab's own judgement of the run (lab.arena.summarize): the best
adambridge configuration at reference quality against the best of each other library, the all-GPU control,
and the capacity probe. Paste the output under "Measured results" in README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "adambridge-lab"))

from lab import arena  # noqa: E402

FAMILIES = ["adambridge", "DeepSpeed", "torchao", "bitsandbytes", "PyTorch"]


def _best(items, family):
    return arena._best(items, family)


def _cell(i, ref=None):
    if i is None:
        return "–"
    tps = f"{i['tokens_per_s']:,.0f} tokens/s" if i.get("tokens_per_s") else "trains"
    return tps


def main(path: str) -> None:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [r for r in report["results"] if r["kind"] == "llm_arena" and r["status"] == "done" and r.get("data")]
    if not runs:
        raise SystemExit("no finished arena run in this report")
    d = max(runs, key=lambda r: len(r["data"]["rows"]))["data"]
    s = arena.summarize(d)
    lines = [f"Arena report `{Path(path).name}` (generated {report['generated'][:16].replace('T', ' ')}): "
             f"{d['steps']} optimizer steps of {d['batch']} × {d['seq_len']} tokens on WikiText-2 per run, activation "
             f"checkpointing on for every library, {d['gpu']} ({d['gpu_memory_gib']:.0f} GiB) with "
             f"{d.get('host_memory_gib') or 0:.0f} GiB of host memory. Each cell is the library's fastest configuration "
             f"at reference quality (the same final validation loss as the reference, within the run-to-run noise); "
             f"a dash means no configuration of it fitted or reached reference quality.", ""]
    lines.append("| model | " + " | ".join(FAMILIES) + " | adambridge vs DeepSpeed |")
    lines.append("|---|" + "---:|" * (len(FAMILIES) + 1))
    for m in s["models"]:
        best = {f: _best(m["items"], f) for f in FAMILIES}
        ab, ds = best["adambridge"], best["DeepSpeed"]
        if m["reference"] is None and not m["speed_only"]:
            fits = [i for i in m["items"] if i["acc"] == 1 and i["fits"] and i.get("tokens_per_s")]
            row = {f: max((i for i in fits if i["family"] == f), key=lambda i: i["tokens_per_s"], default=None)
                   for f in FAMILIES}
            cells = [(_cell(row[f]) + " (not judged)" if row[f] else "–") for f in FAMILIES]
            lines.append(f"| {m['short']} ({m['params_b'] or 0:.1f}B) | " + " | ".join(cells) + " | no reference fits |")
            continue
        vs = "–"
        if ab and ds:
            r = ab["tokens_per_s"] / ds["tokens_per_s"]
            vs = "a tie" if abs(r - 1) <= 0.03 else f"{(r - 1) * 100:+.0f}%"
        elif ab:
            vs = "only adambridge trains it"
        lines.append(f"| {m['short']} ({m['params_b'] or 0:.1f}B) | " + " | ".join(_cell(best[f]) for f in FAMILIES)
                     + f" | {vs} |")
    if s["probe"]:
        lines += ["", "Largest model each library trains here (random-initialised Qwen2.5-3B-shaped models with more "
                      f"layers, 3 steps at micro-batch 1, bisected to ±{d.get('probe_tolerance_pct', 8)}%):", ""]
        lines.append("| library | largest model | stopped by |")
        lines.append("|---|---:|---|")
        lim = {"gpu": "GPU memory", "host": "host memory", "not reached": "the probe's ceiling", "error": "an error"}
        for x in sorted(s["probe"], key=lambda x: -x["max_params_b"]):
            lines.append(f"| {x['label']} | {x['max_params_b']:.2f}B | {lim.get(x['limit'], x['limit'])} |")
    if s["disturbed"]:
        lines += ["", "Runs whose timing the report flagged (left out above): " + "; ".join(
            f"{k}: {v}" for k, v in s["disturbed"].items()) + "."]
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
