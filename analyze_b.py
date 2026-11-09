"""Applies the fixed rule to Sarvam run B, and summarises the post hoc top-k pass. It writes summary_runB.json. When logs/ is present it also
refuses to run unless decision_rule.json and logs/addendum_runB.json match their logged hashes. The published repository has
no logs/ and no per-chunk results, so this script documents the analysis rather than rerunning it.

    python3 analyze_b.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import analyze as a
import cert_core as c

HERE = Path(__file__).resolve().parent
if (HERE / "logs" / "decision_rule_amendment1.log").exists():
    rule = hashlib.sha256((HERE / "decision_rule.json").read_bytes()).hexdigest()
    assert rule == (HERE / "logs" / "decision_rule_amendment1.log").read_text().split()[0], "rule changed"
    add = hashlib.sha256((HERE / "logs" / "addendum_runB.json").read_bytes()).hexdigest()
    assert add == (HERE / "logs" / "addendum_runB_sha256.log").read_text().split()[0], "addendum changed"
prompts = [json.loads(x) for x in (HERE / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
group_of = {r["prompt"]: r["group"] for r in prompts}
folder = HERE / "results_runB"
name = "sarvam30b"
cfg = a.MODELS[name]
data, ctls = a.load_model(folder, name, group_of)
ctl = a.controls(ctls, folder, name, data)
st = HERE / "logs" / "selftest.log"
ctl["C9_selftest"] = dict(ok=("selftest passed" in st.read_text()) if st.exists() else bool(c.selftest()))
info = json.loads((folder / name / "dump_info.json").read_text())
n_meas = sum(len(data[m]["prompt"]) for m in data if "prompt" in data[m])
broken = [k for k in ("C1_alignment", "C2_soundness", "C3_exactness", "C4_oracle_order", "C6_planted", "C8_sampling", "C9_selftest") if not ctl[k]["ok"]]
table = a.cert_table(data, cfg)
elig = [x for x in a.ELIGIBLE if ctl["C5_dither"]["ok"] or "dither" not in x]
best = {r: a.best_cert(rows, elig) for r, rows in table.items()}
per_rule = {r: a.rule_status(table[r][b], cfg["V"]) for r, b in best.items() if b}
both = [x for x in elig if a.gates_ok(table["greedy"][x], cfg["V"]) and a.gates_ok(table["gumbel_t1.0"][x], cfg["V"])]
kill = all(min(table[r][x]["bytes"]["median"] for x in elig) >= a.KILL_BYTES for r in ("greedy", "gumbel_t1.0"))
out = dict(run_B=dict(info=info, controls=ctl, table=table, best=best, per_rule_status=per_rule, confirm_certs=both, kill=kill, broken=broken,
                      states_measured=n_meas, states_dumped=info.get("n_states"), ratios=a.ratios(data, name)))
post = {}
for tagname in ("gemma4e4b_main", "qwen3_8b_main", "sarvam30b_main", "sarvam30b_main_b"):
    d = HERE / "results" / "posthoc_topk" / tagname / "posthoc_topk"
    if not d.exists():
        d = HERE / "results" / "posthoc_topk" / tagname
    n = tagname.split("_main")[0]
    V, D = a.MODELS[n]["V"], a.MODELS[n]["d"]
    parts = {}
    for p in sorted(d.glob("*.npz")):
        with np.load(p) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    arr = {k: np.concatenate(v) for k, v in parts.items()}
    groups = np.array([group_of[int(i)] for i in arr["prompt"]])
    b4 = c.total_bytes("int4d", "dither", D, V, arr["R|int4d|dither+int8"], arr["S|int4d|dither+int8"])
    b8 = c.total_bytes("int8", "resid", D, V, arr["R|int8|resid+bf16"])
    row = dict(states=len(groups), mismatch=json.loads((d / "mismatch.json").read_text()),
               cascade=dict(R=a.stats(arr["R|int4d|dither+int8"]), S=a.stats(arr["S|int4d|dither+int8"]), bytes=a.stats(b4)),
               int8=dict(R=a.stats(arr["R|int8|resid+bf16"]), bytes=a.stats(b8)), by_group={})
    for g in a.GROUPS:
        m = groups == g
        row["by_group"][g] = dict(R=a.stats(arr["R|int4d|dither+int8"][m]), S=a.stats(arr["S|int4d|dither+int8"][m]), bytes=a.stats(b4[m]), int8_bytes=a.stats(b8[m]))
    post[tagname] = row
out["posthoc_topk"] = post
(HERE / "summary_runB.json").write_text(json.dumps(out, indent=1))
print("run B confirm certs", both, "kill", kill, "broken", broken, "status", per_rule)

# Add run B's status and the post hoc pass to summary.json, without changing what analyze.py computed.
s = json.loads((HERE / "summary.json").read_text())
s["run_B_sarvam_thinking_closed"] = dict(status="Reported beside the main run under logs/addendum_runB.json. It does not replace the main outcome. Full tables are in summary_runB.json.",
                                         confirm_certs=both, kill=kill, broken=broken, per_rule_status=per_rule, c7_ok=ctl["C7_language"]["ok"], best=best)
s["posthoc_topk"] = post
if (HERE / "results" / "billing.json").exists():
    s["billing"] = json.loads((HERE / "results" / "billing.json").read_text())
(HERE / "summary.json").write_text(json.dumps(s, indent=1))
