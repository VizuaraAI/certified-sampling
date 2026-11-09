"""Exp01 analysis. Reads results/ and prompts.jsonl and applies the rule as written in decision_rule.json.

    python3 analyze.py                  uses results/ and writes summary.json
    python3 analyze.py <folder>         uses <folder> in place of results/ and writes there, for testing on a fixture

It refuses to run on results/ unless decision_rule.json has the SHA-256 logged when it was written, when that log exists.
The published repository has no logs/, so the check is skipped there and the self tests run in place of logs/selftest.log.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cert_core as c  # noqa: E402

MODELS = dict(gemma4e4b=dict(V=262144, d=2560), qwen3_8b=dict(V=151936, d=4096), sarvam30b=dict(V=262144, d=4096))
M262 = ("gemma4e4b", "sarvam30b")
GROUPS = ["english", "code", "hindi", "bengali", "tamil", "telugu"]
SCRIPT = dict(hindi=(0x900, 0x97F), bengali=(0x980, 0x9FF), tamil=(0xB80, 0xBFF), telugu=(0xC00, 0xC7F))
ELIGIBLE = ["int4|worst", "int4|resid", "int4d|worst", "int4d|resid", "int4d|dither", "int4h|worst", "int4h|resid", "nvfp4|worst",
            "nvfp4|resid", "int4|worst+int8", "int4|resid+int8", "int4d|dither+int8", "int4h|resid+int8", "nvfp4|resid+int8"]
NARROW = ["int4|worst", "int4d|worst", "int4d|dither"]
BASELINES = ["int8|worst", "int8|resid"]
ORACLES = ["int8|oracle", "int4|oracle", "int4d|oracle", "int4h|oracle", "nvfp4|oracle"]
RULE_STATES = {"greedy": "greedy", "gumbel_t0.6": "sampled", "gumbel_t1.0": "sampled", "topk50_t1.0": "sampled", "minp0.05_t1.0": "sampled"}
GATE_R50, GATE_R99, GATE_BYTES, KILL_BYTES = 0.001, 0.01, 0.35, 0.5


def pct(x, q):
    return float(np.percentile(x, q, method="inverted_cdf"))


def split_cert(cert):
    base = cert.split("+")[0]
    head, bound = base.split("|")
    return head, bound, "+" in cert


def load_model(folder, name, group_of):
    root = folder / name / "measure"
    data, ctls = {"greedy": {}, "sampled": {}}, []
    for mode in data:
        parts = {}
        for p in sorted(root.glob(f"{mode}_*.npz")):
            with np.load(p) as z:
                for k in z.files:
                    parts.setdefault(k, []).append(z[k])
        data[mode] = {k: np.concatenate(v) for k, v in parts.items()}
    for p in sorted(root.glob("*.json")):
        ctls.append(json.loads(p.read_text()))
    for mode in data:
        if "prompt" in data[mode]:
            data[mode]["group"] = np.array([group_of[int(i)] for i in data[mode]["prompt"]])
    return data, ctls


def stats(x):
    return dict(median=float(np.median(x)), p99=pct(x, 99), max=float(np.max(x)), mean=float(np.mean(x)))


def cert_table(data, cfg):
    """For every certificate and rule: set sizes and bytes, over all states and by group."""
    V, d = cfg["V"], cfg["d"]
    out = {}
    for rule, mode in list(RULE_STATES.items()) + [("greedy@sampled", "sampled")]:
        rname = rule.split("@")[0]
        D = data[mode]
        if "prompt" not in D:
            continue
        rows = {}
        for cert in ELIGIBLE + BASELINES + ORACLES:
            k = f"R|{cert}|{rname}"
            if k not in D:
                continue
            head, bound, casc = split_cert(cert)
            R = D[k]
            S = D.get(f"S|{cert}|{rname}") if casc else None
            B = c.total_bytes(head, bound, d, V, R, S)
            row = dict(R=stats(R), bytes=stats(B), by_group={})
            if S is not None:
                row["S"] = stats(S)
            for g in GROUPS:
                m = D["group"] == g
                if m.sum() == 0:
                    continue
                row["by_group"][g] = dict(n=int(m.sum()), R=stats(R[m]), bytes=stats(B[m]))
                if S is not None:
                    row["by_group"][g]["S"] = stats(S[m])
            rows[cert] = row
        out[rule] = rows
    return out


def gates_ok(row, V):
    return all(g["R"]["median"] <= GATE_R50 * V and g["R"]["p99"] <= GATE_R99 * V and g["bytes"]["median"] <= GATE_BYTES
               for g in row["by_group"].values())


def rule_status(row, V):
    groups = row["by_group"].values()
    if row["bytes"]["median"] >= KILL_BYTES:
        return "DEAD"
    if gates_ok(row, V):
        return "GATES_PASS"
    if all(g["bytes"]["median"] <= GATE_BYTES for g in groups):
        return "CHEAP"
    if all(g["bytes"]["median"] < KILL_BYTES for g in groups):
        return "UNDER_HALF"
    return "MIXED"


def best_cert(rows, certs):
    have = [x for x in certs if x in rows]
    if not have:
        return None
    return min(have, key=lambda x: (rows[x]["bytes"]["median"], rows[x]["R"]["p99"]))


def controls(ctls, folder, name, data):
    out = dict(n_chunks=len(ctls))
    viol = {}
    mism = {}
    ties = {}
    orc = {}
    tau = {}
    for x in ctls:
        for k, v in x["viol"].items():
            viol[k] = viol.get(k, 0) + v
        for k, v in x["mismatch"].items():
            mism[k] = mism.get(k, 0) + v
        for k, v in x["tie_mismatch"].items():
            ties[k] = ties.get(k, 0) + v
        for k, v in x["oracle_exceeded"].items():
            orc[k] = orc.get(k, 0) + v
        for k, v in x["tau_share"].items():
            tau[k] = max(tau.get(k, 0.0), v)
    is_dither = lambda k: "dither" in k  # noqa: E731
    gn = sum(x["n"] for x in ctls if x["mode"] == "greedy")
    em = sum(x["emitted_is_argmax_bf16"] for x in ctls if x["mode"] == "greedy")
    ems = sum(x["emitted_is_argmax_bf16_shifted"] for x in ctls if x["mode"] == "greedy")
    e32 = sum(x["emitted_is_argmax"] for x in ctls if x["mode"] == "greedy")
    a3216 = sum(x["fp32_argmax_is_bf16_argmax"] for x in ctls if x["mode"] == "greedy")
    out["C1_alignment"] = dict(rate_bf16=em / max(gn, 1), shifted_rate_bf16=ems / max(gn, 1), states=gn, ok=gn > 0 and em / gn >= 0.95 and ems / gn <= 0.05,
                               readout_fp32_argmax_is_emitted=e32 / max(gn, 1), readout_fp32_argmax_is_bf16_argmax=a3216 / max(gn, 1))
    n2 = sum(v for k, v in viol.items() if not is_dither(k))
    out["C2_soundness"] = dict(violations=n2, ok=n2 == 0 and len(viol) > 0)
    n3 = sum(v for k, v in mism.items() if not is_dither(k))
    t3 = sum(v for k, v in ties.items() if not is_dither(k))
    out["C3_exactness"] = dict(mismatches=n3, on_exact_ties=t3, decisions_checked=len([k for k in mism if not is_dither(k)]), ok=n3 - t3 == 0)
    n4 = sum(orc.values())
    out["C4_oracle_order"] = dict(exceeded=n4, ok=n4 == 0)
    zz = sum(x["dither_z2_sum"] for x in ctls)
    zn = sum(x["dither_z2_n"] for x in ctls)
    dv = sum(v for k, v in viol.items() if is_dither(k))
    dm = sum(v for k, v in mism.items() if is_dither(k))
    dt = sum(v for k, v in ties.items() if is_dither(k))
    ratio = zz / max(zn, 1)
    out["C5_dither"] = dict(violations=dv, mismatches=dm, on_exact_ties=dt, variance_ratio=ratio, rows_checked=zn, ok=dv == 0 and dm - dt == 0 and 0.85 <= ratio <= 1.15)
    pcs = [x["positive_controls"] for x in ctls if "positive_controls" in x]
    ok6 = len(pcs) >= 1 and all(p["small_bound_violations"] > 0 and p["permuted_rows_violations"] > 0 and p["dropped_winner_mismatch_share"] == 1.0
                                and p["exact_head_share_R1"] >= 0.95 and p["empty_head_median_R_share"] >= 0.99 and p["empty_head_violations"] == 0
                                and p["tau_over_fp32_err"] >= 1.0 for p in pcs)
    out["C6_planted"] = dict(values=pcs, ok=ok6)
    outs = {}
    ok7 = True
    for mode in ("greedy", "sampled"):
        f = folder / name / f"outputs_{mode}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        for g, (a, b) in SCRIPT.items():
            shares = []
            for r in rows:
                if r["group"] != g:
                    continue
                letters = [ch for ch in r["text"] if ch.isalpha()]
                shares.append(sum(a <= ord(ch) <= b for ch in letters) / max(len(letters), 1))
            if shares:
                outs[f"{g}_{mode}"] = float(np.median(shares))
                ok7 = ok7 and outs[f"{g}_{mode}"] >= 0.5
        lens = {}
        for r in rows:
            lens.setdefault(r["group"], []).append(r["n"])
        outs[f"median_tokens_{mode}"] = {g: float(np.median(v)) for g, v in lens.items()}
    out["C7_language"] = dict(values=outs, ok=ok7)
    nll = sum(x.get("nll_sum", 0.0) for x in ctls)
    ent = sum(x.get("entropy_sum", 0.0) for x in ctls)
    out["C8_sampling"] = dict(nll_over_entropy=nll / max(ent, 1e-9), ok=0.9 <= nll / max(ent, 1e-9) <= 1.1)
    out["slack_share_of_bound_max"] = tau
    return out


def ratios(data, name):
    out = {}
    for mode in ("greedy", "sampled"):
        D = data[mode]
        if "prompt" not in D:
            continue
        row = {}
        for k in D:
            if k.startswith("b_max|"):
                label = k.split("|", 1)[1]
                head = label.split("|")[0]
                r = D[k] / np.maximum(D[f"err_max|{head}"], 1e-12)
                row[label] = dict(bound_over_true_max_median=float(np.median(r)), p99=pct(r, 99), mean_bound_median=float(np.median(D[f"b_mean|{label}"])),
                                  max_bound_median=float(np.median(D[k])), true_max_err_median=float(np.median(D[f"err_max|{head}"])))
        if "b_mean|int4|worst" in D:
            gain = D["b_mean|int4|worst"] / D["b_mean|int4d|dither"]
            row["_dither_gain_mean_worst_over_mean_dither"] = dict(median=float(np.median(gain)), p01=pct(gain, 1), p99=pct(gain, 99))
        row["_state_facts"] = dict(margin_median=float(np.median(D["margin"])), margin_p01=pct(D["margin"], 1), h_l2_median=float(np.median(D["h_l2"])),
                                   l1_over_sqrtd_l2_median=float(np.median(D["h_l1"] / D["h_l2"] / np.sqrt(MODELS[name]["d"]))))
        out[mode] = row
    return out


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    if len(sys.argv) == 1 and (HERE / "logs" / "decision_rule_sha256_at_write.log").exists():
        now = hashlib.sha256((HERE / "decision_rule.json").read_bytes()).hexdigest()
        logged = (HERE / "logs" / "decision_rule_sha256_at_write.log").read_text().split()[0]
        amend = HERE / "logs" / "decision_rule_amendment1.log"
        if amend.exists():
            logged = amend.read_text().split()[0]
        if now != logged:
            sys.exit(f"decision_rule.json has changed: {now} against {logged}")
    prompts = [json.loads(x) for x in (HERE / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    group_of = {r["prompt"]: r["group"] for r in prompts}
    plan = json.loads((folder / "plan.json").read_text()) if (folder / "plan.json").exists() else {}
    summary = dict(plan=plan, models={})
    st = HERE / "logs" / "selftest.log"
    selftest_ok = ("selftest passed" in st.read_text()) if st.exists() else bool(c.selftest())
    if (HERE / "logs" / "byte_share.json").exists():
        summary["byte_share"] = json.loads((HERE / "logs" / "byte_share.json").read_text())
    status = {}
    for name, cfg in MODELS.items():
        if not (folder / name / "measure").exists():
            continue
        data, ctls = load_model(folder, name, group_of)
        info_p = folder / name / "dump_info.json"
        info = json.loads(info_p.read_text()) if info_p.exists() else {}
        n_meas = sum(len(data[m]["prompt"]) for m in data if "prompt" in data[m])
        ctl = controls(ctls, folder, name, data)
        ctl["C9_selftest"] = dict(ok=selftest_ok)
        complete = info.get("n_states") == n_meas and n_meas > 0
        broken = [k for k in ("C1_alignment", "C2_soundness", "C3_exactness", "C4_oracle_order", "C6_planted", "C8_sampling", "C9_selftest") if not ctl[k]["ok"]]
        table = cert_table(data, cfg)
        elig = [x for x in ELIGIBLE if ctl["C5_dither"]["ok"] or "dither" not in x]
        narrow = [x for x in NARROW if ctl["C5_dither"]["ok"] or "dither" not in x]
        best, per_rule = {}, {}
        for rule, rows in table.items():
            b = best_cert(rows, elig)
            if b:
                best[rule] = b
                per_rule[rule] = rule_status(rows[b], cfg["V"]) if rule != "greedy@sampled" else rule_status(rows[b], cfg["V"])
        both = [x for x in elig if "greedy" in table and "gumbel_t1.0" in table and x in table["greedy"] and x in table["gumbel_t1.0"]
                and gates_ok(table["greedy"][x], cfg["V"]) and gates_ok(table["gumbel_t1.0"][x], cfg["V"])]
        both_narrow = [x for x in both if x in narrow]
        kill = all(min(table[r][x]["bytes"]["median"] for x in elig if x in table[r]) >= KILL_BYTES for r in ("greedy", "gumbel_t1.0") if r in table) and "greedy" in table and "gumbel_t1.0" in table
        status[name] = dict(complete=complete, broken=broken, states_measured=n_meas, states_dumped=info.get("n_states"), confirm_certs=both, confirm_certs_narrow=both_narrow, kill=kill)
        summary["models"][name] = dict(info=info, controls=ctl, table=table, best=best, per_rule_status=per_rule, status=status[name], ratios=ratios(data, name))

    g = status.get("gemma4e4b")
    if g is None or not g["complete"]:
        outcome = "INCOMPLETE"
    elif g["broken"]:
        outcome = "MEASUREMENT_BROKEN"
    else:
        ran = [m for m in M262 if m in status and status[m]["complete"] and not status[m]["broken"]]
        if all(status[m]["confirm_certs"] for m in ran):
            outcome = "CONFIRMED"
        elif all(status[m]["kill"] for m in ran):
            outcome = "KILLED"
        else:
            outcome = "MIDDLE"
        ran_narrow = all(status[m]["confirm_certs_narrow"] for m in ran)
        summary["confirm_with_review_certificates_only"] = ran_narrow
    summary["outcome"] = outcome
    summary["status"] = status
    bill_p = folder / "billing.json"
    if bill_p.exists():
        summary["billing"] = json.loads(bill_p.read_text())
    (folder.parent / "summary.json" if len(sys.argv) == 1 else folder / "summary.json").write_text(json.dumps(summary, indent=1))
    print("outcome", outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
