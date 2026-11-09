"""Exp01's Modal and file chores in one python3 entry point. Modal calls use your active Modal profile (set up with
`modal setup`), or the one named in the environment variable MODAL_PROFILE if you set it. Output of each launch goes to
logs/<name>.log. The spending guard is EXP01_BUDGET_USD (default 20); stages refuse to start once it is used up.

    python3 ctl.py lock
    python3 ctl.py selftest
    python3 ctl.py prep                 write prompts.jsonl and pull it here
    python3 ctl.py fetch [names]        download pinned models to our volume on a CPU container
    python3 ctl.py hash                 write the SHA-256 of decision_rule.json and the code to logs/ (refuses if it exists)
    python3 ctl.py check
    python3 ctl.py smoke [names]
    python3 ctl.py plan                 apply the cut rule from the smoke throughput and write plan.json to the volume
    python3 ctl.py dump NAME            detached, under a wall-clock limit from the money left
    python3 ctl.py measure NAME         detached, likewise
    python3 ctl.py pull [NAME]          copy measures, outputs and cost logs from the volume into results/
    python3 ctl.py status | costs | guard | stop | apps
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS, RES = HERE / "logs", HERE / "results"
PROF = ["--profile", os.environ["MODAL_PROFILE"]] if os.environ.get("MODAL_PROFILE") else []
VOLUME = APP = "irb07-exp01"
OUR_APPS = (APP, "irb07-exp01-lock")
NAMES = ("gemma4e4b", "qwen3_8b", "sarvam30b")
USD_PER_S = 0.000694 + 8 * 0.0000131 + 64 * 0.00000222
USD_PER_S_SARVAM = 0.000694 + 8 * 0.0000131 + 96 * 0.00000222
BILLING_FACTOR = 1.03
GUARD_USD = float(os.environ.get("EXP01_BUDGET_USD", "20"))   # stop line; plan.json aims to finish under PLAN_USD
PLAN_USD = round(0.875 * GUARD_USD, 2)


def launch(name, args):
    LOGS.mkdir(exist_ok=True)
    cmd = ["modal", "run", *PROF] + args
    with open(LOGS / f"{name}.log", "a") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"# exit code {rc}\n")
    print(name, "exit code", rc)
    return rc


def volume_get(src, dest):
    return subprocess.run(["modal", "volume", "get", "--force", *PROF, VOLUME, src, str(dest)], capture_output=True, text=True)


def costlog():
    with tempfile.TemporaryDirectory() as tmp:
        r = volume_get("costlog", tmp + "/")
        if r.returncode:
            return []
        return sorted((json.loads(p.read_text()) for p in Path(tmp).rglob("*.json")), key=lambda x: x["start"])


def billed():
    out = subprocess.run(["modal", "billing", "report", "--for", "today", "--json", *PROF], capture_output=True, text=True)
    if out.returncode:
        return None, []
    rows = [r for r in json.loads(out.stdout) if r.get("description") in OUR_APPS]
    return round(sum(float(r["cost"]) for r in rows), 4), rows


def used():
    return max(billed()[0] or 0.0, BILLING_FACTOR * sum(r["usd_est"] for r in costlog()))


def plan():
    """The cut rule in decision_rule.json. It uses throughput and cost only, and no measure."""
    smokes = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in NAMES:
            if volume_get(f"smoke/{n}/smoke_summary.json", f"{tmp}/{n}.json").returncode == 0:
                smokes[n] = json.loads(Path(f"{tmp}/{n}.json").read_text())
    full = 240 * 2 * 128

    def project(n, prompts=240):
        s = smokes[n]
        states = full * prompts / 240
        rate = USD_PER_S_SARVAM if n == "sarvam30b" else USD_PER_S
        dump_s = s["info"]["s_per_state"] * states + s["info"]["load_s"] + 60
        c = s["controls"]
        meas_s = c["build_s"] + 60 + (c["seconds_per_state"]["greedy"] + c["seconds_per_state"]["sampled"]) * states / 2
        return dict(dump_s=round(dump_s), measure_s=round(meas_s), usd=round((dump_s * rate + meas_s * USD_PER_S) * BILLING_FACTOR, 2))
    now = used()
    table = {n: project(n) for n in smokes}
    two = sum(table[n]["usd"] for n in ("gemma4e4b", "qwen3_8b"))
    if "sarvam30b" in table and now + two + table["sarvam30b"]["usd"] <= PLAN_USD:
        pick = dict(models=list(NAMES), prompts_per_group=None)
    elif now + two <= PLAN_USD:
        pick = dict(models=["gemma4e4b", "qwen3_8b"], prompts_per_group=None)
    else:
        half = {n: project(n, 120) for n in ("gemma4e4b", "qwen3_8b")}
        ok = now + sum(v["usd"] for v in half.values()) <= PLAN_USD
        pick = dict(models=["gemma4e4b", "qwen3_8b"] if ok else ["gemma4e4b"], prompts_per_group=20)
        table["half"] = half
    out = dict(used_usd=round(now, 2), table=table, plan_usd=PLAN_USD, **pick)
    RES.mkdir(exist_ok=True)
    p = RES / "plan.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return subprocess.run(["modal", "volume", "put", "--force", *PROF, VOLUME, str(p), "/plan.json"]).returncode


def pull(names):
    RES.mkdir(exist_ok=True)
    for n in names:
        (RES / n).mkdir(exist_ok=True)
        for src in ("measure", "outputs_greedy.jsonl", "outputs_sampled.jsonl", "dump_info.json", "measure_timing.json"):
            r = volume_get(f"main/{n}/{src}", str(RES / n) + "/")
            print(n, src, "ok" if r.returncode == 0 else "absent")
    for n in NAMES:
        (RES / "smoke" / n).mkdir(parents=True, exist_ok=True)
        volume_get(f"smoke/{n}/smoke_summary.json", str(RES / "smoke" / n) + "/")
    return costs()


def costs():
    recs = costlog()
    RES.mkdir(exist_ok=True)
    with open(RES / "costlog.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    est = round(sum(r["usd_est"] for r in recs), 2)
    total, rows = billed()
    with open(RES / "billing.json", "w") as f:
        json.dump(dict(rows=rows, total_usd=total, our_estimate_usd=est), f, indent=1)
    for r in recs:
        print(f"{r['fn']:14s} {r['key']:10s} {r['seconds'] / 60:7.1f} min  ${r['usd_est']:.2f}")
    print(f"our estimate ${est:.2f}; Modal billing for our apps today ${total} over {len(rows)} rows")
    return 0


def our_apps():
    out = subprocess.run(["modal", "app", "list", "--json", *PROF], capture_output=True, text=True).stdout
    return [a for a in json.loads(out) if a.get("description") in OUR_APPS and a.get("state") != "stopped"]


def status():
    print("apps:", [(a.get("app_id"), a.get("description"), a.get("state"), a.get("tasks")) for a in our_apps()])
    recs = costlog()
    print("our estimate USD:", round(sum(r["usd_est"] for r in recs), 2), "; billed USD:", billed()[0], "; guard", GUARD_USD)
    return 0


def stop():
    for a in our_apps():
        print("stopping", a["app_id"], subprocess.run(["modal", "app", "stop", a["app_id"], *PROF]).returncode)
    left = our_apps()
    print("still running:", left)
    return 0 if not left else 1


def guard():
    b = billed()[0] or 0.0
    est = BILLING_FACTOR * sum(r["usd_est"] for r in costlog())
    print(f"billed {b:.2f}, estimate x 1.03 {est:.2f}, guard {GUARD_USD:.2f}")
    if max(b, est) >= GUARD_USD:
        print("GUARD: stopping our apps")
        return stop()
    return 0


def limit_for(rate, share=1.0):
    u = used()
    if u >= GUARD_USD:
        sys.exit(f"guard: {u:.2f} USD used, refusing to start")
    return int((GUARD_USD - u) / (rate * BILLING_FACTOR) * share)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    job, a = sys.argv[1], sys.argv[2:]
    if job == "lock":
        return launch("lock", ["make_lock.py"])
    if job == "selftest":
        return subprocess.run([sys.executable, str(HERE / "cert_core.py")]).returncode
    if job == "prep":
        return launch("prep", ["exp01_modal.py::prep_local"])
    if job == "fetch":
        rcs = [launch(f"fetch_{n}", ["exp01_modal.py::fetch", "--name", n]) for n in (a or NAMES)]
        return max(rcs)
    if job in ("hash", "check"):
        now = hashlib.sha256((HERE / "decision_rule.json").read_bytes()).hexdigest()
        p = LOGS / "decision_rule_sha256_at_write.log"
        if job == "check":
            amend = LOGS / "decision_rule_amendment1.log"
            logged = (amend if amend.exists() else p).read_text().split()[0]
            print(now, "matches" if now == logged else "DIFFERS FROM", logged)
            return 0 if now == logged else 1
        if p.exists():
            sys.exit("already hashed: " + p.read_text())
        LOGS.mkdir(exist_ok=True)
        ts = subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S%z"], capture_output=True, text=True).stdout.strip()
        lines = [f"{now}  decision_rule.json  written {ts}"]
        for name in ("analyze.py", "exp01_modal.py", "cert_core.py", "ctl.py", "prompts.jsonl", "requirements.lock.txt", "sarvam.lock.txt"):
            lines.append(f"{hashlib.sha256((HERE / name).read_bytes()).hexdigest()}  {name}  at the same time, for the record")
        p.write_text("\n".join(lines) + "\n")
        print(p.read_text())
        return 0
    if job == "smoke":
        rc = 0
        for n in (a or NAMES):
            if n == "sarvam30b":
                rc |= launch("smoke_sarvam30b", ["exp01_modal.py::smoke_sarvam_dump"])
                rc |= launch("smoke_sarvam30b", ["exp01_modal.py::smoke_sarvam_measure"])
            else:
                rc |= launch(f"smoke_{n}", ["exp01_modal.py::smoke_main", "--name", n])
        return rc
    if job == "plan":
        return plan()
    if job == "dump":
        n = a[0]
        if n == "sarvam30b":
            return launch(f"dump_{n}", ["--detach", "exp01_modal.py::dump_sarvam", "--limit-s", str(limit_for(USD_PER_S_SARVAM))])
        return launch(f"dump_{n}", ["--detach", "exp01_modal.py::dump_main", "--name", n, "--limit-s", str(limit_for(USD_PER_S))])
    if job == "measure":
        n = a[0]
        return launch(f"measure_{n}", ["--detach", "exp01_modal.py::measure", "--name", n, "--limit-s", str(limit_for(USD_PER_S))])
    if job == "pull":
        return pull(a or NAMES)
    if job == "runb":
        rc = launch("dump_b_sarvam30b", ["--detach", "exp01_modal.py::dump_sarvam_b", "--limit-s", str(limit_for(USD_PER_S_SARVAM, 0.5))])
        if rc == 0:
            rc = launch("measure_b_sarvam30b", ["--detach", "exp01_modal.py::measure_b", "--limit-s", str(limit_for(USD_PER_S, 0.5))])
        return rc
    if job == "posthoc":
        rc = 0
        for n, t in (("gemma4e4b", "main"), ("qwen3_8b", "main"), ("sarvam30b", "main"), ("sarvam30b", "main_b")):
            rc |= launch(f"posthoc_{n}_{t}", ["--detach", "exp01_modal.py::posthoc_topk", "--name", n, "--tag", t, "--limit-s", str(min(1500, limit_for(USD_PER_S, 0.5)))])
        return rc
    if job == "pullb":
        B = HERE / "results_runB" / "sarvam30b"
        B.mkdir(parents=True, exist_ok=True)
        for src in ("measure", "outputs_greedy.jsonl", "outputs_sampled.jsonl", "dump_info.json", "measure_timing.json"):
            print(src, volume_get(f"main_b/sarvam30b/{src}", str(B) + "/").returncode)
        P = RES / "posthoc_topk"
        for n, t in (("gemma4e4b", "main"), ("qwen3_8b", "main"), ("sarvam30b", "main"), ("sarvam30b", "main_b")):
            d = P / f"{n}_{t}"
            d.mkdir(parents=True, exist_ok=True)
            print(n, t, volume_get(f"{t}/{n}/posthoc_topk", str(d) + "/").returncode)
        return costs()
    if job == "costs":
        return costs()
    if job == "status":
        return status()
    if job == "guard":
        return guard()
    if job == "stop":
        return stop()
    if job == "apps":
        print(our_apps())
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
