"""Runs the planned stages one after another. ctl.py refuses a stage once EXP01_BUDGET_USD is used, and each stage has a
wall-clock limit from the money left, so this cannot run past the guard by more than the last chunk.

    python3 drive.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / "results" / "plan.json").read_text())
(HERE / "logs").mkdir(exist_ok=True)
log = open(HERE / "logs" / "drive.log", "a")


def run(*args):
    r = subprocess.run([sys.executable, str(HERE / "ctl.py"), *args], capture_output=True, text=True)
    log.write(f"{args} rc={r.returncode} {r.stdout.strip()[-300:]} {r.stderr.strip()[-300:]}\n")
    log.flush()
    return r.returncode


for name in plan["models"]:
    if run("dump", name) != 0:
        log.write(f"dump failed for {name}\n")
        log.flush()
        continue
    run("guard")
    run("measure", name)
    run("guard")
log.write("drive done\n")
log.close()
