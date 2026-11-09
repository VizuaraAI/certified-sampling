"""Build requirements.lock.txt with exact versions and file hashes, for Linux x86-64 and Python 3.12.

    python3 ctl.py lock

The resolver runs in a small Modal CPU container, so nothing is installed on this Mac. uv's --exclude-newer refuses any
release uploaded in the 14 days before the run date (5 November 2026). sarvam.lock.txt is a copy of topic 08's lock.
"""

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
UV = "uv==0.8.4"
TOP = """torch==2.13.0
transformers==5.19.0
tokenizers==0.23.3
safetensors==0.8.0
accelerate
numpy==2.3.5
huggingface-hub
hf-xet==1.7.0
pyarrow==21.0.0
jinja2==3.1.6
sentencepiece==0.2.2
protobuf==7.36.2
"""
app = modal.App("irb07-exp01-lock")
image = modal.Image.debian_slim(python_version="3.12").pip_install(UV)


@app.function(image=image, cpu=2, memory=4096, timeout=900)
def resolve(top):
    Path("/tmp/in.txt").write_text(top)
    r = subprocess.run(["uv", "pip", "compile", "/tmp/in.txt", "--python-version", "3.12", "--python-platform",
                        "x86_64-unknown-linux-gnu", "--generate-hashes", "--exclude-newer", "2026-10-22T00:00:00Z",
                        "--no-header", "--annotation-style", "line", "-o", "/tmp/out.txt"],
                       capture_output=True, text=True)
    return dict(rc=r.returncode, stderr=r.stderr[-6000:], lock=Path("/tmp/out.txt").read_text() if r.returncode == 0 else "")


@app.local_entrypoint()
def main():
    o = resolve.remote(TOP)
    print("rc", o["rc"])
    print(o["stderr"])
    if o["rc"] == 0:
        (HERE / "requirements.lock.txt").write_text(o["lock"])
        print(sum(1 for x in o["lock"].splitlines() if "==" in x and not x.startswith(" ")), "packages")
