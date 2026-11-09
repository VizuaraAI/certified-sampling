"""A dry run of the measure step and analyze.py on the CPU, on synthetic data, before any GPU is paid for. It writes to a
fixture folder under the scratch path given as the first argument. No real model or state is touched.

    python3 test_local.py <folder>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cert_core as c  # noqa: E402
import exp01_modal as e  # noqa: E402

fix = Path(sys.argv[1])
name = "gemma4e4b"
V, d = 4000, 1024
cfg = dict(V=V, d=d, cap=30.0)
torch.manual_seed(1)
W = torch.randn(V, d) * 0.02
heads, rot, rownorm = e.build_heads(W, cfg, "cpu")
prompts = [json.loads(x) for x in (HERE / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
root = fix / name / "measure"
root.mkdir(parents=True, exist_ok=True)
outs = {}
n_all = 0
for mode_i, mode in enumerate(("greedy", "sampled")):
    n = 96
    pr = torch.tensor([prompts[i * 2 + 1]["prompt"] for i in range(n)])
    st = torch.zeros(n, dtype=torch.int64)
    H = torch.randn(n, d) * torch.exp(torch.randn(n, d) * 0.3) * 0.5
    H[:, :8] *= 6
    keys = c.state_key(pr, st, mode_i)
    ref = c.phi(H @ W.T, cfg["cap"])
    if mode == "greedy":
        tok = ref.argmax(1)
    else:
        tok = torch.multinomial(torch.softmax(ref, 1), 1).flatten()
    arrs, ctl = e.measure_chunk(H, W, heads, rot, rownorm, cfg, keys, tok, mode, True, Hb=H.bfloat16(), Wb=W.bfloat16())
    arrs["prompt"] = pr.numpy()
    arrs["step"] = st.numpy()
    ctl.update(mode=mode, chunk=0, n=n, seconds=1.0)
    np.savez(root / f"{mode}_0000.npz", **arrs)
    (root / f"{mode}_0000.json").write_text(json.dumps(ctl))
    n_all += n
    with open(fix / name / f"outputs_{mode}.jsonl", "w", encoding="utf-8") as f:
        for g, s in (("hindi", "नमस्ते दुनिया"), ("bengali", "নমস্কার"), ("tamil", "வணக்கம்"), ("telugu", "నమస్కారం"), ("english", "hello")):
            f.write(json.dumps(dict(prompt=0, group=g, n=4, text=s), ensure_ascii=False) + "\n")
(fix / name / "dump_info.json").write_text(json.dumps(dict(n_states=n_all)))
print(json.dumps({k: v for k, v in ctl["positive_controls"].items()}, indent=0))
print("keys in one chunk:", len(arrs))
print("violations", sum(ctl["viol"].values()), "mismatches", sum(ctl["mismatch"].values()), "oracle", sum(ctl["oracle_exceeded"].values()))
