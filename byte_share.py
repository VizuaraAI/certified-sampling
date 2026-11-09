"""Step 1 of the review's E0, on a laptop: the head's share of weight bytes read per token at batch 1. It reads only the
safetensors headers, through byte-range requests at the pinned revisions, and writes logs/byte_share.json.

    python3 byte_share.py
"""

import json
import struct
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOS = dict(gemma4e4b=("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"),
             qwen3_8b=("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
             sarvam30b=("sarvamai/sarvam-30b", "071ae95e933605ca1104a6b4524a36a98488efa4"))
EXPERTS_ACTIVE = 6 / 128        # Sarvam-30B routes each token to 6 of 128 experts; the shared expert always runs


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "research", **({"Range": rng} if rng else {})})
    return urllib.request.urlopen(req, timeout=60).read()


def header(repo, rev, f):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{f}"
    n = struct.unpack("<Q", get(url, "bytes=0-7"))[0]
    return json.loads(get(url, f"bytes=8-{7 + n}"))


out = {}
for name, (repo, rev) in REPOS.items():
    info = json.loads(get(f"https://huggingface.co/api/models/{repo}/revision/{rev}"))
    files = sorted(s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".safetensors"))
    embed = head = body = vision = expert = per_layer = 0
    for f in files:
        for k, v in header(repo, rev, f).items():
            if k == "__metadata__":
                continue
            n = 1
            for x in v["shape"]:
                n *= x
            low = k.lower()
            if "vision" in low or "audio" in low or "embed_vision" in low or "embed_audio" in low:
                vision += n
            elif "lm_head" in low:
                head += n
            elif "embed_tokens_per_layer" in low:
                per_layer += n      # Gemma 4's per-layer token inputs: an input-side table, read one row per token
            elif "embed_tokens" in low or "word_embeddings" in low:
                embed += n
            elif ".experts." in low and "shared" not in low:
                expert += n
            else:
                body += n
    if head == 0:
        head = embed        # tied
    active = body + expert * EXPERTS_ACTIVE
    row = dict(head_params=head, input_embedding_params=embed, body_active_params=int(active), vision_audio_params=vision, per_layer_input_table_params=per_layer,
               head_share_bf16_body=round(head / (head + active), 3), head_share_4bit_body=round(head * 2 / (head * 2 + active * 0.5), 3))
    out[name] = row
    print(name, row)
out["note"] = "Reads one embedding row per token at batch 1, which we ignore, and ignores KV-cache reads, which grow with context."
(HERE / "logs").mkdir(exist_ok=True)
(HERE / "logs" / "byte_share.json").write_text(json.dumps(out, indent=1))
