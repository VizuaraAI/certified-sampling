"""Exp01 for topic 07, the review's E0: how wide is a 4-bit certificate on real decode states?

Steps run through ctl.py, which uses your active Modal profile. Nothing here prints a rule measure: the sets and bytes go to
the volume, and analyze.py reads them only after the decision rule is hashed. See decision_rule.json.

    prep        write the prompts (CPU)
    fetch       download the pinned models to the volume (CPU)
    smoke       a few states per model, all controls, throughput for the cut rule
    stage       dump the states of one model, then measure them, under a wall-clock limit
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "irb07-exp01"
app = modal.App(APP)
vol = modal.Volume.from_name(APP, create_if_missing=True)
VOL = "/vol"
ENV = {"PYTHONUNBUFFERED": "1", "HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "true",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
img_main = (modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(HERE / "requirements.lock.txt"))
            .env(ENV).add_local_file(str(HERE / "cert_core.py"), "/root/cert_core.py"))
# Sarvam-30B's own code was written for transformers 4.57.2. This lock is topic 08's, which already ran this revision.
img_sarvam = (modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(str(HERE / "sarvam.lock.txt"))
              .env(ENV).add_local_file(str(HERE / "cert_core.py"), "/root/cert_core.py"))

MODELS = {
    "gemma4e4b": dict(repo="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c", V=262144, d=2560, cap=30.0),
    "qwen3_8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218", V=151936, d=4096, cap=None),
    "sarvam30b": dict(repo="sarvamai/sarvam-30b", rev="071ae95e933605ca1104a6b4524a36a98488efa4", V=262144, d=4096, cap=None),
}
SARVAM_CODE = {"configuration_sarvam_moe.py": "105ba18705cd28a075231f2b4b8764c3171256616187470f9da4b55c315a1360",
               "modeling_sarvam_moe.py": "87b0983d7094de03df116219a3d75ae083dc4aac5b9b70455c2ac60706b7e4e3"}
FASTCHAT = "b494d0c6b4e7935f1764f8439e75da3e66beccc7"
HUMANEVAL = "463c980b59e818ace59f6f9803cd92c749ceae61"
NEWS_REPO, NEWS_REV = "PleIAs/US-PD-Newspapers", "39dc0a2a0113609fdcc2f865a25a6d8bafd1f5e6"
GROUPS = ["english", "code", "hindi", "bengali", "tamil", "telugu"]
TRANSLATE = {"hindi": "Hindi", "bengali": "Bengali", "tamil": "Tamil", "telugu": "Telugu"}

BATCH, MAX_NEW, CHUNK = 16, 128, 1024
GPU_NAME = "A100-80GB"
PRICES = {"A100-80GB": 0.000694, "none": 0.0}             # modal.com/pricing and `modal billing rates`, 5 Nov 2026
CPU_RATE, MEM_RATE = 0.0000131, 0.00000222
CPU_N, MEM_GIB, MEM_GIB_SARVAM = 8, 64, 96
SEED = 20261105
CASCADE_BASES = [("int4", "worst"), ("int4", "resid"), ("int4d", "dither"), ("int4h", "resid"), ("nvfp4", "resid")]


# ------------------------------------------------------------------ cost log and clock, as topic 09

class Ledger:
    def __init__(self, fn, key, gpu, cpu=CPU_N, mem=MEM_GIB):
        self.t0, self.last = time.time(), 0.0
        self.rec = dict(fn=fn, key=key, gpu=gpu, cpu=cpu, mem_gib=mem, start=self.t0, seconds=0.0, usd_est=0.0)
        os.makedirs(f"{VOL}/costlog", exist_ok=True)
        self.path = f"{VOL}/costlog/{fn}_{key}_{int(self.t0)}.json"
        self.beat(force=True)

    def beat(self, force=False):
        now = time.time()
        if force or now - self.last > 60:
            s = now - self.t0
            r = self.rec
            r["seconds"] = round(s, 1)
            r["usd_est"] = round(s * (PRICES[r["gpu"]] + r["cpu"] * CPU_RATE + r["mem_gib"] * MEM_RATE), 4)
            with open(self.path, "w") as f:
                json.dump(r, f)
            vol.commit()
            self.last = now


class OutOfBudget(Exception):
    pass


class Clock:
    def __init__(self, limit_s, ledger):
        self.t0, self.limit, self.ledger = time.time(), limit_s, ledger

    def check(self):
        self.ledger.beat()
        if time.time() - self.t0 > self.limit:
            raise OutOfBudget(f"limit {self.limit:.0f}s reached")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                return h.hexdigest()
            h.update(b)


# ------------------------------------------------------------------ prompts

BLOCK = re.compile(r"\b(negro|negroes|nigger|colored|chinaman|chinamen|indians?|savage|savages|murder\w*|lynch\w*|rape\w*|"
                   r"kill\w*|hanged|whiskey|jap|japs|jew|jews|slave\w*|corpse|suicide|dead|death)\b", re.I)


def news_passages(n):
    """Five consecutive clean English sentences from a CC0 newspaper corpus. OCR noise is filtered by a word list built from
    the same files: a word must appear at least 50 times, and sentences must look like ordinary sentences."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from collections import Counter
    files = sorted(f for f in HfApi().list_repo_files(NEWS_REPO, revision=NEWS_REV, repo_type="dataset") if f.endswith(".parquet"))
    picks = files[::len(files) // 12][:12]
    pages, counts = [], Counter()
    for f in picks:
        p = hf_hub_download(NEWS_REPO, f, revision=NEWS_REV, repo_type="dataset")
        t = pq.read_table(p, columns=["date", "text"])
        dates, texts = t["date"].to_pylist(), t["text"].to_pylist()
        for dt, tx in zip(dates, texts):
            if dt[:4] < "1900" or not tx:
                continue
            tx = re.sub(r"-\n", "", tx)
            tx = re.sub(r"\s+", " ", tx)
            pages.append((f, dt, tx))
            counts.update(w.lower() for w in re.findall(r"[A-Za-z']+", tx))
        os.remove(p)
        if len(pages) > 4000:
            break
    good = {w for w, c in counts.items() if c >= 50 and (len(w) > 1 or w in ("a", "i"))}

    def clean(s):
        if not (s.endswith(".") and s[0].isupper() and re.fullmatch(r"[A-Za-z ,;'.\-]+", s)):
            return False
        w = re.findall(r"[A-Za-z']+", s)
        if not 8 <= len(w) <= 28 or sum(x[0].isupper() for x in w) > 0.3 * len(w):
            return False
        return all(x.lower() in good for x in w) and not BLOCK.search(s)

    cands = []
    for f, dt, tx in pages:
        ss = re.split(r"(?<=[.!?])\s+", tx)
        i = 0
        while i + 5 <= len(ss):
            if all(clean(x) for x in ss[i:i + 5]):
                passage = " ".join(ss[i:i + 5])
                cands.append((hashlib.sha256(passage.encode()).hexdigest(), passage, f, dt))
                i += 5
            else:
                i += 1
    cands.sort()
    seen, out = set(), []
    for _, passage, f, dt in cands:
        key = (f, dt)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(text=passage, source=f, date=dt))
        if len(out) == n:
            break
    return out, dict(files=picks, pages=len(pages), candidates=len(cands))


@app.function(image=img_main, cpu=2, memory=16 * 1024, volumes={VOL: vol}, timeout=1800)
def prep():
    import gzip
    import urllib.request
    led = Ledger("prep", "all", "none", cpu=2, mem=16)

    def get(url):
        return urllib.request.urlopen(url, timeout=60).read()
    mt = [json.loads(x) for x in get(f"https://raw.githubusercontent.com/lm-sys/FastChat/{FASTCHAT}/fastchat/llm_judge/data/mt_bench/question.jsonl").decode().splitlines()]
    he = [json.loads(x) for x in gzip.decompress(get(f"https://raw.githubusercontent.com/openai/human-eval/{HUMANEVAL}/data/HumanEval.jsonl.gz")).decode().splitlines()]
    rows = []
    for q in mt[::2][:40]:
        rows.append(dict(group="english", text=q["turns"][0], source=f"mt_bench/{q['question_id']}"))
    for q in he[:40]:
        rows.append(dict(group="code", text="Complete the following Python function. Write only the code.\n\n" + q["prompt"], source=q["task_id"]))
    passages, info = news_passages(40)
    assert len(passages) == 40, f"only {len(passages)} passages"
    for g, name in TRANSLATE.items():
        for j, p in enumerate(passages):
            rows.append(dict(group=g, text=f"Translate the following English passage into {name}. Write only the translation.\n\n{p['text']}",
                             source=f"{p['source']}|{p['date']}|passage{j}"))
    for i, r in enumerate(rows):
        r["prompt"] = i
    os.makedirs(f"{VOL}/data", exist_ok=True)
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    Path(f"{VOL}/data/prompts.jsonl").write_text(body, encoding="utf-8")
    led.beat(force=True)
    vol.commit()
    return dict(body=body, sha256=hashlib.sha256(body.encode()).hexdigest(), info=info, n=len(rows))


@app.local_entrypoint()
def prep_local():
    o = prep.remote()
    (HERE / "prompts.jsonl").write_text(o["body"], encoding="utf-8")
    print(json.dumps(dict(sha256=o["sha256"], n=o["n"], info=o["info"])))


@app.function(image=img_main, cpu=2, memory=8 * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def fetch(name: str):
    from huggingface_hub import snapshot_download
    cfg = MODELS[name]
    led = Ledger("fetch", name, "none", cpu=2, mem=8)
    path = f"{VOL}/models/{name}"
    ok = f"{path}/.complete"
    if not os.path.exists(ok):
        snapshot_download(cfg["repo"], revision=cfg["rev"], local_dir=path, max_workers=8,
                          ignore_patterns=["*.gguf", "*.bin", "*.pth", "*.msgpack", "*.h5", "*.onnx", "*.tflite"])
        Path(ok).write_text(cfg["rev"])
    led.beat(force=True)
    vol.commit()
    return name, round(time.time() - led.t0, 1)


# ------------------------------------------------------------------ states

def load_model(name):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    cfg = MODELS[name]
    path = f"{VOL}/models/{name}"
    assert Path(f"{path}/.complete").read_text() == cfg["rev"]
    t0 = time.time()
    if name == "sarvam30b":
        for f, sha in SARVAM_CODE.items():
            assert sha_file(f"{path}/{f}") == sha, f"{f} differs from the file we read"
        # As topic 08: AutoTokenizer fails for this local folder under 4.57.2, and tokenizer_config.json names this class.
        tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json", pad_token="<pad>", unk_token="<unk>",
                                      clean_up_tokenization_spaces=False)
        tok.chat_template = Path(f"{path}/chat_template.jinja").read_text()
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
                                                     attn_implementation="sdpa")
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok.padding_side = "left"
    info = dict(load_s=round(time.time() - t0, 1), cls=type(model).__name__, torch=torch.__version__, transformers=transformers.__version__)
    return torch, tok, model, info


def eos_ids(model, name):
    e = model.generation_config.eos_token_id
    if name == "sarvam30b":
        e = 26 if e is None else e
    return sorted(set(e if isinstance(e, list) else [e]))


def render(tok, text, prefill=""):
    s = tok.apply_chat_template([dict(role="user", content=text)], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s + prefill, add_special_tokens=False)["input_ids"]


def dump_states(name, rows, max_new, tag, clock, prefill=""):
    """BF16 h at the head for every decision, one greedy and one T = 1 continuation per prompt. States after EOS are dropped."""
    torch, tok, model, info = load_model(name)
    torch.set_grad_enabled(False)
    head = model.get_output_embeddings()
    W = head.weight.detach()
    cfg = MODELS[name]
    assert W.shape == (cfg["V"], cfg["d"]), f"head shape {tuple(W.shape)}"
    root = f"{VOL}/{tag}/{name}"
    os.makedirs(root, exist_ok=True)
    wcpu = W.to("cpu").bfloat16()
    torch.save(wcpu, f"{root}/head.pt")
    info.update(head_shape=list(W.shape), head_sha256=hashlib.sha256(wcpu.float().numpy().tobytes()).hexdigest(), eos=eos_ids(model, name),
                gpu=torch.cuda.get_device_name(0), vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
                tied_to_input=bool(W.data_ptr() == model.get_input_embeddings().weight.data_ptr()))
    pad = tok.pad_token_id
    enc = [(r, render(tok, r["text"], prefill)) for r in rows]
    enc.sort(key=lambda x: (-len(x[1]), x[0]["prompt"]))
    info["prompt_tokens_max"] = len(enc[0][1])
    info["eos_text"] = [tok.decode([t]) for t in info["eos"]]
    info["sample_prompt_tail"] = tok.decode(enc[-1][1][-12:])
    stash = []

    def pre(_m, args):
        stash.append(args[0][:, -1, :].detach().to("cpu", dtype=torch.bfloat16))
    hook = head.register_forward_pre_hook(pre)
    t_all = time.time()
    n_states = 0
    for mode_i, mode in enumerate(("greedy", "sampled")):
        torch.manual_seed(SEED + mode_i)
        torch.cuda.manual_seed_all(SEED + mode_i)
        hs, ps, ss, ts, outs = [], [], [], [], []
        for b in range(0, len(enc), BATCH):
            clock.check()
            part = enc[b:b + BATCH]
            L = max(len(x[1]) for x in part)
            ids = torch.tensor([[pad] * (L - len(x[1])) + x[1] for x in part], device="cuda")
            mask = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in part], device="cuda")
            kw = dict(do_sample=False) if mode == "greedy" else dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
            stash.clear()
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, repetition_penalty=1.0,
                                 pad_token_id=pad, eos_token_id=info["eos"], use_cache=True, **kw)
            new = out[:, L:].to("cpu")
            assert len(stash) == new.shape[1], f"{len(stash)} head calls for {new.shape[1]} tokens"
            H = torch.stack(stash, 1)
            for i, (r, _) in enumerate(part):
                toks = new[i].tolist()
                n = next((j + 1 for j, t in enumerate(toks) if t in info["eos"]), len(toks))
                hs.append(H[i, :n])
                ps += [r["prompt"]] * n
                ss += list(range(n))
                ts += toks[:n]
                outs.append(dict(prompt=r["prompt"], group=r["group"], n=n, text=tok.decode(toks[:n], skip_special_tokens=True)))
        Hall = torch.cat(hs, 0)
        order = sorted(range(len(ps)), key=lambda i: (ps[i], ss[i]))
        idx = torch.tensor(order)
        torch.save(dict(h=Hall[idx], prompt=torch.tensor(ps)[idx], step=torch.tensor(ss)[idx], token=torch.tensor(ts)[idx]), f"{root}/states_{mode}.pt")
        with open(f"{root}/outputs_{mode}.jsonl", "w", encoding="utf-8") as f:
            for o in sorted(outs, key=lambda o: o["prompt"]):
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        n_states += len(ps)
        clock.ledger.beat(force=True)
    hook.remove()
    info.update(dump_s=round(time.time() - t_all, 1), n_states=n_states, s_per_state=(time.time() - t_all) / max(1, n_states))
    with open(f"{root}/dump_info.json", "w") as f:
        json.dump(info, f, indent=1)
    vol.commit()
    return info


# ------------------------------------------------------------------ measurement

def build_heads(W, cfg, dev):
    import torch
    import cert_core as c
    d = cfg["d"]
    rot = c.Rotation(d, SEED + 1, dev)
    heads = {"int8": c.int_head(W, c.A_INT8, c.Q_INT8, d),
             "int4": c.int_head(W, c.A_INT4, c.Q_INT4, c.GROUP),
             "int4d": c.int_head(W, c.A_INT4_DITHER, c.Q_INT4, c.GROUP, c.dither_pattern(d, SEED + 2, dev)),
             "int4h": c.int_head(rot(W), c.A_INT4, c.Q_INT4, c.GROUP),
             "nvfp4": c.nvfp4_head(W)}
    return heads, rot, W.norm(dim=1)


def measure_chunk(H, W, heads, rot, rownorm, cfg, keys, tokens, mode, first, Hb=None, Wb=None):
    """One chunk: sets per certificate and rule, and every control count. Returns arrays by state and a dict of counters."""
    import torch
    import cert_core as c
    cap, V = cfg["cap"], cfg["V"]
    dev = H.device
    ref_raw = H @ W.T
    ref = c.phi(ref_raw, cap)
    tau = c.fp_slack(H, rownorm)
    rules = [c.RULE["greedy"]] if mode == "greedy" else c.RULES
    g = c.gumbel(keys, V, dev) if mode == "sampled" else None
    reftok = {r["name"]: c.ref_token(ref, r, g) for r in rules}
    arrs, ctl = {}, dict(viol={}, mismatch={}, tie_mismatch={}, oracle_exceeded={}, tau_share={})
    n = H.shape[0]

    # Facts about the states
    top2 = ref.topk(2, dim=1).values
    arrs["margin"] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    arrs["h_l2"] = H.norm(dim=1).cpu().numpy()
    arrs["h_l1"] = H.abs().sum(1).cpu().numpy()
    tok = tokens.to(dev)
    ctl["emitted_is_argmax"] = int((ref.argmax(1) == tok).sum())
    ctl["emitted_is_argmax_shuffled"] = int((ref.argmax(1) == tok.roll(1)).sum())
    if Wb is not None:      # the model's own product, in BF16: amendment 1 of the rule
        am16 = c.phi((Hb @ Wb.T).float(), cap).argmax(1)
        ctl["emitted_is_argmax_bf16"] = int((am16 == tok).sum())
        ctl["emitted_is_argmax_bf16_shifted"] = int((am16 == tok.roll(1)).sum())
        ctl["fp32_argmax_is_bf16_argmax"] = int((am16 == ref.argmax(1)).sum())
        del am16
    if mode == "sampled":
        lp = torch.log_softmax(ref, dim=1)
        ctl["nll_sum"] = float(-lp.gather(1, tok.view(-1, 1)).double().sum())
        ctl["entropy_sum"] = float(-(lp.exp() * lp).sum(1).double().sum())
        del lp

    def apply_rules(L, U, label, base_sets=None):
        sets = {}
        for r in rules:
            R, aux = c.rule_set(L, U, r, g)
            arrs[f"R|{label}|{r['name']}"] = R.sum(1).int().cpu().numpy()
            if "|oracle" not in label:
                t = c.cert_token(ref, R, aux, r, g)
                bad = t != reftok[r["name"]]
                ctl["mismatch"][f"{label}|{r['name']}"] = int(bad.sum())
                if bad.any() and r["kind"] == "topk":
                    vals = ref[bad].topk(r["k"] + 1, dim=1).values
                    ctl["tie_mismatch"][f"{label}|{r['name']}"] = int((vals[:, -1] == vals[:, -2]).sum())
            sets[r["name"]] = R
        return sets

    L8 = U8 = None
    for hname in ("int8", "int4", "int4d", "int4h", "nvfp4"):
        hd = heads[hname]
        Hq = rot(H) if hname == "int4h" else H
        approx = Hq @ hd["W"].T
        err = (approx - ref_raw).abs()
        H1, H2 = c.group_norms(Hq, hd["group"])
        Lo, Uo = c.interval(approx, err, cap)
        oset = apply_rules(Lo, Uo, f"{hname}|oracle")
        ogreedy = oset["greedy"].sum(1)
        del oset, Lo, Uo
        bounds = dict(worst=c.bound_worst(hd, H1), resid=c.bound_resid(hd, H2))
        if hname == "int4d":
            bounds["dither"] = c.bound_dither(hd, H2)
            sd = c.predicted_sd(hd, H2)
            zz = ((approx - ref_raw) / sd) ** 2
            ctl["dither_z2_sum"], ctl["dither_z2_n"] = float(zz.sum()), zz.numel()
            del sd, zz
        emax = err.amax(1)
        emean = err.mean(1).clamp_min(1e-12)
        arrs[f"err_max|{hname}"] = emax.cpu().numpy()
        arrs[f"err_mean|{hname}"] = emean.cpu().numpy()
        for bname, b in bounds.items():
            label = f"{hname}|{bname}"
            ctl["viol"][label] = int((err > b + tau).sum())
            ctl["tau_share"][label] = float((tau.mean(1) / b.mean(1)).amax())
            arrs[f"b_mean|{label}"] = b.mean(1).cpu().numpy()
            arrs[f"b_max|{label}"] = b.amax(1).cpu().numpy()
            L, U = c.interval(approx, b + tau, cap)
            sets = apply_rules(L, U, label)
            ctl["oracle_exceeded"][label] = int((ogreedy > sets["greedy"].sum(1)).sum())
            if hname == "int8" and bname == "resid":
                L8, U8 = L, U
            if (hname, bname) in CASCADE_BASES:
                for r in rules:
                    S = sets[r["name"]]
                    L2, U2 = c.refine(L, U, S, L8, U8)
                    R2, aux = c.rule_set(L2, U2, r, g)
                    lab = f"{label}+int8"
                    arrs[f"R|{lab}|{r['name']}"] = R2.sum(1).int().cpu().numpy()
                    arrs[f"S|{lab}|{r['name']}"] = S.sum(1).int().cpu().numpy()
                    bad = c.cert_token(ref, R2, aux, r, g) != reftok[r["name"]]
                    ctl["mismatch"][f"{lab}|{r['name']}"] = int(bad.sum())
                    if bad.any() and r["kind"] == "topk":
                        vals = ref[bad].topk(r["k"] + 1, dim=1).values
                        ctl["tie_mismatch"][f"{lab}|{r['name']}"] = int((vals[:, -1] == vals[:, -2]).sum())
                    del R2, L2, U2
            del L, U, sets
        del approx, err, bounds

    if first:
        pc = {}
        # Planted faults. Each of these must fire, or the counters above mean nothing.
        hd = heads["int4"]
        H1, H2 = c.group_norms(H, hd["group"])
        approx = H @ hd["W"].T
        err = (approx - ref_raw).abs()
        pc["small_bound_violations"] = int((err > 0.02 * c.bound_resid(hd, H2)).sum())
        gen = torch.Generator(device="cpu").manual_seed(SEED + 3)
        perm = torch.randperm(V, generator=gen).to(dev)
        ap = H @ hd["W"][perm].T
        b = c.bound_resid(hd, H2) + tau
        pc["permuted_rows_violations"] = int(((ap - ref_raw).abs() > b).sum())
        L, U = c.interval(ap, b, cap)
        R, aux = c.rule_set(L, U, c.RULE["greedy"], g)
        pc["permuted_rows_mismatch"] = int((c.cert_token(ref, R, aux, c.RULE["greedy"], g) != reftok["greedy"]).sum())
        # A set with the winner removed must give a wrong token on every state, or the exactness count means nothing.
        R0 = R.clone()
        R0.scatter_(1, reftok["greedy"].view(-1, 1), False)
        pc["dropped_winner_mismatch_share"] = float((c.cert_token(ref, R0, aux, c.RULE["greedy"], g) != reftok["greedy"]).float().mean())
        del R0
        # Brackets. An exact head leaves one row; an empty head leaves very many.
        L, U = c.interval(ref_raw, 0.0 * ref_raw, cap)
        pc["exact_head_share_R1"] = float((c.rule_set(L, U, c.RULE["greedy"], g)[0].sum(1) == 1).float().mean())
        b0 = torch.outer(H.norm(dim=1), rownorm) + tau
        L, U = c.interval(torch.zeros_like(ref_raw), b0, cap)
        Rz = c.rule_set(L, U, c.RULE["greedy"], g)[0]
        pc["empty_head_median_R_share"] = float(Rz.sum(1).float().median() / V)
        pc["empty_head_violations"] = int((ref_raw.abs() > b0).sum())
        # FP32 against FP64 on a few states
        k = min(64, n)
        r64 = H[:k].double() @ W.double().T
        d3264 = (r64 - ref_raw[:k].double()).abs()
        pc["fp32_fp64_max_abs"] = float(d3264.max())
        pc["fp32_fp64_argmax_agree"] = float((r64.argmax(1) == ref_raw[:k].argmax(1)).double().mean())
        pc["tau_max_on_those"] = float(tau[:k].max())
        pc["tau_over_fp32_err"] = float((tau[:k].double() / d3264.clamp_min(1e-12)).amin())
        ctl["positive_controls"] = pc
    return arrs, ctl


def measure_states(name, tag, clock, chunk):
    import numpy as np
    import torch
    import cert_core as c
    torch.set_grad_enabled(False)
    cfg = MODELS[name]
    dev = "cuda"
    root = f"{VOL}/{tag}/{name}"
    out = f"{root}/measure"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    Wb = torch.load(f"{root}/head.pt", map_location="cpu").to(dev)
    W = Wb.float()
    heads, rot, rownorm = build_heads(W, cfg, dev)
    t_build = time.time() - t0
    timing = dict(build_s=round(t_build, 1), chunks=[], peak_gib=0.0)
    for mode_i, mode in enumerate(("greedy", "sampled")):
        s = torch.load(f"{root}/states_{mode}.pt", map_location="cpu")
        n = s["h"].shape[0]
        for ci, a in enumerate(range(0, n, chunk)):
            path = f"{out}/{mode}_{ci:04d}.npz"
            if os.path.exists(path):
                continue
            clock.check()
            t1 = time.time()
            z = min(a + chunk, n)
            H = s["h"][a:z].to(dev).float()
            keys = c.state_key(s["prompt"][a:z], s["step"][a:z], mode_i)
            arrs, ctl = measure_chunk(H, W, heads, rot, rownorm, cfg, keys, s["token"][a:z], mode, first=(ci == 0), Hb=s["h"][a:z].to(dev), Wb=Wb)
            arrs["prompt"] = s["prompt"][a:z].numpy()
            arrs["step"] = s["step"][a:z].numpy()
            ctl.update(mode=mode, chunk=ci, n=z - a, seconds=round(time.time() - t1, 2))
            np.savez(path + ".tmp.npz", **arrs)
            os.replace(path + ".tmp.npz", path)
            with open(f"{out}/{mode}_{ci:04d}.json", "w") as f:
                json.dump(ctl, f)
            timing["chunks"].append(dict(mode=mode, n=z - a, s=round(time.time() - t1, 2)))
            timing["peak_gib"] = max(timing["peak_gib"], round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
            vol.commit()
            torch.cuda.empty_cache()
    with open(f"{root}/measure_timing.json", "w") as f:
        json.dump(timing, f)
    vol.commit()
    return timing


def load_rows(n_per_group=None):
    rows = [json.loads(x) for x in Path(f"{VOL}/data/prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    if n_per_group is None:
        return rows
    kept, seen = [], {}
    for r in rows:
        seen[r["group"]] = seen.get(r["group"], 0) + 1
        if seen[r["group"]] <= n_per_group:
            kept.append(r)
    return kept


def control_digest(name, tag):
    """Pass or fail flags and timing only. No set sizes, no bound widths."""
    root = f"{VOL}/{tag}/{name}"
    viol, mism, ties, orc, pos, em, ems, nst, nll, ent, zz, zn = 0, 0, 0, 0, None, 0, 0, 0, 0.0, 0.0, 0.0, 0
    gn = 0
    per = {"greedy": [0, 0.0], "sampled": [0, 0.0]}
    for p in sorted(Path(f"{root}/measure").glob("*.json")):
        c = json.loads(p.read_text())
        viol += sum(v for k, v in c["viol"].items() if not k.endswith("dither"))
        dv = sum(v for k, v in c["viol"].items() if k.endswith("dither"))
        mism += sum(c["mismatch"].values())
        ties += sum(c["tie_mismatch"].values())
        orc += sum(c["oracle_exceeded"].values())
        pos = c.get("positive_controls", pos) if pos is None else pos
        if c["mode"] == "greedy":
            em += c["emitted_is_argmax"]
            ems += c["emitted_is_argmax_shuffled"]
            gn += c["n"]
        else:
            nll += c["nll_sum"]
            ent += c["entropy_sum"]
        zz += c["dither_z2_sum"]
        zn += c["dither_z2_n"]
        nst += c["n"]
        per[c["mode"]][0] += c["n"]
        per[c["mode"]][1] += c["seconds"]
    return dict(states=nst, sound_violations=viol, mismatches=mism, tie_mismatches=ties, oracle_exceeded=orc, positive=pos,
                emitted_argmax_rate=em / max(1, gn), emitted_shuffled_rate=ems / max(1, gn), nll_over_entropy=nll / max(1e-9, ent),
                dither_variance_ratio=zz / max(1, zn), seconds_per_state={k: v[1] / max(1, v[0]) for k, v in per.items()})


# ------------------------------------------------------------------ the steps, by image

def _smoke(name, gpu_mem):
    led = Ledger("smoke", name, GPU_NAME, mem=gpu_mem)
    clock = Clock(1500, led)
    info = dump_states(name, load_rows(2), 12, "smoke", clock)
    return info, led


def _write_smoke(name, info, digest):
    p = f"{VOL}/smoke/{name}/smoke_summary.json"
    with open(p, "w") as f:
        json.dump(dict(info=info, controls=digest), f, indent=1)
    vol.commit()


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=1800)
def smoke_main(name: str):
    info, led = _smoke(name, MEM_GIB)
    timing = measure_states(name, "smoke", Clock(1200, led), 128)
    digest = control_digest(name, "smoke")
    digest["build_s"] = timing["build_s"]
    digest["peak_gib"] = timing["peak_gib"]
    _write_smoke(name, info, digest)
    led.beat(force=True)
    print(json.dumps(dict(info=info, controls=digest)), flush=True)


@app.function(image=img_sarvam, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB_SARVAM * 1024, volumes={VOL: vol}, timeout=1800)
def smoke_sarvam_dump():
    info, led = _smoke("sarvam30b", MEM_GIB_SARVAM)
    led.beat(force=True)
    print(json.dumps(info), flush=True)


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=1800)
def smoke_sarvam_measure():
    led = Ledger("smoke_measure", "sarvam30b", GPU_NAME)
    timing = measure_states("sarvam30b", "smoke", Clock(1200, led), 128)
    digest = control_digest("sarvam30b", "smoke")
    digest["build_s"] = timing["build_s"]
    digest["peak_gib"] = timing["peak_gib"]
    info = json.loads(Path(f"{VOL}/smoke/sarvam30b/dump_info.json").read_text())
    _write_smoke("sarvam30b", info, digest)
    led.beat(force=True)
    print(json.dumps(dict(info=info, controls=digest)), flush=True)


def _dump_full(name, limit_s, mem):
    led = Ledger("dump", name, GPU_NAME, mem=mem)
    plan = json.loads(Path(f"{VOL}/plan.json").read_text()) if os.path.exists(f"{VOL}/plan.json") else {}
    rows = load_rows(plan.get("prompts_per_group"))
    info = dump_states(name, rows, MAX_NEW, "main", Clock(limit_s, led))
    led.beat(force=True)
    print(json.dumps(info), flush=True)


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=2 * 3600)
def dump_main(name: str, limit_s: float):
    _dump_full(name, limit_s, MEM_GIB)


@app.function(image=img_sarvam, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB_SARVAM * 1024, volumes={VOL: vol}, timeout=2 * 3600)
def dump_sarvam(limit_s: float):
    _dump_full("sarvam30b", limit_s, MEM_GIB_SARVAM)


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def measure(name: str, limit_s: float):
    led = Ledger("measure", name, GPU_NAME)
    try:
        timing = measure_states(name, "main", Clock(limit_s, led), CHUNK)
        print(json.dumps(dict(chunks=len(timing["chunks"]), build_s=timing["build_s"], peak_gib=timing["peak_gib"])), flush=True)
    except OutOfBudget as e:
        print("stopped:", e, flush=True)
    led.beat(force=True)
    print(json.dumps(control_digest(name, "main")), flush=True)


# ------------------------------------------------------------------ a diagnostic added after the smoke test, for control C8 only

def _diag(name):
    """Does generate sample from the distribution the reference says? It compares the logits generate returned with the head
    applied to the hooked h, with and without the soft-cap, and scores the sampled tokens. It computes no set and no bound."""
    import cert_core as c
    led = Ledger("diag", name, GPU_NAME)
    torch, tok, model, info = load_model(name)
    torch.set_grad_enabled(False)
    head = model.get_output_embeddings()
    W = head.weight.detach().float()
    cap = MODELS[name]["cap"]
    rows = load_rows(4)
    enc = sorted([render(tok, r["text"]) for r in rows], key=len)
    L = max(len(x) for x in enc)
    pad = tok.pad_token_id
    ids = torch.tensor([[pad] * (L - len(x)) + x for x in enc], device="cuda")
    mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in enc], device="cuda")
    stash = []
    hook = head.register_forward_pre_hook(lambda _m, a: stash.append(a[0][:, -1, :].detach().float()))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=64, repetition_penalty=1.0, pad_token_id=pad, eos_token_id=eos_ids(model, name),
                         use_cache=True, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, return_dict_in_generate=True, output_logits=True, output_scores=True)
    hook.remove()
    new = out.sequences[:, L:]
    eos = set(eos_ids(model, name))
    rep = dict(name=name, steps=len(stash), batch=len(enc), gen_cfg_cap=getattr(model.config.get_text_config(), "final_logit_softcapping", None))
    agg = {k: [] for k in ("nll_capped", "ent_capped", "var_capped", "nll_raw", "ent_raw", "var_raw")}
    dmax_cap, dmax_raw, sdiff = 0.0, 0.0, 0.0
    for t, h in enumerate(stash):
        raw = h @ W.T
        capd = c.phi(raw, cap)
        got = out.logits[t].float()
        sc = out.scores[t].float()
        alive = torch.tensor([all(x not in eos for x in new[i, :t].tolist()) for i in range(new.shape[0])], device="cuda")
        if not alive.any():
            break
        dmax_cap = max(dmax_cap, float((got[alive] - capd[alive]).abs().max()))
        dmax_raw = max(dmax_raw, float((got[alive] - raw[alive]).abs().max()))
        sdiff = max(sdiff, float((sc[alive] - got[alive]).abs().max()))
        for lab, x in (("capped", capd), ("raw", raw)):
            lp = torch.log_softmax(x[alive].double(), 1)
            nll = -lp.gather(1, new[alive, t].view(-1, 1)).flatten()
            ent = -(lp.exp() * lp).sum(1)
            var = (lp.exp() * lp * lp).sum(1) - ent * ent
            agg[f"nll_{lab}"] += nll.tolist()
            agg[f"ent_{lab}"] += ent.tolist()
            agg[f"var_{lab}"] += var.tolist()
    rep.update(max_abs_generate_logits_vs_capped_head=dmax_cap, max_abs_generate_logits_vs_raw_head=dmax_raw, max_abs_scores_vs_logits=sdiff)
    for lab in ("capped", "raw"):
        n = len(agg[f"nll_{lab}"])
        a, b, v = sum(agg[f"nll_{lab}"]), sum(agg[f"ent_{lab}"]), sum(agg[f"var_{lab}"])
        rep[lab] = dict(states=n, nll_mean=a / n, ent_mean=b / n, ratio=a / b, z=(a - b) / math.sqrt(max(v, 1e-12)))
    led.beat(force=True)
    print(json.dumps(rep), flush=True)


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=900)
def diag_sampling(name: str):
    _diag(name)


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=900)
def diag_alignment(tag: str = "smoke"):
    """Is a greedy token that differs from the FP32 argmax a broken mapping, or BF16 rounding in the model's own head? It repeats
    the head product in BF16 from the stored h. It computes no set and no bound."""
    import torch
    import cert_core as c
    torch.set_grad_enabled(False)
    led = Ledger("diag_align", tag, GPU_NAME)
    for name, cfg in MODELS.items():
        root = f"{VOL}/{tag}/{name}"
        s = torch.load(f"{root}/states_greedy.pt", map_location="cpu")
        Wb = torch.load(f"{root}/head.pt", map_location="cpu").to("cuda")
        H = s["h"].to("cuda")
        tok = s["token"].to("cuda")
        n = H.shape[0]
        a32, a16, rawmax, flipgap, ulp = [], [], [], [], []
        for i in range(0, n, 512):
            h = H[i:i + 512]
            t = tok[i:i + 512]
            r32 = c.phi(h.float() @ Wb.float().T, cfg["cap"])
            r16 = c.phi((h @ Wb.T).float(), cfg["cap"])
            am32, am16 = r32.argmax(1), r16.argmax(1)
            a32.append((am32 == t).float())
            a16.append((am16 == t).float())
            top = r32.topk(1, dim=1).values.flatten()
            rawmax.append(top)
            bad = am32 != t
            if bad.any():
                flipgap.append(top[bad] - r32[bad].gather(1, t[bad].view(-1, 1)).flatten())
        a32, a16, top = torch.cat(a32), torch.cat(a16), torch.cat(rawmax)
        rep = dict(name=name, states=n, fp32_argmax_is_emitted=float(a32.mean()), bf16_argmax_is_emitted=float(a16.mean()), top_logit_median=float(top.median()), top_logit_max=float(top.max()))
        if flipgap:
            f = torch.cat(flipgap)
            rep.update(flips=int(f.numel()), flip_gap_median=float(f.median()), flip_gap_max=float(f.max()))
        print(json.dumps(rep), flush=True)
        del Wb, H
        torch.cuda.empty_cache()
    led.beat(force=True)


# ------------------------------------------------------------------ additions after the main run: see logs/addendum_runB.json

SARVAM_PREFILL = "<think></think>\n"     # how Sarvam's own chat template renders an assistant turn with no reasoning


@app.function(image=img_sarvam, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB_SARVAM * 1024, volumes={VOL: vol}, timeout=2 * 3600)
def dump_sarvam_b(limit_s: float):
    led = Ledger("dump_b", "sarvam30b", GPU_NAME, mem=MEM_GIB_SARVAM)
    info = dump_states("sarvam30b", load_rows(None), MAX_NEW, "main_b", Clock(limit_s, led), prefill=SARVAM_PREFILL)
    led.beat(force=True)
    print(json.dumps(info), flush=True)


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=3 * 3600)
def measure_b(limit_s: float):
    led = Ledger("measure_b", "sarvam30b", GPU_NAME)
    timing = measure_states("sarvam30b", "main_b", Clock(limit_s, led), CHUNK)
    led.beat(force=True)
    print(json.dumps(dict(chunks=len(timing["chunks"]), peak_gib=timing["peak_gib"])), flush=True)


def stepwise_topk(Lo, Uo, L8, U8, ref, rule, g, rounds=4):
    """Post hoc read plan for top-k. Read only rows that can still win the race, tighten them, and ask again. Rows of uncertain
    membership are read only for states where a candidate still has uncertain membership after that."""
    import math as m
    import torch
    import cert_core as c
    k, T = rule["k"], rule["T"]
    S = torch.zeros_like(ref, dtype=torch.bool)
    L, U = Lo, Uo
    for _ in range(rounds):
        C = U >= L.topk(k, dim=1).values[:, -1:]
        D = L > U.topk(k + 1, dim=1).values[:, -1:]
        best = torch.where(D, L / T + g, -m.inf).amax(1, keepdim=True)
        race = C & (U / T + g >= best)
        if not (race & ~S).any():
            break
        S = S | race
        L, U = c.refine(Lo, Uo, S, L8, U8)
    C = U >= L.topk(k, dim=1).values[:, -1:]
    D = L > U.topk(k + 1, dim=1).values[:, -1:]
    best = torch.where(D, L / T + g, -m.inf).amax(1, keepdim=True)
    race = C & (U / T + g >= best)
    amb = (race & ~D).any(1, keepdim=True)
    S = S | (amb & C & ~D)
    L, U = c.refine(Lo, Uo, S, L8, U8)
    R, aux = c.rule_set(L, U, rule, g)
    return S, R, aux


@app.function(image=img_main, gpu=GPU_NAME, cpu=CPU_N, memory=MEM_GIB * 1024, volumes={VOL: vol}, timeout=3600)
def posthoc_topk(name: str, tag: str, limit_s: float):
    import numpy as np
    import torch
    import cert_core as c
    torch.set_grad_enabled(False)
    led = Ledger("posthoc_topk", f"{name}_{tag}", GPU_NAME)
    clock = Clock(limit_s, led)
    cfg = MODELS[name]
    cap, V = cfg["cap"], cfg["V"]
    root = f"{VOL}/{tag}/{name}"
    out = f"{root}/posthoc_topk"
    os.makedirs(out, exist_ok=True)
    W = torch.load(f"{root}/head.pt", map_location="cpu").to("cuda").float()
    heads, rot, rownorm = build_heads(W, cfg, "cuda")
    rule = c.RULE["topk50_t1.0"]
    s = torch.load(f"{root}/states_sampled.pt", map_location="cpu")
    n = s["h"].shape[0]
    bad = {"int4d|dither+int8": 0, "int8|resid+bf16": 0}
    for ci, a in enumerate(range(0, n, CHUNK)):
        clock.check()
        z = min(a + CHUNK, n)
        H = s["h"][a:z].to("cuda").float()
        g = c.gumbel(c.state_key(s["prompt"][a:z], s["step"][a:z], 1), V, "cuda")
        raw = H @ W.T
        ref = c.phi(raw, cap)
        tau = c.fp_slack(H, rownorm)
        want = c.ref_token(ref, rule, g)
        h8, h4 = heads["int8"], heads["int4d"]
        L8, U8 = c.interval(H @ h8["W"].T, c.bound_resid(h8, c.group_norms(H, h8["group"])[1]) + tau, cap)
        L4, U4 = c.interval(H @ h4["W"].T, c.bound_dither(h4, c.group_norms(H, h4["group"])[1]) + tau, cap)
        S, R, aux = stepwise_topk(L4, U4, L8, U8, ref, rule, g)
        arrs = {"S|int4d|dither+int8": S.sum(1).int().cpu().numpy(), "R|int4d|dither+int8": R.sum(1).int().cpu().numpy()}
        bad["int4d|dither+int8"] += int((c.cert_token(ref, R, aux, rule, g) != want).sum())
        # The INT8 baseline under the same plan: rows it reads are BF16, so their intervals become points.
        Le, Ue = ref - c.ABS_SLACK, ref + c.ABS_SLACK
        S2, R2, aux2 = stepwise_topk(L8, U8, Le, Ue, ref, rule, g)
        arrs["R|int8|resid+bf16"] = (S2 | R2).sum(1).int().cpu().numpy()
        bad["int8|resid+bf16"] += int((c.cert_token(ref, R2, aux2, rule, g) != want).sum())
        arrs["prompt"] = s["prompt"][a:z].numpy()
        np.savez(f"{out}/{ci:04d}.npz", **arrs)
        del g, raw, ref, tau, L8, U8, L4, U4, S, R, Le, Ue, S2, R2
        torch.cuda.empty_cache()
    with open(f"{out}/mismatch.json", "w") as f:
        json.dump(bad, f)
    led.beat(force=True)
    vol.commit()
    print(json.dumps(bad), flush=True)
