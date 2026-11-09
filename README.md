# Certified sampling with a 4-bit LM head: Exp01

This is the code for Exp01 of topic 07 in the Vizuara Inference Research Bootcamp. The experiment asks one question. If a model reads a cheap 4-bit copy of its output head first, how many rows of the full BF16 head must it still read to pick exactly the token the BF16 head would have picked?

The idea is a certificate. A low-bit head gives an approximate logit for every vocabulary row, and a bound says how far that approximation can be from the true logit. Each row then has an interval. For a given sampling rule (greedy, Gumbel-max at a temperature, top-k or min-p), most rows can be ruled out from their intervals alone. Only the rows that remain need an exact BF16 read, and the token that comes out is the same as the reference. We measured the size of that remaining set, and the bytes it costs, on real decode states from three models.

## Results in five lines

1. Under the decision rule written before the run, the outcome is MIDDLE: the claim is neither confirmed nor killed.
2. Deterministic 4-bit bounds (worst-case and residual-norm) did not beat an INT8 head on any model. Median bytes were 0.87 to 1.29 of the BF16 head.
3. A dithered INT4 head with a high-probability (Hoeffding) bound, followed by an INT8 shortlist, met every gate on Gemma 4 E4B and Qwen3-8B for greedy, Gumbel-max and min-p: median 0.258 of BF16 head bytes, against 0.501 for INT8.
4. It does not work on Sarvam-30B (median 0.46 to 0.48, 99th percentile near 0.77), and top-k fails the rule on both 262K-vocabulary models.
5. No row exceeded its bound in 51.4 billion rows checked per head variant, and every certified token matched the FP32 reference.

The full lab note, with tables, controls and caveats, is in [docs/exp01-results.md](docs/exp01-results.md).

## Files

| File | What it is |
|-|-|
| `cert_core.py` | The method itself, in plain PyTorch: the five head variants (INT8, INT4, dithered INT4, rotated INT4, NVFP4 emulation), the bounds, the candidate sets for each sampling rule, counter-based Gumbel noise, and the byte model. `python3 cert_core.py` runs its self tests on a CPU. |
| `exp01_modal.py` | The Modal app. It writes the prompts, downloads the pinned models, runs generation with a hook that saves the hidden state at the head for every decision, and measures every certificate and control on those states on a GPU. |
| `ctl.py` | One command-line entry point for every step: lock, selftest, prep, fetch, hash, smoke, plan, dump, measure, pull, and a spending guard. |
| `drive.py` | Runs the planned dump and measure stages for each model, one after another, with the guard between them. |
| `analyze.py` | Applies `decision_rule.json` to the measured results and writes `summary.json`. |
| `analyze_b.py` | The same for the follow-up Sarvam-30B run with thinking closed (run B) and the post hoc top-k pass. Writes `summary_runB.json`. |
| `byte_share.py` | Reads only the safetensors headers from Hugging Face and works out what share of the weight bytes read per token belongs to the head. Runs on a laptop. |
| `make_lock.py` | Rebuilds `requirements.lock.txt` inside a small Modal CPU container. |
| `test_local.py` | A dry run of the measure step on synthetic data, on a CPU, before any GPU is paid for. |
| `requirements.lock.txt` | Pinned, hashed dependencies for the main image (Python 3.12, torch 2.13.0, transformers 5.19.0). |
| `sarvam.lock.txt` | Pinned dependencies for the Sarvam-30B image (Python 3.11, torch 2.5.1, transformers 4.57.2), which its custom model code needs. |
| `prompts.jsonl` | The 240 prompts: 40 English, 40 code, and 40 each of Hindi, Bengali, Tamil and Telugu translation. |
| `decision_rule.json` | The pre-registered rule: models, heads, bounds, rules, controls, and the confirm and kill gates. |
| `summary.json`, `summary_runB.json` | The analysis output that the lab note is based on. |
| `results/`, `results_runB/` | Per model: the generated text (`outputs_*.jsonl`), facts about the run (`dump_info.json`) and timing (`measure_timing.json`). |
| `docs/exp01-results.md` | The lab note. |

## Try it on a CPU first

You need Python 3.11 or later with `torch`, `numpy` and `modal` installed. No GPU and no Modal account are needed for this part.

```
python3 cert_core.py                    # self tests, about a minute
python3 test_local.py /tmp/exp01-fixture
python3 analyze.py /tmp/exp01-fixture   # analysis on the synthetic fixture
```

`test_local.py` should print the planted-fault controls and end with `violations 0 mismatches 0 oracle 0`. The synthetic fixture uses random weights, so the outcome that `analyze.py` prints for it means nothing.

## Run it on Modal with your own account

1. Install Modal and sign in: `pip install modal`, then `modal setup`. Every command below uses your active Modal profile. If you keep several profiles, set `MODAL_PROFILE` to the one you want.
2. Set a spending guard in US dollars, for example `export EXP01_BUDGET_USD=20`. `ctl.py` gives each stage a wall-clock limit from the money left and refuses to start a stage once the guard is used up. Our full run, all three models plus run B and the post hoc pass, cost about 6 USD at Modal list prices on A100-80GB GPUs.
3. Run the steps from this folder:

```
python3 ctl.py selftest
python3 ctl.py prep           # writes prompts.jsonl (the one here is what we used)
python3 ctl.py fetch          # downloads the three pinned models to a Modal volume
python3 ctl.py hash           # records the SHA-256 of decision_rule.json and the code in logs/
python3 ctl.py smoke          # a few states per model, every control, and throughput
python3 ctl.py plan           # chooses which models fit the budget, from throughput only
python3 drive.py              # dump and measure each planned model
python3 ctl.py pull           # copies results from the volume into results/
python3 analyze.py            # applies the rule and writes summary.json
```

For run B and the post hoc top-k pass, run `python3 ctl.py runb`, `python3 ctl.py posthoc`, `python3 ctl.py pullb`, then `python3 analyze_b.py`. `python3 ctl.py status`, `costs` and `stop` show what is running, what it has cost, and stop our apps.

All three models are public on Hugging Face and not gated, so no Hugging Face token is needed. The models are pinned to exact revisions in `exp01_modal.py`, and Sarvam-30B's custom code is checked against a SHA-256 before it loads. Sarvam-30B needs 96 GiB of host memory on its container.

## What is not in this repository

- The per-chunk measurement arrays (about 110 MB) and the hidden states and head weights saved on the Modal volume. Rerunning the steps above recreates them.
- The `logs/` folder from our run: launch logs, the hash records, diagnostics and the run B addendum. The lab note refers to some of these files. `analyze.py` and `analyze_b.py` check the hashes only when those logs exist, and otherwise run the self tests in place of the logged result.
- The literature review that the lab note calls "the review".

`decision_rule.json` here differs from the hashed original in three fields (`status`, `budget` and `cut_rule`), which were reworded to remove internal notes and spending figures. No definition, threshold or gate changed, but its SHA-256 no longer matches the one logged at the time. `summary.json` has the Modal billing rows and planning budget removed, and keeps the total cost.

## Licences

The code in this repository is under the MIT licence; see [LICENSE](LICENSE).

Models used (weights are downloaded from Hugging Face, not redistributed here):

| Model | Revision | Licence |
|-|-|-|
| `google/gemma-4-E4B-it` | 4ecc2ba6 | Apache 2.0 |
| `Qwen/Qwen3-8B` | b968826d | Apache 2.0 |
| `sarvamai/sarvam-30b` | 071ae95e | Apache 2.0 |

Datasets used for the prompts in `prompts.jsonl`:

| Source | Used for | Licence |
|-|-|-|
| MT-Bench questions, from `lm-sys/FastChat` at b494d0c6 | 40 English prompts (first turns) | Apache 2.0 |
| HumanEval, from `openai/human-eval` at 463c980b | 40 code prompts (tasks 0 to 39) | MIT |
| `PleIAs/US-PD-Newspapers` at 39dc0a2a | English source passages for the 160 translation prompts | CC0 1.0 |

The text in `results/` and `results_runB/` is model output on these prompts. The Flores-IN benchmark was not used, because its CC BY-SA 4.0 terms did not fit the bootcamp's licence rule. The packages pinned in the lockfiles each carry their own licences.
