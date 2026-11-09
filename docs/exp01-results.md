# Exp01, how wide is a 4-bit certificate, topic 07: lab note

Run by the Vizuara research team on 5 November 2026.

## In short

The rule's outcome is MIDDLE. It is neither confirmed nor killed.

Deterministic 4-bit bounds did not help on any of the three models. The worst-case and residual-norm bounds cannot exclude most rows, so a first pass reads the 4-bit head and then most or all of the BF16 head. Median bytes are 0.87 to 1.29 of BF16, and adding an INT8 stage does not bring any of them under the INT8 baseline.

One 4-bit certificate works, and only on two of the three models. It uses a dithered first pass with a high-probability bound, followed by an INT8 shortlist. On Gemma 4 E4B it meets every confirm gate for greedy and T = 1 in all six language groups: median total head bytes 0.258 of BF16, 99th percentile 0.31 over all states and at most 0.32 in any group, one BF16 row at the median. The INT8 baseline reads 0.501. Qwen3-8B, which is outside the rule, does better still.

It does not work on Sarvam-30B, the model the paper most wants. Median bytes for greedy and T = 1 are 0.46 to 0.48 in the main run and 0.32 to 0.36 with its thinking closed, and the 99th percentile is near 0.77 either way, against 0.53 for INT8. Top-k fails the rule on both 262K models. Min-p fails it on Sarvam-30B.

The measurement itself held. No row exceeded a bound over 51.4 billion rows checked per head variant, and every certified token matched the reference.

## What we ran

This is the review's E0 (`literature-review.md`, section 6). Each model got 240 prompts: 40 English (MT-Bench), 40 code (HumanEval), and 40 each of Hindi, Bengali, Tamil and Telugu. Each prompt got one greedy and one T = 1 continuation of up to 128 tokens at batch 16 in BF16. A hook saved h at the head for every decision, and states after EOS were dropped.

| Model | Head | Licence, gated | Decode states |
|-|-|-|-|
| Gemma 4 E4B at 4ecc2ba6 | 262,144 x 2,560, tied, soft-cap 30 | Apache 2.0, no | 48,051 |
| Qwen3-8B at b968826d | 151,936 x 4,096, untied | Apache 2.0, no | 55,298 |
| Sarvam-30B at 071ae95e, main run | 262,144 x 4,096, untied | Apache 2.0, no | 61,440 |
| Sarvam-30B, run B, thinking closed | the same | the same | 54,504 |

We built five head variants: INT8 per row, INT4 groups of 128, dithered INT4, rotated INT4 and NVFP4. Each got worst-case and residual-norm bounds, the dithered head also got the Hoeffding bound, and each got the true error as an oracle. Five bases also got the INT8 cascade. All rules ran on all of these with a counter-based Gumbel noise. `decision_rule.json` has the definitions and `cert_core.py` has the code.

The reference is the FP32-accumulated product of the BF16 row and the BF16 h in our fixed matmul. "Exact" below means equal to this reference, and not to a stock serving kernel.

**Flores-IN was skipped.** It is CC BY-SA 4.0, whose text says that exercising the licensed rights binds the user to its terms, and the brief's licence rule excludes that. The translation prompts use five consecutive sentences from `PleIAs/US-PD-Newspapers` at 39dc0a2a (CC0, not gated), filtered for OCR noise. Some OCR breaks remain, such as "coun try". The states we measure are the model's own output in the target script, so the source text matters less than the output, but this is not the benchmark the review named.

## Main table

Each row is the dithered INT4 head with the INT8 stage, which is the best eligible certificate on every row or ties it to three places. S is the rows read in INT8, R the rows read in BF16, and each pair is median then 99th percentile. Bytes are fractions of the BF16 head. Greedy is scored on greedy states and the other rules on T = 1 states.

| Model | Rule | Status | S | R | Bytes median, p99, mean | INT8 bytes median, p99 |
|-|-|-|-|-|-|-|
| Gemma 4 E4B | greedy | GATES_PASS | 37, 23,089 | 1, 13 | 0.258, 0.302, 0.262 | 0.501, 0.501 |
| Gemma 4 E4B | Gumbel T = 0.6 | GATES_PASS | 43, 25,826 | 1, 16 | 0.258, 0.308, 0.263 | 0.501, 0.501 |
| Gemma 4 E4B | Gumbel T = 1 | GATES_PASS | 49, 25,654 | 1, 18 | 0.258, 0.307, 0.263 | 0.501, 0.501 |
| Gemma 4 E4B | top-50 | DEAD | 126,659, 201,225 | 1, 8,829 | 0.503, 0.644, 0.450 | 0.501, 0.535 |
| Gemma 4 E4B | min-p 0.05 | GATES_PASS | 165, 42,447 | 1, 60 | 0.259, 0.340, 0.266 | 0.501, 0.501 |
| Qwen3-8B | greedy | GATES_PASS | 3, 206 | 1, 10 | 0.258, 0.259, 0.258 | 0.501, 0.501 |
| Qwen3-8B | Gumbel T = 0.6 | GATES_PASS | 3, 208 | 1, 10 | 0.258, 0.259, 0.258 | 0.501, 0.501 |
| Qwen3-8B | Gumbel T = 1 | GATES_PASS | 3, 213 | 1, 10 | 0.258, 0.259, 0.258 | 0.501, 0.501 |
| Qwen3-8B | top-50 | UNDER_HALF | 8, 85,489 | 1, 16 | 0.258, 0.540, 0.320 | 0.501, 0.501 |
| Qwen3-8B | min-p 0.05 | GATES_PASS | 6, 752 | 1, 20 | 0.258, 0.261, 0.258 | 0.501, 0.501 |
| Sarvam-30B main | greedy | MIXED | 113,748, 262,072 | 36, 6,554 | 0.475, 0.761, 0.465 | 0.501, 0.526 |
| Sarvam-30B main | Gumbel T = 0.6 | MIXED | 111,154, 261,846 | 34, 7,886 | 0.470, 0.765, 0.462 | 0.501, 0.531 |
| Sarvam-30B main | Gumbel T = 1 | UNDER_HALF | 107,080, 259,121 | 36, 7,349 | 0.463, 0.759, 0.457 | 0.501, 0.529 |
| Sarvam-30B main | top-50 | DEAD | 262,024, 262,144 | 14,882, 112,280 | 0.814, 1.187, 0.806 | 0.558, 0.929 |
| Sarvam-30B main | min-p 0.05 | DEAD | 151,729, 262,144 | 90, 30,703 | 0.548, 0.872, 0.515 | 0.501, 0.618 |
| Sarvam-30B run B | greedy | MIXED | 30,348, 262,107 | 10, 6,511 | 0.316, 0.769, 0.406 | 0.501, 0.526 |
| Sarvam-30B run B | Gumbel T = 0.6 | MIXED | 58,796, 262,038 | 17, 8,731 | 0.370, 0.775, 0.429 | 0.501, 0.534 |
| Sarvam-30B run B | Gumbel T = 1 | MIXED | 53,636, 260,495 | 19, 7,681 | 0.361, 0.766, 0.422 | 0.501, 0.530 |
| Sarvam-30B run B | top-50 | DEAD | 261,538, 262,144 | 3,736, 116,923 | 0.768, 1.205, 0.803 | 0.515, 0.947 |
| Sarvam-30B run B | min-p 0.05 | DEAD | 137,832, 262,144 | 61, 43,626 | 0.522, 0.922, 0.507 | 0.501, 0.667 |

For comparison, the deterministic bounds. On single-stage 4-bit heads the best median is 1.226 on Gemma 4 greedy, 0.866 on Qwen3-8B at T = 1 (both NVFP4 with the residual bound), and 1.266 on Sarvam-30B. With an INT8 stage they stay above INT8 alone on every rule and model. `summary.json` gives each value.

On Gemma 4 and Qwen3-8B the dither alone is enough for the medians, but on Gemma 4 its 99th percentile is 23,089 rows on greedy, or 8.8% of the vocabulary. That fails the 1% gate, so the INT8 stage is what passes it.

## The decision rule

The rule was written before any smoke run, dump or measure. Its SHA-256 (2deb9636) was logged at 10:24 IST. One amendment followed the smoke runs and came before any main run, giving 2d32b983 at 10:42 IST (`logs/decision_rule_amendment1.log`). `analyze.py` refuses to run if the file differs from that hash, and it still matches. The amendment is described under Controls.

| Item | Outcome |
|-|-|
| Controls C1 to C6, C8, C9 | Passed on all three models and on run B |
| C7, language | Passed on Gemma 4, Qwen3-8B and run B. Failed on Sarvam-30B's main run, with a 0.0 share of target script in every group |
| Confirm, Gemma 4 E4B | Holds for one certificate, the dithered head with INT8 |
| Confirm, Sarvam-30B | Fails |
| Kill, Gemma 4 E4B | Does not hold |
| Kill, Sarvam-30B | Does not hold. Greedy and T = 1 stay under 0.5 over all states |
| Outcome | **MIDDLE** |
| Same test on worst-case and dithered single-stage certificates only | Not confirmed |

The outcome text fixed in advance reads that the claim narrows to the rules and languages that work. Run B does not change this outcome. Under the addendum written before it ran (`logs/addendum_runB.json`, hash ff93478f), run B also fails confirm and does not meet kill.

The review's wording was "best sound certificate, worst-case or dithered". We widened this in advance to include the residual norm and the cascades. The last row of the table shows that this matters: read literally, confirm fails on Gemma 4 too, because the dithered bound alone misses the 1% gate.

## Sarvam-30B, and why there are two runs

C7 fired for a real reason. In the main run Sarvam-30B wrote an English `<think>` block for all 128 tokens in every group, although the prompt carried `<|nothink|>`. So its Indian-language groups held English reasoning about translation, and its code group ran the full 128 tokens as well. The rule keeps a flagged group in the outcome, and we kept it there.

After reading the main results we wrote an addendum and hashed it, then ran B. It ends the prompt with `<think></think>` and a newline, which is how Sarvam's template renders a turn with no reasoning. Nothing else changed, and C7 now passes: at least 94% of letters are in the target script on the median prompt. Because we had seen the main run first, run B is a follow-up and not a blind test.

By group, for run B at T = 1 on the dithered head with INT8:

| Group | Median S | Bytes median | Bytes mean | R p99 |
|-|-|-|-|-|
| English | 141,842 | 0.529 | 0.507 | 11,307 |
| Code | 95,434.5 | 0.440 | 0.467 | 7,183 |
| Hindi | 44,215 | 0.343 | 0.394 | 2,802 |
| Bengali | 35,089 | 0.325 | 0.397 | 6,886 |
| Tamil | 37,472 | 0.330 | 0.391 | 7,794 |
| Telugu | 18,119 | 0.293 | 0.361 | 5,703 |

So Sarvam-30B's Indian-language output is cheaper than its English and code. Means stay near 0.36 to 0.40 against about 0.50 for INT8. That is a real saving in the average, but with long tails, and it is far from the gates.

We do not know why Sarvam-30B is so much harder. Two readouts look relevant. Its top logit is near 200 on the smoke states against about 25 (capped) on Gemma 4 and about 38 on Qwen3-8B. Its h is more concentrated: l1 over sqrt(d) l2 has a median of 0.36 to 0.38, against 0.62 and 0.70, and 0.80 for a Gaussian. A shared offset in the head rows could produce both, which is what ablation A3 tests, but we did not test it here.

## Top-k (post hoc)

Top-k is DEAD on both 262K models under the pre-registered read plan. That plan reads every row of uncertain membership as soon as any race candidate has uncertain membership. We asked whether this was the plan's cost or the bound's. This pass was fixed in the same addendum, has no rule, and does not change any status above.

The stepwise plan reads INT8 only for rows that can still win the race, tightens them and asks again, and reads the rows of uncertain membership only for states that remain uncertain after that. Tokens matched the reference on every state. For dithered INT4 with INT8:

| States | S median, p99 | Bytes median, p99 | Bytes median by group: English, code, Hindi, Bengali, Tamil, Telugu |
|-|-|-|-|
| Gemma 4 E4B | 79, 196,660 | 0.258, 0.637 | 0.258, 0.258, 0.258, 0.540, 0.519, 0.547 |
| Qwen3-8B | 3, 26,596 | 0.258, 0.346 | 0.258 in all six |
| Sarvam-30B main | 262,023, 262,144 | 0.814, 1.187 | 0.806, 0.778, 0.836, 0.833, 0.822, 0.824 |
| Sarvam-30B run B | 261,537, 262,144 | 0.768, 1.205 | 0.825, 0.827, 0.767, 0.755, 0.767, 0.763 |

So part of the top-k failure was the plan. Qwen3-8B now reads 0.258 at the median in all six groups, and Gemma 4 does so in three. Bengali, Tamil and Telugu on Gemma 4 still read more than the INT8 baseline, and Sarvam-30B does not move. We think this is because a race candidate near rank 50 stays uncertain until the rank-50 value itself is known, and a wide first pass cannot give that. We did not test this explanation.

## Bound widths, and the brief's estimate

The review expected the dithered bound to be about 6 times tighter than the worst case on Gaussian-looking states at d = 4,096, and expected real states to shrink that. They do. The brief's sqrt(d) estimate of 64 is far too high.

| Readout on greedy states, medians | Gemma 4 | Qwen3-8B | Sarvam main | Sarvam run B |
|-|-|-|-|-|
| Mean worst-case bound over mean dithered bound | 3.35 | 4.87 | 2.50 | 2.61 |
| Worst-case bound over true largest error | 17.3 | 26.2 | 5.8 | 7.7 |
| Dithered bound over true largest error | 6.3 | 5.5 | 7.0 | 7.1 |
| True largest error, INT4, in raw logits | 4.94 | 2.19 | 18.14 | 10.35 |
| Largest dithered bound, in raw logits | 34.1 | 12.6 | 43.6 | 36.4 |
| Top-two margin of the capped reference | 5.60 | 8.16 | 5.83 | 4.73 |

The oracle shows what remains: the true error alone leaves a median of 1 row and a 99th percentile of 2 to 8 rows for greedy and T = 1 on the dithered head. Almost all of the set size comes from the gap between a bound and the error it covers.

## Language and temperature

On Gemma 4, the dithered shortlist is far wider for three Indian languages. Median S on greedy is 2 for English, 1 for code, 18 for Hindi, 490 for Bengali, 576 for Tamil and 326 for Telugu. They still pass all gates, since 576 rows is 0.2% of the head and the shortlist is read in INT8. On Qwen3-8B the groups are flat, at 1 to 4. On Sarvam-30B run B the order reverses, as above. Language therefore matters, and the direction depends on the model. Three models cannot say why.

Across T = 0.6 and T = 1, the median of the dithered set moves from 43 to 49 on Gemma 4, stays at 3 on Qwen3-8B, and falls from 58,796 to 53,636 on Sarvam-30B run B. The review expected the set to shrink as T rises, and this does not show that on Gemma 4. The T = 0.6 scores use T = 1 states, so this is weak evidence, and ablation A2 is the proper test.

## Head share of weight bytes read

From the safetensors headers, with one row per token for input tables and six of 128 experts for Sarvam-30B (`logs/byte_share.json`). KV-cache reads are ignored. The review had left Gemma 4 uncomputed.

| Model | Head share with a BF16 body | With a 4-bit body |
|-|-|-|
| Gemma 4 E4B | 14.3% | 40.0% |
| Qwen3-8B | 8.2% | 26.4% |
| Sarvam-30B | 31.2% | 64.4% |

So the certificate works best where the head matters less, and it does poorly on the model where the head matters most. By our arithmetic, with a 4-bit body, reading 0.26 of the head saves about 30% of weight bytes on Gemma 4 and 19% on Qwen3-8B. Bytes are not time.

## What this means for the paper

- The review's claim holds only in part: exact tokens at a quarter to a third of the head's bytes, on Gemma 4 E4B and Qwen3-8B, for greedy, Gumbel-max and min-p. The paper may claim this for this reference, with a dithered head and an INT8 stage.
- It may not claim that a 4-bit certificate works on 262K-vocabulary models in general. Sarvam-30B is the counterexample, and on min-p the INT8 certificate beats 4 bits there.
- The guarantee is high-probability. On these models no deterministic 4-bit certificate beat INT8 on bytes. So the question for the first discussion, "deterministic or 1-delta", comes down to this: 4 bits only pays with a 1-delta guarantee. Whether reviewers and Sarvam AI accept that stays open.
- The dithered head with a shared pattern is what makes the result possible. The review found no public work doing this as of 3 November. We did not search again, so that needs a fresh check before the paper relies on it.
- Top-k under this plan is not claimable. The stepwise plan is worth writing up as an ablation, and it does not rescue Sarvam-30B.
- The next useful test is why Sarvam-30B is wide. The mean-row shift (A3), which the review limits to Qwen3-8B, should also run on Sarvam-30B.
- FP32 and the stock BF16 head disagree more than we expected on Sarvam-30B, which bears on what "identical" means. The FP32 reference argmax equals the emitted greedy token on 99.5% of Gemma 4 states, 99.5% of Qwen3-8B states, and 95.4% and 97.0% of Sarvam-30B states in the two runs.

## Controls

**Alignment (C1).** The BF16 product from the stored h equals the emitted greedy token on 99.7%, 100.0%, 97.9% and 98.6% of states (Gemma 4, Qwen3-8B, Sarvam main, run B), and the same test shifted by one state agrees on under 0.1%.

**The amendment.** C1 first compared the emitted token with the FP32 argmax. On Sarvam-30B's 144 smoke states that was 93.1%, under the 95% gate. `logs/diag_alignment_smoke.log` shows why: its top logit is about 200, where one BF16 step is 1.0, and the BF16 product agrees on 99.3% there. We changed the product under test and kept the threshold. This was a relaxation made after seeing a control fail, on data that held no set size or byte figure. On the main run, the original test would have passed at 95.4%.

**Soundness (C2) and exactness (C3).** Zero violations of the worst-case and residual bounds, on every head and state, over 51.4 billion rows per head variant. Zero token mismatches across all non-dither certificates and rules, and none sat on an exact tie.

**Dither (C5).** Zero rows beyond the dithered bound, and zero token mismatches. The mean squared normalised error is 1.004, 1.001, 0.997 and 0.985 against a prediction of 1, over 12.6, 8.4, 16.1 and 14.3 billion rows. This is one dither draw.

**Planted faults (C6).** On the first chunk of each mode: 2% of the residual bound gave tens of millions of violations, a head with shuffled rows gave violations, a set with the winner removed gave a wrong token on every state, an exact head with a zero bound left one row on at least 99.9% of states, and an empty head left all rows. FP32 against FP64 differed by at most 0.0011, and the slack was at least 200 times the observed FP32 error.

**Sampling (C8).** Mean negative log probability of the sampled token over mean entropy is 1.007, 1.015, 1.008 and 1.006. Gemma 4's smoke gave 1.50 on 144 states, so we checked it before the main run: `logs/diag_gemma4e4b.log` shows 0.994 on 1,438 states against the capped head, z = -0.12, and 3.69 against the raw head, so the soft-cap is where we had it. That diagnostic changed no rule.

**Self tests (C9).** `cert_core.py` compares the vectorised sets with a plain Python definition on every rule, with and without the soft-cap (`logs/selftest.log`).

## Caveats

- This counts bytes. It measures no kernel, no speed-up and no scattered-read cost. The author of certhead reports that saved bytes did not turn into proportional time on an Apple GPU. The optional L4 timing in the review was not run.
- Keeping a BF16 head, a dithered INT4 copy and an INT8 copy adds about three quarters of the head's BF16 size to memory.
- "Exact" means equal to our FP32 reference, not to vLLM or SGLang.
- One dither draw, one seed for sampling, and one machine type per model. The bound fails with probability about 0.3% over 10 million steps by the union bound, as the review computed. We observed none.
- The FP32 slack is conservative. At its largest it is 18% of an INT8 worst-case bound, 8.2% of an INT8 residual bound and 2.6% of a dithered one, which slightly favours the 4-bit side.
- Newspaper OCR replaces Flores-IN. MT-Bench and HumanEval are subsets. The prompts are short, so KV-cache and long-context states are absent.
- Sarvam-30B ran on torch 2.5.1 and transformers 4.57.2, from topic 08's lock, because of its custom code. The others ran on torch 2.13.0 and transformers 5.19.0. All measures ran on torch 2.13.0.
- Run B and the post hoc pass were designed after we read the main results.

## Files

`decision_rule.json`, `prompts.jsonl`, `cert_core.py`, `exp01_modal.py`, `ctl.py`, `drive.py`, `analyze.py`, `analyze_b.py`, `byte_share.py`, `test_local.py`, `make_lock.py`, `requirements.lock.txt`, `sarvam.lock.txt`, `summary.json`, `summary_runB.json`. Raw per-state set sizes and control counts are in `results/` and `results_runB/`, one file per chunk, with the model outputs. Launch logs, hashes and diagnostics are in `logs/`.
