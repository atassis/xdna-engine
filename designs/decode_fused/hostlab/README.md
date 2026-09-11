# hostlab -- measuring what a weight FORMAT costs the model, without the device

`eval_llm_perplexity.py` next door is the device gate: it drives the real graph, one NPU dispatch
per token, on a single-tenant box. That makes it the right instrument for confirming ONE decision
and the wrong one for choosing between formats, because a format sweep is thirty arms and each
needs its own xclbin.

A weight format only changes which numbers the weights hold. The information it destroys is a
property of the weights, not of the NPU, so a CPU forward pass with the format applied measures it
in about twenty seconds an arm. The device then confirms the point you picked.

## The instrument is anchored, not assumed

Two gates, both cheap, both in this directory. Run them before trusting a number out of here.

    validate_formats.py   the symmetric path must be BIT-IDENTICAL to the shipped packer
                          (iron/common/quant.py) and must reproduce the recorded rel-L2
                          sweep at every group size.
    fold_check.py         the AWQ fold must be neutral on a NON-TRIVIAL scale. awq_eval's own
                          gate reported 0.0 and proved nothing: with no quantization the alpha
                          search picks 0, the scale is all-ones, and the fold is the identity.

Beyond those, as of 2026-09-10 the lab reproduces both recorded device results inside their CIs
(MLP int4/g128 +11.24% t=5.67 against the device's +11.51%; lm-head +2.50% against +2.63%) and its
bf16 control lands within 0.71% of the device harness's absolute.

## What each metric can and cannot see

No single number decides a format. These are ordered by how much statistical power they have per
position, which is the opposite of the order they are usually quoted in.

| metric | sees | is blind to |
|---|---|---|
| `kl_mean` KL(bf16 ‖ arm) | the whole distributional distortion the format caused, with no ground truth involved | whether that distortion matters to anything downstream |
| `top1_agree` | whether greedy decoding would pick the same token -- the metric that predicts generation divergence | how wrong the runner-up ordering got |
| perplexity, paired | distributional shift against the TEXT | anything that compounds; every position is scored on the TRUE prefix, so trajectory drift is invisible by construction |
| `divergence.py` | how far a greedy generation runs before it forks from the bf16 one | it is chaotic at ties -- read the DISTRIBUTION of the fork position, never one number, and never read "the outputs differ" as failure |
| `capability.py` ARC | whether the model still ranks a correct ANSWER above distractors | generation quality, format compliance |
| `capability.py` compliance | instruction-following floor, mechanically checkable | anything about open-ended quality. Small N: a tripwire, not a percentage |
| `ref_rank_mean` | -- | **do not use.** Heavy-tailed; its mean is dominated by a handful of positions and is non-monotonic in group size, which is a property of the statistic and not of the format |

Two things the perplexity number specifically deserves:

* The RATIO is `exp(delta mean NLL)`, so it does not depend on the corpus's absolute perplexity.
  Base perplexity spans 7x across the four corpora here; the int4 delta spans 1.45x.
* Compare two ARMS with `pairwise.py`, never by differencing their two control-relative
  percentages. The arms share a corpus and share positions; throwing that pairing away turns a
  t=3.6 result into two overlapping confidence intervals, which is exactly how an earlier read of
  affine-vs-symmetric came out with the wrong sign.

## Layout

    wq_formats.py     the formats: sym / affine, group size, scale and min width, and whether the
                      affine grid is constrained to contain exact zero. Kernel-faithful rounding.
    wq_eval.py        one arm: teacher-forced NLL, top-1, and the reference comparison
    sweep.py          many arms against one control, one model load; per-class specs for mixed
                      precision
    awq.py            activation-aware channel scaling and the fold that makes it free
    pairwise.py       paired arm-vs-arm t on saved per-position NLL
    divergence.py     greedy generation fork position
    capability.py     ARC-Easy multiple choice + a format-compliance tripwire
    bias_probe.py     splits a format's matvec error into its coherent and incoherent parts
    make_corpora.py   builds the four-genre corpus set

Corpora and run outputs live under `$QLAB_WORK` (default `/mnt/data/qlab`), not in the repo: a
6000-position full-vocab logprob memmap is 3.6 GB.

## Environment

Needs torch + transformers + safetensors, which `.venv-iron` deliberately does not carry (the
device harness has its own tokenizer for exactly that reason). Use `.venv-export`. `capability.py`
additionally needs pyarrow to read the ARC parquet.
