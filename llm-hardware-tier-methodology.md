# Methodology: Mapping LLMs to Hardware Tiers

## Goal
For each LLM in our dataset, assign the **smallest hardware tier** on which the model can realistically run inference, so the visualization reflects *deployability*.

## What We Have
Per-model inputs:
- `size` — total parameters (raw count, e.g. `2.0e10` for a 20B model).
- `size_active` — parameters active per token (MoE only; empty for dense models).
  **Recorded but no longer used for tier assignment** — see "MoE models" below.

## What Drives the Constraint

At inference, two resources matter — RAM and storage:

```
RAM_GB     ≈ params_for_RAM     × bytes_per_param × (1 + overhead)
Storage_GB ≈ params_for_storage × bytes_per_param
```

We standardize on **INT8 (1 byte / param) for all tiers**, including Cloud. Rationale: INT8 is a defensible "middle" assumption — accurate for Workstation / Tablet deployment, conservative for Mobile (real-world is usually INT4), and generous for Cloud (real-world is usually FP16 + batching). A single quantization keeps tier comparisons consistent and reflects what *parameter count alone* tells us. Overhead is set to **25%** for KV cache + activations + framework, assuming short-to-medium context.

For **MoE models**, total parameters drive RAM, exactly as for dense models. We previously used `size_active` for RAM, on the assumption that only the active experts need to be loaded. That assumption is wrong in practice: the router can pick any expert for any token, so the whole model has to be in memory — offloading the inactive experts to disk is possible but collapses throughput, so it is not the deployment pattern to plan around.

The evidence that prompted the change is gpt-oss-120b (117B total, 5.1B active). If active parameters governed RAM it would fit a 12 GB tablet; in reality it needs roughly 64 GB of unified memory to run at usable speed ([r/LocalLLM report](https://www.reddit.com/r/LocalLLM/comments/1mjc494/you_can_now_run_openais_gptoss_model_on_your/)), and OpenAI's own figure is "within 80 GB of memory" for the 120b and 16 GB for the 20b at native MXFP4 ([introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)). Both track total parameters, not active ones.

## Hardware Tiers

RAM caps are *inference-available memory* (after OS and other processes), not total device RAM. Storage ranges are typical on-device free space realistically available for model files.

| Tier | RAM cap | Storage (typical free) | Quantization | Indicative device |
|---|---|---|---|---|
| **Cloud** | > 35 GB | > 1 TB (effectively unlimited) | INT8 | Datacenter GPU / server |
| **Workstation** | 12 – 35 GB | 100 GB – 2 TB | INT8 | Laptop, local workstation |
| **Tablet** | 4 – 12 GB | 32 – 256 GB | INT8 | High-end phone, tablet |
| **Mobile** | < 4 GB | 4 – 32 GB | INT8 | Basic Android, edge device |

The RAM caps map cleanly to hardware: 4 GB ≈ entry-level phone after OS overhead; 12 GB ≈ premium phone / iPad-Pro-class unified memory; 35 GB ≈ ceiling of a single high-end consumer GPU (RTX 5090 has 32 GB VRAM). Above 35 GB requires datacenter cards or multi-GPU.

## Formula

For each model row in the dataset:

```
# 1. Effective parameter counts
params_RAM     = size          # total params, dense and MoE alike
params_storage = size          # full model always on disk

# 2. Memory and disk requirements (INT8, 25% RAM overhead)
RAM_GB     = params_RAM     × 1.25
Storage_GB = params_storage × 1.0

# 3. Assign the smallest tier where BOTH constraints fit
if   RAM_GB ≤ 4   and Storage_GB ≤ 32:    tier = "Mobile"
elif RAM_GB ≤ 12  and Storage_GB ≤ 256:   tier = "Tablet"
elif RAM_GB ≤ 35  and Storage_GB ≤ 2048:  tier = "Workstation"
else:                                      tier = "Cloud"

# 4. Models with empty `size`:
#      - if `open` is False (closed-weights, proprietary): assign tier = "Cloud".
#        Rationale: closed models are served exclusively via vendor APIs running
#        on datacenter hardware, so their *deployable* tier is Cloud regardless
#        of the (unpublished) parameter count.
#      - otherwise (open-weights model with missing size in the dataset):
#        label "size unknown" and exclude.
```

In practice RAM is now the binding constraint for every row: with both requirements driven by total parameters, `RAM_GB ≤ cap` is always the tighter test (Mobile admits ≤ 3.2B params, well inside its 32 GB storage cap). The storage check is retained because it documents a real limit and would bind again under a lower-precision assumption.

## Caveats to Surface in the Visualization

- **INT8 is a single fixed assumption.** Cloud models in production typically run FP16 (so the chart is *generous* to Cloud); Mobile models in production typically run INT4 (so the chart is *conservative* on Mobile — real-world capability there is somewhat better than shown).
- **Overhead is 25%, short context.** Long-context workloads (> 8K tokens) inflate KV cache and may push a model up one tier.
- **MoE gets no discount.** Sparse models are charged for their full parameter count, so a 117B/5.1B model sits in Cloud rather than Tablet. Setups that page experts from disk or CPU can beat this, at a large cost in tokens/sec — the tiers are deliberately conservative for MoE.
- **Natively low-precision releases beat the INT8 assumption.** gpt-oss ships in MXFP4 (~0.5 bytes/param), so our INT8 figure overstates its RAM by roughly 2×. This is a per-model exception, not a reason to change the shared assumption.
- **Size unknown ≠ small.** Models with unreported parameter counts are excluded from the chart, not silently assigned to a default tier.
