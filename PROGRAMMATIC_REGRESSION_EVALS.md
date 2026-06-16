# Programmatic Regression Evals — Collaborator Guide

This guide walks a collaborator (and their coding assistant) through running the
three-benchmark regression harness on NYU HPC. It was built to be robust to
slightly different setups — it fails fast on bad configuration rather than
silently doing the wrong thing.

> **Scope**: this doc is about `hpc/run_programmatic_regression_evals.sh` and
> its two SLURM backends. It is **not** the broader training pipeline doc
> (see the repo [README.md](README.md) for that).

## What it does

A single shell entrypoint takes one required flag — `--checkpoint` — and runs
three evaluations on it:

| Eval | Dataset | Size | Format | Metric |
|---|---|---|---|---|
| **GPQA-Diamond** | `Idavidrein/gpqa` (gated) | 198 | 4-option MC | accuracy + by_subdomain |
| **IFEval** | `google/IFEval` | 541 | instruction-following | strict/loose prompt + instruction accuracy, by category |
| **TruthfulQA new MC** (Jan 2025) | [`sylinrl/TruthfulQA`](https://github.com/sylinrl/TruthfulQA) raw CSV | ~790 | 2-option MC (Best Answer vs Best Incorrect Answer, randomized) | accuracy + by_category |

The shell script classifies the checkpoint (Tinker URI / local LoRA / local
merged model), auto-detects the base model without loading GPU weights,
validates it against an allowlist, auto-prepares missing datasets, and submits
the right SLURM job:

| Checkpoint type | Base model | Backend | Inference script |
|---|---|---|---|
| `tinker://...` | (must pass `--base-model`) | Tinker API | `eval_sageeval.py` on CPU SLURM |
| local LoRA dir | `meta-llama/Llama-3.1-8B-Instruct` | `local` | `local_inference.py` on H100 (transformers + PEFT) |
| local LoRA dir | `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B` | `local` | `local_inference.py` on H100 |
| local LoRA dir | `openai/gpt-oss-20b` | `vllm` | auto-merge via `merge_tinker_lora.py`, then `vllm_inference.py` in verl-vllm container on H100 |
| local merged dir | (same allowlist) | `local` or `vllm` | skips merge step |

See the [Sanity-check accuracy numbers](#sanity-check-accuracy-numbers-from-initial-validation)
section below for reference values measured on the project owner's sweep1
checkpoints during the pipeline's initial end-to-end validation. The full
per-tier validation narrative lives in
`eval_results/regression-programmatic/TEST_PROGRESS.md`, which is under
`.gitignore` — ask the project owner if you want to read it.

---

## Prerequisites

### 1. NYU HPC access

- Greene access with at least one account that grants H100 GPU allocation
  (`torch_pr_230_tandon_priority`, `torch_pr_882_tandon_advanced`, or similar —
  check `sshare -U` if unsure).
- Ability to submit SLURM jobs (`sbatch` on a login node).
- `uv` installed on the login node (`module load uv` or install via curl — the
  cluster-wide install is in most collaborators' PATH already).
- `jq` on the login node (used by the shell script for `adapter_config.json`
  inspection).

### 2. Git + submodules

```bash
cd /scratch/$USER
git clone git@github.com:andyqhan/consistency-opd.git
cd consistency-opd
git submodule update --init --recursive
```

This pulls in `tinker-cookbook` which `eval_sageeval.py` depends on. The
`third_party/ifeval/` directory is already in-repo (vendored from
`google-research/google-research`, commit `09446d32`) so no extra submodule
init for that.

### 3. Secrets via `.env`

Copy `.env.example` to `.env` (if present) or create a new `.env` in the
project root with the following keys:

```bash
# REQUIRED if you plan to evaluate Tinker checkpoints via the tinker:// mode.
# Can be omitted if you only eval local LoRA / merged models.
TINKER_API_KEY=tml-...

# REQUIRED to download the GPQA-Diamond dataset on first run (gated repo).
# Also required to download base models like meta-llama/Llama-3.1-8B-Instruct
# from HuggingFace on first run (gated repo).
HF_TOKEN=hf_...

# OPTIONAL. Only needed if you plan to log to wandb from other scripts.
WANDB_API_KEY=...
```

**Important**: the shell script and both SLURM jobs automatically
`set -a; source .env; set +a` before invoking the container, so as long as the
file exists in the project root, the variables propagate correctly.

**Never commit `.env`.** The repo's `.gitignore` already excludes it. Your
coding assistant should never read `.env` contents directly — if it needs to
verify a secret is set, have it run something like:

```bash
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN is set" || echo "HF_TOKEN is missing"
```

### 4. HuggingFace gated-repo access

For first-time runs, you need HuggingFace accept-terms access to:

- `Idavidrein/gpqa` (for the GPQA-Diamond dataset)
- `meta-llama/Llama-3.1-8B-Instruct` (if you'll evaluate Llama checkpoints)
- `openai/gpt-oss-20b` (if you'll evaluate gpt-oss checkpoints)
- The Qwen models are not gated but HF_TOKEN still helps with rate limits.

Visit each model/dataset page on huggingface.co while logged in, click
"Request access", wait for approval, then ensure your `HF_TOKEN` has the
"read public gated repos" scope.

Datasets that are not gated (IFEval, TruthfulQA) work without any credentials.

---

## Setup: Python environment

This project has **two** venv locations that get used at different times:

1. **Project-local `.venv`** — created/managed by `uv sync` from `pyproject.toml`.
   Used by `uv run python ...` on the login node for dataset prep and smoke
   tests.
2. **Container venv** — lives on scratch, mounted into the Apptainer container
   during SLURM jobs. The SLURM scripts `source <VENV_DIR>/bin/activate` before
   running inference and scoring.

Both venvs need the same dependencies. Below is the recommended per-collaborator
setup. **Option A (recommended)**: set up your own container venv. **Option B**:
try to reuse the project owner's shared venv — only works if they've chmoded
their scratch dir to allow traversal.

### Option A: set up your own container venv (recommended)

This is the cleanest, most reproducible path. Start on a login node.

```bash
cd /scratch/$USER/consistency-opd

# Sync the project-local .venv (gets transformers, torch, peft, tinker_cookbook,
# nltk, langdetect, absl-py, immutabledict, etc.)
uv sync

# Create a separate container venv on scratch for SLURM jobs. The container
# needs its venv at a path that's bind-mounted inside Apptainer — /scratch/$USER
# is already in APPTAINER_BINDPATH, so this works.
uv venv /scratch/$USER/venvs/consistency-opd-env --python 3.12
uv pip install --python /scratch/$USER/venvs/consistency-opd-env/bin/python \
    -e . \
    flash-attn==2.8.3+cu12torch2.9cxx11abiTRUE \
    --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/

# The flash-attn line may fail because pip can't dereference GitHub release
# URLs as index. If it does, fall back to the wheel-download path below.
```

If the flash-attn line fails (it usually does because pip doesn't treat the
release URL as an index), do it manually:

```bash
# Download the exact wheel matching torch 2.9.1+cu128 + python 3.12 + cxx11abi=True
cd /tmp
curl -LO "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --python /scratch/$USER/venvs/consistency-opd-env/bin/python \
    ./flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
cd /scratch/$USER/consistency-opd
```

Then download the NLTK data IFEval needs:

```bash
/scratch/$USER/venvs/consistency-opd-env/bin/python -c \
    "import nltk; nltk.download('punkt_tab')"
```

This saves to `~/nltk_data` which Apptainer will bind-mount into the container
automatically (`$HOME` is bound by default).

Now verify both venvs are healthy:

```bash
# Project .venv
uv run python -c "import peft, transformers, torch, nltk, langdetect, absl, immutabledict; print('project venv OK')"

# Container venv
/scratch/$USER/venvs/consistency-opd-env/bin/python -c "
import peft, transformers, torch, nltk, langdetect, absl, immutabledict, flash_attn
print(f'container venv OK: peft={peft.__version__}, flash_attn={flash_attn.__version__}')
"
```

### Option B: reuse the project owner's shared venv (only if they've granted access)

The project owner's venv at `/scratch/ah7660/venvs/consistency-opd-env/`
contains the exact deps this pipeline needs, including flash-attn 2.8.3. You
can share it **only if** the owner has done:

```bash
# Owner must run this once to allow traversal and read:
chmod +x /scratch/ah7660
chmod +x /home/ah7660                                    # so /home/ah7660/nltk_data is reachable
chmod -R +rx /scratch/ah7660/venvs/consistency-opd-env   # usually already correct
chmod -R +rx /home/ah7660/nltk_data
```

Verify you can read it:

```bash
ls /scratch/ah7660/venvs/consistency-opd-env/bin/python && \
ls /home/ah7660/nltk_data/tokenizers/punkt_tab >/dev/null && \
echo "shared venv reachable"
```

If that works, you can skip most of Option A but still need to either:

1. Edit the SLURM scripts to point `VENV_DIR` at the shared path (see
   [Customizing for your setup](#customizing-for-your-setup) below), OR
2. Bind-mount your own symlink at `/scratch/$USER/venvs/consistency-opd-env`
   pointing to the owner's path.

**Most collaborators should just use Option A.** Your own venv is self-contained
and doesn't depend on the owner's permissions.

### The vLLM container image (only if evaluating gpt-oss-20b)

The `vllm` backend uses a separate Apptainer image (`verl-vllm012.sif`, ~13GB).
This is only needed if you plan to evaluate `openai/gpt-oss-20b` checkpoints.
For Llama/Qwen checkpoints the standard `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`
image (which is on the cluster for everyone) is sufficient.

If you need the vLLM image:

- **Easiest**: ask the project owner to `chmod +r /scratch/ah7660/singularity_images/verl-vllm012.sif`
  and adjust `VLLM_CONTAINER=...` in your copy of
  `hpc/run_programmatic_regression_evals_local.slurm`.
- **Self-sufficient**: build your own from the verl project's Dockerfile (this
  is a pytorch-2.9 + vllm-0.12 + transformers-4.57 image) and convert to SIF
  with `singularity build`. This takes ~20 min of login-node work and should
  only be done once.
- **Shortcut**: if you already have a vLLM container for another project at
  the right versions (vllm ≥ 0.12, torch ≥ 2.9, transformers ≥ 4.57), just
  point `VLLM_CONTAINER` at it. The inference script
  (`vllm_inference.py`) uses stock vLLM APIs.

---

## Customizing for your setup

The shell script takes all the usual parameters via flags and the SLURM
scripts read most settings from the top of the file. You'll need to change a
few owner-specific hardcodes. Pick one of the two strategies below.

### Strategy 1 (recommended): edit in place, don't commit

Make a local copy of the three pipeline files under your own name or just
edit them in place — whichever is easier for your git hygiene:

```bash
cp hpc/run_programmatic_regression_evals_tinker.slurm{,.mine}
cp hpc/run_programmatic_regression_evals_local.slurm{,.mine}
```

Then open each `.slurm.mine` and change:

| Line / setting | Owner's value | Change to |
|---|---|---|
| `#SBATCH --mail-user=` | `ah7660@nyu.edu` | `<your NYU net-id>@nyu.edu` (or delete the line if you don't want mail) |
| `#SBATCH -A` | `torch_pr_230_tandon_priority` | your default account, **OR** leave it and use `--sbatch-opts "--account <your-acct>"` at submit time (easier, doesn't modify the file) |
| `PROJECT_DIR=` | `/scratch/ah7660/consistency-opd` | `/scratch/<your-username>/consistency-opd` |
| `VENV_DIR=` | `/scratch/ah7660/venvs/consistency-opd-env` | `/scratch/<your-username>/venvs/consistency-opd-env` (Option A) or leave pointing at the owner's (Option B) |
| `CACHE_DIR=` | `/scratch/ah7660/hf_cache` | `/scratch/<your-username>/hf_cache` |
| `VLLM_CONTAINER=` (in `_local.slurm` only) | `/scratch/ah7660/singularity_images/verl-vllm012.sif` | path to your own verl-vllm image, or leave unchanged if you're only running Llama/Qwen |

Run with the `.mine` variants:

```bash
# Modify the script name used by the shell wrapper:
sed -i 's|run_programmatic_regression_evals_local.slurm|run_programmatic_regression_evals_local.slurm.mine|' \
    hpc/run_programmatic_regression_evals.sh
sed -i 's|run_programmatic_regression_evals_tinker.slurm|run_programmatic_regression_evals_tinker.slurm.mine|' \
    hpc/run_programmatic_regression_evals.sh
```

(Or just edit the `SLURM_SCRIPT_TINKER` / `SLURM_SCRIPT_LOCAL` constants at the
top of `hpc/run_programmatic_regression_evals.sh`.)

### Strategy 2: fork the repo

If you want everything under your own git, fork the repo and commit your
adaptations on your fork. The owner files will be different, so make your
changes to the three `run_programmatic_regression_evals*` files and any other
settings you need. A `git rebase upstream/main` will then pull in new
pipeline work without conflicts on your custom settings (which will only
conflict at the top-of-file constants block).

### The `PROJECT_DIR` sanity check

The shell script asserts it's running from `/scratch/ah7660/consistency-opd`
(the value baked in at the top of the script). Change that constant to your
own path **once** — it's used for a few `cd` checks and for `realpath`
traversal stopping conditions when detecting base models:

```bash
# in hpc/run_programmatic_regression_evals.sh, near the top:
PROJECT_DIR="/scratch/<your-username>/consistency-opd"
```

---

## Running the pipeline

### Tinker checkpoint (CPU SLURM, ~5–30 min depending on dataset)

```bash
./hpc/run_programmatic_regression_evals.sh \
    --checkpoint tinker://abcd1234:train:0/sampler_weights/000700 \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --sbatch-opts "--account <your-account>"
```

`--base-model` is **required** for Tinker URIs because the adapter file isn't
on local disk so auto-detection can't walk up to find `config.json`.

### Local Llama / Qwen LoRA (H100, 3–60 min)

Auto-detection walks up from the LoRA directory to find `logs/<run>/config.json`
and reads the `model_name` field. Just point at the adapter folder:

```bash
./hpc/run_programmatic_regression_evals.sh \
    --checkpoint logs/my-llama-run/downloaded_checkpoints/000700/sampler_weights \
    --sbatch-opts "--account <your-account>"
```

`--base-model` is optional — pass it only if detection fails (e.g., if your
checkpoint lives outside the `logs/<run>/downloaded_checkpoints/<step>/sampler_weights`
layout the detection walker expects).

### Local gpt-oss-20b LoRA (H100, ~5 min total: ~3 min CPU merge + ~2 min vLLM)

```bash
./hpc/run_programmatic_regression_evals.sh \
    --checkpoint logs/my-gptoss-run/downloaded_checkpoints/001400/sampler_weights \
    --sbatch-opts "--account <your-account>"
```

**The pipeline automatically applies `--thinking-mode=enable` and
`--reasoning-effort=low` when the detected base model is `openai/gpt-oss-20b`,
unless you pass them explicitly.** All sweep1 gpt-oss-20b training runs in
this repo used `thinking_mode=enable, reasoning_effort=low`, so this matches
the training-time distribution. Evaluating gpt-oss-20b with reasoning off
would be out-of-distribution and misrepresents the checkpoint's actual
capability. You'll see two `[run_programmatic_regression_evals] Defaulting ...`
log lines in the shell output when this fires.

**If your gpt-oss-20b training used different reasoning settings**, pass
`--thinking-mode` and `--reasoning-effort` explicitly to override the default:

```bash
# Override: evaluate with reasoning disabled (probe non-reasoning mode)
./hpc/run_programmatic_regression_evals.sh \
    --checkpoint logs/my-gptoss-run/downloaded_checkpoints/001400/sampler_weights \
    --thinking-mode disable \
    --sbatch-opts "--account <your-account>"

# Override: evaluate at higher reasoning effort
./hpc/run_programmatic_regression_evals.sh \
    --checkpoint logs/my-gptoss-run/downloaded_checkpoints/001400/sampler_weights \
    --thinking-mode enable \
    --reasoning-effort high \
    --sbatch-opts "--account <your-account>"
```

The first run for a given step will merge inline and create
`logs/my-gptoss-run/merged_models/001400/` (~38GB). The second run for the
same step will detect the existing merge and skip the merge step (~2 min total
run time).

### Already-merged gpt-oss-20b (skip merge entirely, ~2 min)

```bash
# Auto-defaults thinking_mode=enable, reasoning_effort=low for gpt-oss-20b.
./hpc/run_programmatic_regression_evals.sh \
    --checkpoint logs/my-gptoss-run/merged_models/001400 \
    --sbatch-opts "--account <your-account>"
```

### Running only specific evals

```bash
# GPQA only
./hpc/run_programmatic_regression_evals.sh --checkpoint ... --only gpqa

# IFEval + TruthfulQA, skip GPQA
./hpc/run_programmatic_regression_evals.sh --checkpoint ... --only ifeval,truthfulqa
```

### Everything overridable

```bash
./hpc/run_programmatic_regression_evals.sh --help
```

Full flag list: `--checkpoint`, `--eval-name`, `--base-model`, `--thinking-mode`,
`--reasoning-effort`, `--only`, `--output-dir`, `--max-tokens`, `--temperature`,
`--seed`, `--sbatch-opts`.

---

## Default hyperparameters

All defaults below apply to every backend (Tinker CPU, local H100, vllm H100)
unless you override via the corresponding flag on the shell entrypoint.

### Global defaults (shared across all three evals)

| Parameter | Default | Overridable via |
|---|---|---|
| `temperature` | **`0.0`** (greedy decoding) | `--temperature <float>` |
| `seed` | **`47`** | `--seed <int>` |

### Per-eval `max_tokens` defaults

Different evals have different output-length expectations, so the SLURM
scripts hardcode different `max_tokens` per eval:

| Eval | Default `max_tokens` | Rationale |
|---|---|---|
| **GPQA-Diamond** | **`512`** | Enough for both direct-answer models and a short CoT chain before the final letter. Was 256 originally, bumped to 512 after observing that Llama-8B OPD checkpoints (step-by-step reasoners) ran out of budget mid-reasoning. |
| **IFEval** | **`2048`** | Essays, long responses, multi-paragraph answers — needs the headroom. |
| **TruthfulQA (new MC)** | **`256`** | 2-option A/B MC — 256 is generous. |

The `--max-tokens <int>` shell flag overrides **all three** evals at once. If
you need per-eval overrides, edit the hardcoded `256`/`512`/`2048` values in
`hpc/run_programmatic_regression_evals_local.slurm` and
`hpc/run_programmatic_regression_evals_tinker.slurm` directly.

### Reasoning defaults — gpt-oss-20b auto-defaults, Llama/Qwen unset

| Base model family | `--thinking-mode` default | `--reasoning-effort` default |
|---|---|---|
| `openai/gpt-oss-20b` | **`enable`** (auto-applied) | **`low`** (auto-applied) |
| Llama-3.1-8B-Instruct, Qwen3-4B*, Qwen3-8B | *unset* (inference script's own default) | *unset* |

**Why gpt-oss-20b gets auto-defaulted**: all sweep1 gpt-oss-20b training runs
in this repo used `thinking_mode=enable, reasoning_effort=low`. If you
evaluated gpt-oss-20b with reasoning off, the model would be used out-of-distribution
relative to its training and the numbers would misrepresent its actual
capability. The shell script detects `BASE_MODEL == "openai/gpt-oss-20b"`
after model detection and injects both flags **only if** you didn't pass them
explicitly.

You'll see these log lines when the auto-default fires:

```
[run_programmatic_regression_evals] Defaulting --thinking-mode=enable for openai/gpt-oss-20b (pass --thinking-mode explicitly to override)
[run_programmatic_regression_evals] Defaulting --reasoning-effort=low for openai/gpt-oss-20b (pass --reasoning-effort explicitly to override)
```

**To override** (e.g. to probe non-reasoning behavior or a different effort
level), pass the flags explicitly on the command line — your values take
precedence over the auto-defaults. See the
[Local gpt-oss-20b LoRA](#local-gpt-oss-20b-lora-h100-5-min-total-3-min-cpu-merge--2-min-vllm)
section above for override examples.

**Llama and Qwen don't get auto-defaults** because the Qwen3 Instruct-2507
models don't have thinking mode at all (`renderer_utils.py` errors out if
you pass `--thinking-mode enable` for them), and Llama models don't support
thinking-mode control either. For Qwen3-8B (the hybrid-thinking model)
you'd need to pass the flag explicitly if you want thinking on.

---

## Monitoring a running job

The shell script **submits and exits** — it prints the job ID and exits
immediately. To watch progress:

```bash
# Watch queue status
squeue -u $USER

# Tail the SLURM log live
tail -F logs/prog_regr_local_<jobid>.log   # or prog_regr_tinker for Tinker backend

# Peek at partial results during inference
wc -l eval_results/regression-programmatic/<eval-name>/gpqa-diamond/results.jsonl
wc -l eval_results/regression-programmatic/<eval-name>/ifeval/results.jsonl
wc -l eval_results/regression-programmatic/<eval-name>/truthfulqa-new-mc/results.jsonl

# Final summary once the job completes:
cat eval_results/regression-programmatic/<eval-name>/regression_summary.json | jq .
```

If the job runs into `QOSGrpGRES` or `QOSMaxGRESPerUser` in the queue-reason
column, that's the H100 slot limit — **be patient**. Per the cluster CLAUDE.md
rule, do not cancel jobs for this reason.

---

## Output format

Each eval run creates a self-contained directory under
`./eval_results/regression-programmatic/<eval-name>/`:

```
<eval-name>/
├── regression_summary.json          # Top-line metrics across all three evals
├── slurm_<jobid>.log                # Copy of the SLURM stdout
├── gpqa-diamond/
│   ├── config.json                  # Full chz config dumped by the inference script
│   ├── results.jsonl                # 198 lines, one per example
│   └── gpqa_scores.json             # Letter extraction + by_subdomain accuracy
├── ifeval/
│   ├── config.json
│   ├── results.jsonl                # 541 lines
│   └── ifeval_scores.json           # Strict + loose prompt/instruction accuracy, by_category
└── truthfulqa-new-mc/
    ├── config.json
    ├── results.jsonl                # ~790 lines
    └── truthfulqa_scores.json       # A/B extraction + by_category accuracy
```

`regression_summary.json` is the quick-look file for comparing across
checkpoints. Example structure:

```json
{
  "eval_name": "my-run-000700",
  "base_model": "Qwen/Qwen3-4B-Instruct-2507",
  "checkpoint": "/scratch/alice/consistency-opd/logs/my-run/downloaded_checkpoints/000700/sampler_weights",
  "timestamp": "2026-04-10T22:51:26.766109+00:00",
  "gpqa": {
    "accuracy": 0.288,
    "correct": 57,
    "total": 198,
    "extraction_rate": 0.91,
    "by_subdomain": { ... }
  },
  "ifeval": {
    "prompt_level_strict_acc": 0.824,
    "prompt_level_loose_acc": 0.865,
    "inst_level_strict_acc": 0.881,
    "inst_level_loose_acc": 0.908,
    "n": 541,
    "by_instruction_category_strict": { ... }
  },
  "truthfulqa": {
    "accuracy": 0.740,
    "correct": 585,
    "total": 790,
    "extraction_rate": 1.0
  },
  "missing_evals": []
}
```

If an eval crashes or was filtered via `--only`, its section is `null` and the
name appears in `missing_evals`.

---

## Resume behavior

All three inference scripts (`eval_sageeval.py`, `local_inference.py`,
`vllm_inference.py`) support resume-from-partial-results — they count the
existing lines in `results.jsonl` and continue from there. This means you can:

- Re-run a failed job and it'll pick up where it crashed.
- Re-run with `--only ifeval` after an earlier `--only gpqa` run on the same
  `eval_name`, and GPQA results will be kept while IFEval is newly computed.
- Re-run after a partial OOM or timeout.

**Be careful not to change hyperparameters between runs with the same output
dir.** The inference config is saved to `<eval-dir>/<eval>/config.json` but
`results.jsonl` is append-only on resume — if you change `max_tokens`, `seed`,
`temperature`, or the model, existing results will be mixed with new ones.
When in doubt, delete the output dir and start fresh.

---

## Troubleshooting

### `flash_attn seems to be not installed` during Step 1 of `local` backend

Your container venv is missing flash-attn. See [Option A setup](#option-a-set-up-your-own-container-venv-recommended)
for the wheel install. Verify with:

```bash
/scratch/$USER/venvs/consistency-opd-env/bin/python -c "import flash_attn; print(flash_attn.__version__)"
```

### `CUDA out of memory` during GPQA/IFEval prefill

The per-eval batch sizes in `hpc/run_programmatic_regression_evals_local.slurm::pick_local_batch_size`
are conservative defaults tuned on H100 80GB. If you're on a smaller GPU, edit
that function to use smaller numbers for your base model. Current values:

| Base model | GPQA | IFEval | TruthfulQA |
|---|---|---|---|
| Llama-3.1-8B-Instruct, Qwen3-8B | 16 | 8 | 32 |
| Qwen3-4B, Qwen3-4B-Instruct-2507 | 48 | 16 | 64 |

If you add a new base model to the allowlist, also add a case to
`pick_local_batch_size` or the script will fall back to the ultra-safe
`batch_size=8`.

### `Could not detect base model` error

The shell script tries three paths for detection:

1. `adapter_config.json::base_model_name_or_path` (works for Unsloth-style adapters)
2. Walk up from the LoRA dir to the first ancestor containing a
   `config.json` with a `.model_name` field (works for `logs/<run>/downloaded_checkpoints/<step>/sampler_weights`
   layouts where `<run>/config.json` was saved by the training script)
3. For merged models, `config.json::_name_or_path`

If all three fail, pass `--base-model <canonical-name>` explicitly.
Canonical allowed names are:

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-4B-Instruct-2507`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `openai/gpt-oss-20b`

### `Not in allowlist`

The base model doesn't match any supported case. If you want to add a new one,
edit `ALLOWED_MODELS` and `VLLM_MODELS` arrays at the top of
`hpc/run_programmatic_regression_evals.sh`, plus
`pick_local_batch_size` in `_local.slurm` if it's a new size class.

### `GatedRepoError` from HuggingFace

You need to accept the terms for the dataset/model on the HuggingFace website,
and your `HF_TOKEN` needs `read public gated repos` scope. See the
[Prerequisites](#4-huggingface-gated-repo-access) section.

### `QOSGrpGRES` or `QOSMaxGRESPerUser` in `squeue -u $USER`

This is the GPU queue limit. **Wait.** The cluster's `ah7660/.claude/CLAUDE.md`
rule is "be patient, do not cancel". Jobs will start when a slot frees up.

### `Dataset not found` from the SLURM job

The shell script is supposed to auto-prep datasets on the login node before
submitting. If you're seeing this inside the SLURM job, either:

- The login-node prep silently failed (check your `uv sync` and `HF_TOKEN`).
- You're pointing at a non-standard dataset path. Verify
  `./datasets/{gpqa-diamond,ifeval,truthfulqa-new-mc}/` exist by running the
  shell script with `--only gpqa` once (auto-preps GPQA), then
  `--only ifeval` (auto-preps IFEval), etc.

Or manually:

```bash
uv run python prepare_gpqa_dataset.py          # requires HF_TOKEN
uv run python prepare_ifeval_dataset.py
uv run python prepare_truthfulqa_dataset.py
```

### LoRA vs merged accuracy drift (e.g., +4pp on GPQA)

This is **expected bf16 numerical noise**, not a bug. At inference time:

- LoRA computes `base_bf16(x) + lora_B_bf16(lora_A_bf16(x)) * scaling` — two
  bf16 matmuls plus a bf16 addition, each with rounding.
- Merged computes `merged_bf16(x)` as a single bf16 matmul, where the weight
  addition happened once in fp32 during merge.

These are NOT bit-equal at bf16, and at `temperature=0` greedy decoding tiny
logit differences cascade into different token choices. For quantitative
regression tracking, **pick one path and stick with it** (either always LoRA
or always merged).

### GPQA accuracy below 25% (random baseline)

Almost certainly a max-tokens truncation issue. CoT-style models like
OPD-trained Llama-8B emit `## Step 1: ... ## Step 2: ...` reasoning and can
run out of budget before producing a final letter. The pipeline default is
`max_tokens=512` for GPQA (bumped from 256 after initial validation), which
is enough for most models but still tight for verbose reasoners. Bump it
higher:

```bash
./hpc/run_programmatic_regression_evals.sh --checkpoint ... --max-tokens 2048 ...
```

(`--max-tokens` is a global override — it applies to all three evals at once.
If you want to bump only GPQA, edit the `512` hardcoded in
`hpc/run_programmatic_regression_evals_local.slurm` and
`hpc/run_programmatic_regression_evals_tinker.slurm` directly.)

A low extraction rate (shown as `extraction_rate` in `gpqa_scores.json` and
in the "by_subdomain" summary table) is the diagnostic signal for this
problem. If extraction < 80%, your responses aren't reaching the letter and
you need a bigger budget.

### `IFEval` scoring crashes with `TypeError: build_description() got an unexpected keyword argument`

The HuggingFace `datasets` library normalizes the per-row `kwargs` schema
across rows and fills missing fields with `None`. The vendored IFEval scorer
rejects unknown kwargs. The fix is already applied in `score_ifeval.py::build_inputs`
— if you see this, someone reverted the `None`-stripping dict-comprehension.
Find the line:

```python
cleaned_kwargs_list = [
    {k: v for k, v in (d or {}).items() if v is not None}
    for d in raw_kwargs_list
]
```

and make sure it's there.

### Responses full of `Loading weights:` / `Generating:` lines in the log

Those are tqdm progress bars from transformers and vLLM. They're noisy but
harmless. Use `grep -vE "Loading weights:|Generating:" logs/prog_regr_*.log | tail -30`
to see actual progress.

---

## Sanity-check accuracy numbers (from initial validation)

Measured during the pipeline's tier-by-tier end-to-end validation against real
sweep1 training runs. **These numbers were measured with the original
`max_tokens=256` GPQA default**, which has since been bumped to 512 in the
committed pipeline — so if you re-run the same checkpoints today, expect
modestly higher extraction rates and somewhat different accuracies,
especially for CoT-style models that were being truncated at 256. The
magnitudes (gpt-oss >> Qwen > Llama) should still hold.

| Tier | Checkpoint | Backend | Eval | Walltime | Result (with historical max_tokens=256 for GPQA) |
|---|---|---|---|---|---|
| 1a | `sweep1-opd-llama8b/.../000700/sampler_weights` (Llama-3.1-8B LoRA) | `local` | GPQA | 3.7 min | 8.6% (see caveat below) |
| 1b | `sweep1-opd-qwen4b/.../000700/sampler_weights` (Qwen3-4B-Instruct LoRA) | `local` | GPQA | 3 min | 28.8% (extraction 91%) |
| 2 | `sweep1-opd-llama8b/merged_models/000700` (Llama merged, same training step as 1a) | `local` | GPQA | 3.2 min | 12.6% (extraction 54%) |
| 3 | `sweep1-evc-gptoss20b-v2/.../001400/sampler_weights` + pre-merged | `vllm` | GPQA | 2.2 min | 43.9% (extraction 73%) |
| 4 | `sweep1-evc-gptoss20b-v2/.../000720/sampler_weights` (no pre-merge, triggered inline merge) | `vllm` | GPQA | 5.6 min (3 min merge + 2 min vLLM) | 42.9% (extraction 74%) |
| 5 | `sweep1-opd-qwen4b/.../000700/sampler_weights` | `local` | **all three** | 51 min | GPQA 28.8%, IFEval strict-prompt **82.4%** / loose-prompt **86.5%** / strict-inst **88.1%** / loose-inst **90.8%**, TruthfulQA **74.0%** (extraction 100%) |

**Caveats and observations from the validation:**

- **Llama-8B's 8.6% GPQA is below the 25% random baseline.** This is because
  the OPD-trained Llama-8B uses step-by-step reasoning (`## Step 1: ...`) that
  ran out of the historical `max_tokens=256` budget before producing a final
  letter. Only ~48% of responses contained an extractable A/B/C/D letter.
  The GPQA default is now `max_tokens=512` which should reach the letter for
  most CoT chains; for very verbose reasoners you may still need
  `--max-tokens 1024` or higher. Qwen3-4B-Instruct in the same situation
  answered directly and hit 91% extraction even at the old 256 budget.

- **Tier 1a vs Tier 2 shows a 4pp accuracy delta (8.6% vs 12.6%) despite
  running the same training step.** This is NOT a pipeline bug — it's
  expected bf16 numerical drift between the LoRA path
  (`base_bf16(x) + lora_B_bf16(lora_A_bf16(x))*scaling`) and the merged path
  (`merged_bf16(x)` where the weight addition happened once in fp32 before
  casting). At `temperature=0` greedy decoding, tiny logit differences flip
  top-1 token choices and cascade. Normally this is invisible end-to-end,
  but when the max_tokens budget is truncating most responses mid-reasoning,
  the letter extraction is sitting on noise-sensitive edge cases. **For
  quantitative regression tracking, pick one path and stick with it.**

- **gpt-oss-20b is well above random on GPQA (~43%)** because it was trained
  with thinking enabled and produces direct letter answers after its
  reasoning chain.

- **Qwen3-4B-Instruct's IFEval numbers (82.4% strict prompt, 86.5% loose
  prompt)** are consistent with published Qwen3-4B-Instruct results. If
  your own IFEval numbers on a Qwen3-4B-sized model are dramatically lower,
  check whether the model's chat template is being applied correctly.

- **Resume logic was verified**: Tier 5 reused Tier 1b's existing 198 GPQA
  `results.jsonl` lines instead of re-running GPQA inference.

## What the pipeline assumes about your training runs

The walk-up base-model detector expects one of these layouts:

```
logs/<run-name>/
├── config.json                            # written by train_distillation.py etc — has .model_name
├── downloaded_checkpoints/
│   └── <step>/
│       └── sampler_weights/               # <-- the adapter_config.json + adapter_model.safetensors live here
│           ├── adapter_config.json        # may have null base_model_name_or_path (common for Tinker downloads)
│           └── adapter_model.safetensors
└── merged_models/
    └── <step>/                            # produced by merge_tinker_lora.py
        ├── config.json                    # has ._name_or_path = "<base model name>"
        ├── model-00001-of-00N.safetensors
        └── ...
```

If your training run uses a different layout, pass `--base-model` explicitly.
If you want the detector to handle your layout natively, edit
`walk_up_for_model_name` in `hpc/run_programmatic_regression_evals.sh`.

---

## File inventory

Scripts added by this pipeline (everything below is safe to read/modify):

### Top-level Python
- `prepare_ifeval_dataset.py` — downloads `google/IFEval`, saves to `./datasets/ifeval/`
- `prepare_truthfulqa_dataset.py` — downloads `TruthfulQA.csv` from GitHub raw, builds new-MC (A/B) prompts
- `score_ifeval.py` — wraps `third_party.ifeval` evaluation_lib; writes `ifeval_scores.json`
- `score_truthfulqa.py` — A/B letter extraction mirroring `score_gpqa.py`
- `write_regression_summary.py` — aggregates per-eval JSONs into `regression_summary.json`

### Pre-existing (used by this pipeline)
- `prepare_gpqa_dataset.py` — downloads Idavidrein/gpqa (gated), saves to `./datasets/gpqa-diamond/`
- `score_gpqa.py` — 4-pattern letter extraction for GPQA
- `eval_sageeval.py`, `local_inference.py`, `vllm_inference.py` — inference backends
- `merge_tinker_lora.py` — converts LoRA → merged HF model (used by gpt-oss vllm path)

### HPC scripts
- `hpc/run_programmatic_regression_evals.sh` — the shell dispatcher (start here)
- `hpc/run_programmatic_regression_evals_tinker.slurm` — CPU SLURM for Tinker backend
- `hpc/run_programmatic_regression_evals_local.slurm` — H100 SLURM for local/vllm backends

### Vendored scorer
- `third_party/ifeval/evaluation_lib.py` — Google IFEval scoring entry points
- `third_party/ifeval/evaluation_main.py` — upstream CLI (unused; kept for reference)
- `third_party/ifeval/instructions.py` — ~25 instruction verifiers
- `third_party/ifeval/instructions_registry.py` — maps instruction IDs to verifier classes
- `third_party/ifeval/instructions_util.py` — NLTK-based text utilities
- `third_party/ifeval/LICENSE` — Apache-2.0 from google-research
- `third_party/ifeval/README.md` — provenance (upstream commit hash + the 4 relative-import fixes I applied)

---

## For coding assistants

If you're a coding assistant helping a collaborator adapt this for their
checkpoints, here's the checklist:

1. **Don't read `.env`.** Instead, run a tiny script like
   `[ -n "${HF_TOKEN:-}" ] && echo set` to verify secrets exist without
   exposing values.

2. **Always source `.env` explicitly** when you need HF_TOKEN or TINKER_API_KEY
   in an interactive shell:
   ```bash
   set -a; source .env; set +a
   ```

3. **Use `--sbatch-opts "--account <acct>"`** to override the default SLURM
   account rather than editing the `#SBATCH -A` line — it's easier to undo
   and doesn't create merge conflicts against upstream.

4. **Never submit SLURM jobs without the collaborator's explicit authorization**.
   Submitting a job spends their compute quota. Instead, show the exact
   `sbatch` command you're about to run and wait for confirmation.

5. **Do not cancel jobs reason `QOSGrpGRES`** — that's the queue limit, not an
   error. The cluster CLAUDE.md explicitly says to be patient.

6. **If the pipeline crashes with a `local_inference.py` or `merge_tinker_lora.py`
   error, remember those files may have uncommitted modifications in the
   collaborator's working tree.** Check `git status` before blaming the
   pipeline.

7. **Pipeline bugs should be reproducible.** If you see inconsistent results
   across runs on the same checkpoint, check for:
   - Partial `results.jsonl` from a previous failed run (resume artifact)
   - Different `batch_size` values between runs (`pick_local_batch_size`
     changes)
   - LoRA vs merged numerical drift (not a bug — see Troubleshooting)

8. **The [Sanity-check accuracy numbers](#sanity-check-accuracy-numbers-from-initial-validation)
   section above** has reference values for the project owner's sweep1
   checkpoints across all 5 validation tiers. Use them as a sanity check on
   your own runs. The full narrative is in
   `eval_results/regression-programmatic/TEST_PROGRESS.md` on the owner's
   machine (gitignored, so not in the repo).

---

## Questions not covered here

Ask the project owner directly. The pipeline was built in one session with
heavy end-to-end testing; the full validation narrative (with per-tier job IDs,
failure modes, and fixes) is in
`eval_results/regression-programmatic/TEST_PROGRESS.md` on the owner's working
tree (gitignored, so not in the repo — ask if you want to read it).
