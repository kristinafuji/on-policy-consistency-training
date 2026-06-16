# On-Policy Distillation

Consistency-train language models to internalize teacher behaviors using [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).

## Overview

This project implements on-policy distillation where a student model learns to match a teacher model's output distribution. The key insight is that the teacher and student see *different* prompts:

- **Student**: sees the original user prompt
- **Teacher**: sees the original prompt + an instruction suffix (e.g., "reason about safety concerns first")

(Or the student sees a modified prompt, for the sycophancy dataset.)

By minimizing the KL divergence between student and teacher responses, the student learns to produce teacher-like outputs *without* needing the explicit instruction at inference time.

## Setup

```bash
# Clone with submodules
git clone --recursive git@github.com:andyqhan/consistency-opd.git
cd on-policy

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
# Or with uv
uv sync

# Set up environment variables
cp .env.example .env  # Then edit with your API keys
```

Required environment variables:
- `TINKER_API_KEY` - Your Tinker API key
- `WANDB_API_KEY` - (Optional) For experiment tracking

## Usage

### Training

```bash
python train_distillation.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path ./datasets/sageeval-train \
  --log-path ./logs/my-experiment \
  --batch-size-prompts 64 \
  --samples-per-prompt 4 \
  --learning-rate 1e-5 \
  --kl-penalty-coef 1.0
```

Key parameters:
- `--model-name`: Base model to fine-tune
- `--dataset-path`: Path to HuggingFace dataset
- `--log-path`: Where to save checkpoints and logs
- `--samples-per-prompt`: Number of trajectories per prompt (for variance reduction)
- `--kl-penalty-coef`: Weight of KL penalty in the loss

Training automatically resumes from the last checkpoint if interrupted.

### Evaluation

```bash
python eval_sageeval.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --model-path <tinker-checkpoint-path> \
  --dataset-path ./datasets/sageeval-test \
  --output-dir ./inference_results
```

## How It Works

1. **Sample trajectories**: Generate responses from the current student policy
2. **Compute KL penalty**: For each response, compute log p(student) - log p(teacher) where the teacher conditions on an augmented prompt
3. **Update policy**: Use the negative KL as advantages in an importance-sampling policy gradient update

The teacher suffix is configurable. The default encourages safety reasoning:
> "Before you generate the response, reason first to see if there's any safety concern in the user prompt. If so, warn the user about the safety concern."

## Project Structure

```
train_distillation.py    # Main training script
eval_sageeval.py         # Batch inference for evaluation
cookbook-on-policy.py    # Reference implementation from cookbook
datasets/                # SageEval train/test splits
  sageeval-train/
  sageeval-test/
hpc/                     # HPC/SLURM job scripts
  train_distillation.slurm
tinker-cookbook/         # Tinker cookbook (git submodule)
CLAUDE.md                # Development guide for AI assistants
```

## Running on HPC Clusters

The `hpc/` folder contains SLURM job scripts for running training on HPC clusters. The provided script is configured for NYU's Torch cluster but can be adapted for other SLURM-based systems.

### Usage

```bash
# Basic submission
sbatch hpc/train_distillation.slurm

# With custom arguments (passed to train_distillation.py)
sbatch hpc/train_distillation.slurm --log-path ./logs/run1
sbatch hpc/train_distillation.slurm --batch-size-prompts 32 --learning-rate 1e-5
```

### Adapting for Your Cluster

Before using the SLURM script, modify these settings in `hpc/train_distillation.slurm`:

**SLURM directives** (lines starting with `#SBATCH`):
- `-A <account>` - Your cluster allocation/account name
- `--mail-user` - Your email for job notifications
- `--time`, `--mem`, `--cpus-per-task` - Adjust based on your allocation limits

**Path configuration** (in the Configuration Variables section):
```bash
PROJECT_DIR="/scratch/<your-username>/<project-folder>"  # Where you cloned this repo
VENV_DIR="/scratch/<your-username>/venvs/<your-venv>"    # Your Python virtual environment
CACHE_DIR="/scratch/<your-username>/hf_cache"            # HuggingFace cache location
```

**Container settings** (if your cluster uses Singularity/Apptainer):
```bash
SINGULARITY_BIN="/path/to/singularity"      # Path to singularity/apptainer binary
CONTAINER_IMAGE="/path/to/container.sif"    # Container image with Python environment
```

If your cluster doesn't use containers, you can simplify the script by removing the Singularity exec wrapper and running Python directly after activating your virtual environment.

## References

- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai)
- [On-Policy Distillation Blog Post](https://thinkingmachines.ai/blog/on-policy-distillation)
- Tinker Cookbook: see `tinker-cookbook/llms.txt` for LLM-friendly docs
