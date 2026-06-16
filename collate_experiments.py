#!/usr/bin/env python3
"""
Collate experiment results from training logs and evaluation folders.
Combines data from local machine and HPC into a single CSV.
"""

import csv
import glob
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


# Configuration
LOCAL_LOGS_DIR = "./logs"
LOCAL_EVAL_PATTERN = "./eval_results_*"
HPC_HOST = "ah7660@torch"
HPC_BASE_PATH = "/scratch/ah7660/consistency-opd"
SAGEEVAL_JUDGE_SCRIPT = "./sageeval-judge/sageeval-judge.py"
OUTPUT_CSV = "experiment_results.csv"

# Expected total steps for a complete run
COMPLETE_RUN_STEPS = 87

# CSV columns
COLUMNS = [
    "run_folder",
    "source",
    "last_modified",
    "starting_teacher_kl",
    "ending_teacher_kl",
    "teacher_kl_improvement",
    "teacher_mode",
    "training_mode",
    "model_name",
    "thinking_mode",
    "num_epochs",
    "learning_rate",
    "loss_fn",
    "num_substeps",
    "batch_size_prompts",
    "samples_per_prompt",
    "max_ppo_clip_fraction",
    "total_steps",
    "completion_status",
    "eval_folder",
    "sageeval_99",
    "ausc",
    "model_safety_score",
    "claude_notes",
]


def run_ssh_command(cmd: str) -> str:
    """Run a command on HPC via SSH and return output."""
    full_cmd = f'ssh {HPC_HOST} "{cmd}"'
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"SSH command timed out: {cmd}")
        return ""
    except Exception as e:
        print(f"SSH error: {e}")
        return ""


def read_json_file(path: str, source: str = "local") -> dict | None:
    """Read a JSON file, either locally or via SSH."""
    try:
        if source == "local":
            with open(path) as f:
                return json.load(f)
        else:
            content = run_ssh_command(f"cat {path}")
            if content:
                return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {path}: {e}")
    return None


def read_jsonl_file(path: str, source: str = "local") -> list[dict]:
    """Read a JSONL file, either locally or via SSH."""
    lines = []
    try:
        if source == "local":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(json.loads(line))
        else:
            content = run_ssh_command(f"cat {path}")
            if content:
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        lines.append(json.loads(line))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {path}: {e}")
    return lines


def get_file_mtime(path: str, source: str = "local") -> str:
    """Get file modification time as ISO string."""
    try:
        if source == "local":
            mtime = os.path.getmtime(path)
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        else:
            # Use stat on HPC
            output = run_ssh_command(f"stat -c '%Y' {path} 2>/dev/null")
            if output:
                mtime = int(output)
                return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        print(f"Error getting mtime for {path}: {e}")
    return ""


def detect_teacher_mode(config: dict, folder_name: str) -> str:
    """Detect if teacher mode is 'user' or 'system'."""
    # Check folder name first
    folder_lower = folder_name.lower()
    if "system" in folder_lower:
        return "system"
    if "user" in folder_lower:
        return "user"

    # Check config for system_prompt presence
    if config.get("system_prompt"):
        return "system"

    return "user"


def detect_training_mode(folder_name: str) -> str:
    """Detect training mode from folder name."""
    folder_lower = folder_name.lower()
    if "overfit" in folder_lower:
        return "overfit"
    if "offpolicy" in folder_lower or "off-policy" in folder_lower or "off_policy" in folder_lower:
        return "off_policy"
    return "on_policy"


def check_completion_status(
    metrics: list[dict], checkpoints_path: str, source: str
) -> str:
    """Check if run is complete, early stopped, or incomplete."""
    if not metrics:
        return "incomplete"

    total_steps = len(metrics)

    # Check for "final" checkpoint
    checkpoints = read_jsonl_file(checkpoints_path, source)
    has_final = any(cp.get("name") == "final" for cp in checkpoints)

    if has_final or total_steps >= COMPLETE_RUN_STEPS:
        return "complete"
    elif total_steps > 0:
        return "early_stopped"
    return "incomplete"


def extract_training_data(folder_path: str, source: str = "local") -> dict | None:
    """Extract training data from a log folder."""
    folder_name = os.path.basename(folder_path)

    # Required files
    config_path = os.path.join(folder_path, "config.json")
    metrics_path = os.path.join(folder_path, "metrics.jsonl")
    logs_path = os.path.join(folder_path, "logs.log")
    checkpoints_path = os.path.join(folder_path, "checkpoints.jsonl")

    # Check if config exists
    config = read_json_file(config_path, source)
    if not config:
        return None

    # Read metrics
    metrics = read_jsonl_file(metrics_path, source)

    # Check completion status
    completion_status = check_completion_status(metrics, checkpoints_path, source)
    if completion_status == "incomplete":
        return None

    # Extract KL values (find first and last entries that have teacher_kl)
    starting_kl = None
    ending_kl = None
    if metrics:
        # Find first entry with teacher_kl
        for m in metrics:
            if m.get("teacher_kl") is not None:
                starting_kl = m.get("teacher_kl")
                break
        # Find last entry with teacher_kl (iterate backwards)
        for m in reversed(metrics):
            if m.get("teacher_kl") is not None:
                ending_kl = m.get("teacher_kl")
                break

    # Extract max PPO clip fraction (only for PPO runs)
    max_clip = None
    if config.get("loss_fn") == "ppo":
        clip_fractions = [m.get("optim/ppo_clip_fraction") for m in metrics if m.get("optim/ppo_clip_fraction") is not None]
        if clip_fractions:
            max_clip = max(clip_fractions)

    # Get last modified time
    last_modified = get_file_mtime(logs_path, source)
    if not last_modified:
        last_modified = get_file_mtime(metrics_path, source)

    # Calculate KL improvement percentage: (start - end) / start * 100
    # Positive = improvement (KL decreased), Negative = regression (KL increased)
    kl_improvement = None
    if starting_kl is not None and ending_kl is not None and starting_kl != 0:
        kl_improvement = (starting_kl - ending_kl) / starting_kl * 100

    return {
        "run_folder": folder_name,
        "source": source,
        "last_modified": last_modified,
        "starting_teacher_kl": f"{starting_kl:.6f}" if starting_kl is not None else "",
        "ending_teacher_kl": f"{ending_kl:.6f}" if ending_kl is not None else "",
        "teacher_kl_improvement": f"{kl_improvement:+.1f}%" if kl_improvement is not None else "",
        "teacher_mode": detect_teacher_mode(config, folder_name),
        "training_mode": detect_training_mode(folder_name),
        "model_name": config.get("model_name", ""),
        "thinking_mode": config.get("thinking_mode") or "none",
        "num_epochs": config.get("num_epochs", ""),
        "learning_rate": config.get("learning_rate", ""),
        "loss_fn": config.get("loss_fn", ""),
        "num_substeps": config.get("num_substeps", ""),
        "batch_size_prompts": config.get("batch_size_prompts", ""),
        "samples_per_prompt": config.get("samples_per_prompt", ""),
        "max_ppo_clip_fraction": f"{max_clip:.4f}" if max_clip is not None else "",
        "total_steps": len(metrics),
        "completion_status": completion_status,
        "eval_folder": "",
        "sageeval_99": "",
        "ausc": "",
        "model_safety_score": "",
        "claude_notes": "",
        "_checkpoints": read_jsonl_file(checkpoints_path, source),  # For UUID matching
    }


def run_verdict_only(eval_folder: str, source: str = "local") -> dict:
    """Run sageeval-judge.py --verdict-only and parse output."""
    result = {"sageeval_99": "", "ausc": "", "model_safety_score": ""}

    judge_results_path = os.path.join(eval_folder, "sageeval_judge_results.jsonl")

    # Check if judge results exist
    if source == "local":
        if not os.path.exists(judge_results_path):
            return result
    else:
        check = run_ssh_command(f"test -f {judge_results_path} && echo exists")
        if "exists" not in check:
            return result

    try:
        if source == "local":
            cmd = f"python {SAGEEVAL_JUDGE_SCRIPT} --input-dir {eval_folder} --verdict-only"
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            output = proc.stdout
        else:
            cmd = f"cd {HPC_BASE_PATH} && python {SAGEEVAL_JUDGE_SCRIPT} --input-dir {eval_folder} --verdict-only"
            output = run_ssh_command(cmd)

        if not output:
            return result

        # Parse AUSC
        ausc_match = re.search(r"Area Under Safety Curve \(AUSC\):\s*([0-9.]+)", output)
        if ausc_match:
            result["ausc"] = ausc_match.group(1)

        # Parse Model Safety Score (100% threshold)
        safety_match = re.search(r"(\d+)\s*/\s*(\d+)\s*safety facts fully passed\s*=\s*([0-9.]+)%", output)
        if safety_match:
            result["model_safety_score"] = f"{safety_match.group(1)}/{safety_match.group(2)} ({safety_match.group(3)}%)"

        # Parse SAGE-Eval @ 99%
        threshold_match = re.search(r"Threshold\s+99%:\s*(\d+)/(\d+)\s*facts\s*\(([0-9.]+)%\)", output)
        if threshold_match:
            result["sageeval_99"] = f"{threshold_match.group(1)}/{threshold_match.group(2)} ({threshold_match.group(3)}%)"

    except Exception as e:
        print(f"Error running verdict-only for {eval_folder}: {e}")

    return result


def extract_eval_data(folder_path: str, source: str = "local") -> dict | None:
    """Extract evaluation data from an eval folder."""
    folder_name = os.path.basename(folder_path)
    config_path = os.path.join(folder_path, "config.json")

    config = read_json_file(config_path, source)
    if not config:
        return None

    # Get model_path to determine if baseline
    model_path = config.get("model_path")
    is_baseline = model_path is None

    # Run verdict-only to get eval metrics
    eval_metrics = run_verdict_only(folder_path, source)

    return {
        "eval_folder": folder_name,
        "model_path": model_path,
        "is_baseline": is_baseline,
        "base_model": config.get("base_model", ""),
        "supervision_key": config.get("supervision_key"),
        "thinking_mode": config.get("thinking_mode") or "none",
        **eval_metrics,
    }


def extract_tinker_uuid(model_path: str) -> str | None:
    """Extract UUID from tinker:// model path."""
    if not model_path:
        return None
    match = re.search(r"tinker://([a-f0-9-]+):", model_path)
    return match.group(1) if match else None


def find_matching_training_run(
    eval_data: dict, training_runs: list[dict]
) -> dict | None:
    """Find training run that matches eval's model_path."""
    model_path = eval_data.get("model_path")
    if not model_path:
        return None

    eval_uuid = extract_tinker_uuid(model_path)
    if not eval_uuid:
        return None

    for run in training_runs:
        checkpoints = run.get("_checkpoints", [])
        for cp in checkpoints:
            cp_path = cp.get("state_path", "") or cp.get("sampler_path", "")
            cp_uuid = extract_tinker_uuid(cp_path)
            if cp_uuid and cp_uuid == eval_uuid:
                return run

    return None


def scan_local_log_folders() -> list[str]:
    """Scan local logs directory for training folders."""
    folders = []
    if os.path.exists(LOCAL_LOGS_DIR):
        for item in os.listdir(LOCAL_LOGS_DIR):
            full_path = os.path.join(LOCAL_LOGS_DIR, item)
            if os.path.isdir(full_path):
                folders.append(full_path)
    return folders


def scan_local_eval_folders() -> list[str]:
    """Scan for local eval folders."""
    return glob.glob(LOCAL_EVAL_PATTERN)


def scan_hpc_log_folders() -> list[str]:
    """Scan HPC logs directory for training folders."""
    folders = []
    hpc_logs_dir = f"{HPC_BASE_PATH}/logs"
    output = run_ssh_command(f"ls -1 {hpc_logs_dir}")
    if output:
        for item in output.split("\n"):
            item = item.strip()
            if item and not item.startswith("."):
                folders.append(f"{hpc_logs_dir}/{item}")
    return folders


def scan_hpc_eval_folders() -> list[str]:
    """Scan HPC for eval folders."""
    folders = []
    output = run_ssh_command(f"ls -1d {HPC_BASE_PATH}/eval_results_* 2>/dev/null")
    if output:
        for item in output.split("\n"):
            item = item.strip()
            if item:
                folders.append(item)
    return folders


def create_baseline_row(eval_data: dict, source: str) -> dict:
    """Create a row for a baseline evaluation."""
    # Determine type from folder name and config
    folder_name = eval_data.get("eval_folder", "")
    is_teacher = eval_data.get("supervision_key") == "safety_fact" or "teacher" in folder_name.lower()

    baseline_type = "teacher_baseline" if is_teacher else "student_baseline"

    return {
        "run_folder": f"[{baseline_type}]",
        "source": source,
        "last_modified": "",
        "starting_teacher_kl": "",
        "ending_teacher_kl": "",
        "teacher_kl_improvement": "",
        "teacher_mode": "",
        "training_mode": "",
        "model_name": eval_data.get("base_model", ""),
        "thinking_mode": eval_data.get("thinking_mode", ""),
        "num_epochs": "",
        "learning_rate": "",
        "loss_fn": "",
        "num_substeps": "",
        "batch_size_prompts": "",
        "samples_per_prompt": "",
        "max_ppo_clip_fraction": "",
        "total_steps": "",
        "completion_status": "",
        "eval_folder": eval_data.get("eval_folder", ""),
        "sageeval_99": eval_data.get("sageeval_99", ""),
        "ausc": eval_data.get("ausc", ""),
        "model_safety_score": eval_data.get("model_safety_score", ""),
        "claude_notes": "",
    }


def main():
    print("=" * 60)
    print("Collating Experiment Results")
    print("=" * 60)

    # Collect all training runs
    print("\n[1/6] Scanning local log folders...")
    local_log_folders = scan_local_log_folders()
    print(f"  Found {len(local_log_folders)} local log folders")

    print("\n[2/6] Scanning HPC log folders...")
    hpc_log_folders = scan_hpc_log_folders()
    print(f"  Found {len(hpc_log_folders)} HPC log folders")

    # Extract training data
    print("\n[3/6] Extracting training data...")
    training_runs = []

    for folder in local_log_folders:
        data = extract_training_data(folder, "local")
        if data:
            training_runs.append(data)
            print(f"  + {data['run_folder']} (local, {data['completion_status']})")

    for folder in hpc_log_folders:
        # Skip if we already have this folder locally
        folder_name = os.path.basename(folder)
        if any(r["run_folder"] == folder_name and r["source"] == "local" for r in training_runs):
            print(f"  - {folder_name} (skipping, exists locally)")
            continue

        data = extract_training_data(folder, "hpc")
        if data:
            training_runs.append(data)
            print(f"  + {data['run_folder']} (hpc, {data['completion_status']})")

    print(f"\n  Total training runs: {len(training_runs)}")

    # Collect eval folders
    print("\n[4/6] Scanning eval folders...")
    local_eval_folders = scan_local_eval_folders()
    hpc_eval_folders = scan_hpc_eval_folders()
    print(f"  Found {len(local_eval_folders)} local eval folders")
    print(f"  Found {len(hpc_eval_folders)} HPC eval folders")

    # Extract eval data and link to training runs
    print("\n[5/6] Extracting eval data and linking to training runs...")
    training_run_rows = {}  # Dict keyed by run_folder to handle multiple evals per run
    baseline_rows = []

    # Process local evals
    for folder in local_eval_folders:
        eval_data = extract_eval_data(folder, "local")
        if not eval_data:
            continue

        if eval_data["is_baseline"]:
            baseline_rows.append(create_baseline_row(eval_data, "local"))
            print(f"  + {eval_data['eval_folder']} (baseline)")
        else:
            # Try to link to training run
            matching_run = find_matching_training_run(eval_data, training_runs)
            if matching_run:
                run_key = matching_run["run_folder"]
                if run_key not in training_run_rows:
                    # First eval for this training run
                    row = {k: v for k, v in matching_run.items() if not k.startswith("_")}
                    row["eval_folder"] = eval_data.get("eval_folder", "")
                    row["sageeval_99"] = eval_data.get("sageeval_99", "")
                    row["ausc"] = eval_data.get("ausc", "")
                    row["model_safety_score"] = eval_data.get("model_safety_score", "")
                    training_run_rows[run_key] = row
                    print(f"  + {eval_data['eval_folder']} -> {matching_run['run_folder']}")
                else:
                    # Additional eval for same training run - append to eval_folder
                    existing = training_run_rows[run_key]
                    existing["eval_folder"] += f"; {eval_data.get('eval_folder', '')}"
                    # Keep best eval scores (non-empty preferred)
                    for key in ["sageeval_99", "ausc", "model_safety_score"]:
                        if not existing.get(key) and eval_data.get(key):
                            existing[key] = eval_data.get(key)
                    print(f"  + {eval_data['eval_folder']} -> {matching_run['run_folder']} (merged)")
            else:
                # Couldn't link, add as standalone eval
                print(f"  ? {eval_data['eval_folder']} (no matching training run)")

    # Process HPC evals (skip duplicates)
    for folder in hpc_eval_folders:
        folder_name = os.path.basename(folder)
        # Check if this eval already exists locally
        all_eval_folders = [r.get("eval_folder", "") for r in training_run_rows.values()] + \
                          [r.get("eval_folder", "") for r in baseline_rows]
        if any(folder_name in ef for ef in all_eval_folders):
            print(f"  - {folder_name} (skipping, exists locally)")
            continue

        eval_data = extract_eval_data(folder, "hpc")
        if not eval_data:
            continue

        if eval_data["is_baseline"]:
            baseline_rows.append(create_baseline_row(eval_data, "hpc"))
            print(f"  + {eval_data['eval_folder']} (baseline, hpc)")
        else:
            matching_run = find_matching_training_run(eval_data, training_runs)
            if matching_run:
                run_key = matching_run["run_folder"]
                if run_key not in training_run_rows:
                    row = {k: v for k, v in matching_run.items() if not k.startswith("_")}
                    row["eval_folder"] = eval_data.get("eval_folder", "")
                    row["sageeval_99"] = eval_data.get("sageeval_99", "")
                    row["ausc"] = eval_data.get("ausc", "")
                    row["model_safety_score"] = eval_data.get("model_safety_score", "")
                    training_run_rows[run_key] = row
                    print(f"  + {eval_data['eval_folder']} -> {matching_run['run_folder']} (hpc)")
                else:
                    # Merge with existing
                    existing = training_run_rows[run_key]
                    existing["eval_folder"] += f"; {eval_data.get('eval_folder', '')}"
                    for key in ["sageeval_99", "ausc", "model_safety_score"]:
                        if not existing.get(key) and eval_data.get(key):
                            existing[key] = eval_data.get(key)
                    print(f"  + {eval_data['eval_folder']} -> {matching_run['run_folder']} (hpc, merged)")

    # Build final rows list
    rows = list(training_run_rows.values())

    # Add training runs without linked evals
    linked_run_folders = set(training_run_rows.keys())
    for run in training_runs:
        if run["run_folder"] not in linked_run_folders:
            row = {k: v for k, v in run.items() if not k.startswith("_")}
            rows.append(row)

    # Add baseline rows
    rows.extend(baseline_rows)

    # Sort: training runs by last_modified (descending), then baselines at the end
    def sort_key(row):
        is_baseline = "baseline" in row.get("run_folder", "")
        last_mod = row.get("last_modified", "")
        # Baselines sort to end (is_baseline=1), training runs first (is_baseline=0)
        # Within training runs, sort by date descending (invert by using "~" prefix trick won't work, use tuple)
        return (1 if is_baseline else 0, last_mod if not is_baseline else "")

    rows.sort(key=sort_key)
    # Reverse just the training runs portion by date
    training_rows = [r for r in rows if "baseline" not in r.get("run_folder", "")]
    baseline_rows_final = [r for r in rows if "baseline" in r.get("run_folder", "")]
    training_rows.sort(key=lambda r: r.get("last_modified", ""), reverse=True)
    rows = training_rows + baseline_rows_final

    # Write CSV
    print(f"\n[6/6] Writing {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            # Only write columns that exist in COLUMNS
            filtered_row = {k: row.get(k, "") for k in COLUMNS}
            writer.writerow(filtered_row)

    print(f"\n  Written {len(rows)} rows to {OUTPUT_CSV}")
    print("\nDone!")


if __name__ == "__main__":
    main()
