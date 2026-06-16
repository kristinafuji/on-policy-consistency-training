#!/usr/bin/env python3
"""Tinker checkpoint manager: inventory, select, download, and delete checkpoints."""

import argparse
import io
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import requests


LOGS_DIR = "./logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_uuid(tinker_path: str) -> str | None:
    """Extract UUID from tinker://UUID:train:N/..."""
    m = re.match(r"tinker://([^:]+):", tinker_path)
    return m.group(1) if m else None


def find_runs(run_filter: str | None = None) -> list[dict]:
    """Find all log runs that have tinker:// checkpoints.

    Returns list of dicts with keys: name, log_dir, checkpoints, uuid.
    """
    runs = []
    logs_path = Path(LOGS_DIR)
    if not logs_path.exists():
        return runs

    for ckpt_file in sorted(logs_path.glob("**/checkpoints.jsonl")):
        log_dir = str(ckpt_file.parent)
        run_name = str(ckpt_file.parent.relative_to(logs_path))

        if run_filter and run_filter != run_name:
            continue

        checkpoints = read_jsonl(str(ckpt_file))
        # Filter to entries with tinker:// sampler_path
        tinker_ckpts = [
            c for c in checkpoints
            if c.get("sampler_path", "").startswith("tinker://")
        ]
        if not tinker_ckpts:
            continue

        uuid = extract_uuid(tinker_ckpts[0]["sampler_path"])
        runs.append({
            "name": run_name,
            "log_dir": log_dir,
            "checkpoints": tinker_ckpts,
            "uuid": uuid,
        })

    return runs


def create_rest_clients() -> list:
    """Create REST clients from available API keys.

    Tries TINKER_API_KEY from environment/.env first,
    then all keys from .env-tinker if present.
    Returns list of (label, rest_client) tuples.
    """
    from dotenv import load_dotenv
    import tinker

    clients = []

    # Primary key from .env
    load_dotenv()
    primary_key = os.environ.get("TINKER_API_KEY")
    if primary_key:
        try:
            sc = tinker.ServiceClient()
            clients.append(("default", sc.create_rest_client()))
        except Exception:
            pass

    # Additional keys from .env-tinker
    env_tinker = Path(".env-tinker")
    if env_tinker.exists():
        for line in env_tinker.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            var_name, _, key_value = line.partition("=")
            key_value = key_value.strip().strip("'\"")
            if not key_value.startswith("tml-"):
                continue
            label = var_name.strip().replace("TINKER_API_KEY_", "").lower()
            try:
                os.environ["TINKER_API_KEY"] = key_value
                sc = tinker.ServiceClient()
                clients.append((label, sc.create_rest_client()))
            except Exception:
                pass

    # Restore original key
    if primary_key:
        os.environ["TINKER_API_KEY"] = primary_key

    return clients


def get_working_client(clients: list, uuid: str, sample_tinker_path: str | None = None):
    """Find a client that can access the given training run UUID.

    If sample_tinker_path is provided, validates by fetching the archive URL
    (works even when list_checkpoints returns 404 for deleted models).
    Otherwise falls back to list_checkpoints.

    Returns (label, rest_client) or (None, None).
    """
    for label, client in clients:
        try:
            if sample_tinker_path:
                client.get_checkpoint_archive_url_from_tinker_path(sample_tinker_path).result()
            else:
                client.list_checkpoints(uuid).result()
            return label, client
        except Exception:
            continue
    return None, None


def is_downloaded(log_dir: str, name: str) -> bool:
    """Check if a checkpoint has been downloaded (sentinel file exists)."""
    sentinel = Path(log_dir) / "downloaded_checkpoints" / name / "sampler_weights" / "checkpoint_complete"
    return sentinel.exists()


def fmt_size(b: int) -> str:
    """Format byte count as human-readable size."""
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    elif b >= 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b} B"


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def read_metrics(log_dir: str) -> list[dict]:
    """Read metrics.jsonl from a log directory."""
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return []
    return read_jsonl(metrics_path)


def detect_epoch_boundaries(metrics: list[dict], config: dict | None) -> list[int]:
    """Detect epoch boundary batch numbers from metrics.

    Returns list of batch numbers where each epoch ends (the last batch of that epoch).
    """
    boundaries = []

    # Try progress/epoch field first (most runs)
    epoch_key = None
    if metrics and "progress/epoch" in metrics[0]:
        epoch_key = "progress/epoch"
    elif metrics and "epoch" in metrics[0]:
        epoch_key = "epoch"

    if epoch_key:
        for i in range(len(metrics) - 1):
            curr_epoch = metrics[i].get(epoch_key)
            next_epoch = metrics[i + 1].get(epoch_key)
            if curr_epoch is not None and next_epoch is not None and next_epoch > curr_epoch:
                # Epoch changed - this batch is the last of curr_epoch
                batch = metrics[i].get("progress/batch", metrics[i].get("batch"))
                if batch is not None:
                    boundaries.append(int(batch))
        return boundaries

    # Fallback: use done_frac + num_epochs from config
    if config and metrics:
        num_epochs = config.get("num_epochs") or config.get("opd_num_epochs")
        if num_epochs and num_epochs > 1:
            for epoch_end in range(1, int(num_epochs)):
                target_frac = epoch_end / num_epochs
                # Find closest batch to this fraction
                best_batch = None
                best_dist = float("inf")
                for m in metrics:
                    frac = m.get("progress/done_frac")
                    if frac is not None:
                        dist = abs(frac - target_frac)
                        if dist < best_dist:
                            best_dist = dist
                            batch = m.get("progress/batch", m.get("batch"))
                            if batch is not None:
                                best_batch = int(batch)
                                best_dist = dist
                if best_batch is not None:
                    boundaries.append(best_batch)

    return boundaries


def find_lowest_kl_batch(metrics: list[dict]) -> int | None:
    """Find batch number with the lowest teacher_kl."""
    best_batch = None
    best_kl = float("inf")
    for m in metrics:
        kl = m.get("teacher_kl")
        if kl is not None and kl < best_kl:
            best_kl = kl
            batch = m.get("progress/batch", m.get("batch"))
            if batch is not None:
                best_batch = int(batch)
                best_kl = kl
    return best_batch


def closest_checkpoint(checkpoints: list[dict], target_batch: int) -> dict | None:
    """Find checkpoint closest to target batch number."""
    if not checkpoints:
        return None
    best = None
    best_dist = float("inf")
    for c in checkpoints:
        dist = abs(c["batch"] - target_batch)
        if dist < best_dist:
            best_dist = dist
            best = c
    return best


def select_checkpoints(run: dict) -> list[dict]:
    """Select important checkpoints for a run.

    Returns list of dicts with keys: checkpoint, reasons, downloaded.
    Each checkpoint may have multiple reasons (e.g., "last + lowest_kl").
    """
    checkpoints = run["checkpoints"]
    if not checkpoints:
        return []

    log_dir = run["log_dir"]
    metrics = read_metrics(log_dir)

    config = None
    config_path = os.path.join(log_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Map from checkpoint name -> {checkpoint, reasons}
    selected: dict[str, dict] = {}

    def add(ckpt: dict, reason: str):
        name = ckpt["name"]
        if name not in selected:
            selected[name] = {
                "checkpoint": ckpt,
                "reasons": [],
                "downloaded": is_downloaded(log_dir, name),
            }
        selected[name]["reasons"].append(reason)

    # 1. Last checkpoint
    add(checkpoints[-1], "last")

    # 2. 50% checkpoint (middle by index)
    if len(checkpoints) >= 3:
        mid_idx = len(checkpoints) // 2
        add(checkpoints[mid_idx], "50%")

    # 3. Epoch boundaries
    epoch_boundaries = detect_epoch_boundaries(metrics, config)
    for i, boundary_batch in enumerate(epoch_boundaries):
        ckpt = closest_checkpoint(checkpoints, boundary_batch)
        if ckpt:
            add(ckpt, f"epoch_{i}_end")

    # 4. Lowest KL
    lowest_kl_batch = find_lowest_kl_batch(metrics)
    if lowest_kl_batch is not None:
        ckpt = closest_checkpoint(checkpoints, lowest_kl_batch)
        if ckpt:
            add(ckpt, "lowest_kl")

    # Sort by batch number
    result = sorted(selected.values(), key=lambda x: x["checkpoint"]["batch"])
    return result


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_checkpoint(rest_client, checkpoint: dict, log_dir: str, dry_run: bool = False) -> bool:
    """Download a single checkpoint's sampler_weights.

    Returns True on success, False on failure.
    """
    name = checkpoint["name"]
    sampler_path = checkpoint["sampler_path"]
    dest_dir = Path(log_dir) / "downloaded_checkpoints" / name / "sampler_weights"
    sentinel = dest_dir / "checkpoint_complete"

    if sentinel.exists():
        return True

    if dry_run:
        print(f"  [dry-run] Would download {name} -> {dest_dir}")
        return True

    print(f"  Downloading {name}...", end=" ", flush=True)
    try:
        response = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
        r = requests.get(response.url)
        r.raise_for_status()

        # Extract tar archive
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:*") as tar:
            tar.extractall(path=str(dest_dir))

        # Write sentinel
        sentinel.write_text("")
        print("done")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_inventory(args):
    runs = find_runs(args.run)
    if not runs:
        print("No runs with tinker:// checkpoints found.")
        return

    # Try to query cloud for counts
    clients = create_rest_clients()
    if not clients:
        print("Warning: No Tinker API keys found. Cloud counts will be unavailable.\n")

    print(f"{'Run Name':<45} {'UUID (short)':<15} {'In JSONL':>10} {'On Cloud':>10} {'Downloaded':>12}")
    print("-" * 95)

    for run in runs:
        uuid_short = (run["uuid"] or "???")[:12] + "..."
        n_jsonl = len(run["checkpoints"])

        # Count downloaded
        n_downloaded = sum(
            1 for c in run["checkpoints"]
            if is_downloaded(run["log_dir"], c["name"])
        )

        # Query cloud
        n_cloud = "?"
        sample_path = run["checkpoints"][0].get("sampler_path") if run["checkpoints"] else None
        if clients and run["uuid"]:
            _, client = get_working_client(clients, run["uuid"], sample_tinker_path=sample_path)
            if client:
                try:
                    resp = client.list_checkpoints(run["uuid"]).result()
                    n_cloud = len(resp.checkpoints)
                except Exception:
                    n_cloud = "err"

        print(f"{run['name']:<45} {uuid_short:<15} {n_jsonl:>10} {str(n_cloud):>10} {n_downloaded:>12}")


def cmd_select(args):
    runs = find_runs(args.run)
    if not runs:
        print("No runs with tinker:// checkpoints found.")
        return

    for run in runs:
        selections = select_checkpoints(run)
        print(f"\n{'='*70}")
        print(f"Run: {run['name']}  ({len(run['checkpoints'])} total checkpoints)")
        print(f"{'='*70}")

        if not selections:
            print("  No checkpoints selected.")
            continue

        print(f"  {'Name':<15} {'Batch':>8} {'Reasons':<35} {'Downloaded'}")
        print(f"  {'-'*70}")
        for sel in selections:
            c = sel["checkpoint"]
            reasons = " + ".join(sel["reasons"])
            dl = "yes" if sel["downloaded"] else "no"
            print(f"  {c['name']:<15} {c['batch']:>8} {reasons:<35} {dl}")


def cmd_download(args):
    runs = find_runs(args.run)
    if not runs:
        print("No runs with tinker:// checkpoints found.")
        return

    clients = create_rest_clients()
    if not clients and not args.dry_run:
        print("Error: No Tinker API keys found. Set TINKER_API_KEY or create .env-tinker.")
        sys.exit(1)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for run in runs:
        selections = select_checkpoints(run)
        to_download = [s for s in selections if not s["downloaded"]]

        if not to_download:
            print(f"\n{run['name']}: all {len(selections)} selected checkpoints already downloaded")
            total_skipped += len(selections)
            continue

        print(f"\n{run['name']}: {len(to_download)} to download, {len(selections) - len(to_download)} already downloaded")

        if not args.dry_run:
            sample_path = to_download[0]["checkpoint"].get("sampler_path")
            _, client = get_working_client(clients, run["uuid"], sample_tinker_path=sample_path)
            if not client:
                print(f"  Error: No working API key for UUID {run['uuid']}")
                total_failed += len(to_download)
                continue
        else:
            client = None

        for sel in to_download:
            ok = download_checkpoint(client, sel["checkpoint"], run["log_dir"], dry_run=args.dry_run)
            if ok:
                total_downloaded += 1
            else:
                total_failed += 1

    print(f"\nSummary: {total_downloaded} downloaded, {total_skipped} already present, {total_failed} failed")


def cmd_storage(args):
    """Show exact cloud storage usage by querying the Tinker API."""
    from collections import defaultdict

    clients = create_rest_clients()
    if not clients:
        print("Error: No Tinker API keys found. Set TINKER_API_KEY or create .env-tinker.")
        sys.exit(1)

    # Build UUID -> run name mapping from local checkpoints.jsonl files
    uuid_to_run = {}
    runs = find_runs()
    for run in runs:
        if run["uuid"]:
            uuid_to_run[run["uuid"]] = run["name"]

    all_checkpoints = fetch_all_cloud_checkpoints(clients)

    if not all_checkpoints:
        print("\nNo checkpoints found on cloud.")
        return

    # Group by training run UUID
    run_data: dict[str, dict] = defaultdict(lambda: {"size_bytes": 0, "count": 0, "missing_size": 0})
    total_bytes = 0
    total_count = 0
    total_missing_size = 0

    for tinker_path, ckpt in all_checkpoints.items():
        uuid = extract_uuid(tinker_path)
        if not uuid:
            continue
        run_data[uuid]["count"] += 1
        total_count += 1
        if ckpt.size_bytes is not None:
            run_data[uuid]["size_bytes"] += ckpt.size_bytes
            total_bytes += ckpt.size_bytes
        else:
            run_data[uuid]["missing_size"] += 1
            total_missing_size += 1

    # Sort by size descending
    sorted_runs = sorted(run_data.items(), key=lambda x: x[1]["size_bytes"], reverse=True)

    # Print results
    print(f"\n{'Run Name':<55} {'Ckpts':>6} {'Size':>10}")
    print("-" * 75)

    for uuid, data in sorted_runs:
        run_name = uuid_to_run.get(uuid, f"(unknown) {uuid[:12]}...")
        size_str = fmt_size(data["size_bytes"])
        if data["missing_size"] > 0:
            size_str += f" (+{data['missing_size']} unknown)"
        print(f"{run_name:<55} {data['count']:>6} {size_str:>10}")

    print("-" * 75)
    total_str = fmt_size(total_bytes)
    if total_missing_size > 0:
        total_str += f"  ({total_missing_size} ckpts missing size)"
    print(f"{'TOTAL':<55} {total_count:>6} {total_str:>10}")
    print()


def fetch_all_cloud_checkpoints(clients: list) -> dict[str, object]:
    """Paginate list_user_checkpoints across all API keys, deduplicating by tinker_path."""
    all_checkpoints: dict[str, object] = {}
    for label, client in clients:
        print(f"Querying API key '{label}'...", end=" ", flush=True)
        offset = 0
        page_size = 200
        key_count = 0
        while True:
            try:
                resp = client.list_user_checkpoints(limit=page_size, offset=offset).result()
            except Exception as e:
                print(f"error: {e}")
                break

            for ckpt in resp.checkpoints:
                if ckpt.tinker_path not in all_checkpoints:
                    all_checkpoints[ckpt.tinker_path] = ckpt
                    key_count += 1

            offset += len(resp.checkpoints)
            total = resp.cursor.total_count if resp.cursor else None
            if len(resp.checkpoints) < page_size or (total and offset >= total):
                break

        print(f"{key_count} new checkpoints (total on this key: {total or offset})")
    return all_checkpoints


def fetch_training_runs_metadata(clients: list) -> dict[str, object]:
    """Paginate list_training_runs across all API keys. Returns dict[training_run_id, TrainingRun]."""
    all_runs: dict[str, object] = {}
    for label, client in clients:
        offset = 0
        page_size = 200
        while True:
            try:
                resp = client.list_training_runs(limit=page_size, offset=offset).result()
            except Exception:
                break

            for tr in resp.training_runs:
                if tr.training_run_id not in all_runs:
                    all_runs[tr.training_run_id] = tr

            offset += len(resp.training_runs)
            total = resp.cursor.total_count if resp.cursor else None
            if len(resp.training_runs) < page_size or (total and offset >= total):
                break
    return all_runs


def cmd_list_cloud(args):
    """List all checkpoints on the cloud with full details, grouped by training run."""
    from collections import defaultdict

    clients = create_rest_clients()
    if not clients:
        print("Error: No Tinker API keys found. Set TINKER_API_KEY or create .env-tinker.")
        sys.exit(1)

    # Build local cross-reference: UUID -> {run_name, log_dir}
    uuid_to_local: dict[str, dict] = {}
    for run in find_runs():
        if run["uuid"]:
            uuid_to_local[run["uuid"]] = {"name": run["name"], "log_dir": run["log_dir"]}

    # If --run filter, resolve to UUID
    filter_uuids: set[str] | None = None
    if args.run:
        filter_uuids = set()
        for uuid, local in uuid_to_local.items():
            if local["name"] == args.run:
                filter_uuids.add(uuid)
        if not filter_uuids:
            print(f"No local run named '{args.run}' found.")
            return

    # Fetch cloud data
    all_checkpoints = fetch_all_cloud_checkpoints(clients)
    if not all_checkpoints:
        print("\nNo checkpoints found on cloud.")
        return

    training_runs = fetch_training_runs_metadata(clients)

    # Group by training run UUID
    by_run: dict[str, list] = defaultdict(list)
    for tinker_path, ckpt in all_checkpoints.items():
        uuid = extract_uuid(tinker_path)
        if not uuid:
            continue
        if filter_uuids and uuid not in filter_uuids:
            continue
        if args.type != "all" and ckpt.checkpoint_type != args.type:
            continue
        by_run[uuid].append((tinker_path, ckpt))

    if not by_run:
        print("\nNo checkpoints match the filters.")
        return

    # Sort runs: known local runs first (alphabetical), then unknown
    def run_sort_key(uuid):
        local = uuid_to_local.get(uuid)
        return (0, local["name"]) if local else (1, uuid)

    grand_total_count = 0
    grand_total_bytes = 0

    for uuid in sorted(by_run.keys(), key=run_sort_key):
        ckpts = by_run[uuid]
        local = uuid_to_local.get(uuid)
        run_name = local["name"] if local else f"(unknown) {uuid[:12]}..."
        tr = training_runs.get(f"{uuid}:train:0")

        # Extract the training_run_id suffix (e.g. ":train:0") from a tinker_path
        # to look up the correct training run if the simple lookup failed
        if tr is None:
            for tp, _ in ckpts:
                m = re.match(r"tinker://([^/]+)/", tp)
                if m:
                    tr = training_runs.get(m.group(1))
                    if tr:
                        break

        # Header
        print(f"\n{'=' * 80}")
        header = f"Run: {run_name}  ({uuid[:12]}...)"
        print(header)
        if tr:
            lora_str = f"LoRA r{tr.lora_rank}" if tr.is_lora and tr.lora_rank else ("LoRA" if tr.is_lora else "full")
            print(f"Base: {tr.base_model}  |  {lora_str}")
        print(f"{'=' * 80}")

        # Parse checkpoint name from tinker_path: .../weights/NAME or .../sampler_weights/NAME
        def ckpt_name(tinker_path: str) -> str:
            return tinker_path.rsplit("/", 1)[-1]

        def ckpt_sort_key(item):
            tp, c = item
            name = ckpt_name(tp)
            # Sort "final" last, otherwise by name, then type (sampler before training)
            is_final = name == "final"
            type_order = 0 if c.checkpoint_type == "sampler" else 1
            return (is_final, name, type_order)

        ckpts_sorted = sorted(ckpts, key=ckpt_sort_key)

        # Per-type accumulators
        run_sampler_count = 0
        run_sampler_bytes = 0
        run_training_count = 0
        run_training_bytes = 0

        if not args.summary_only:
            print(f"  {'Name':<15} {'Type':<10} {'Size':>10} {'Downloaded':<12} {'Public':<8} {'Expires'}")
            print(f"  {'-' * 72}")

        for tinker_path, ckpt in ckpts_sorted:
            name = ckpt_name(tinker_path)
            size = ckpt.size_bytes or 0
            is_sampler = ckpt.checkpoint_type == "sampler"

            if is_sampler:
                run_sampler_count += 1
                run_sampler_bytes += size
            else:
                run_training_count += 1
                run_training_bytes += size

            if not args.summary_only:
                size_str = fmt_size(size) if ckpt.size_bytes is not None else "?"
                # Downloaded status: only meaningful for sampler checkpoints with a local run
                if is_sampler and local:
                    dl_str = "yes" if is_downloaded(local["log_dir"], name) else "no"
                else:
                    dl_str = "-"
                public_str = "yes" if ckpt.public else "no"
                expires_str = str(ckpt.expires_at.date()) if ckpt.expires_at else "-"
                print(f"  {name:<15} {ckpt.checkpoint_type:<10} {size_str:>10} {dl_str:<12} {public_str:<8} {expires_str}")

        # Per-run summary
        run_total = run_sampler_count + run_training_count
        run_bytes = run_sampler_bytes + run_training_bytes
        parts = []
        if run_sampler_count:
            parts.append(f"{run_sampler_count} sampler ({fmt_size(run_sampler_bytes)})")
        if run_training_count:
            parts.append(f"{run_training_count} training ({fmt_size(run_training_bytes)})")
        summary = " + ".join(parts) if parts else "0 checkpoints"
        print(f"\n  {summary} = {run_total} ckpts ({fmt_size(run_bytes)})")

        grand_total_count += run_total
        grand_total_bytes += run_bytes

    # Grand total
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {grand_total_count} checkpoints ({fmt_size(grand_total_bytes)})")
    print()


def cmd_delete_all(args):
    runs = find_runs(args.run)
    if not runs:
        print("No runs with tinker:// checkpoints found.")
        return

    # Show what will be deleted
    total_cloud = 0
    for run in runs:
        n_ckpts = len(run["checkpoints"])
        # Each checkpoint has state_path + sampler_path = 2 cloud objects
        n_cloud = sum(
            (1 if c.get("state_path", "").startswith("tinker://") else 0) +
            (1 if c.get("sampler_path", "").startswith("tinker://") else 0)
            for c in run["checkpoints"]
        )

        print(f"{run['name']}: {n_ckpts} checkpoints ({n_cloud} cloud objects)")
        total_cloud += n_cloud

    print(f"\nTotal: {total_cloud} cloud objects to delete")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return

    if not args.yes:
        answer = input("\nType 'yes' to confirm deletion: ")
        if answer.strip().lower() != "yes":
            print("Aborted.")
            return

    clients = create_rest_clients()
    if not clients:
        print("Error: No Tinker API keys found.")
        sys.exit(1)

    for run in runs:
        print(f"\nDeleting {run['name']}...")

        # Delete cloud checkpoints
        sample_path = run["checkpoints"][0].get("sampler_path") if run["checkpoints"] else None
        _, client = get_working_client(clients, run["uuid"], sample_tinker_path=sample_path)
        if client:
            for c in run["checkpoints"]:
                for path_key in ("state_path", "sampler_path"):
                    tinker_path = c.get(path_key, "")
                    if tinker_path.startswith("tinker://"):
                        try:
                            client.delete_checkpoint_from_tinker_path(tinker_path).result()
                            print(f"  Deleted cloud: {c['name']}/{path_key.split('_')[0]}")
                        except Exception as e:
                            print(f"  Failed to delete {tinker_path}: {e}")
        else:
            print(f"  Warning: No working API key for UUID {run['uuid']}, skipping cloud deletion")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage Tinker training checkpoints: inventory, select, download, delete."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inventory
    p_inv = subparsers.add_parser("inventory", help="List all runs with tinker:// checkpoints")
    p_inv.add_argument("--run", help="Filter to a specific run name")

    # select
    p_sel = subparsers.add_parser("select", help="Show which checkpoints would be selected for download")
    p_sel.add_argument("--run", help="Filter to a specific run name")

    # download
    p_dl = subparsers.add_parser("download", help="Download selected checkpoints")
    p_dl.add_argument("--run", help="Filter to a specific run name")
    p_dl.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")

    # storage
    subparsers.add_parser("storage", help="Show exact cloud storage usage from Tinker API")

    # list-cloud
    p_lc = subparsers.add_parser("list-cloud", help="List all checkpoints on the cloud with full details")
    p_lc.add_argument("--run", help="Filter to a specific local run name")
    p_lc.add_argument("--type", choices=["sampler", "training", "all"], default="all", help="Filter by checkpoint type")
    p_lc.add_argument("--summary-only", action="store_true", help="Show per-run summaries only")

    # delete-all
    p_del = subparsers.add_parser("delete-all", help="Delete ALL checkpoints (cloud + local)")
    p_del.add_argument("--run", help="Filter to a specific run name")
    p_del.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    p_del.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.command == "inventory":
        cmd_inventory(args)
    elif args.command == "select":
        cmd_select(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "storage":
        cmd_storage(args)
    elif args.command == "list-cloud":
        cmd_list_cloud(args)
    elif args.command == "delete-all":
        cmd_delete_all(args)


if __name__ == "__main__":
    main()
