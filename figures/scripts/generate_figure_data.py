"""Generate .dat files under figures/data/ from eval_results.

Reads the same directory mappings view_sweep1.py uses (so the figures match
the dashboard one-to-one) and writes plain-text tables consumable by
pgfplots `\\addplot table {...}`.

Outputs (under figures/data/):

  safety_<variant>_<family_slug>_<method_slug>.dat
      # threshold_idx  pct  ci_lo  ci_hi
      variant = sage | safe_sage
      For the `sage` variant we read eval_results/<dir>/gemini31_...jsonl
      directly. For `safe_sage` we read
      eval_results/regression/safe-sage-eval-test/<dir>/gemini31_...jsonl
      with a tinker-judge fallback annotated via the sidecar manifest.

  ausc_<variant>_<family_slug>.dat
      # method_idx  method_slug  ausc  model_score

  heatmap_<variant>_<family_slug>.dat
      # fact_idx  method_idx  pass_rate
      Facts are the union across all methods (stable order per file).

  regression_<eval_slug>.dat
      # family_idx  method_idx  accuracy   (accuracy in [0,100]; NaN = missing)

  manifest.json
      Cross-file metadata: method/family orderings, fact labels, baseline
      accuracies, TODO markers listing (variant, family, method) cells with
      no data, and judge-fallback notes.

Run from repo root:
    uv run python figures/scripts/generate_figure_data.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAGEEVAL_JUDGE_DIR = REPO_ROOT / "sageeval-judge"
sys.path.insert(0, str(SAGEEVAL_JUDGE_DIR))

# Pulled directly from view_sweep1 to guarantee parity with the dashboard.
import view_sweep1 as vs  # noqa: E402

# Bootstrap config — 95 % CI, 1000 resamples, fixed seed for reproducibility.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 47
BOOTSTRAP_CI = 0.95
# A single rng shared across calls so re-runs are reproducible across every
# bootstrap call site (order of calls matters).
_BOOT_RNG = np.random.default_rng(BOOTSTRAP_SEED)


def _ausc_from_pass_rates(pass_rates: np.ndarray) -> float:
    """Replicates view_sweep1.compute_ausc but takes an np.ndarray."""
    n = len(pass_rates)
    if n == 0:
        return 0.0
    total = 0.0
    # AUSC = mean over 101 thresholds of fraction(pass_rate >= threshold).
    for t in range(101):
        threshold = t / 100.0
        total += np.sum(pass_rates >= threshold) / n
    return float(total / 101)


def bootstrap_ausc_ci(pass_rates: list[float]) -> tuple[float, float, float]:
    """Return (ausc, ausc_lo, ausc_hi). Resamples facts with replacement."""
    arr = np.asarray(pass_rates, dtype=float)
    ausc = _ausc_from_pass_rates(arr)
    n = len(arr)
    if n < 2:
        return ausc, ausc, ausc
    samples = _BOOT_RNG.choice(arr, size=(BOOTSTRAP_N, n), replace=True)
    boot_ausc = np.array([_ausc_from_pass_rates(s) for s in samples])
    alpha = (1 - BOOTSTRAP_CI) / 2
    lo = float(np.quantile(boot_ausc, alpha))
    hi = float(np.quantile(boot_ausc, 1 - alpha))
    return ausc, lo, hi


def bootstrap_proportion_ci(n_correct: int, n_total: int) -> tuple[float, float, float]:
    """Binomial bootstrap CI on a proportion. Returns (p, lo, hi) in [0, 1]."""
    if n_total <= 0:
        return 0.0, 0.0, 0.0
    p = n_correct / n_total
    if n_total < 2:
        return p, p, p
    outcomes = np.zeros(n_total, dtype=np.int8)
    outcomes[:n_correct] = 1
    samples = _BOOT_RNG.choice(outcomes, size=(BOOTSTRAP_N, n_total), replace=True)
    boot_p = samples.mean(axis=1)
    alpha = (1 - BOOTSTRAP_CI) / 2
    lo = float(np.quantile(boot_p, alpha))
    hi = float(np.quantile(boot_p, 1 - alpha))
    return p, lo, hi

FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR = FIGURES_DIR / "data"

# Ordered list of method labels we plot (methods + baselines).
METHODS_ORDERED: list[str] = list(vs.METHOD_ORDER)
FAMILIES_ORDERED: list[str] = list(vs.MODELS.keys())

# Short machine-safe slugs used in filenames.
FAMILY_SLUGS = {
    "Llama 8B": "llama8b",
    "Qwen 4B": "qwen4b",
    "GPT-OSS 20B": "gptoss20b",
}

METHOD_SLUGS = {
    "OPD (e1)":                "opd_e1",
    "OPD — final":             "opd_final",
    "SFT":                     "sft",
    "SFT (e1)":                "sft_e1",
    "SFT + OPD":               "sft_opd",
    "Teacher Cheat Oracle":    "teacher",
    "Vanilla (no safety fact)": "vanilla",
}

assert set(METHOD_SLUGS) == set(METHODS_ORDERED), (
    "METHOD_SLUGS must cover every entry in METHOD_ORDER"
)


def _write_dat(path: Path, header: list[str], rows: list[list]) -> None:
    # pgfplots reads the first row as column names by default — no `#`
    # prefix or it'll be parsed as a literal column label.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("  ".join(header) + "\n")
        for row in rows:
            f.write("  ".join(_fmt_cell(v) for v in row) + "\n")


def _fmt_cell(v) -> str:
    if v is None:
        return "nan"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.6g}"
    return str(v)


def _load_judge_rows(folder: Path) -> tuple[list[dict], str] | None:
    """Return (rows, judge_variant) or None.

    Strict gemini31 only — mirrors view_sweep1.py's authoritative-judge
    policy. Dirs that only have a tinker/openrouter/john judge file are
    treated as missing here, so they show up as TODO in the figures.
    """
    authoritative = folder / vs.AUTHORITATIVE_JUDGE_FILENAME
    if authoritative.is_file():
        return vs.load_judged_results(authoritative), "gemini31"
    return None


def _safe_sage_folder(dir_name: str | None) -> Path | None:
    if dir_name is None:
        return None
    p = vs.SAFE_SAGE_DIR / dir_name
    return p if p.is_dir() else None


def _sage_folder(dir_name: str | None) -> Path | None:
    if dir_name is None:
        return None
    p = vs.EVAL_RESULTS_DIR / dir_name
    return p if p.is_dir() else None


def _resolve_sage_cell(family: str, method: str) -> tuple[str | None, Path | None]:
    cfg = vs.MODELS[family]
    all_methods = {**cfg["methods"], **cfg.get("baselines", {})}
    dir_name = all_methods.get(method)
    if dir_name is None:  # genuine n/a (e.g. SFT (e1) for gptoss)
        return None, None
    return dir_name, _sage_folder(dir_name)


# Safe-sage directory mapping: REGRESSION_ROWS covers the 14 method cells,
# but baselines need hand-mapping because view_sweep1 never tracked safe-sage
# baselines. We re-use the sageeval baseline dir names — on disk only the two
# Llama ones actually exist today; the rest are TODO.
_SAFE_SAGE_REGRESSION_ROW_LOOKUP: dict[tuple[str, str], str] = {
    (fam, method): run_dir
    for fam, method, _ckpt, run_dir in vs.REGRESSION_ROWS
}


def _resolve_safe_sage_cell(family: str, method: str) -> tuple[str | None, Path | None]:
    if method in vs.BASELINE_NAMES:
        # Baselines: pull the sageeval dir name and try that same name under
        # safe-sage-eval-test (llama8b-teacher-system, sweep1-baseline-llama8b
        # are the only ones that exist today).
        cfg = vs.MODELS[family]
        dir_name = cfg.get("baselines", {}).get(method)
        if dir_name is None:
            return None, None
        return dir_name, _safe_sage_folder(dir_name)
    # Training-method rows: REGRESSION_ROWS is the source of truth. Treat a
    # missing entry as n/a (e.g. SFT (e1) for gptoss).
    dir_name = _SAFE_SAGE_REGRESSION_ROW_LOOKUP.get((family, method))
    if dir_name is None:
        return None, None
    return dir_name, _safe_sage_folder(dir_name)


def _curve_from_folder(folder: Path) -> tuple[list[float], list[float], str] | None:
    """(percentages, ci_bounds, judge_variant) for a single judged run."""
    loaded = _load_judge_rows(folder)
    if loaded is None:
        return None
    rows, variant = loaded
    if not rows:
        return None
    verdicts = vs.compute_fact_verdicts(rows)
    pcts, cis = vs.compute_threshold_curve(verdicts, vs.THRESHOLDS)
    return pcts, cis, variant


def _model_metrics_from_folder(folder: Path) -> tuple[float, float, float, float, str] | None:
    """(ausc, ausc_lo, ausc_hi, model_score, judge_variant) for a single run.

    CI is a 95 % nonparametric bootstrap over facts (resample per-fact pass
    rates with replacement; recompute AUSC on each resample).
    """
    loaded = _load_judge_rows(folder)
    if loaded is None:
        return None
    rows, variant = loaded
    if not rows:
        return None
    fact_scores = vs.compute_fact_level_scores(rows)
    pass_rates = [s["pass_rate"] for s in fact_scores.values()]
    ausc, lo, hi = bootstrap_ausc_ci(pass_rates)
    _, _, model_score = vs.compute_model_level_score(fact_scores)
    return ausc, lo, hi, model_score, variant


def _fact_scores_from_folder(folder: Path) -> tuple[dict[str, dict], str] | None:
    loaded = _load_judge_rows(folder)
    if loaded is None:
        return None
    rows, variant = loaded
    if not rows:
        return None
    return vs.compute_fact_level_scores(rows), variant


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _variant_resolver(variant: str):
    if variant == "sage":
        return _resolve_sage_cell
    if variant == "safe_sage":
        return _resolve_safe_sage_cell
    raise ValueError(variant)


def write_safety_curves(variant: str, todos: list[dict], fallbacks: list[dict]) -> None:
    resolver = _variant_resolver(variant)
    # Sentinel rows for TODO cells. pct is NaN (main line renders nothing),
    # but lo=hi=0 gives a real-valued degenerate fill-between path so the
    # pgfplots fillbetween lib doesn't blow up on an empty named path. The
    # degenerate path is zero-area so nothing visible ends up on the figure.
    n_thresh = len(vs.THRESHOLDS)
    empty_rows = [[i, None, 0.0, 0.0] for i in range(n_thresh)]
    for family in FAMILIES_ORDERED:
        fam_slug = FAMILY_SLUGS[family]
        for method in METHODS_ORDERED:
            slug = METHOD_SLUGS[method]
            dat = DATA_DIR / f"safety_{variant}_{fam_slug}_{slug}.dat"
            dir_name, folder = resolver(family, method)
            if dir_name is None:
                _write_dat(dat, ["idx", "pct", "lo", "hi"], empty_rows)
                continue
            if folder is None:
                _write_dat(dat, ["idx", "pct", "lo", "hi"], empty_rows)
                todos.append({
                    "kind": "safety_curve", "variant": variant,
                    "family": family, "method": method,
                    "reason": f"directory {dir_name!r} missing on disk",
                })
                continue
            res = _curve_from_folder(folder)
            if res is None:
                _write_dat(dat, ["idx", "pct", "lo", "hi"], empty_rows)
                todos.append({
                    "kind": "safety_curve", "variant": variant,
                    "family": family, "method": method,
                    "reason": f"no gemini31/tinker judge in {dir_name!r}",
                })
                continue
            pcts, cis, used_variant = res
            if used_variant != "gemini31":
                fallbacks.append({
                    "kind": "safety_curve", "variant": variant,
                    "family": family, "method": method,
                    "dir": dir_name, "judge_used": used_variant,
                })
            rows = [
                [i, pcts[i], pcts[i] - cis[i], pcts[i] + cis[i]]
                for i in range(len(pcts))
            ]
            _write_dat(dat, ["idx", "pct", "lo", "hi"], rows)


def write_ausc(variant: str, todos: list[dict], fallbacks: list[dict]) -> None:
    resolver = _variant_resolver(variant)
    columns = ["method_idx", "method", "ausc", "ausc_lo", "ausc_hi",
               "model_score"]
    for family in FAMILIES_ORDERED:
        fam_slug = FAMILY_SLUGS[family]
        dat = DATA_DIR / f"ausc_{variant}_{fam_slug}.dat"
        rows: list[list] = []
        for method_idx, method in enumerate(METHODS_ORDERED):
            slug = METHOD_SLUGS[method]
            dir_name, folder = resolver(family, method)
            if dir_name is None or folder is None:
                rows.append([method_idx, slug, None, None, None, None])
                if dir_name is not None:
                    todos.append({
                        "kind": "ausc", "variant": variant,
                        "family": family, "method": method,
                        "reason": (
                            f"directory {dir_name!r} missing on disk"
                            if folder is None else "cell n/a"
                        ),
                    })
                continue
            res = _model_metrics_from_folder(folder)
            if res is None:
                rows.append([method_idx, slug, None, None, None, None])
                todos.append({
                    "kind": "ausc", "variant": variant,
                    "family": family, "method": method,
                    "reason": f"no judge rows in {dir_name!r}",
                })
                continue
            ausc, ausc_lo, ausc_hi, model_score, used_variant = res
            if used_variant != "gemini31":
                fallbacks.append({
                    "kind": "ausc", "variant": variant,
                    "family": family, "method": method,
                    "dir": dir_name, "judge_used": used_variant,
                })
            rows.append([method_idx, slug, ausc, ausc_lo, ausc_hi, model_score])
        _write_dat(dat, columns, rows)


def write_heatmaps(variant: str, todos: list[dict], fallbacks: list[dict],
                    fact_labels_out: dict[str, list[str]]) -> None:
    resolver = _variant_resolver(variant)
    # Pass 1: harvest per-cell fact scores for every (family, method) that
    # has data. Accumulate the variant-wide fact union so every family
    # panel shares the same x-axis (needed for pgfplots mesh plots to have
    # a consistent mesh/cols count even when some panels are entirely TODO).
    per_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for family in FAMILIES_ORDERED:
        for method in METHODS_ORDERED:
            dir_name, folder = resolver(family, method)
            if dir_name is None or folder is None:
                continue
            res = _fact_scores_from_folder(folder)
            if res is None:
                continue
            fact_scores, used_variant = res
            per_cell[(family, method)] = fact_scores
            if used_variant != "gemini31":
                fallbacks.append({
                    "kind": "heatmap", "variant": variant,
                    "family": family, "method": method,
                    "dir": dir_name, "judge_used": used_variant,
                })

    # Variant-wide fact union (alphabetical for reproducibility).
    fact_union: set[str] = set()
    for scores in per_cell.values():
        fact_union.update(scores.keys())
    facts = sorted(fact_union)

    # Pass 2: emit dense per-family .dat. Every grid cell is present (nan
    # where missing) so `mesh/cols` stays valid. Empty data → all-NaN grid.
    for family in FAMILIES_ORDERED:
        fam_slug = FAMILY_SLUGS[family]
        fact_labels_out[f"{variant}:{family}"] = facts
        rows: list[list] = []
        for method_idx, method in enumerate(METHODS_ORDERED):
            scores = per_cell.get((family, method))
            for fact_idx, fact in enumerate(facts):
                rate = None
                if scores is not None:
                    entry = scores.get(fact)
                    rate = entry["pass_rate"] if entry else None
                rows.append([fact_idx, method_idx, rate])
            if scores is None:
                # cell is either n/a (true — no dir_name) or TODO (dir_name
                # exists but folder/judge missing). Only flag TODO for the
                # latter.
                dir_name, folder = resolver(family, method)
                if dir_name is not None:
                    todos.append({
                        "kind": "heatmap", "variant": variant,
                        "family": family, "method": method,
                        "reason": (
                            f"directory {dir_name!r} missing"
                            if folder is None else
                            f"no judge rows in {dir_name!r}"
                        ),
                    })
        _write_dat(
            DATA_DIR / f"heatmap_{variant}_{fam_slug}.dat",
            ["fact_idx", "method_idx", "pass_rate"],
            rows,
        )
        labels_dat = DATA_DIR / f"heatmap_{variant}_{fam_slug}_labels.dat"
        with open(labels_dat, "w") as f:
            for idx, fact in enumerate(facts):
                # Fact strings can embed raw newlines (multi-paragraph
                # additions). Collapse them to spaces so line-count of the
                # labels file equals fact count.
                collapsed = " ".join(fact.splitlines()).strip()
                f.write(f"{idx}\t{collapsed}\n")


def _read_score_and_counts(scores_path: Path, scores_key: str) -> tuple[float | None, int | None, int | None]:
    """Return (score, n_correct, n_total). Handles the three scorers' flat-
    JSON layouts: gpqa/math have `accuracy`/`correct`/`total`; ifeval has
    `prompt_level_strict_acc`/`n` plus nested `strict.prompt_correct`.
    """
    if not scores_path.is_file():
        return None, None, None
    with open(scores_path) as f:
        data = json.load(f)
    if scores_key not in data:
        return None, None, None
    score = data[scores_key]
    if not isinstance(score, (int, float)):
        return None, None, None
    # Total item count — `total` for gpqa/math, `n` for ifeval.
    n_total = data.get("total", data.get("n"))
    # Correct count — `correct` for gpqa/math; ifeval's prompt-level strict
    # count lives under `strict.prompt_correct`. Fall back to round(score*n)
    # so we always have a count to resample from.
    n_correct = data.get("correct")
    if n_correct is None:
        strict = data.get("strict")
        if isinstance(strict, dict):
            n_correct = strict.get("prompt_correct")
    if n_correct is None and isinstance(n_total, (int, float)):
        n_correct = int(round(float(score) * n_total))
    if isinstance(n_total, (int, float)) and isinstance(n_correct, (int, float)):
        return float(score), int(n_correct), int(n_total)
    return float(score), None, None


def write_regression(todos: list[dict]) -> dict:
    """Per-eval `.dat` + a baselines dict returned for the manifest.

    Adds a 95 % CI column pair per method. CI is a binomial bootstrap over
    per-item outcomes (n_correct vs n_total reconstructed from the scores
    JSON) with 1000 resamples.
    """
    lookup: dict[tuple[str, str], str] = {
        (fam, m): run for fam, m, _ckpt, run in vs.REGRESSION_ROWS
    }
    baselines: dict[str, dict[str, dict | None]] = {}
    non_baseline_methods = [m for m in METHODS_ORDERED if m not in vs.BASELINE_NAMES]
    method_slug_order = [METHOD_SLUGS[m] for m in non_baseline_methods]
    for slug, display, _expected, scores_file, scores_key in vs.REGRESSION_EVALS:
        scores_grid: dict[tuple[str, str], tuple[float, float, float] | None] = {}
        for family in FAMILIES_ORDERED:
            for method in non_baseline_methods:
                run_dir = lookup.get((family, method))
                if run_dir is None:
                    scores_grid[(family, method)] = None
                    continue
                scores_path = vs.REGRESSION_DIR / run_dir / slug / scores_file
                score, n_corr, n_tot = _read_score_and_counts(scores_path, scores_key)
                if score is None:
                    scores_grid[(family, method)] = None
                    todos.append({
                        "kind": "regression", "variant": slug,
                        "family": family, "method": method,
                        "reason": (
                            f"{scores_path.relative_to(REPO_ROOT)} missing"
                        ),
                    })
                    continue
                if n_corr is None or n_tot is None:
                    # No CI possible without counts; emit point estimate only.
                    scores_grid[(family, method)] = (score, score, score)
                    continue
                p, lo, hi = bootstrap_proportion_ci(n_corr, n_tot)
                scores_grid[(family, method)] = (p, lo, hi)

        # Long format (one row per (family, method)) with CI.
        long_rows: list[list] = []
        for fam_idx, family in enumerate(FAMILIES_ORDERED):
            for m_idx, method in enumerate(non_baseline_methods):
                entry = scores_grid[(family, method)]
                if entry is None:
                    long_rows.append([fam_idx, m_idx, METHOD_SLUGS[method],
                                        None, None, None])
                else:
                    p, lo, hi = entry
                    long_rows.append([fam_idx, m_idx, METHOD_SLUGS[method],
                                        p * 100, lo * 100, hi * 100])
        _write_dat(
            DATA_DIR / f"regression_{slug}.dat",
            ["family_idx", "method_idx", "method", "accuracy_pct",
             "acc_lo_pct", "acc_hi_pct"],
            long_rows,
        )

        # Wide format with explicit `<m>_lo` / `<m>_hi` columns per method.
        wide_cols = ["family_idx"]
        for slug_m in method_slug_order:
            wide_cols.extend([slug_m, f"{slug_m}_lo", f"{slug_m}_hi"])
        wide_rows: list[list] = []
        for fam_idx, family in enumerate(FAMILIES_ORDERED):
            row: list = [fam_idx]
            for method in non_baseline_methods:
                entry = scores_grid[(family, method)]
                if entry is None:
                    row.extend([None, None, None])
                else:
                    p, lo, hi = entry
                    row.extend([p * 100, lo * 100, hi * 100])
            wide_rows.append(row)
        _write_dat(
            DATA_DIR / f"regression_{slug}_wide.dat",
            wide_cols,
            wide_rows,
        )

        # Per-family baselines: bootstrap from baseline-dir scores.
        baselines[display] = {}
        for family in FAMILIES_ORDERED:
            baseline_dir = vs.REGRESSION_BASELINE_DIRS.get(family)
            if baseline_dir is None:
                baselines[display][family] = None
                continue
            b_path = vs.REGRESSION_DIR / baseline_dir / slug / scores_file
            b_score, b_corr, b_tot = _read_score_and_counts(b_path, scores_key)
            if b_score is None:
                baselines[display][family] = None
                todos.append({
                    "kind": "regression_baseline", "variant": slug,
                    "family": family, "method": "baseline",
                    "reason": (
                        f"baseline-dir scores missing under "
                        f"regression-programmatic/{baseline_dir}"
                    ),
                })
                continue
            if b_corr is not None and b_tot is not None:
                p, lo, hi = bootstrap_proportion_ci(b_corr, b_tot)
                baselines[display][family] = {
                    "accuracy_pct": p * 100,
                    "acc_lo_pct": lo * 100,
                    "acc_hi_pct": hi * 100,
                }
            else:
                baselines[display][family] = {
                    "accuracy_pct": b_score * 100,
                    "acc_lo_pct": b_score * 100,
                    "acc_hi_pct": b_score * 100,
                }
    # Baselines .dat with CI columns (for overlay markers + optional bar).
    rows_b: list[list] = []
    for eval_idx, (_slug, display, *_rest) in enumerate(vs.REGRESSION_EVALS):
        for fam_idx, family in enumerate(FAMILIES_ORDERED):
            entry = baselines[display][family]
            if entry is None:
                rows_b.append([eval_idx, fam_idx, display, None, None, None])
            else:
                rows_b.append([eval_idx, fam_idx, display,
                                entry["accuracy_pct"],
                                entry["acc_lo_pct"],
                                entry["acc_hi_pct"]])
    _write_dat(
        DATA_DIR / "regression_baselines.dat",
        ["eval_idx", "family_idx", "eval",
         "accuracy_pct", "acc_lo_pct", "acc_hi_pct"],
        rows_b,
    )
    return baselines


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    todos: list[dict] = []
    fallbacks: list[dict] = []
    fact_labels: dict[str, list[str]] = {}

    write_safety_curves("sage", todos, fallbacks)
    write_safety_curves("safe_sage", todos, fallbacks)
    write_ausc("sage", todos, fallbacks)
    write_ausc("safe_sage", todos, fallbacks)
    write_heatmaps("sage", todos, fallbacks, fact_labels)
    write_heatmaps("safe_sage", todos, fallbacks, fact_labels)
    baselines = write_regression(todos)

    manifest = {
        "methods_ordered": METHODS_ORDERED,
        "method_slugs": METHOD_SLUGS,
        "baseline_methods": sorted(vs.BASELINE_NAMES),
        "families_ordered": FAMILIES_ORDERED,
        "family_slugs": FAMILY_SLUGS,
        "thresholds": vs.THRESHOLDS,
        "threshold_labels": [f"{int(t * 100)}%" for t in vs.THRESHOLDS],
        "regression_evals": [
            {"slug": slug, "display": display, "expected_rows": exp}
            for slug, display, exp, _sf, _sk in vs.REGRESSION_EVALS
        ],
        "regression_baselines": baselines,
        "fact_labels": fact_labels,
        "todos": todos,
        "judge_fallbacks": fallbacks,
    }
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Human summary
    print(f"Wrote {len(list(DATA_DIR.glob('*.dat')))} .dat files + manifest.json")
    print(f"TODO cells: {len(todos)}")
    for t in todos:
        print(
            f"  [{t['kind']}/{t.get('variant', '-')}] "
            f"{t['family']} / {t['method']}: {t['reason']}"
        )
    if fallbacks:
        print(f"\nJudge fallbacks (gemini31 missing, used tinker): {len(fallbacks)}")
        for f in fallbacks:
            print(
                f"  [{f['kind']}/{f.get('variant', '-')}] "
                f"{f['family']} / {f['method']} — {f['dir']}"
            )


if __name__ == "__main__":
    main()
