import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM, ARMS
from mutation_ab.records import read_generations

COLORS = {ARM_DIFFERENCE: "#1f77b4", ARM_MIXTURE: "#ff7f0e", ARM_RANDOM: "#7f7f7f"}
LABELS = {ARM_DIFFERENCE: "A: differential", ARM_MIXTURE: "B: 10% Gaussian", ARM_RANDOM: "Random search"}
COLLAPSE_FRACTION = 0.01
MIN_IMPROVED = 20
USEFUL_SHIFT_M = 0.10
FALLBACK_GENERATIONS = 120


class IncompleteRuns(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmRun:
    seed: int
    arm: str
    generations: dict[str, np.ndarray]
    children: list[dict]
    config: dict


def load_runs(root: Path, exclude: set[int] | None = None) -> dict[int, dict[str, ArmRun]]:
    exclude = exclude or set()
    seed_dirs = [
        d
        for d in sorted(Path(root).glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
        if int(d.name.split("_")[1]) not in exclude
    ]
    if not seed_dirs:
        raise IncompleteRuns(f"no seed directories under {root}")
    missing = [str(d / arm) for d in seed_dirs for arm in ARMS if not (d / arm / "COMPLETE").exists()]
    if missing:
        raise IncompleteRuns(f"runs without COMPLETE marker: {missing}")

    runs: dict[int, dict[str, ArmRun]] = {}
    reference = None
    reference_hashes = None
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[1])
        runs[seed] = {}
        for arm in ARMS:
            run_dir = seed_dir / arm
            meta = json.loads((run_dir / "config.json").read_text())
            shared = {k: v for k, v in meta["config"].items() if k != "seed"}
            if reference is None:
                reference = shared
                reference_hashes = meta["hashes"]
            elif shared != reference:
                raise ValueError(f"{run_dir} uses a different configuration")
            elif meta["hashes"] != reference_hashes:
                raise ValueError(f"{run_dir} uses different hashes {meta['hashes']} != {reference_hashes}")
            children = [json.loads(line) for line in (run_dir / "children.jsonl").read_text().splitlines()]
            runs[seed][arm] = ArmRun(seed, arm, read_generations(run_dir / "generations.csv"), children, meta["config"])
    return runs


def _last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def _rate(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def seed_metrics(run: ArmRun) -> dict:
    g = run.generations
    metrics: dict = {"best_final": float(g["best_so_far"][-1])}
    if run.arm == ARM_RANDOM:
        return metrics

    diversity = g["diversity"] / g["diversity"][0]
    step = g["diff_proposal_rms"]
    born = [c for c in run.children if c["kind"] not in ("init", "random")]
    improved = [c["distance"] < c["parent_distance"] for c in born]
    shifted = [math.dist(c["final_xy"], c["parent_xy"]) >= USEFUL_SHIFT_M for c in born]
    collapsed = np.flatnonzero(g["unique"] == 1)
    step_after_start = step[1:]
    metrics.update(
        {
            "D_bar": float(np.mean(diversity[1:])),
            "D_final": float(diversity[-1]),
            "step_mean": float(np.nanmean(step_after_start)) if np.isfinite(step_after_start).any() else float("nan"),
            "step_first": float(step[1]),
            "step_final": _last_finite(step),
            "collapse_generation": int(g["generation"][collapsed[0]]) if collapsed.size else None,
            "improved_count": int(sum(improved)),
            "improve_rate": _rate(improved),
            "useful_rate": _rate([i and s for i, s in zip(improved, shifted, strict=True)]),
            "mean_gain": float(np.mean([c["parent_distance"] - c["distance"] for c in born])),
            "improve_rate_difference": _rate([i for c, i in zip(born, improved, strict=True) if c["kind"] == "difference"]),
            "improve_rate_gaussian": _rate([i for c, i in zip(born, improved, strict=True) if c["kind"] == "gaussian"]),
        }
    )
    return metrics


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if np.allclose(diffs, 0.0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(diffs, zero_method="wilcox", method="exact").pvalue)
        except ValueError:
            # exact method cannot handle ties with zero under some scipy builds; auto falls back to normal approx.
            p = float(stats.wilcoxon(diffs, zero_method="wilcox", method="auto").pvalue)
    return {
        "mean_a": float(np.mean(a)),
        "sd_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "sd_b": float(np.std(b, ddof=1)),
        "mean_diff": float(np.mean(diffs)),
        "p": p,
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def plateau_generation(
    curves: list[np.ndarray], start: int = 40, every: int = 10, window: int = 15, gain: float = 0.005
) -> int | None:
    last = min(len(curve) for curve in curves) - 1
    for generation in range(start, last + 1, every):
        if all(curve[generation - window] - curve[generation] < gain for curve in curves):
            return generation
    return None


def _metric_array(runs, arm: str, key: str) -> np.ndarray:
    return np.array([seed_metrics(runs[seed][arm])[key] for seed in sorted(runs)], dtype=float)


def statistical_tests(runs) -> dict:
    families = {
        "diversity": {k: (ARM_DIFFERENCE, ARM_MIXTURE, k) for k in ("D_bar", "D_final")},
        "step": {k: (ARM_DIFFERENCE, ARM_MIXTURE, k) for k in ("step_mean", "step_final")},
        "fitness": {
            "B_vs_A": (ARM_DIFFERENCE, ARM_MIXTURE, "best_final"),
            "A_vs_random": (ARM_RANDOM, ARM_DIFFERENCE, "best_final"),
            "B_vs_random": (ARM_RANDOM, ARM_MIXTURE, "best_final"),
        },
    }
    report = {}
    for family, contrasts in families.items():
        results = {
            name: paired(_metric_array(runs, a, key), _metric_array(runs, b, key))
            for name, (a, b, key) in contrasts.items()
        }
        adjusted = holm({name: r["p"] for name, r in results.items()})
        for name, result in results.items():
            result["p_holm"] = adjusted[name]
        report[family] = results
    return report


def _mean_curve(runs, arm: str, key: str) -> np.ndarray:
    return np.mean([runs[seed][arm].generations[key] for seed in runs], axis=0)


def go_no_go(runs, *, final_seeds: int, workers: int, deadline: date, now: datetime) -> dict:
    seeds = sorted(runs)
    needed = math.ceil(2 * len(seeds) / 3)
    config = runs[seeds[0]][ARM_DIFFERENCE].config
    plateau = plateau_generation([_mean_curve(runs, arm, "best_so_far") for arm in (ARM_DIFFERENCE, ARM_MIXTURE)])
    g_final = plateau if plateau is not None else FALLBACK_GENERATIONS

    walls = [c["wall_s"] for s in seeds for arm in ARMS for c in runs[s][arm].children if c["kind"] != "init"]
    per_seed = config["population_size"] + (config["population_size"] - 1) * g_final * len(ARMS)
    seed_batches = math.ceil(final_seeds / workers)
    hours = seed_batches * per_seed * float(np.mean(walls)) / 3600
    finish = now + timedelta(hours=hours)

    diff = {s: seed_metrics(runs[s][ARM_DIFFERENCE]) for s in seeds}
    mix = {s: seed_metrics(runs[s][ARM_MIXTURE]) for s in seeds}
    collapsed = sum(m["step_final"] < COLLAPSE_FRACTION * m["step_first"] for m in diff.values())
    alive = sum(m["step_final"] >= COLLAPSE_FRACTION * m["step_first"] for m in mix.values())
    fewest_improved = min(m["improved_count"] for m in (*diff.values(), *mix.values()))

    criteria = {
        "speed": {
            "passed": finish.date() <= deadline,
            "value": {"hours": hours, "finish": finish.isoformat(timespec="minutes"), "mean_eval_s": float(np.mean(walls))},
            "rule": f"ceil({final_seeds} seeds / {workers} workers) x {per_seed} evals per seed finish by {deadline}",
        },
        "a_collapses": {
            "passed": collapsed >= needed,
            "value": collapsed,
            "rule": f"A final step < {COLLAPSE_FRACTION:.0%} of generation-1 step in >= {needed}/{len(seeds)} seeds",
        },
        "b_stays_alive": {
            "passed": alive >= needed,
            "value": alive,
            "rule": f"B final step >= {COLLAPSE_FRACTION:.0%} of generation-1 step in >= {needed}/{len(seeds)} seeds",
        },
        "h4_measurable": {
            "passed": fewest_improved >= MIN_IMPROVED,
            "value": fewest_improved,
            "rule": f">= {MIN_IMPROVED} parent-improving children per arm per seed",
        },
    }
    return {"criteria": criteria, "g_final": g_final, "plateau_found": plateau is not None}


def _band(
    ax, runs, arm: str, key: str, normalise: bool = False, log: bool = False, bounds: tuple[float, float] | None = None
) -> None:
    curves = np.array([runs[s][arm].generations[key] for s in sorted(runs)])
    if normalise:
        curves = curves / curves[:, :1]
    # generation 0 has no differential step yet, so diff_proposal_rms is nan there for every
    # seed; nanmean/nanstd warn on that all-nan column, which is expected, not a bug.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        mean = np.nanmean(curves, axis=0)
        sd = np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    x = runs[sorted(runs)[0]][arm].generations["generation"]
    lower = np.clip(mean - sd, 1e-12, None) if log else mean - sd
    upper = mean + sd
    if bounds is not None:
        lower, upper = np.clip(lower, *bounds), np.clip(upper, *bounds)
    ax.plot(x, mean, color=COLORS[arm], label=LABELS[arm])
    ax.fill_between(x, lower, upper, color=COLORS[arm], alpha=0.2, linewidth=0)
    if log:
        ax.set_yscale("log")


def _step_panel(ax, runs, arm: str) -> float:
    seeds = sorted(runs)
    curves = np.array([runs[s][arm].generations["diff_proposal_rms"] for s in seeds])
    x = runs[seeds[0]][arm].generations["generation"]
    positive = np.where(curves > 0, curves, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN (slice|axis) encountered")
        median = np.nanmedian(positive, axis=0)
        q1 = np.nanpercentile(positive, 25, axis=0)
        q3 = np.nanpercentile(positive, 75, axis=0)
    ax.plot(x, median, color=COLORS[arm], label=LABELS[arm])
    ax.fill_between(x, q1, q3, color=COLORS[arm], alpha=0.2, linewidth=0)
    return float(median[1]) if median.size > 1 else float("nan")


def _collapse_generations(runs, arm: str) -> list[int]:
    collapses = []
    for seed in sorted(runs):
        generations = runs[seed][arm].generations
        zero = np.flatnonzero(generations["diff_proposal_rms"] == 0)
        if zero.size:
            collapses.append(int(generations["generation"][zero[0]]))
    return collapses


def paired_seed_rows(runs) -> list[dict]:
    rows = []
    for seed in sorted(runs):
        a = seed_metrics(runs[seed][ARM_DIFFERENCE])
        b = seed_metrics(runs[seed][ARM_MIXTURE])
        rows.append(
            {
                "seed": seed,
                "best_difference": a["best_final"],
                "best_mixture": b["best_final"],
                "best_random": seed_metrics(runs[seed][ARM_RANDOM])["best_final"],
                "improve_a": a["improve_rate"],
                "improve_b_difference": b["improve_rate_difference"],
                "improve_b_gaussian": b["improve_rate_gaussian"],
            }
        )
    return rows


def _step_panel_ylim_bottom(gen1_medians: list[float]) -> float:
    finite = [m for m in gen1_medians if math.isfinite(m) and m > 0]
    baseline = max(finite) if finite else 1e-2
    return 1e-4 * baseline


def figures(runs, out: Path) -> None:
    ea_arms = (ARM_DIFFERENCE, ARM_MIXTURE)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for arm in ea_arms:
        _band(left, runs, arm, "diversity", normalise=True)
    for arm in ARMS:
        _band(right, runs, arm, "best_so_far")
    left.set(xlabel="generation", ylabel="genotype diversity / initial")
    right.set(xlabel="generation", ylabel="best distance to target (m)")
    left.legend(frameon=False, fontsize=7)
    right.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig1.{suffix}", dpi=200)
    plt.close(fig)

    population = runs[sorted(runs)[0]][ARM_DIFFERENCE].config["population_size"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    gen1_medians = []
    for arm in ea_arms:
        gen1_medians.append(_step_panel(left, runs, arm))
        _band(right, runs, arm, "unique", bounds=(1, population))
    bottom = _step_panel_ylim_bottom(gen1_medians)
    left.set_yscale("log")
    left.set_ylim(bottom=bottom)
    collapses = _collapse_generations(runs, ARM_DIFFERENCE)
    if collapses:
        left.plot(collapses, [bottom * 1.6] * len(collapses), linestyle="none", marker="|", markersize=9,
                  color=COLORS[ARM_DIFFERENCE], label="A: seed collapsed (step = 0)")
    left.set(xlabel="generation", ylabel="differential step RMS")
    left.legend(frameon=False, fontsize=7, loc="upper right")
    right.set(xlabel="generation", ylabel="unique genomes", ylim=(0, population + 0.5))
    right.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    right.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig2.{suffix}", dpi=200)
    plt.close(fig)

    rows = paired_seed_rows(runs)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    positions = {ARM_DIFFERENCE: 0, ARM_MIXTURE: 1, ARM_RANDOM: 2}
    for row in rows:
        values = [row[f"best_{arm}"] for arm in ARMS]
        left.plot(list(positions.values()), values, color="#bbbbbb", linewidth=0.8, zorder=1)
    for arm, position in positions.items():
        values = [row[f"best_{arm}"] for row in rows]
        left.scatter([position] * len(values), values, color=COLORS[arm], s=14, zorder=2)
        left.hlines(np.mean(values), position - 0.25, position + 0.25, color=COLORS[arm], linewidth=2, zorder=3)
    left.set_xticks(list(positions.values()), ["A", "B", "Random"])
    left.set(ylabel="final best distance (m)", xlim=(-0.5, 2.5))

    groups = [
        ("A\nchildren", "improve_a", ARM_DIFFERENCE),
        ("B difference\nchildren", "improve_b_difference", ARM_MIXTURE),
        ("B Gaussian\nchildren", "improve_b_gaussian", ARM_MIXTURE),
    ]
    jitter = np.linspace(-0.12, 0.12, len(rows)) if len(rows) > 1 else np.zeros(1)
    for position, (_, key, arm) in enumerate(groups):
        values = np.array([row[key] for row in rows if row[key] is not None], dtype=float)
        if values.size:
            right.scatter(position + jitter[: values.size], values, color=COLORS[arm], s=14, zorder=2)
            right.hlines(values.mean(), position - 0.25, position + 0.25, color=COLORS[arm], linewidth=2, zorder=3)
    right.set_xticks(range(len(groups)), [label for label, _, _ in groups])
    right.set(ylabel="children beating their parent", xlim=(-0.5, len(groups) - 0.5))
    right.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig3.{suffix}", dpi=200)
    plt.close(fig)


H4_SUMMARY_KEYS = ("improve_rate_difference", "improve_rate_gaussian", "improve_rate", "useful_rate", "mean_gain")


def h4_summary(per_seed: list[dict]) -> list[dict]:
    rows = []
    for arm in (ARM_DIFFERENCE, ARM_MIXTURE):
        arm_rows = [row for row in per_seed if row["arm"] == arm]
        summary = {"arm": arm}
        for key in H4_SUMMARY_KEYS:
            values = np.array([row[key] for row in arm_rows if row[key] is not None], dtype=float)
            summary[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary[f"{key}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        rows.append(summary)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a mutation A/B results root")
    parser.add_argument("root", type=Path)
    parser.add_argument("--poc", action="store_true", help="also evaluate the POC go/no-go criteria")
    parser.add_argument("--final-seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--deadline", type=date.fromisoformat, default=date(2026, 10, 8))
    parser.add_argument("--exclude", default="", help="comma-separated seeds to leave out, e.g. a seed with a FAILED arm")
    args = parser.parse_args(argv)

    exclude = {int(seed) for seed in args.exclude.split(",") if seed.strip()}
    runs = load_runs(args.root, exclude=exclude)
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)

    per_seed = [
        {"seed": seed, "arm": arm, **seed_metrics(runs[seed][arm])} for seed in sorted(runs) for arm in ARMS
    ]
    write_csv(out / "seed_metrics.csv", per_seed)
    h4 = [
        {"seed": row["seed"], "arm": row["arm"], **{k: row[k] for k in ("improve_rate_difference", "improve_rate_gaussian", "improved_count", "useful_rate", "mean_gain")}}
        for row in per_seed
        if row["arm"] != ARM_RANDOM
    ]
    write_csv(out / "h4.csv", h4)
    report = {"seeds": sorted(runs), "excluded_seeds": sorted(exclude), "tests": statistical_tests(runs)}
    if args.poc:
        report["go_no_go"] = go_no_go(
            runs, final_seeds=args.final_seeds, workers=args.workers, deadline=args.deadline, now=datetime.now()
        )
        report["exploratory"] = True
        report["exploratory_note"] = "POC p-values are exploratory: small sample, calibration run, not confirmatory evidence."
        write_csv(out / "h4_summary.csv", h4_summary(per_seed))
    (out / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
    figures(runs, out)

    print(json.dumps(report["tests"], indent=2))
    if args.poc:
        for name, criterion in report["go_no_go"]["criteria"].items():
            verdict = "PASS" if criterion["passed"] else "FAIL"
            print(f"{verdict}  {name}: {criterion['value']}  ({criterion['rule']})")
        g = report["go_no_go"]
        print(f"G_final = {g['g_final']} ({'plateau' if g['plateau_found'] else 'no plateau by POC end; use cap'})")
        print(f"NOTE: {report['exploratory_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
