"""Analysis of the step size x step direction experiment (see PROTOCOL.md).

    uv run python -m mutation_ab.analysis mutation_ab/results/final_spider

Writes the tables, figures and report.txt to <root>/analysis/, or to --out.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from mutation_ab.config import (
    ARM_DE,
    ARM_DE_MATCHED,
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_MIXTURE,
    ARM_NORMALISED,
    ARM_RANDOM,
    ARM_SIZE_MATCHED,
    FACTORIAL_ARMS,
)
from mutation_ab.records import read_generations

REFERENCES = (ARM_MIXTURE, ARM_DE, ARM_DE_MATCHED)
# Order in tables and figures: shrinking-size cells, fixed-size cells, references, baseline.
ARM_ORDER = (ARM_DIFFERENCE, ARM_SIZE_MATCHED, ARM_NORMALISED, ARM_GAUSSIAN, *REFERENCES, ARM_RANDOM)
SHORT = {
    ARM_DIFFERENCE: "A: difference", ARM_NORMALISED: "B: normalised", ARM_SIZE_MATCHED: "C: size-matched",
    ARM_GAUSSIAN: "D: Gaussian", ARM_MIXTURE: "Mixture", ARM_DE: "DE (textbook)",
    ARM_DE_MATCHED: "DE (EA's F, Cr)", ARM_RANDOM: "Random search",
}
# Colour-blind-safe palette; dashed lines = shrinking step size, solid = fixed.
COLORS = {
    ARM_DIFFERENCE: "#2a78d6", ARM_SIZE_MATCHED: "#eb6834", ARM_NORMALISED: "#1baf7a", ARM_GAUSSIAN: "#eda100",
    ARM_MIXTURE: "#e87ba4", ARM_DE: "#008300", ARM_DE_MATCHED: "#4a3aa7", ARM_RANDOM: "#52514e",
}
STYLES = {
    ARM_DIFFERENCE: "--", ARM_SIZE_MATCHED: "--", ARM_NORMALISED: "-", ARM_GAUSSIAN: "-",
    ARM_MIXTURE: ":", ARM_DE: "-.", ARM_DE_MATCHED: "-.", ARM_RANDOM: "-",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BOOTSTRAP = 20_000


@dataclass(frozen=True)
class Run:
    seed: int
    arm: str
    directory: Path
    generations: dict[str, np.ndarray]
    children: list[dict]
    config: dict


def load(root: Path) -> dict[str, dict[int, Run]]:
    """arm -> seed -> Run for every completed run. Runs must share one setup and one seed set."""
    runs: dict[str, dict[int, Run]] = {}
    setups: dict[str, list[str]] = {}
    for seed_dir in sorted(Path(root).glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        seed = int(seed_dir.name.split("_")[1])
        for arm_dir in sorted(seed_dir.iterdir()):
            if not (arm_dir / "COMPLETE").exists():
                continue
            meta = json.loads((arm_dir / "config.json").read_text())
            setup = json.dumps({"config": {k: v for k, v in meta["config"].items() if k != "seed"},
                                "hashes": meta["hashes"]}, sort_keys=True)
            setups.setdefault(setup, []).append(str(arm_dir))
            children = [json.loads(line) for line in (arm_dir / "children.jsonl").read_text().splitlines()]
            runs.setdefault(arm_dir.name, {})[seed] = Run(
                seed, arm_dir.name, arm_dir, read_generations(arm_dir / "generations.csv"), children, meta["config"]
            )
    if len(setups) > 1:
        examples = [dirs[0] for dirs in setups.values()]
        raise ValueError(f"runs under {root} use {len(setups)} different setups, e.g. {examples}")
    seed_sets = {arm: tuple(sorted(by_seed)) for arm, by_seed in runs.items()}
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"arms cover different seeds: {seed_sets}")
    return {arm: runs[arm] for arm in ARM_ORDER if arm in runs}


def seed_metrics(run: Run) -> dict:
    g = run.generations
    bred = [c for c in run.children if c["kind"] not in ("init", "random")]
    metrics = {
        "seed": run.seed,
        "arm": run.arm,
        "best_final": float(g["best_so_far"][-1]),
        "evaluations": int(g["evaluations"][-1]),
    }
    if run.arm == ARM_RANDOM:
        return metrics
    single = np.flatnonzero(g["unique"] == 1)
    clones = np.array([c["change_rms"] == 0 for c in bred])
    better = np.array([c["distance"] < c["parent_distance"] for c in bred])
    metrics |= {
        "collapse_generation": int(g["generation"][single[0]]) if single.size else None,
        "unique_final": int(g["unique"][-1]),
        "clone_share": float(clones.mean()),
        "success_rate": float(better.mean()),
        "success_rate_excluding_clones": float(better[~clones].mean()) if (~clones).any() else None,
        "improving_children": int(better.sum()),
    }
    return metrics


def finals(runs, arm: str) -> np.ndarray:
    return np.array([runs[arm][seed].generations["best_so_far"][-1] for seed in sorted(runs[arm])])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(BOOTSTRAP, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


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


def signed_test(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Exact two-sided Wilcoxon signed-rank against zero, with a bootstrap CI of the mean."""
    p = float(wilcoxon(values, method="exact").pvalue) if np.any(values != 0) else 1.0
    low, high = bootstrap_ci(values, rng)
    return {"mean": float(values.mean()), "ci_low": low, "ci_high": high, "positive": int(np.sum(values > 0)),
            "n": len(values), "p": p}


def statistical_tests(runs) -> list[dict]:
    """Holm within each family; reference comparisons unadjusted. Positive = first-named is worse."""
    rng = np.random.default_rng(0)
    rows: list[dict] = []

    def family(name: str, tests: dict[str, np.ndarray], adjust: bool) -> None:
        results = {label: signed_test(values, rng) for label, values in tests.items()}
        adjusted = holm({label: r["p"] for label, r in results.items()}) if adjust else {}
        for label, result in results.items():
            rows.append({"family": name, "test": label, **result, "p_holm": adjusted.get(label)})

    if all(arm in runs for arm in FACTORIAL_ARMS):
        a, b, c, d = (finals(runs, arm) for arm in (ARM_DIFFERENCE, ARM_NORMALISED, ARM_SIZE_MATCHED, ARM_GAUSSIAN))
        family("factorial", {
            "size: shrinking - fixed = (A+C)/2 - (B+D)/2": (a + c) / 2 - (b + d) / 2,
            "direction: population - random = (A+B)/2 - (C+D)/2": (a + b) / 2 - (c + d) / 2,
            "interaction: (A-C) - (B-D)": (a - c) - (b - d),
        }, adjust=True)
        family("direction within size", {
            "shrinking size: A - C": a - c,
            "fixed size: B - D": b - d,
        }, adjust=True)
    if ARM_RANDOM in runs:
        random = finals(runs, ARM_RANDOM)
        family("vs random search", {
            f"{SHORT[arm]} - random": finals(runs, arm) - random for arm in FACTORIAL_ARMS if arm in runs
        }, adjust=True)
    pairs = [(ARM_MIXTURE, ARM_DIFFERENCE), (ARM_MIXTURE, ARM_RANDOM), (ARM_MIXTURE, ARM_GAUSSIAN),
             (ARM_DE, ARM_GAUSSIAN), (ARM_DE, ARM_RANDOM), (ARM_DE_MATCHED, ARM_RANDOM),
             (ARM_DE, ARM_DE_MATCHED)]
    family("references (unadjusted)", {
        f"{SHORT[x]} - {SHORT[y]}": finals(runs, x) - finals(runs, y) for x, y in pairs if x in runs and y in runs
    }, adjust=False)
    return rows


def mean_curve(runs, arm: str, key: str = "best_so_far") -> np.ndarray:
    return np.mean([run.generations[key] for run in runs[arm].values()], axis=0)


def plateaus(runs) -> list[dict]:
    """PROTOCOL.md rule (plateau_generation) on each arm's mean best-so-far curve."""
    rows = []
    for arm in runs:
        curve = mean_curve(runs, arm)
        generation = plateau_generation([curve])
        evaluations = mean_curve(runs, arm, "evaluations")
        rows.append({
            "arm": arm,
            "plateau_generation": generation,
            "plateau_evaluations": int(evaluations[generation]) if generation is not None else None,
            "gain_last_50_generations": float(curve[-min(51, len(curve))] - curve[-1]),
            "gain_last_100_generations": float(curve[-min(101, len(curve))] - curve[-1]),
        })
    return rows


def in_span_fraction(adults: np.ndarray, generations: np.ndarray, deltas: np.ndarray) -> dict[int, tuple[float, int]]:
    """Per generation: mean share of step length inside the parents' span, and the span's rank."""
    out: dict[int, tuple[float, int]] = {}
    for generation in np.unique(generations):
        parents = adults[generation - 1]
        centred = parents - parents.mean(axis=0)
        _, singular, basis = np.linalg.svd(centred, full_matrices=False)
        keep = singular > 1e-9 * max(float(singular.max(initial=0.0)), 1.0)
        steps = deltas[generations == generation].astype(float)
        norms = np.einsum("ij,ij->i", steps, steps)
        steps, norms = steps[norms > 0], norms[norms > 0]
        if not keep.any() or not len(steps):
            continue
        projected = steps @ basis[keep].T
        out[int(generation)] = (float(np.mean(np.einsum("ij,ij->i", projected, projected) / norms)), int(keep.sum()))
    return out


def step_span(runs) -> list[dict]:
    rows = []
    for arm in runs:
        for run in runs[arm].values():
            steps_file, adults_file = run.directory / "steps.npz", run.directory / "adults.npz"
            if not (steps_file.exists() and adults_file.exists()):
                continue
            steps, adults = np.load(steps_file), np.load(adults_file)["adults"]
            length = adults.shape[2]
            for generation, (share, rank) in in_span_fraction(adults, steps["generation"], steps["delta"]).items():
                rows.append({"arm": arm, "seed": run.seed, "generation": generation, "in_span_share": share,
                             "span_rank": rank, "isotropic_share": rank / length})
    return rows


def step_shape(runs) -> list[dict]:
    """Zero-step share, weights changed per step and change per changed weight, per arm."""
    rows = []
    for arm in runs:
        deltas = [np.load(run.directory / "steps.npz")["delta"].astype(float)
                  for run in runs[arm].values() if (run.directory / "steps.npz").exists()]
        if not deltas:
            continue
        steps = np.concatenate(deltas)
        changed = steps != 0
        counts = changed.sum(axis=1)
        moving = counts > 0
        per_changed = np.sqrt((steps[moving] ** 2).sum(axis=1) / counts[moving])
        rows.append({
            "arm": arm,
            "steps": len(steps),
            "zero_step_share": float(np.mean(~moving)),
            "weights_changed_median": float(np.median(counts[moving])) if moving.any() else 0.0,
            "step_rms_median": float(np.median(np.sqrt(np.mean(steps[moving] ** 2, axis=1)))) if moving.any() else 0.0,
            "change_per_changed_weight_median": float(np.median(per_changed)) if moving.any() else 0.0,
        })
    return rows


# --- Figures ---------------------------------------------------------------- #
def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def equivalent_generations(evaluations: np.ndarray, population_size: int) -> np.ndarray:
    """The EA generation that has used this many evaluations (used to place canonical DE)."""
    return (np.asarray(evaluations, dtype=float) - population_size) / (population_size - 1)


def _band(ax, runs, arm: str, key: str) -> None:
    values = np.array([run.generations[key] for run in runs[arm].values()])
    population = next(iter(runs[arm].values())).config["population_size"]
    x = equivalent_generations(mean_curve(runs, arm, "evaluations"), population)
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[arm], alpha=0.12, linewidth=0)
    ax.plot(x, mean, color=COLORS[arm], linestyle=STYLES[arm], linewidth=2, label=SHORT[arm])


def _shared_legend(fig, axes, ncol: int) -> None:
    """One legend row under the panels, so no legend covers data."""
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False, fontsize=8)


def _save(fig, out: Path, name: str) -> None:
    # No timestamps, so unchanged results give byte-identical files in git.
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fig_fitness(runs, out: Path) -> None:
    """Best-so-far distance per generation, mean ± sd over seeds; canonical DE at equal evaluations."""
    population = next(iter(next(iter(runs.values())).values())).config["population_size"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    panels = [(FACTORIAL_ARMS, "(a) Step size × step direction"), ((ARM_GAUSSIAN, *REFERENCES), "(b) References")]
    for ax, (arms, title) in zip(axes, panels, strict=True):
        for arm in (*arms, ARM_RANDOM):
            if arm in runs:
                _band(ax, runs, arm, "best_so_far")
        _style(ax, "generation", "best distance to target (m)" if ax is axes[0] else "")
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=22)
        top = ax.secondary_xaxis(
            "top",
            functions=(lambda g: (population + (population - 1) * g) / 1000,
                       lambda e: (e * 1000 - population) / (population - 1)),
        )
        top.set_xlabel("evaluations (thousands)", color=MUTED, fontsize=8)
        top.tick_params(colors=MUTED, labelsize=7)
        top.spines["top"].set_color(MUTED)
    fig.text(0.5, -0.13, "Canonical DE is plotted at the EA generation with the same number of evaluations.",
             ha="center", fontsize=7.5, color=MUTED)
    _shared_legend(fig, axes, ncol=4)
    _save(fig, out, "fig_fitness")


def fig_mechanism(runs, span_rows: list[dict], out: Path) -> None:
    """(a) population difference size, (b) distinct genomes, (c) share of steps inside the population span."""
    ea_arms = [arm for arm in (*FACTORIAL_ARMS, ARM_MIXTURE, ARM_DE, ARM_DE_MATCHED) if arm in runs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    floor = 1e-6
    for arm in ea_arms:
        # Generation 0 is the initial population: no steps yet, so its difference size is NaN.
        values = np.array([run.generations["difference_rms"][1:] for run in runs[arm].values()])
        generation = mean_curve(runs, arm, "generation")
        axes[0].plot(generation[1:], np.clip(np.nanmedian(values, axis=0), floor, None),
                     color=COLORS[arm], linestyle=STYLES[arm], linewidth=2, label=SHORT[arm])
        axes[1].plot(generation, mean_curve(runs, arm, "unique"),
                     color=COLORS[arm], linestyle=STYLES[arm], linewidth=2, label=SHORT[arm])
    axes[0].set_yscale("log")
    _style(axes[0], "generation", f"RMS of F(b − c), median (floor {floor:g})")
    axes[0].set_title("(a) Population difference size", loc="left", fontsize=10, color=INK)
    population = next(iter(runs[ea_arms[0]].values())).config["population_size"]
    _style(axes[1], "generation", f"distinct genomes (of {population}), mean")
    axes[1].set_title("(b) Collapse", loc="left", fontsize=10, color=INK)

    for arm in ea_arms:
        rows = [r for r in span_rows if r["arm"] == arm]
        if not rows:
            continue
        generations = sorted({r["generation"] for r in rows})
        share = [np.mean([r["in_span_share"] for r in rows if r["generation"] == g]) for g in generations]
        axes[2].plot(generations, share, color=COLORS[arm], linestyle=STYLES[arm], linewidth=2, label=SHORT[arm])
    iso = [r["isotropic_share"] for r in span_rows if r["arm"] == ARM_GAUSSIAN]
    if iso:
        axes[2].axhline(float(np.mean(iso)), color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
        axes[2].annotate("isotropic expectation", (0.02, float(np.mean(iso))), xycoords=("axes fraction", "data"),
                         xytext=(0, 4), textcoords="offset points", fontsize=7.5, color=MUTED)
    _style(axes[2], "generation", "share of step inside population span")
    axes[2].set_title("(c) Step direction", loc="left", fontsize=10, color=INK)
    _shared_legend(fig, axes, ncol=7)
    _save(fig, out, "fig_mechanism")


def fig_seeds(runs, out: Path) -> None:
    """Final best distance per seed for the 2x2 and random search; grey lines join one seed."""
    arms = [arm for arm in (ARM_DIFFERENCE, ARM_SIZE_MATCHED, ARM_NORMALISED, ARM_GAUSSIAN, ARM_RANDOM) if arm in runs]
    values = np.array([finals(runs, arm) for arm in arms])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for column in values.T:
        ax.plot(range(len(arms)), column, color=GRID, linewidth=1, zorder=1)
    for i, arm in enumerate(arms):
        ax.scatter(np.full(values.shape[1], i), values[i], s=36, color=COLORS[arm],
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.hlines(values[i].mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.annotate(f"{values[i].mean():.2f}", (i + 0.28, values[i].mean()), fontsize=8, color=INK, va="center")
    ax.set_xticks(range(len(arms)), [SHORT[arm].replace(": ", ":\n") for arm in arms], fontsize=8)
    _style(ax, "", "final best distance to target (m)")
    ax.set_title("Final distance per seed (bar = mean)", loc="left", fontsize=10, color=INK)
    _save(fig, out, "fig_seeds")


# --- Tables and report -------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(per_seed: list[dict]) -> list[dict]:
    rows = []
    for arm in ARM_ORDER:
        mine = [r for r in per_seed if r["arm"] == arm]
        if not mine:
            continue
        best = np.array([r["best_final"] for r in mine])
        row = {"arm": arm, "label": SHORT[arm], "n": len(mine), "best_mean": float(best.mean()),
               "best_sd": float(best.std(ddof=1)), "best_median": float(np.median(best)),
               "best_min": float(best.min()), "best_max": float(best.max()),
               "evaluations": mine[0]["evaluations"]}
        if arm != ARM_RANDOM:
            collapsed = [r["collapse_generation"] for r in mine if r["collapse_generation"] is not None]
            excluding = [r["success_rate_excluding_clones"] for r in mine if r["success_rate_excluding_clones"] is not None]
            row |= {
                "collapsed_seeds": len(collapsed),
                "collapse_generation_range": f"{min(collapsed)}-{max(collapsed)}" if collapsed else "",
                "clone_share_mean": float(np.mean([r["clone_share"] for r in mine])),
                "success_rate_mean": float(np.mean([r["success_rate"] for r in mine])),
                "success_rate_excluding_clones_mean": float(np.mean(excluding)) if excluding else None,
            }
        rows.append(row)
    return rows


def report(summary: list[dict], stats: list[dict], plateau: list[dict], span: list[dict], shape: list[dict]) -> str:
    lines = ["Final best distance to target (m), mean ± sd over seeds"]
    for r in summary:
        extra = (f"  collapsed {r['collapsed_seeds']}/{r['n']} {r['collapse_generation_range']}"
                 f"  clones {r['clone_share_mean']:.1%}  success excl. clones "
                 f"{r['success_rate_excluding_clones_mean'] or 0:.1%}") if "clone_share_mean" in r else ""
        lines.append(f"  {SHORT[r['arm']]:20s} {r['best_mean']:.3f} ± {r['best_sd']:.3f}  "
                     f"min {r['best_min']:.3f}{extra}")
    lines.append("\nTests (positive mean = first-named worse; exact two-sided Wilcoxon; 95% bootstrap CI)")
    for r in stats:
        holm_text = f"  Holm p={r['p_holm']:.4f}" if r["p_holm"] is not None else ""
        lines.append(f"  [{r['family']}] {r['test']}: {r['mean']:+.3f} m [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                     f"  positive {r['positive']}/{r['n']}  p={r['p']:.4f}{holm_text}")
    lines.append("\nPlateau (PROTOCOL.md rule on the mean curve)")
    for r in plateau:
        lines.append(f"  {SHORT[r['arm']]:20s} generation {r['plateau_generation']}  "
                     f"gain last 50 gens {r['gain_last_50_generations']:.4f} m")
    if span:
        lines.append("\nShare of step length inside the population span (mean over logged generations and seeds)")
        for arm in dict.fromkeys(r["arm"] for r in span):
            mine = [r for r in span if r["arm"] == arm]
            lines.append(f"  {SHORT[arm]:20s} {np.mean([r['in_span_share'] for r in mine]):.3f}  "
                         f"(isotropic {np.mean([r['isotropic_share'] for r in mine]):.3f}, n={len(mine)})")
    if shape:
        lines.append("\nRealised steps (medians over non-zero logged steps)")
        for r in shape:
            lines.append(f"  {SHORT[r['arm']]:20s} zero steps {r['zero_step_share']:.1%}  "
                         f"weights changed {r['weights_changed_median']:.0f}  step RMS {r['step_rms_median']:.4f}  "
                         f"per changed weight {r['change_per_changed_weight_median']:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step size x step direction analysis")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, help="output folder (default: <root>/analysis)")
    args = parser.parse_args(argv)
    runs = load(args.root)
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    per_seed = [seed_metrics(run) for arm in runs for run in runs[arm].values()]
    summary = summary_rows(per_seed)
    stats = statistical_tests(runs)
    plateau = plateaus(runs)
    span = step_span(runs)
    shape = step_shape(runs)
    for name, rows in (("per_seed", per_seed), ("summary", summary), ("stats", stats),
                       ("plateau", plateau), ("step_span", span), ("step_shape", shape)):
        write_csv(out / f"{name}.csv", rows)

    fig_fitness(runs, out)
    fig_mechanism(runs, span, out)
    fig_seeds(runs, out)
    text = report(summary, stats, plateau, span, shape)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
