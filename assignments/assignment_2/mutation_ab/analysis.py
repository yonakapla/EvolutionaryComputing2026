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
from matplotlib.ticker import MaxNLocator
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
from mutation_ab.operators import KIND_GAUSSIAN
from mutation_ab.records import read_generations

REFERENCES = (ARM_MIXTURE, ARM_DE, ARM_DE_MATCHED)
# Order in tables and figures: shrinking-size cells, fixed-size cells, references, baseline.
ARM_ORDER = (ARM_DIFFERENCE, ARM_SIZE_MATCHED, ARM_NORMALISED, ARM_GAUSSIAN, *REFERENCES, ARM_RANDOM)
# Names as in the report's tables.
NAMES = {
    ARM_DIFFERENCE: "A difference", ARM_NORMALISED: "B normalised", ARM_SIZE_MATCHED: "C size-matched",
    ARM_GAUSSIAN: "D Gaussian", ARM_MIXTURE: "Mixture", ARM_DE: "DE (0.5, 0.9)",
    ARM_DE_MATCHED: "DE matched", ARM_RANDOM: "Random search",
}
# The 2x2 is encoded visually: colour = direction (population blue, random orange),
# line style and marker fill = size (dashed/hollow shrinking, solid/filled fixed).
POPULATION_DIRECTION, RANDOM_DIRECTION = "#1f5fbf", "#e07000"
MUTED, LIGHT = "#555555", "#d0d0d0"
COLORS = {
    ARM_DIFFERENCE: POPULATION_DIRECTION, ARM_NORMALISED: POPULATION_DIRECTION,
    ARM_SIZE_MATCHED: RANDOM_DIRECTION, ARM_GAUSSIAN: RANDOM_DIRECTION,
    ARM_MIXTURE: "#b8479a", ARM_DE: "#1a8a3a", ARM_DE_MATCHED: "#7a5cc4", ARM_RANDOM: MUTED,
}
DASHED, DASH_DOT = (0, (4, 2)), (0, (5, 1.5, 1, 1.5))
STYLES = {
    ARM_DIFFERENCE: DASHED, ARM_SIZE_MATCHED: DASHED, ARM_NORMALISED: "-", ARM_GAUSSIAN: "-",
    ARM_MIXTURE: (0, (1, 1.5)), ARM_DE: DASH_DOT, ARM_DE_MATCHED: DASH_DOT, ARM_RANDOM: "-",
}
SHRINKING_SIZE, FIXED_SIZE = (ARM_DIFFERENCE, ARM_SIZE_MATCHED), (ARM_NORMALISED, ARM_GAUSSIAN)
# Figures are drawn at their printed size in the GECCO sigconf layout, so fonts print at nominal size.
TEXT_WIDTH, COLUMN_WIDTH = 7.0, 3.33
FIGURE_STYLE = {
    "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "pdf.fonttype": 42,
}
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


SIGMA_FREE_ARMS = (ARM_DIFFERENCE, ARM_SIZE_MATCHED, ARM_RANDOM)


def borrow_sigma_free_arms(runs, reference_root: Path) -> dict[str, dict[int, Run]]:
    """Add A, C and random search from another run of the same seeds. None of them uses the Gaussian
    step size, so a run that only varies gaussian_sd can reuse them for the full 2x2 contrasts."""
    reference = load(reference_root)
    seeds = sorted(next(iter(runs.values())))
    own = next(iter(next(iter(runs.values())).values())).config
    for arm in SIGMA_FREE_ARMS:
        borrowed = {seed: reference[arm][seed] for seed in seeds}
        for run in borrowed.values():
            differing = {k for k in own if k not in ("seed", "gaussian_sd") and own[k] != run.config.get(k)}
            if differing:
                raise ValueError(f"{run.directory} differs in {sorted(differing)}")
        runs[arm] = borrowed
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
    final = np.load(run.directory / "adults.npz")["adults"][-1]
    clones = np.array([c["change_rms"] == 0 for c in bred])
    better = np.array([c["distance"] < c["parent_distance"] for c in bred])
    metrics |= {
        "collapse_generation": int(g["generation"][single[0]]) if single.size else None,
        "unique_final": int(g["unique"][-1]),
        "frozen_genes_final": int(np.all(final == final[0], axis=0).sum()),
        "gaussian_step_share": float(np.mean([c["kind"] == KIND_GAUSSIAN for c in bred])),
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
    curve: np.ndarray, start: int = 40, every: int = 10, window: int = 15, gain: float = 0.005
) -> int | None:
    for generation in range(start, len(curve), every):
        if curve[generation - window] - curve[generation] < gain:
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
    elif ARM_NORMALISED in runs and ARM_GAUSSIAN in runs:
        family("direction within size", {
            "fixed size: B - D": finals(runs, ARM_NORMALISED) - finals(runs, ARM_GAUSSIAN),
        }, adjust=True)
    if ARM_RANDOM in runs:
        random = finals(runs, ARM_RANDOM)
        family("vs random search", {
            f"{NAMES[arm]} - random": finals(runs, arm) - random for arm in FACTORIAL_ARMS if arm in runs
        }, adjust=True)
    pairs = [(ARM_MIXTURE, ARM_DIFFERENCE), (ARM_MIXTURE, ARM_RANDOM), (ARM_MIXTURE, ARM_GAUSSIAN),
             (ARM_DE, ARM_GAUSSIAN), (ARM_DE, ARM_RANDOM), (ARM_DE_MATCHED, ARM_RANDOM),
             (ARM_DE, ARM_DE_MATCHED)]
    family("references (unadjusted)", {
        f"{NAMES[x]} - {NAMES[y]}": finals(runs, x) - finals(runs, y) for x, y in pairs if x in runs and y in runs
    }, adjust=False)
    return rows


def mean_curve(runs, arm: str, key: str = "best_so_far") -> np.ndarray:
    return np.mean([run.generations[key] for run in runs[arm].values()], axis=0)


def plateaus(runs) -> list[dict]:
    """PROTOCOL.md rule (plateau_generation) on each arm's mean best-so-far curve."""
    rows = []
    for arm in runs:
        curve = mean_curve(runs, arm)
        generation = plateau_generation(curve)
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
def equivalent_generations(evaluations: np.ndarray, population_size: int) -> np.ndarray:
    """The EA generation that has used this many evaluations (used to place canonical DE)."""
    return (np.asarray(evaluations, dtype=float) - population_size) / (population_size - 1)


def evaluations_at(generations: np.ndarray, population_size: int) -> np.ndarray:
    return population_size + (population_size - 1) * np.asarray(generations, dtype=float)


def _population(runs, arm: str) -> int:
    return next(iter(runs[arm].values())).config["population_size"]


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=4)


def _band(ax, runs, arm: str) -> None:
    """Mean ± sd of best-so-far over seeds."""
    values = np.array([run.generations["best_so_far"] for run in runs[arm].values()])
    x = equivalent_generations(mean_curve(runs, arm, "evaluations"), _population(runs, arm))
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[arm], alpha=0.10, linewidth=0)
    ax.plot(x, mean, color=COLORS[arm], linestyle=STYLES[arm], label=NAMES[arm])


def _legend_below(fig, axes, ncol: int) -> None:
    """One legend row under the panels, in ARM_ORDER, so no legend covers data."""
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    labels = [NAMES[arm] for arm in ARM_ORDER if NAMES[arm] in handles]
    fig.legend([handles[label] for label in labels], labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=min(ncol, len(labels)), frameon=False, handlelength=2.2, handletextpad=0.5, columnspacing=1.0)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _save(fig, out: Path, name: str) -> None:
    # No timestamps, so unchanged results give byte-identical files in git.
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.05, metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05, metadata={"CreationDate": None})
    plt.close(fig)


def fig_fitness(runs, out: Path) -> None:
    """Best-so-far distance per generation, mean ± sd over seeds; canonical DE at equal evaluations."""
    ea_arm = next(arm for arm in runs if arm not in (ARM_DE, ARM_DE_MATCHED))
    population = _population(runs, ea_arm)
    panels = [(tuple(a for a in ARM_ORDER if a in FACTORIAL_ARMS and a in runs), "(a) Step size × step direction")]
    references = tuple(a for a in (ARM_GAUSSIAN, *REFERENCES) if a in runs)
    if any(a in runs for a in REFERENCES):
        panels.append((references, "(b) References"))
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH if len(panels) > 1 else COLUMN_WIDTH, 2.9),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (arms, text) in zip(axes, panels, strict=True):
            for arm in (ARM_RANDOM, *arms):
                if arm in runs:
                    _band(ax, runs, arm)
            ax.margins(x=0)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("generation")
            top = ax.secondary_xaxis(
                "top",
                functions=(lambda g: evaluations_at(g, population) / 1000,
                           lambda e: equivalent_generations(e * 1000, population)),
            )
            top.set_xlabel("evaluations (thousands)", color=MUTED, labelpad=2)
            top.tick_params(colors=MUTED, labelsize=6.5)
            _title(ax, text)
        axes[0].set_ylabel("best distance to target (m)")
        fig.tight_layout(w_pad=1.5)
        _legend_below(fig, axes, ncol=8)
        _save(fig, out, "fig_fitness")


def fig_mechanism(runs, span_rows: list[dict], out: Path) -> None:
    """(a) population difference size, (b) distinct genotypes, (c) share of step length inside the population span."""
    ea_arms = [arm for arm in (*FACTORIAL_ARMS, ARM_MIXTURE, ARM_DE, ARM_DE_MATCHED) if arm in runs]
    population = _population(runs, ea_arms[0])
    floor = 1e-6
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
        for arm in ea_arms:
            generation = mean_curve(runs, arm, "generation")
            # Generation 0 is the initial population: no steps yet, so its difference size is NaN.
            size = np.array([run.generations["difference_rms"][1:] for run in runs[arm].values()])
            axes[0].plot(generation[1:], np.clip(np.nanmedian(size, axis=0), floor, None),
                         color=COLORS[arm], linestyle=STYLES[arm], label=NAMES[arm])
            axes[1].plot(generation, _moving_average(mean_curve(runs, arm, "unique"), 9),
                         color=COLORS[arm], linestyle=STYLES[arm], label=NAMES[arm])
        axes[0].set_yscale("log")
        axes[0].set_ylim(floor / 2, 2)
        axes[0].set_ylabel("RMS of $F(b-c)$, median")
        axes[0].set_xlabel("generation")
        _title(axes[0], "(a) Difference-step size")
        axes[1].set_ylim(0, population + 0.8)
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axes[1].set_ylabel(f"distinct genotypes (of {population})")
        axes[1].set_xlabel("generation")
        _title(axes[1], "(b) Collapse")

        ax = axes[2]
        span_arms = [arm for arm in ea_arms if any(r["arm"] == arm for r in span_rows)]
        for y, arm in enumerate(span_arms):
            mine = [r for r in span_rows if r["arm"] == arm]
            share = float(np.mean([r["in_span_share"] for r in mine]))
            isotropic = float(np.mean([r["isotropic_share"] for r in mine]))
            ax.plot([isotropic, share], [y, y], color=LIGHT, linewidth=1, zorder=1)
            ax.scatter(isotropic, y, marker="|", s=40, color=MUTED, zorder=2)
            ax.scatter(share, y, s=22, color="white" if arm in SHRINKING_SIZE else COLORS[arm],
                       edgecolor=COLORS[arm], linewidth=1.2, zorder=3)
            ax.annotate(f"{share:.2f}", (share, y), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=6.5)
        ax.set_yticks(range(len(span_arms)), [NAMES[arm] for arm in span_arms])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("share inside population span")
        _title(ax, "(c) Step direction")
        fig.tight_layout(w_pad=1.2)
        _legend_below(fig, axes[:2], ncol=7)
        _save(fig, out, "fig_mechanism")


def fig_seeds(runs, out: Path) -> None:
    """Final best distance per seed for the 2x2 and random search; grey lines join one seed."""
    arms = [arm for arm in ARM_ORDER if (arm in FACTORIAL_ARMS or arm == ARM_RANDOM) and arm in runs]
    values = np.array([finals(runs, arm) for arm in arms])
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.95))
        for column in values.T:
            ax.plot(range(len(arms)), column, color=LIGHT, linewidth=0.7, zorder=1)
        for i, arm in enumerate(arms):
            ax.scatter(np.full(values.shape[1], i), values[i], s=14,
                       color="white" if arm in SHRINKING_SIZE else COLORS[arm],
                       edgecolor=COLORS[arm], linewidth=1.0, zorder=3)
            ax.hlines(values[i].mean(), i - 0.22, i + 0.22, color="black", linewidth=1.4, zorder=4)
            ax.annotate(f"{values[i].mean():.2f}", (i + 0.25, values[i].mean()), fontsize=6.5, va="center")
        groups = {"shrinking size": [i for i, a in enumerate(arms) if a in SHRINKING_SIZE],
                  "fixed size": [i for i, a in enumerate(arms) if a in FIXED_SIZE]}
        for text, positions in groups.items():
            if positions:
                ax.text(np.mean(positions), 0.97, text, transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=6.5, color=MUTED)
                ax.axvline(max(positions) + 0.5, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_xticks(range(len(arms)), [NAMES[arm].split()[0] if arm != ARM_RANDOM else "Random" for arm in arms])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.4, len(arms) - 0.3)
        ax.set_ylim(-0.05, values.max() * 1.18)
        ax.set_ylabel("final distance (m)")
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
        row = {"arm": arm, "label": NAMES[arm], "n": len(mine), "best_mean": float(best.mean()),
               "best_sd": float(best.std(ddof=1)), "best_median": float(np.median(best)),
               "best_min": float(best.min()), "best_max": float(best.max()),
               "evaluations": mine[0]["evaluations"]}
        if arm != ARM_RANDOM:
            collapsed = [r["collapse_generation"] for r in mine if r["collapse_generation"] is not None]
            excluding = [r["success_rate_excluding_clones"] for r in mine]
            excluding = [value for value in excluding if value is not None]
            row |= {
                "collapsed_seeds": len(collapsed),
                "collapse_generation_range": f"{min(collapsed)}-{max(collapsed)}" if collapsed else "",
                "frozen_genes_final_median": float(np.median([r["frozen_genes_final"] for r in mine])),
                "gaussian_step_share_mean": float(np.mean([r["gaussian_step_share"] for r in mine])),
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
                 f"  frozen genes {r['frozen_genes_final_median']:.0f}"
                 f"  Gaussian steps {r['gaussian_step_share_mean']:.1%}"
                 f"  clones {r['clone_share_mean']:.1%}  success excl. clones "
                 f"{r['success_rate_excluding_clones_mean'] or 0:.1%}") if "clone_share_mean" in r else ""
        lines.append(f"  {NAMES[r['arm']]:20s} {r['best_mean']:.3f} ± {r['best_sd']:.3f}  "
                     f"min {r['best_min']:.3f}{extra}")
    lines.append("\nTests (positive mean = first-named worse; exact two-sided Wilcoxon; 95% bootstrap CI)")
    for r in stats:
        holm_text = f"  Holm p={r['p_holm']:.4f}" if r["p_holm"] is not None else ""
        lines.append(f"  [{r['family']}] {r['test']}: {r['mean']:+.3f} m [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                     f"  positive {r['positive']}/{r['n']}  p={r['p']:.4f}{holm_text}")
    lines.append("\nPlateau (PROTOCOL.md rule on the mean curve)")
    for r in plateau:
        lines.append(f"  {NAMES[r['arm']]:20s} generation {r['plateau_generation']}  "
                     f"gain last 50 gens {r['gain_last_50_generations']:.4f} m")
    if span:
        lines.append("\nShare of step length inside the population span (mean over logged generations and seeds)")
        for arm in dict.fromkeys(r["arm"] for r in span):
            mine = [r for r in span if r["arm"] == arm]
            lines.append(f"  {NAMES[arm]:20s} {np.mean([r['in_span_share'] for r in mine]):.3f}  "
                         f"(isotropic {np.mean([r['isotropic_share'] for r in mine]):.3f}, n={len(mine)})")
    if shape:
        lines.append("\nRealised steps (medians over non-zero logged steps)")
        for r in shape:
            lines.append(f"  {NAMES[r['arm']]:20s} zero steps {r['zero_step_share']:.1%}  "
                         f"weights changed {r['weights_changed_median']:.0f}  step RMS {r['step_rms_median']:.4f}  "
                         f"per changed weight {r['change_per_changed_weight_median']:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step size x step direction analysis")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, help="output folder (default: <root>/analysis)")
    parser.add_argument("--sigma-free-arms-from", type=Path,
                        help="results of the same seeds to take A, C and random search from (for sigma runs)")
    args = parser.parse_args(argv)
    runs = load(args.root)
    if args.sigma_free_arms_from:
        runs = borrow_sigma_free_arms(runs, args.sigma_free_arms_from)
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
