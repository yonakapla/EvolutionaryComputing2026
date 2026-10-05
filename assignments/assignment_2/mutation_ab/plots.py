"""The three report figures, drawn at their printed size in the GECCO layout."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from mutation_ab.conditions import (
    ARM_ORDER,
    COLORS,
    FIXED_SIZE,
    LIGHT,
    MUTED,
    NAMES,
    REFERENCES,
    SHRINKING_SIZE,
    STYLES,
)
from mutation_ab.config import (
    ARM_DE,
    ARM_DE_MATCHED,
    ARM_GAUSSIAN,
    ARM_MIXTURE,
    ARM_RANDOM,
    FACTORIAL_ARMS,
)
from mutation_ab.records import finals, mean_curve

# Widths of the sigconf page, so that fonts print at their nominal size.
TEXT_WIDTH, COLUMN_WIDTH = 7.0, 3.33
FIGURE_STYLE = {
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
}


def equivalent_generations(evaluations: np.ndarray, population_size: int) -> np.ndarray:
    """The EA generation that has used this many evaluations.

    Canonical DE spends one more evaluation per generation than the EA arms, so
    it is plotted at the EA generation with the same number of evaluations.
    """
    return (np.asarray(evaluations, dtype=float) - population_size) / (
        population_size - 1
    )


def evaluations_at(generations: np.ndarray, population_size: int) -> np.ndarray:
    return population_size + (population_size - 1) * np.asarray(
        generations, dtype=float
    )


def _population(runs, arm: str) -> int:
    return next(iter(runs[arm].values())).config["population_size"]


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=4)


def _band(ax, runs, arm: str) -> None:
    """Mean ± sd of best-so-far over seeds."""
    values = np.array([run.generations["best_so_far"] for run in runs[arm].values()])
    x = equivalent_generations(
        mean_curve(runs, arm, "evaluations"), _population(runs, arm)
    )
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[arm], alpha=0.10, linewidth=0)
    ax.plot(x, mean, color=COLORS[arm], linestyle=STYLES[arm], label=NAMES[arm])


def _legend_below(fig, axes, ncol: int) -> None:
    """One legend row under the panels, so that no legend covers data."""
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    labels = [NAMES[arm] for arm in ARM_ORDER if NAMES[arm] in handles]
    fig.legend(
        [handles[label] for label in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(ncol, len(labels)),
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.0,
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _save(fig, out: Path, name: str) -> None:
    # No timestamps, so unchanged results give byte-identical files in git.
    fig.savefig(
        out / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={"Software": None},
    )
    fig.savefig(
        out / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={"CreationDate": None},
    )
    plt.close(fig)


def fig_fitness(runs, out: Path) -> None:
    """Best-so-far distance per generation, mean ± sd over seeds."""
    ea_arm = next(arm for arm in runs if arm not in (ARM_DE, ARM_DE_MATCHED))
    population = _population(runs, ea_arm)
    panels = [
        (
            tuple(a for a in ARM_ORDER if a in FACTORIAL_ARMS and a in runs),
            "(a) Step size × step direction",
        )
    ]
    references = tuple(a for a in (ARM_GAUSSIAN, *REFERENCES) if a in runs)
    if any(a in runs for a in REFERENCES):
        panels.append((references, "(b) References"))
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(TEXT_WIDTH if len(panels) > 1 else COLUMN_WIDTH, 2.9),
            sharey=True,
            squeeze=False,
        )
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
                functions=(
                    lambda g: evaluations_at(g, population) / 1000,
                    lambda e: equivalent_generations(e * 1000, population),
                ),
            )
            top.set_xlabel("evaluations (thousands)", color=MUTED, labelpad=2)
            top.tick_params(colors=MUTED, labelsize=6.5)
            _title(ax, text)
        axes[0].set_ylabel("best distance to target (m)")
        fig.tight_layout(w_pad=1.5)
        _legend_below(fig, axes, ncol=8)
        _save(fig, out, "fig_fitness")


def fig_mechanism(runs, span_rows: list[dict], out: Path) -> None:
    """(a) Size of F(b - c), (b) distinct genotypes, (c) share of step length
    inside the population span."""
    ea_arms = [
        arm
        for arm in (*FACTORIAL_ARMS, ARM_MIXTURE, ARM_DE, ARM_DE_MATCHED)
        if arm in runs
    ]
    population = _population(runs, ea_arms[0])
    floor = 1e-6
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(TEXT_WIDTH, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.9]}
        )
        for arm in ea_arms:
            generation = mean_curve(runs, arm, "generation")
            # Generation 0 has no steps yet, so its difference size is NaN.
            size = np.array(
                [run.generations["difference_rms"][1:] for run in runs[arm].values()]
            )
            axes[0].plot(
                generation[1:],
                np.clip(np.nanmedian(size, axis=0), floor, None),
                color=COLORS[arm],
                linestyle=STYLES[arm],
                label=NAMES[arm],
            )
            axes[1].plot(
                generation,
                _moving_average(mean_curve(runs, arm, "unique"), 9),
                color=COLORS[arm],
                linestyle=STYLES[arm],
                label=NAMES[arm],
            )
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
            ax.scatter(
                share,
                y,
                s=22,
                color="white" if arm in SHRINKING_SIZE else COLORS[arm],
                edgecolor=COLORS[arm],
                linewidth=1.2,
                zorder=3,
            )
            ax.annotate(
                f"{share:.2f}",
                (share, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=6.5,
            )
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
    """Final best distance per seed for the 2x2 and random search; grey lines
    join the runs of one seed."""
    arms = [
        arm
        for arm in ARM_ORDER
        if (arm in FACTORIAL_ARMS or arm == ARM_RANDOM) and arm in runs
    ]
    values = np.array([finals(runs, arm) for arm in arms])
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.95))
        for column in values.T:
            ax.plot(range(len(arms)), column, color=LIGHT, linewidth=0.7, zorder=1)
        for i, arm in enumerate(arms):
            ax.scatter(
                np.full(values.shape[1], i),
                values[i],
                s=14,
                color="white" if arm in SHRINKING_SIZE else COLORS[arm],
                edgecolor=COLORS[arm],
                linewidth=1.0,
                zorder=3,
            )
            ax.hlines(
                values[i].mean(),
                i - 0.22,
                i + 0.22,
                color="black",
                linewidth=1.4,
                zorder=4,
            )
            ax.annotate(
                f"{values[i].mean():.2f}",
                (i + 0.25, values[i].mean()),
                fontsize=6.5,
                va="center",
            )
        groups = {
            "shrinking size": [i for i, a in enumerate(arms) if a in SHRINKING_SIZE],
            "fixed size": [i for i, a in enumerate(arms) if a in FIXED_SIZE],
        }
        for text, positions in groups.items():
            if positions:
                ax.text(
                    np.mean(positions),
                    0.97,
                    text,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=MUTED,
                )
                ax.axvline(
                    max(positions) + 0.5, color=MUTED, linewidth=0.6, linestyle=":"
                )
        ax.set_xticks(
            range(len(arms)),
            [NAMES[arm].split()[0] if arm != ARM_RANDOM else "Random" for arm in arms],
        )
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.4, len(arms) - 0.3)
        ax.set_ylim(-0.05, values.max() * 1.18)
        ax.set_ylabel("final distance (m)")
        _save(fig, out, "fig_seeds")
