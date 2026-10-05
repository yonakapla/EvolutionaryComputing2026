"""Makes the tables, figures and report of one experiment, using stats.py,
plots.py and conditions.py. See README.md for the design and hypotheses.

    uv run python -m mutation_ab.analysis mutation_ab/results/final_spider

Writes the tables, figures and report.txt to <root>/analysis/, or to --out.
"""

import argparse
from pathlib import Path

import numpy as np

from mutation_ab.conditions import ARM_ORDER, NAMES
from mutation_ab.config import ARM_RANDOM
from mutation_ab.operators import KIND_GAUSSIAN
from mutation_ab.plots import fig_fitness, fig_mechanism, fig_seeds
from mutation_ab.records import Run, borrow_sigma_free_arms, load, write_csv
from mutation_ab.stats import plateaus, statistical_tests


def seed_metrics(run: Run) -> dict:
    """Final result of one run and, for the EA and DE arms, how its population
    collapsed and how often its children improved on their parent."""
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
        "gaussian_step_share": float(
            np.mean([c["kind"] == KIND_GAUSSIAN for c in bred])
        ),
        "clone_share": float(clones.mean()),
        "success_rate": float(better.mean()),
        "success_rate_excluding_clones": float(better[~clones].mean())
        if (~clones).any()
        else None,
        "improving_children": int(better.sum()),
    }
    return metrics


def in_span_fraction(
    adults: np.ndarray, generations: np.ndarray, deltas: np.ndarray
) -> dict[int, tuple[float, int]]:
    """Per logged generation: the mean share of each step's squared length that lies
    in the space spanned by the parents, and the rank of that space.

    An isotropic step in n dimensions puts rank / n of its length in the span;
    a difference step F(b - c) puts all of it there.
    """
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
        out[int(generation)] = (
            float(np.mean(np.einsum("ij,ij->i", projected, projected) / norms)),
            int(keep.sum()),
        )
    return out


def step_span(runs) -> list[dict]:
    """in_span_fraction for every logged generation of every run."""
    rows = []
    for arm in runs:
        for run in runs[arm].values():
            steps_file, adults_file = (
                run.directory / "steps.npz",
                run.directory / "adults.npz",
            )
            if not (steps_file.exists() and adults_file.exists()):
                continue
            steps, adults = np.load(steps_file), np.load(adults_file)["adults"]
            length = adults.shape[2]
            for generation, (share, rank) in in_span_fraction(
                adults, steps["generation"], steps["delta"]
            ).items():
                rows.append(
                    {
                        "arm": arm,
                        "seed": run.seed,
                        "generation": generation,
                        "in_span_share": share,
                        "span_rank": rank,
                        "isotropic_share": rank / length,
                    }
                )
    return rows


def step_shape(runs) -> list[dict]:
    """Per arm: share of zero steps, and for the others, how many weights a step
    changes and by how much."""
    rows = []
    for arm in runs:
        deltas = [
            np.load(run.directory / "steps.npz")["delta"].astype(float)
            for run in runs[arm].values()
            if (run.directory / "steps.npz").exists()
        ]
        if not deltas:
            continue
        steps = np.concatenate(deltas)
        changed = steps != 0
        counts = changed.sum(axis=1)
        moving = counts > 0
        per_changed = np.sqrt((steps[moving] ** 2).sum(axis=1) / counts[moving])
        rows.append(
            {
                "arm": arm,
                "steps": len(steps),
                "zero_step_share": float(np.mean(~moving)),
                "weights_changed_median": float(np.median(counts[moving]))
                if moving.any()
                else 0.0,
                "step_rms_median": float(
                    np.median(np.sqrt(np.mean(steps[moving] ** 2, axis=1)))
                )
                if moving.any()
                else 0.0,
                "change_per_changed_weight_median": float(np.median(per_changed))
                if moving.any()
                else 0.0,
            }
        )
    return rows


def summary_rows(per_seed: list[dict]) -> list[dict]:
    """seed_metrics aggregated over the seeds of each arm."""
    rows = []
    for arm in ARM_ORDER:
        mine = [r for r in per_seed if r["arm"] == arm]
        if not mine:
            continue
        best = np.array([r["best_final"] for r in mine])
        row = {
            "arm": arm,
            "label": NAMES[arm],
            "n": len(mine),
            "best_mean": float(best.mean()),
            "best_sd": float(best.std(ddof=1)),
            "best_median": float(np.median(best)),
            "best_min": float(best.min()),
            "best_max": float(best.max()),
            "evaluations": mine[0]["evaluations"],
        }
        if arm != ARM_RANDOM:
            collapsed = [
                r["collapse_generation"]
                for r in mine
                if r["collapse_generation"] is not None
            ]
            excluding = [r["success_rate_excluding_clones"] for r in mine]
            excluding = [value for value in excluding if value is not None]
            row |= {
                "collapsed_seeds": len(collapsed),
                "collapse_generation_range": f"{min(collapsed)}-{max(collapsed)}"
                if collapsed
                else "",
                "frozen_genes_final_median": float(
                    np.median([r["frozen_genes_final"] for r in mine])
                ),
                "gaussian_step_share_mean": float(
                    np.mean([r["gaussian_step_share"] for r in mine])
                ),
                "clone_share_mean": float(np.mean([r["clone_share"] for r in mine])),
                "success_rate_mean": float(np.mean([r["success_rate"] for r in mine])),
                "success_rate_excluding_clones_mean": float(np.mean(excluding))
                if excluding
                else None,
            }
        rows.append(row)
    return rows


def report(
    summary: list[dict],
    stats: list[dict],
    plateau: list[dict],
    span: list[dict],
    shape: list[dict],
) -> str:
    lines = ["Final best distance to target (m), mean ± sd over seeds"]
    for r in summary:
        extra = ""
        if "clone_share_mean" in r:
            collapsed = (
                f"{r['collapsed_seeds']}/{r['n']} {r['collapse_generation_range']}"
            )
            success = r["success_rate_excluding_clones_mean"] or 0
            extra = (
                f"  collapsed {collapsed}"
                f"  frozen genes {r['frozen_genes_final_median']:.0f}"
                f"  Gaussian steps {r['gaussian_step_share_mean']:.1%}"
                f"  clones {r['clone_share_mean']:.1%}"
                f"  success excl. clones {success:.1%}"
            )
        lines.append(
            f"  {NAMES[r['arm']]:20s} {r['best_mean']:.3f} ± {r['best_sd']:.3f}  "
            f"min {r['best_min']:.3f}{extra}"
        )

    lines.append(
        "\nTests (positive mean = first-named worse; exact two-sided Wilcoxon; "
        "95% bootstrap CI)"
    )
    for r in stats:
        holm_text = f"  Holm p={r['p_holm']:.4f}" if r["p_holm"] is not None else ""
        interval = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        lines.append(
            f"  [{r['family']}] {r['test']}: {r['mean']:+.3f} m {interval}"
            f"  positive {r['positive']}/{r['n']}  p={r['p']:.4f}{holm_text}"
        )

    lines.append("\nPlateau (rule fixed before the final run, on the mean curve)")
    for r in plateau:
        lines.append(
            f"  {NAMES[r['arm']]:20s} generation {r['plateau_generation']}  "
            f"gain last 50 gens {r['gain_last_50_generations']:.4f} m"
        )

    if span:
        lines.append(
            "\nShare of step length inside the population span "
            "(mean over logged generations and seeds)"
        )
        for arm in dict.fromkeys(r["arm"] for r in span):
            mine = [r for r in span if r["arm"] == arm]
            share = np.mean([r["in_span_share"] for r in mine])
            isotropic = np.mean([r["isotropic_share"] for r in mine])
            lines.append(
                f"  {NAMES[arm]:20s} {share:.3f}  "
                f"(isotropic {isotropic:.3f}, n={len(mine)})"
            )

    if shape:
        lines.append("\nRealised steps (medians over non-zero logged steps)")
        for r in shape:
            lines.append(
                f"  {NAMES[r['arm']]:20s} zero steps {r['zero_step_share']:.1%}  "
                f"weights changed {r['weights_changed_median']:.0f}  "
                f"step RMS {r['step_rms_median']:.4f}  "
                f"per changed weight {r['change_per_changed_weight_median']:.3f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step size x step direction analysis")
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--out", type=Path, help="output folder (default: <root>/analysis)"
    )
    parser.add_argument(
        "--sigma-free-arms-from",
        type=Path,
        help="results of the same seeds to take A, C and random search from "
        "(for the sigma runs)",
    )
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
    for name, rows in (
        ("per_seed", per_seed),
        ("summary", summary),
        ("stats", stats),
        ("plateau", plateau),
        ("step_span", span),
        ("step_shape", shape),
    ):
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
