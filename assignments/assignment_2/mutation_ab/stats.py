"""The hypothesis tests and the plateau rule, both fixed before the final run.

Every test is paired by seed: all arms start from the same initial population,
so the difference between two arms is taken per seed and tested against zero.
A positive difference means the first-named arm ends further from the target.
"""

import numpy as np
from scipy.stats import wilcoxon

from mutation_ab.conditions import NAMES
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
from mutation_ab.records import finals, mean_curve

BOOTSTRAP = 20_000
REFERENCE_PAIRS = [
    (ARM_MIXTURE, ARM_DIFFERENCE),
    (ARM_MIXTURE, ARM_RANDOM),
    (ARM_MIXTURE, ARM_GAUSSIAN),
    (ARM_DE, ARM_GAUSSIAN),
    (ARM_DE, ARM_RANDOM),
    (ARM_DE_MATCHED, ARM_RANDOM),
    (ARM_DE, ARM_DE_MATCHED),
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(BOOTSTRAP, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction for multiple tests."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def signed_test(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Exact two-sided Wilcoxon signed-rank test of per-seed differences.

    With ten seeds there are only 2**10 sign patterns, so the p-value comes from
    enumerating all of them. The mean comes with a bootstrap 95% interval.
    """
    p = float(wilcoxon(values, method="exact").pvalue) if np.any(values != 0) else 1.0
    low, high = bootstrap_ci(values, rng)
    return {
        "mean": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "positive": int(np.sum(values > 0)),
        "n": len(values),
        "p": p,
    }


def statistical_tests(runs) -> list[dict]:
    """Holm within each family; the reference comparisons stay unadjusted."""
    rng = np.random.default_rng(0)
    rows: list[dict] = []

    def family(name: str, tests: dict[str, np.ndarray], adjust: bool) -> None:
        results = {label: signed_test(values, rng) for label, values in tests.items()}
        adjusted = (
            holm({label: r["p"] for label, r in results.items()}) if adjust else {}
        )
        for label, result in results.items():
            rows.append(
                {"family": name, "test": label, **result, "p_holm": adjusted.get(label)}
            )

    if all(arm in runs for arm in FACTORIAL_ARMS):
        a, b, c, d = (
            finals(runs, arm)
            for arm in (ARM_DIFFERENCE, ARM_NORMALISED, ARM_SIZE_MATCHED, ARM_GAUSSIAN)
        )
        family(
            "factorial",
            {
                "size: shrinking - fixed = (A+C)/2 - (B+D)/2": (a + c) / 2
                - (b + d) / 2,
                "direction: population - random = (A+B)/2 - (C+D)/2": (a + b) / 2
                - (c + d) / 2,
                "interaction: (A-C) - (B-D)": (a - c) - (b - d),
            },
            adjust=True,
        )
        family(
            "direction within size",
            {
                "shrinking size: A - C": a - c,
                "fixed size: B - D": b - d,
            },
            adjust=True,
        )
    elif ARM_NORMALISED in runs and ARM_GAUSSIAN in runs:
        family(
            "direction within size",
            {
                "fixed size: B - D": finals(runs, ARM_NORMALISED)
                - finals(runs, ARM_GAUSSIAN),
            },
            adjust=True,
        )
    if ARM_RANDOM in runs:
        random = finals(runs, ARM_RANDOM)
        family(
            "vs random search",
            {
                f"{NAMES[arm]} - random": finals(runs, arm) - random
                for arm in FACTORIAL_ARMS
                if arm in runs
            },
            adjust=True,
        )
    family(
        "references (unadjusted)",
        {
            f"{NAMES[x]} - {NAMES[y]}": finals(runs, x) - finals(runs, y)
            for x, y in REFERENCE_PAIRS
            if x in runs and y in runs
        },
        adjust=False,
    )
    return rows


def plateau_generation(
    curve: np.ndarray,
    start: int = 40,
    every: int = 10,
    window: int = 15,
    gain: float = 0.005,
) -> int | None:
    """First checked generation whose best-so-far gained less than `gain` metres
    over the previous `window` generations, or None if the curve never flattens."""
    for generation in range(start, len(curve), every):
        if curve[generation - window] - curve[generation] < gain:
            return generation
    return None


def plateaus(runs) -> list[dict]:
    """The plateau rule on each arm's mean best-so-far curve."""
    rows = []
    for arm in runs:
        curve = mean_curve(runs, arm)
        generation = plateau_generation(curve)
        evaluations = mean_curve(runs, arm, "evaluations")
        rows.append(
            {
                "arm": arm,
                "plateau_generation": generation,
                "plateau_evaluations": int(evaluations[generation])
                if generation is not None
                else None,
                "gain_last_50_generations": float(
                    curve[-min(51, len(curve))] - curve[-1]
                ),
                "gain_last_100_generations": float(
                    curve[-min(101, len(curve))] - curve[-1]
                ),
            }
        )
    return rows
