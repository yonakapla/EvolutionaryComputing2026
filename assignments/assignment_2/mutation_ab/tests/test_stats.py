"""The hypothesis tests, on hand-made final distances."""

from pathlib import Path

import numpy as np
import pytest

from mutation_ab.config import (
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_NORMALISED,
    ARM_SIZE_MATCHED,
)
from mutation_ab.records import Run
from mutation_ab.stats import holm, plateau_generation, statistical_tests

SEEDS = range(6)


def runs_ending_at(finals: dict[str, list[float]]) -> dict[str, dict[int, Run]]:
    """Runs whose best-so-far curve is a single final value per seed."""
    return {
        arm: {
            seed: Run(seed, arm, Path(), {"best_so_far": np.array([value])}, [], {})
            for seed, value in zip(SEEDS, values, strict=True)
        }
        for arm, values in finals.items()
    }


def test_contrasts_are_paired_by_seed():
    """Seeds differ by up to 0.5 m, as much as the size effect itself. Unpaired,
    the four arms would overlap; paired, the shrinking-size arms A and C end
    exactly 1 m further away on every seed, and direction makes no difference."""
    seed_offset = np.linspace(0.0, 0.5, len(SEEDS))
    runs = runs_ending_at(
        {
            ARM_DIFFERENCE: list(1.5 + seed_offset),
            ARM_SIZE_MATCHED: list(1.5 + seed_offset),
            ARM_NORMALISED: list(0.5 + seed_offset),
            ARM_GAUSSIAN: list(0.5 + seed_offset),
        }
    )
    tests = {row["test"]: row for row in statistical_tests(runs)}
    size = tests["size: shrinking - fixed = (A+C)/2 - (B+D)/2"]
    direction = tests["direction: population - random = (A+B)/2 - (C+D)/2"]
    assert size["mean"] == pytest.approx(1.0)
    assert size["positive"] == 6
    assert size["p"] == pytest.approx(2 / 2**6)
    assert direction["mean"] == pytest.approx(0.0)
    assert direction["p"] == 1.0


def test_holm_multiplies_the_smallest_p_by_the_number_of_tests():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_plateau_is_the_first_check_after_the_curve_flattens():
    flat_after_50 = np.concatenate([np.linspace(3.0, 1.0, 51), np.full(30, 1.0)])
    assert plateau_generation(flat_after_50) == 70


def test_a_curve_that_keeps_falling_has_no_plateau():
    assert plateau_generation(np.linspace(3.0, 1.0, 81)) is None
