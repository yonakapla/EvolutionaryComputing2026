"""The fixed setup of the experiment: budgets, step sizes and per-arm settings."""

import numpy as np
import pytest

from mutation_ab.config import (
    ARM_DE,
    ARM_DE_MATCHED,
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_MIXTURE,
    ARM_NORMALISED,
    ARM_SIZE_MATCHED,
    RunConfig,
)
from mutation_ab.metrics import rms


def test_budget_counts_the_shared_initial_population():
    cfg = RunConfig(seed=1, generations=800)
    assert cfg.budget == 12 + 11 * 800 == 8_812


def test_de_never_gets_more_evaluations_than_the_ea_arms():
    for generations in (5, 80, 800):
        cfg = RunConfig(seed=1, generations=generations)
        assert cfg.budget - cfg.population_size < cfg.budget_for(ARM_DE) <= cfg.budget


def test_difference_step_matches_the_gaussian_step_at_initialisation():
    """F is chosen so that F(b - c) of two initial genomes has the size of a
    Gaussian step, which makes A and D start out equal in size."""
    cfg = RunConfig(seed=1)
    b, c = np.random.default_rng(0).normal(0.0, cfg.init_sd, (2, 100_000))
    assert rms(cfg.scale_f * (b - c)) == pytest.approx(cfg.gaussian_sd, rel=0.01)


def test_arm_settings_match_the_protocol():
    cfg = RunConfig(seed=1)
    for arm in (ARM_DIFFERENCE, ARM_NORMALISED, ARM_SIZE_MATCHED):
        assert cfg.replacement_probability_for(arm) == 0.0
    assert cfg.replacement_probability_for(ARM_MIXTURE) == 0.10
    assert cfg.replacement_probability_for(ARM_GAUSSIAN) == 1.0
    assert cfg.de_parameters_for(ARM_DE) == (0.5, 0.9)
    assert cfg.de_parameters_for(ARM_DE_MATCHED) == (cfg.scale_f, cfg.crossover_rate)


@pytest.mark.parametrize(
    "overrides",
    [
        {"population_size": 2},
        {"population_size": 3, "tournament_size": 4},
        {"generations": 0},
        {"crossover_rate": 0.0},
    ],
)
def test_impossible_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        RunConfig(seed=1, **overrides)
