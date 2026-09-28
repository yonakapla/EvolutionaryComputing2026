import math
import subprocess
from pathlib import Path

import pytest

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM, RunConfig
from mutation_ab.streams import make_streams

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_budget_counts_shared_initial_population():
    cfg = RunConfig(seed=1, generations=80)
    assert cfg.children_per_generation == 11
    assert cfg.budget == 12 + 11 * 80


def test_scale_factor_matches_gaussian_step_at_initialisation():
    cfg = RunConfig(seed=1)
    assert cfg.scale_f * math.sqrt(2) * cfg.init_sd == pytest.approx(cfg.gaussian_sd)


@pytest.mark.parametrize(
    "overrides",
    [
        {"population_size": 2},
        {"population_size": 3, "tournament_size": 4},
        {"generations": 0},
        {"duration": 0.0},
        {"replacement_probability": 1.5},
        {"crossover_rate": 0.0},
    ],
)
def test_rejects_invalid_settings(overrides):
    with pytest.raises(ValueError):
        RunConfig(seed=1, **overrides)


def test_replacement_probability_per_arm():
    cfg = RunConfig(seed=1)
    assert cfg.replacement_probability_for(ARM_DIFFERENCE) == 0.0
    assert cfg.replacement_probability_for(ARM_MIXTURE) == 0.10
    with pytest.raises(ValueError):
        cfg.replacement_probability_for(ARM_RANDOM)


def test_config_hash_is_stable_and_seed_sensitive():
    assert RunConfig(seed=1).config_hash() == RunConfig(seed=1).config_hash()
    assert RunConfig(seed=1).config_hash() != RunConfig(seed=2).config_hash()


def test_streams_are_reproducible_and_independent():
    first, second = make_streams(7), make_streams(7)
    assert first.mask.random() == second.mask.random()
    assert first.selection.random() != first.gaussian.random()
    assert make_streams(7).init.random() != make_streams(8).init.random()


def test_src_ariel_is_untouched():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/ariel"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""
