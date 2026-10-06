"""The analysis of a folder of runs: which runs may be compared, and the step
geometry behind the direction result."""

import json
import shutil

import numpy as np
import pytest

from conftest import make_fake_root
from mutation_ab import analysis
from mutation_ab.config import ALL_ARMS
from mutation_ab.plots import equivalent_generations
from mutation_ab.records import borrow_sigma_free_arms, load

SMALL = {"generations": 6, "population_size": 6, "step_log_every": 2}
SEEDS = (11, 12, 13, 14, 15, 16)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return make_fake_root(tmp_path_factory.mktemp("final"), SEEDS, ALL_ARMS, **SMALL)


def test_steps_inside_the_population_span_count_fully_and_orthogonal_ones_not():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(20, 3)))[0].T
    parents = rng.normal(size=(6, 3)) @ basis
    adults = np.stack([parents, parents])
    inside = rng.normal(size=(4, 3)) @ basis
    orthogonal = rng.normal(size=(4, 20))
    orthogonal -= orthogonal @ basis.T @ basis
    for steps, expected in ((inside, 1.0), (orthogonal, 0.0)):
        generation = np.ones(4, dtype=int)
        share, rank = analysis.in_span_fraction(adults, generation, steps)[1]
        assert share == pytest.approx(expected, abs=1e-9)
        assert rank == 3


def test_a_collapsed_population_has_no_span():
    adults = np.ones((2, 6, 10))
    generation = np.ones(3, dtype=int)
    assert analysis.in_span_fraction(adults, generation, np.ones((3, 10))) == {}


def test_canonical_de_is_placed_at_the_ea_generation_with_equal_evaluations():
    # EA arms: 12 initial evaluations, then 11 per generation.
    np.testing.assert_allclose(
        equivalent_generations([12, 23, 12 + 11 * 800], 12), [0, 1, 800]
    )
    # DE spends 12 per generation, so its generation 733 sits just before 800.
    assert equivalent_generations([12 + 12 * 733], 12)[0] == pytest.approx(799.6, 0.01)


def test_a_folder_without_runs_says_so(tmp_path):
    with pytest.raises(ValueError, match="no completed runs"):
        load(tmp_path)


def test_arms_must_cover_the_same_seeds(root, tmp_path):
    shutil.copytree(root / "seed_11" / "gaussian", tmp_path / "seed_1" / "gaussian")
    shutil.copytree(root / "seed_12", tmp_path / "seed_2")
    with pytest.raises(ValueError, match="different seeds"):
        load(tmp_path)


def test_runs_with_different_setups_are_not_compared(root, tmp_path):
    shutil.copytree(root / "seed_11", tmp_path / "seed_11")
    shutil.copytree(root / "seed_12", tmp_path / "seed_12")
    config_file = tmp_path / "seed_12" / "gaussian" / "config.json"
    meta = json.loads(config_file.read_text())
    meta["config"]["body"] = "gecko"
    config_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different setups"):
        load(tmp_path)


def test_sigma_runs_borrow_the_arms_that_do_not_use_sigma(root, tmp_path):
    sigma = make_fake_root(
        tmp_path, SEEDS, ("normalised", "gaussian"), gaussian_sd=0.3, **SMALL
    )
    runs = borrow_sigma_free_arms(load(sigma), root)
    assert list(runs) == [
        "difference",
        "size_matched",
        "normalised",
        "gaussian",
        "random",
    ]


def test_borrowing_refuses_runs_that_differ_in_more_than_sigma(root, tmp_path):
    other = make_fake_root(
        tmp_path, SEEDS, ("normalised", "gaussian"), crossover_rate=0.5, **SMALL
    )
    with pytest.raises(ValueError, match="differs"):
        borrow_sigma_free_arms(load(other), root)


def test_analysis_writes_every_table_and_figure(root):
    assert analysis.main([str(root)]) == 0
    out = root / "analysis"
    for name in ("per_seed", "summary", "stats", "plateau", "step_span", "step_shape"):
        assert (out / f"{name}.csv").stat().st_size > 0
    for figure in ("fig_fitness", "fig_mechanism", "fig_seeds"):
        assert (out / f"{figure}.png").exists()
        assert (out / f"{figure}.pdf").exists()
    assert (out / "report.txt").exists()
