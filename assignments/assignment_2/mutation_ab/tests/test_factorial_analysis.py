import csv

import numpy as np
import pytest
from conftest import FAKE_HASHES, fake_evaluator

from mutation_ab import factorial_analysis as fa
from mutation_ab.config import ALL_ARMS, ARM_RANDOM, DE_ARMS, RunConfig
from mutation_ab.de_arm import run_de
from mutation_ab.ea_arm import run_arm
from mutation_ab.initial import make_initial
from mutation_ab.random_search import run_random
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams

SMALL = {"generations": 6, "population_size": 6, "step_log_every": 2}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("final")
    for seed in (11, 12, 13, 14, 15, 16):
        cfg = RunConfig(seed=seed, **SMALL)
        initial = make_initial(cfg, make_streams(seed), fake_evaluator, 222)
        for arm in ALL_ARMS:
            recorder = RunRecorder(root / f"seed_{seed}" / arm, cfg, arm, FAKE_HASHES)
            runner = run_random if arm == ARM_RANDOM else run_de if arm in DE_ARMS else run_arm
            args = (cfg, initial, fake_evaluator, make_streams(seed), recorder)
            recorder.complete(runner(*args) if arm == ARM_RANDOM else runner(cfg, arm, *args[1:]))
    return root


def test_in_span_fraction_is_one_inside_and_zero_orthogonal():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(20, 3)))[0].T  # 3 orthonormal directions in 20-D
    parents = rng.normal(size=(6, 3)) @ basis
    adults = np.stack([parents, parents])
    inside = rng.normal(size=(4, 3)) @ basis
    orthogonal = rng.normal(size=(4, 20))
    orthogonal -= orthogonal @ basis.T @ basis
    for steps, expected in ((inside, 1.0), (orthogonal, 0.0)):
        share, rank = fa.in_span_fraction(adults, np.ones(4, dtype=int), steps)[1]
        assert share == pytest.approx(expected, abs=1e-9)
        assert rank == 3


def test_identical_population_has_no_span():
    adults = np.ones((2, 6, 10))
    assert fa.in_span_fraction(adults, np.ones(3, dtype=int), np.ones((3, 10))) == {}


def test_signed_test_reports_direction_and_ci():
    result = fa.signed_test(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), np.random.default_rng(0))
    assert result["positive"] == 6
    assert result["p"] == pytest.approx(0.03125)
    assert 0.5 <= result["ci_low"] <= result["mean"] <= result["ci_high"] <= 1.0


def test_analysis_writes_every_output(root):
    assert fa.main([str(root)]) == 0
    out = root / "analysis"
    for name in ("per_seed", "summary", "stats", "plateau", "step_span"):
        assert (out / f"{name}.csv").stat().st_size > 0
    for figure in ("fig_fitness", "fig_mechanism", "fig_seeds"):
        assert (out / f"{figure}.png").exists() and (out / f"{figure}.pdf").exists()
    with (out / "summary.csv").open() as handle:
        arms = [row["arm"] for row in csv.DictReader(handle)]
    assert arms == list(fa.ARM_ORDER)


def test_factorial_family_is_holm_adjusted(root):
    stats = fa.statistical_tests(fa.load(root))
    factorial = [r for r in stats if r["family"] == "factorial"]
    assert len(factorial) == 3
    assert all(r["p_holm"] >= r["p"] for r in factorial)
    assert all(r["p_holm"] is None for r in stats if r["family"] == "references (unadjusted)")


def test_load_rejects_uneven_seed_sets(root, tmp_path):
    (tmp_path / "seed_1" / "gaussian").mkdir(parents=True)
    import shutil

    shutil.copytree(root / "seed_11" / "gaussian", tmp_path / "seed_1" / "gaussian", dirs_exist_ok=True)
    shutil.copytree(root / "seed_12", tmp_path / "seed_2")
    with pytest.raises(ValueError):
        fa.load(tmp_path)
