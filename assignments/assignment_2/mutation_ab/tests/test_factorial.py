import csv
from itertools import pairwise

import numpy as np
import pytest

from conftest import FAKE_HASHES, LENGTH, children, fake_evaluator, population
from mutation_ab.config import (
    ARM_DE,
    ARM_DE_MATCHED,
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_NORMALISED,
    ARM_SIZE_MATCHED,
    RunConfig,
)
from mutation_ab.de_arm import run_de
from mutation_ab.ea_arm import run_ea
from mutation_ab.initial import make_initial
from mutation_ab.metrics import rms
from mutation_ab.operators import KIND_GAUSSIAN, de_trial, propose_child
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams

SMALL = {"generations": 5, "population_size": 6}


def test_size_matched_keeps_difference_size_but_not_direction():
    genomes, fitness = population()
    cfg = RunConfig(seed=2)
    plain = propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, make_streams(2))
    matched = propose_child(genomes, fitness, cfg, ARM_SIZE_MATCHED, make_streams(2))
    assert matched.kind == ARM_SIZE_MATCHED
    assert (matched.parent, matched.donors) == (plain.parent, plain.donors)
    assert matched.proposal_rms == pytest.approx(plain.proposal_rms)
    assert not np.allclose(matched.child, plain.child)


def test_normalised_keeps_difference_direction_at_fixed_size():
    genomes, fitness = population()
    cfg = RunConfig(seed=3)
    plain = propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, make_streams(3))
    normalised = propose_child(genomes, fitness, cfg, ARM_NORMALISED, make_streams(3))
    assert normalised.kind == ARM_NORMALISED
    assert normalised.proposal_rms == pytest.approx(cfg.gaussian_sd)
    parent = genomes[plain.parent]
    changed = plain.child != parent
    ratio = (normalised.child - parent)[changed] / (plain.child - parent)[changed]
    np.testing.assert_allclose(ratio, cfg.gaussian_sd / plain.proposal_rms)


def test_normalised_falls_back_to_gaussian_when_population_is_identical():
    genomes = np.tile(np.random.default_rng(4).normal(size=LENGTH), (12, 1))
    proposal = propose_child(
        genomes, np.zeros(12), RunConfig(seed=4), ARM_NORMALISED, make_streams(4)
    )
    assert proposal.kind == KIND_GAUSSIAN
    assert proposal.change_rms > 0


@pytest.mark.parametrize("arm", [ARM_NORMALISED, ARM_SIZE_MATCHED])
def test_new_arms_consume_streams_like_difference(arm):
    genomes, fitness = population()
    cfg = RunConfig(seed=5)
    control, treatment = make_streams(5), make_streams(5)
    for _ in range(50):
        propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, control)
        propose_child(genomes, fitness, cfg, arm, treatment)
    for name in ("selection", "mask", "gaussian", "replacement"):
        assert getattr(control, name).random() == getattr(treatment, name).random()


def test_de_trial_uses_three_distinct_other_members():
    genomes, _ = population()
    streams = make_streams(6)
    for target in range(12):
        trial = de_trial(genomes, target, 0.5, 0.9, streams)
        assert len(set(trial.donors)) == 3
        assert target not in trial.donors
        a, b, c = trial.donors
        mixed = trial.child != genomes[target]
        np.testing.assert_allclose(
            trial.child[mixed], (genomes[a] + 0.5 * (genomes[b] - genomes[c]))[mixed]
        )
        assert trial.proposal_rms == pytest.approx(rms(0.5 * (genomes[b] - genomes[c])))


@pytest.mark.parametrize("arm", [ARM_DE, ARM_DE_MATCHED])
def test_de_arm_is_one_to_one_and_within_budget(tmp_path, arm):
    cfg = RunConfig(seed=7, **SMALL)
    initial = make_initial(cfg, make_streams(7), fake_evaluator, LENGTH)
    recorder = RunRecorder(tmp_path / arm, cfg, arm, FAKE_HASHES)
    recorder.complete(
        run_de(cfg, arm, initial, fake_evaluator, make_streams(7), recorder)
    )
    records = children(tmp_path / arm)
    assert len(records) == cfg.population_size * (1 + cfg.de_generations) <= cfg.budget
    with (tmp_path / arm / "generations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    # One-to-one replacement: population size is constant and no slot ever gets worse.
    adults = np.load(tmp_path / arm / "adults.npz")["adults"]
    assert adults.shape == (cfg.de_generations + 1, cfg.population_size, LENGTH)
    worst = [float(row["worst"]) for row in rows]
    assert all(later <= earlier for earlier, later in pairwise(worst))
    trials = [r for r in records if r["kind"] == "de"]
    assert all(r["parent_uid"] not in r["donor_uids"] for r in trials)


@pytest.mark.parametrize("arm", [ARM_NORMALISED, ARM_SIZE_MATCHED, ARM_GAUSSIAN])
def test_new_ea_arms_log_their_step_kind_and_steps(tmp_path, arm):
    cfg = RunConfig(seed=8, step_log_every=2, **SMALL)
    initial = make_initial(cfg, make_streams(8), fake_evaluator, LENGTH)
    recorder = RunRecorder(tmp_path / arm, cfg, arm, FAKE_HASHES)
    recorder.complete(
        run_ea(cfg, arm, initial, fake_evaluator, make_streams(8), recorder)
    )
    kinds = {r["kind"] for r in children(tmp_path / arm) if r["kind"] != "init"}
    assert kinds <= {arm, KIND_GAUSSIAN}
    steps = np.load(tmp_path / arm / "steps.npz")
    assert set(steps["generation"]) == {2, 4}
    assert steps["delta"].shape == (2 * cfg.children_per_generation, LENGTH)
