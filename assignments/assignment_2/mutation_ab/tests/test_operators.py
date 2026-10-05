import math

import numpy as np
import pytest

from conftest import LENGTH, population
from mutation_ab.config import ARM_DIFFERENCE, ARM_GAUSSIAN, RunConfig
from mutation_ab.metrics import genotype_diversity, rms, unique_genomes
from mutation_ab.operators import (
    KIND_GAUSSIAN,
    binomial_mask,
    draw_donors,
    elite_index,
    propose_child,
    tournament,
)
from mutation_ab.streams import make_streams


def test_rms_and_diversity_known_values():
    assert rms(np.array([3.0, 4.0])) == pytest.approx(math.sqrt(12.5))
    genomes = np.vstack([np.zeros(LENGTH), np.ones(LENGTH)])
    assert genotype_diversity(genomes) == pytest.approx(1.0)
    assert genotype_diversity(np.ones((5, LENGTH))) == 0.0


def test_unique_genomes_counts_exact_duplicates():
    genomes = np.vstack([np.zeros(LENGTH), np.zeros(LENGTH), np.ones(LENGTH)])
    assert unique_genomes(genomes) == 2


def test_tournament_returns_best_of_full_sample():
    fitness = np.array([3.0, 1.0, 2.0])
    assert tournament(fitness, 3, np.random.default_rng(0)) == 1


def test_tournament_ties_are_deterministic_per_seed():
    fitness = np.zeros(12)
    picks_a = [tournament(fitness, 3, rng) for rng in [np.random.default_rng(4)] * 20]
    picks_b = [tournament(fitness, 3, rng) for rng in [np.random.default_rng(4)] * 20]
    assert picks_a == picks_b


def test_donors_are_distinct_and_exclude_parent():
    rng = np.random.default_rng(1)
    for _ in range(500):
        b, c = draw_donors(3, 1, rng)
        assert {b, c} == {0, 2}


@pytest.mark.parametrize(("rate", "expected"), [(1e-12, 1), (1.0, LENGTH)])
def test_mask_forces_one_coordinate(rate, expected):
    mask = binomial_mask(LENGTH, rate, np.random.default_rng(2))
    assert mask.dtype == bool
    assert mask.sum() == expected


def test_identical_population_gives_zero_difference_step():
    genomes = np.tile(np.random.default_rng(3).normal(size=LENGTH), (12, 1))
    proposal = propose_child(
        genomes, np.zeros(12), RunConfig(seed=1), ARM_DIFFERENCE, make_streams(1)
    )
    assert proposal.kind == ARM_DIFFERENCE
    assert proposal.proposal_rms == 0.0
    assert proposal.change_rms == 0.0
    np.testing.assert_array_equal(proposal.child, genomes[proposal.parent])


@pytest.mark.parametrize(
    ("arm", "kind"), [(ARM_DIFFERENCE, ARM_DIFFERENCE), (ARM_GAUSSIAN, KIND_GAUSSIAN)]
)
def test_pure_arms_propose_one_kind(arm, kind):
    genomes, fitness = population()
    streams = make_streams(5)
    kinds = {
        propose_child(genomes, fitness, RunConfig(seed=5), arm, streams).kind
        for _ in range(200)
    }
    assert kinds == {kind}


def test_child_changes_only_masked_coordinates_by_delta():
    genomes, fitness = population()
    cfg = RunConfig(seed=6)
    proposal = propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, make_streams(6))
    parent = genomes[proposal.parent]
    b, c = proposal.donors
    delta = cfg.scale_f * (genomes[b] - genomes[c])
    changed = proposal.child != parent
    np.testing.assert_allclose(proposal.child[changed], (parent + delta)[changed])
    assert proposal.proposal_rms == pytest.approx(rms(delta))
    assert proposal.change_rms == pytest.approx(rms(proposal.child - parent))


def test_arms_consume_random_streams_identically():
    genomes, fitness = population()
    cfg = RunConfig(seed=9)
    control, treatment = make_streams(9), make_streams(9)
    for _ in range(50):
        propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, control)
        propose_child(genomes, fitness, cfg, ARM_GAUSSIAN, treatment)
    for name in ("selection", "mask", "gaussian", "replacement"):
        assert getattr(control, name).random() == getattr(treatment, name).random()


def test_elite_ties_break_on_lowest_uid():
    assert elite_index(np.array([1.0, 0.0, 0.0]), np.array([5, 9, 3])) == 2
