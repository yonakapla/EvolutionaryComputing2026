"""The worked example from the README: one population difference and what each
cell of the 2x2 makes of it.

F(b - c) = [0.3, 0, 0, -0.4] has RMS 0.25. The Gaussian draw is
[0.15, -0.15, 0.15, -0.15], RMS 0.15, the fixed step size.

    A difference      [0.3, 0, 0, -0.4]            size 0.25, population direction
    B normalised      [0.18, 0, 0, -0.24]          size 0.15, population direction
    C size-matched    [0.25, -0.25, 0.25, -0.25]   size 0.25, random direction
    D Gaussian        [0.15, -0.15, 0.15, -0.15]   size 0.15, random direction

Once the population has collapsed to one genome, F(b - c) = 0: A and C stop
moving, while B and D keep taking steps of 0.15.
"""

import numpy as np
import pytest

from conftest import LENGTH, population
from mutation_ab.config import (
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_MIXTURE,
    ARM_NORMALISED,
    ARM_SIZE_MATCHED,
    RunConfig,
    make_streams,
)
from mutation_ab.metrics import rms, unique_genomes
from mutation_ab.operators import (
    KIND_GAUSSIAN,
    binomial_mask,
    de_trial,
    draw_donors,
    propose_child,
    shaped_step,
)

DIFFERENCE = np.array([0.3, 0.0, 0.0, -0.4])
DRAW = np.array([0.15, -0.15, 0.15, -0.15])
FIXED_SIZE = 0.15


def test_difference_takes_the_population_step_as_it_is():
    step, _ = shaped_step(DIFFERENCE, DRAW, ARM_DIFFERENCE, FIXED_SIZE)
    np.testing.assert_allclose(step, [0.3, 0.0, 0.0, -0.4])
    assert rms(step) == pytest.approx(0.25)


def test_normalised_keeps_the_direction_at_the_fixed_size():
    step, _ = shaped_step(DIFFERENCE, DRAW, ARM_NORMALISED, FIXED_SIZE)
    np.testing.assert_allclose(step, [0.18, 0.0, 0.0, -0.24])
    assert rms(step) == pytest.approx(0.15)


def test_size_matched_keeps_the_size_in_a_random_direction():
    step, _ = shaped_step(DIFFERENCE, DRAW, ARM_SIZE_MATCHED, FIXED_SIZE)
    np.testing.assert_allclose(step, [0.25, -0.25, 0.25, -0.25])
    assert rms(step) == pytest.approx(0.25)


def test_a_collapsed_population_stops_a_and_c_but_not_b():
    zero = np.zeros(4)
    for arm in (ARM_DIFFERENCE, ARM_SIZE_MATCHED):
        step, _ = shaped_step(zero, DRAW, arm, FIXED_SIZE)
        np.testing.assert_array_equal(step, zero)
    step, kind = shaped_step(zero, DRAW, ARM_NORMALISED, FIXED_SIZE)
    np.testing.assert_array_equal(step, DRAW)
    assert kind == KIND_GAUSSIAN


def test_gaussian_children_still_move_in_a_collapsed_population():
    genomes = np.tile(np.random.default_rng(3).normal(size=LENGTH), (12, 1))
    assert unique_genomes(genomes) == 1
    cfg = RunConfig(seed=1)
    for arm, moves in ((ARM_DIFFERENCE, False), (ARM_GAUSSIAN, True)):
        child = propose_child(genomes, np.zeros(12), cfg, arm, make_streams(1))
        assert (child.change_rms > 0) == moves


def test_child_changes_only_masked_weights_by_the_step():
    genomes, fitness = population()
    cfg = RunConfig(seed=6)
    proposal = propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, make_streams(6))
    parent = genomes[proposal.parent]
    b, c = proposal.donors
    step = cfg.scale_f * (genomes[b] - genomes[c])
    changed = proposal.child != parent
    np.testing.assert_allclose(proposal.child[changed], (parent + step)[changed])


@pytest.mark.parametrize(("rate", "expected"), [(1e-12, 1), (1.0, LENGTH)])
def test_mask_always_changes_at_least_one_weight(rate, expected):
    mask = binomial_mask(LENGTH, rate, np.random.default_rng(2))
    assert mask.sum() == expected


def test_donors_are_two_other_members():
    rng = np.random.default_rng(1)
    for _ in range(100):
        assert set(draw_donors(3, 1, rng)) == {0, 2}


def test_de_trial_uses_three_other_members():
    genomes, _ = population()
    streams = make_streams(6)
    for target in range(12):
        trial = de_trial(genomes, target, 0.5, 0.9, streams)
        a, b, c = trial.donors
        assert len({target, a, b, c}) == 4
        mixed = trial.child != genomes[target]
        expected = genomes[a] + 0.5 * (genomes[b] - genomes[c])
        np.testing.assert_allclose(trial.child[mixed], expected[mixed])


@pytest.mark.parametrize(
    "arm", [ARM_NORMALISED, ARM_SIZE_MATCHED, ARM_GAUSSIAN, ARM_MIXTURE]
)
def test_arms_use_the_random_streams_identically(arm):
    """Every arm draws the same random numbers in the same order, so two arms
    differ only in how they turn those numbers into a step."""
    genomes, fitness = population()
    cfg = RunConfig(seed=9)
    control, treatment = make_streams(9), make_streams(9)
    for _ in range(50):
        propose_child(genomes, fitness, cfg, ARM_DIFFERENCE, control)
        propose_child(genomes, fitness, cfg, arm, treatment)
    for name in ("selection", "mask", "gaussian", "replacement"):
        assert getattr(control, name).random() == getattr(treatment, name).random()
