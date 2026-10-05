"""The controller and the MuJoCo episode that scores it (0.2 s episodes)."""

import mujoco as mj
import numpy as np
import pytest

from mutation_ab.config import RunConfig
from mutation_ab.controller import act, genome_length, observe, unpack
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.world import build_model

SHORT = RunConfig(seed=1, duration=0.2)
N_INPUTS, N_OUTPUTS = 33, 8  # spider_8
GENOME = genome_length(N_INPUTS, 6, N_OUTPUTS)


@pytest.fixture(scope="module")
def model():
    return build_model(SHORT)[0]


def test_controller_fits_the_spider(model):
    assert GENOME == 260
    assert model.nu == N_OUTPUTS
    assert observe(mj.MjData(model), SHORT.target_xy, SHORT.phase_hz).shape == (
        N_INPUTS,
    )


def test_actions_stay_within_the_hinge_range():
    layers = unpack(np.full(GENOME, 5.0), N_INPUTS, 6, N_OUTPUTS)
    assert np.all(np.abs(act(layers, np.ones(N_INPUTS))) <= np.pi / 2)


def test_evaluation_is_deterministic(model):
    genome = np.random.default_rng(0).normal(0, 0.5, GENOME)
    assert evaluate(genome, model, SHORT) == evaluate(genome, model, SHORT)


def test_a_still_robot_stays_two_metres_from_the_target(model):
    result = evaluate(np.zeros(GENOME), model, SHORT)
    assert result.distance == pytest.approx(2.0, abs=0.1)


def test_a_diverged_episode_raises_instead_of_scoring(model):
    genome = np.zeros(GENOME)
    genome[0] = np.nan
    with pytest.raises(UnstableSimulation):
        evaluate(genome, model, SHORT)
