import json
import multiprocessing

import mujoco as mj
import numpy as np
import pytest
from conftest import FAKE_HASHES

from mutation_ab.config import RunConfig
from mutation_ab.controller import act, genome_length, observe, unpack
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.records import RunRecorder, genome_sha1
from mutation_ab.world import build_model

SHORT = RunConfig(seed=1, duration=0.2)
N_INPUTS, N_OUTPUTS = 33, 8  # spider_8
GENOME = genome_length(N_INPUTS, 6, N_OUTPUTS)


def model_hash(seed: int) -> str:
    return build_model(RunConfig(seed=seed))[1]["model_sha1"]


@pytest.fixture(scope="module")
def model():
    return build_model(SHORT)[0]


def test_genome_length_matches_design():
    assert GENOME == 260


def test_unpack_rejects_wrong_length():
    with pytest.raises(ValueError):
        unpack(np.zeros(GENOME - 1), N_INPUTS, 6, N_OUTPUTS)


def test_model_dimensions_match_controller(model):
    assert model.nu == N_OUTPUTS
    data = mj.MjData(model)
    assert observe(data, SHORT.target_xy, SHORT.phase_hz).shape == (N_INPUTS,)


def test_actions_are_within_hinge_range():
    layers = unpack(np.full(GENOME, 5.0), N_INPUTS, 6, N_OUTPUTS)
    actions = act(layers, np.ones(N_INPUTS))
    assert actions.shape == (N_OUTPUTS,)
    assert np.all(np.abs(actions) <= np.pi / 2)


def test_evaluation_is_deterministic(model):
    genome = np.random.default_rng(0).normal(0, 0.5, GENOME)
    first, second = evaluate(genome, model, SHORT), evaluate(genome, model, SHORT)
    assert first == second
    assert np.isfinite(first.distance)


def test_zero_controller_stays_near_spawn(model):
    result = evaluate(np.zeros(GENOME), model, SHORT)
    assert result.distance == pytest.approx(2.0, abs=0.1)


def test_nan_genome_raises_instead_of_scoring(model):
    genome = np.zeros(GENOME)
    genome[0] = np.nan
    with pytest.raises(UnstableSimulation) as excinfo:
        evaluate(genome, model, SHORT)
    assert np.array_equal(excinfo.value.genome, genome, equal_nan=True)


def test_failed_recording_carries_the_offending_genome(tmp_path, model):
    genome = np.zeros(GENOME)
    genome[0] = np.nan
    try:
        evaluate(genome, model, SHORT)
    except UnstableSimulation as error:
        RunRecorder(tmp_path / "run", SHORT, "difference", FAKE_HASHES).fail(error)

    report = json.loads((tmp_path / "run" / "FAILED.json").read_text())
    assert report["genome"][0] != report["genome"][0]  # nan
    assert report["genome_sha1"] == genome_sha1(genome)


def test_model_is_identical_across_processes():
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        hashes = pool.map(model_hash, [1, 2])
    assert hashes[0] == hashes[1] == model_hash(3)


@pytest.mark.parametrize("body", ["np", "does_not_exist"])
def test_unknown_body_is_rejected(body):
    with pytest.raises(ValueError, match="unknown John Set body"):
        build_model(RunConfig(seed=1, body=body))
