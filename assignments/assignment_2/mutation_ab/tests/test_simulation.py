import json
import multiprocessing

import mujoco as mj
import numpy as np
import pytest

from mutation_ab.config import RunConfig
from mutation_ab.controller import N_INPUTS, N_OUTPUTS, act, genome_length, observe, unpack
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.records import RunRecorder, genome_sha1
from mutation_ab.world import build_model, terrain_fingerprint

SHORT = RunConfig(seed=1, duration=0.2)


@pytest.fixture(scope="module")
def model():
    return build_model(SHORT)[0]


def test_genome_length_matches_design():
    assert genome_length(N_INPUTS, 6, N_OUTPUTS) == 222


def test_unpack_rejects_wrong_length():
    with pytest.raises(ValueError):
        unpack(np.zeros(221), N_INPUTS, 6, N_OUTPUTS)


def test_model_dimensions_match_controller(model):
    assert model.nu == N_OUTPUTS
    data = mj.MjData(model)
    assert observe(data, SHORT.target_xy, SHORT.phase_hz).shape == (N_INPUTS,)


def test_actions_are_within_hinge_range():
    layers = unpack(np.full(222, 5.0), N_INPUTS, 6, N_OUTPUTS)
    actions = act(layers, np.ones(N_INPUTS))
    assert actions.shape == (N_OUTPUTS,)
    assert np.all(np.abs(actions) <= np.pi / 2)


def test_evaluation_is_deterministic(model):
    genome = np.random.default_rng(0).normal(0, 0.5, 222)
    first, second = evaluate(genome, model, SHORT), evaluate(genome, model, SHORT)
    assert first == second
    assert np.isfinite(first.distance)


def test_zero_controller_stays_near_spawn(model):
    result = evaluate(np.zeros(222), model, SHORT)
    assert result.distance == pytest.approx(2.0, abs=0.1)


def test_nan_genome_raises_instead_of_scoring(model):
    genome = np.zeros(222)
    genome[0] = np.nan
    with pytest.raises(UnstableSimulation) as excinfo:
        evaluate(genome, model, SHORT)
    assert np.array_equal(excinfo.value.genome, genome, equal_nan=True)


def test_failed_recording_carries_the_offending_genome(tmp_path, model):
    genome = np.zeros(222)
    genome[0] = np.nan
    try:
        evaluate(genome, model, SHORT)
    except UnstableSimulation as error:
        RunRecorder(tmp_path / "run", SHORT, "difference", {"terrain_sha1": "t", "model_sha1": "m"}).fail(error)

    report = json.loads((tmp_path / "run" / "FAILED.json").read_text())
    assert report["genome"][0] != report["genome"][0]  # nan
    assert report["genome_sha1"] == genome_sha1(genome)


def test_terrain_is_identical_across_processes():
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        prints = pool.map(terrain_fingerprint, [42, 42])
    assert prints[0] == prints[1]
    assert terrain_fingerprint(42)[0] != terrain_fingerprint(43)[0]
