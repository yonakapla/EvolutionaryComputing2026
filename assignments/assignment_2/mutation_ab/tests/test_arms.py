import csv
from itertools import pairwise

import numpy as np
import pytest
from conftest import FAKE_HASHES, LENGTH, TEST_ARMS, children, fake_evaluator, make_fake_root

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, RunConfig
from mutation_ab.ea_arm import run_ea
from mutation_ab.evaluate import UnstableSimulation
from mutation_ab.initial import make_initial
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams

SMALL = {"generations": 5, "population_size": 6}


def generations(run_dir):
    with (run_dir / "generations.csv").open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def root(tmp_path):
    return make_fake_root(tmp_path, [3], **SMALL)


def test_every_arm_spends_exactly_the_budget(root):
    budget = RunConfig(seed=3, **SMALL).budget
    for arm in TEST_ARMS:
        assert len(children(root / "seed_3" / arm)) == budget


def test_all_arms_share_the_initial_population(root):
    founders = {
        arm: [c["genome_sha1"] for c in children(root / "seed_3" / arm) if c["kind"] == "init"]
        for arm in TEST_ARMS
    }
    assert founders["difference"] == founders["mixture"] == founders["random"]


def test_elitism_never_loses_the_best(root):
    for arm in (ARM_DIFFERENCE, ARM_MIXTURE):
        best = [float(r["best"]) for r in generations(root / "seed_3" / arm)]
        assert all(later <= earlier for earlier, later in pairwise(best))


def test_adult_snapshots_cover_every_generation(root):
    adults = np.load(root / "seed_3" / ARM_DIFFERENCE / "adults.npz")["adults"]
    assert adults.shape == (SMALL["generations"] + 1, SMALL["population_size"], LENGTH)


def test_parents_and_donors_come_from_previous_survivors(root):
    records = children(root / "seed_3" / ARM_MIXTURE)
    uids_by_generation = {}
    for record in records:
        uids_by_generation.setdefault(record["generation"], set()).add(record["uid"])
    alive = set(uids_by_generation[0])
    for generation in range(1, SMALL["generations"] + 1):
        born = [r for r in records if r["generation"] == generation]
        for record in born:
            assert record["parent_uid"] in alive
            assert set(record["donor_uids"]) <= alive
            assert record["parent_uid"] not in record["donor_uids"]
        elite = min((r for r in records if r["uid"] in alive), key=lambda r: (r["distance"], r["uid"]))
        alive = {elite["uid"], *(r["uid"] for r in born)}


def test_zero_probability_mixture_reproduces_difference_arm(tmp_path):
    root = make_fake_root(tmp_path, [4], replacement_probability=0.0, **SMALL)
    hashes = {
        arm: [c["genome_sha1"] for c in children(root / "seed_4" / arm)]
        for arm in (ARM_DIFFERENCE, ARM_MIXTURE)
    }
    assert hashes[ARM_DIFFERENCE] == hashes[ARM_MIXTURE]


def test_full_probability_mixture_uses_only_gaussian_children(tmp_path):
    root = make_fake_root(tmp_path, [5], replacement_probability=1.0, **SMALL)
    kinds = {c["kind"] for c in children(root / "seed_5" / ARM_MIXTURE) if c["kind"] != "init"}
    assert kinds == {"gaussian"}


def test_unstable_evaluation_propagates(tmp_path):
    cfg = RunConfig(seed=6, **SMALL)
    initial = make_initial(cfg, make_streams(6), fake_evaluator, LENGTH)
    calls = {"n": 0}

    def exploding(genome):
        calls["n"] += 1
        if calls["n"] > 3:
            raise UnstableSimulation("blew up")
        return fake_evaluator(genome)

    recorder = RunRecorder(tmp_path / "run", cfg, ARM_DIFFERENCE, FAKE_HASHES)
    with pytest.raises(UnstableSimulation):
        run_ea(cfg, ARM_DIFFERENCE, initial, exploding, make_streams(6), recorder)
