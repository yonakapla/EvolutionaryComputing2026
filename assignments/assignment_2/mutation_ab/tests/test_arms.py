"""Short runs of the arms with a stand-in for the simulator (population 6,
5 generations)."""

import csv
from itertools import pairwise

import numpy as np
import pytest

from conftest import LENGTH, TEST_ARMS, children, fake_evaluator, make_fake_root
from mutation_ab.config import (
    ARM_DE,
    ARM_DIFFERENCE,
    ARM_MIXTURE,
    RunConfig,
    make_streams,
)
from mutation_ab.ea import make_initial, run_de
from mutation_ab.records import RunRecorder

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


def test_all_arms_start_from_the_same_population(root):
    founders = {
        arm: [c["genome_sha1"] for c in children(root / "seed_3" / arm)][
            : SMALL["population_size"]
        ]
        for arm in TEST_ARMS
    }
    assert founders["difference"] == founders["mixture"] == founders["random"]


def test_elitism_never_loses_the_best(root):
    for arm in (ARM_DIFFERENCE, ARM_MIXTURE):
        best = [float(r["best"]) for r in generations(root / "seed_3" / arm)]
        assert all(later <= earlier for earlier, later in pairwise(best))


def test_parents_and_donors_come_from_the_previous_generation(root):
    records = children(root / "seed_3" / ARM_MIXTURE)
    alive = {r["uid"] for r in records if r["generation"] == 0}
    for generation in range(1, SMALL["generations"] + 1):
        born = [r for r in records if r["generation"] == generation]
        for record in born:
            assert record["parent_uid"] in alive
            assert set(record["donor_uids"]) <= alive - {record["parent_uid"]}
        elite = min(
            (r for r in records if r["uid"] in alive),
            key=lambda r: (r["distance"], r["uid"]),
        )
        alive = {elite["uid"], *(r["uid"] for r in born)}


def test_mixture_without_gaussian_steps_is_the_difference_arm(tmp_path):
    root = make_fake_root(tmp_path, [4], replacement_probability=0.0, **SMALL)
    genomes = {
        arm: [c["genome_sha1"] for c in children(root / "seed_4" / arm)]
        for arm in (ARM_DIFFERENCE, ARM_MIXTURE)
    }
    assert genomes[ARM_DIFFERENCE] == genomes[ARM_MIXTURE]


def test_de_replaces_an_adult_only_with_a_trial_at_least_as_good(tmp_path):
    cfg = RunConfig(seed=7, **SMALL)
    initial = make_initial(cfg, make_streams(7), fake_evaluator, LENGTH)
    run_dir = tmp_path / ARM_DE
    recorder = RunRecorder(run_dir, cfg, ARM_DE)
    recorder.complete(
        run_de(cfg, ARM_DE, initial, fake_evaluator, make_streams(7), recorder)
    )
    assert len(children(run_dir)) == cfg.budget_for(ARM_DE)
    adults = np.load(run_dir / "adults.npz")["adults"]
    assert adults.shape == (cfg.de_generations + 1, cfg.population_size, LENGTH)
    worst = [float(row["worst"]) for row in generations(run_dir)]
    assert all(later <= earlier for earlier, later in pairwise(worst))
