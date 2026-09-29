import csv
import json

import numpy as np
import pytest
from conftest import FAKE_HASHES, fake_evaluator

from mutation_ab import run
from mutation_ab.config import (
    ALL_ARMS,
    ARM_DE,
    ARM_DE_MATCHED,
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_NORMALISED,
    ARM_SIZE_MATCHED,
    STEP_DIFFERENCE,
    STEP_NORMALISED,
    STEP_SIZE_MATCHED,
    RunConfig,
)
from mutation_ab.de_arm import run_de
from mutation_ab.ea_arm import run_arm
from mutation_ab.initial import make_initial
from mutation_ab.metrics import rms
from mutation_ab.operators import KIND_GAUSSIAN, KIND_NORMALISED, KIND_SIZE_MATCHED, de_trial, propose_child
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams

LENGTH = 222
SMALL = {"generations": 5, "population_size": 6}


def population(seed=0, size=12):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, (size, LENGTH)), rng.random(size)


def children(run_dir):
    return [json.loads(line) for line in (run_dir / "children.jsonl").read_text().splitlines()]


def test_arm_settings():
    cfg = RunConfig(seed=1)
    assert cfg.replacement_probability_for(ARM_GAUSSIAN) == 1.0
    for arm in (ARM_DIFFERENCE, ARM_NORMALISED, ARM_SIZE_MATCHED):
        assert cfg.replacement_probability_for(arm) == 0.0
    assert cfg.step_for(ARM_DIFFERENCE) == STEP_DIFFERENCE
    assert cfg.step_for(ARM_NORMALISED) == STEP_NORMALISED
    assert cfg.step_for(ARM_SIZE_MATCHED) == STEP_SIZE_MATCHED
    assert cfg.de_parameters_for(ARM_DE) == (0.5, 0.9)
    assert cfg.de_parameters_for(ARM_DE_MATCHED) == (cfg.scale_f, cfg.crossover_rate)


def test_de_budget_never_exceeds_ea_budget():
    for generations in (5, 80, 800):
        cfg = RunConfig(seed=1, generations=generations)
        de_evaluations = cfg.population_size * (1 + cfg.de_generations)
        assert cfg.budget - cfg.population_size < de_evaluations <= cfg.budget


def test_size_matched_keeps_difference_size_but_not_direction():
    genomes, fitness = population()
    cfg = RunConfig(seed=2)
    plain = propose_child(genomes, fitness, cfg, 0.0, make_streams(2), STEP_DIFFERENCE)
    matched = propose_child(genomes, fitness, cfg, 0.0, make_streams(2), STEP_SIZE_MATCHED)
    assert matched.kind == KIND_SIZE_MATCHED
    assert (matched.parent, matched.donors) == (plain.parent, plain.donors)
    assert matched.proposal_rms == pytest.approx(plain.proposal_rms)
    assert not np.allclose(matched.child, plain.child)


def test_normalised_keeps_difference_direction_at_fixed_size():
    genomes, fitness = population()
    cfg = RunConfig(seed=3)
    plain = propose_child(genomes, fitness, cfg, 0.0, make_streams(3), STEP_DIFFERENCE)
    normalised = propose_child(genomes, fitness, cfg, 0.0, make_streams(3), STEP_NORMALISED)
    assert normalised.kind == KIND_NORMALISED
    assert normalised.proposal_rms == pytest.approx(cfg.gaussian_sd)
    parent = genomes[plain.parent]
    changed = plain.child != parent
    ratio = (normalised.child - parent)[changed] / (plain.child - parent)[changed]
    np.testing.assert_allclose(ratio, cfg.gaussian_sd / plain.proposal_rms)


def test_normalised_falls_back_to_gaussian_when_population_is_identical():
    genomes = np.tile(np.random.default_rng(4).normal(size=LENGTH), (12, 1))
    proposal = propose_child(genomes, np.zeros(12), RunConfig(seed=4), 0.0, make_streams(4), STEP_NORMALISED)
    assert proposal.kind == KIND_GAUSSIAN
    assert proposal.change_rms > 0


@pytest.mark.parametrize("step", [STEP_NORMALISED, STEP_SIZE_MATCHED])
def test_step_modes_consume_streams_like_difference(step):
    genomes, fitness = population()
    cfg = RunConfig(seed=5)
    control, treatment = make_streams(5), make_streams(5)
    for _ in range(50):
        propose_child(genomes, fitness, cfg, 0.0, control, STEP_DIFFERENCE)
        propose_child(genomes, fitness, cfg, 0.0, treatment, step)
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
        np.testing.assert_allclose(trial.child[mixed], (genomes[a] + 0.5 * (genomes[b] - genomes[c]))[mixed])
        assert trial.proposal_rms == pytest.approx(rms(0.5 * (genomes[b] - genomes[c])))


@pytest.mark.parametrize("arm", [ARM_DE, ARM_DE_MATCHED])
def test_de_arm_is_one_to_one_and_within_budget(tmp_path, arm):
    cfg = RunConfig(seed=7, **SMALL)
    initial = make_initial(cfg, make_streams(7), fake_evaluator, LENGTH)
    recorder = RunRecorder(tmp_path / arm, cfg, arm, FAKE_HASHES)
    summary = run_de(cfg, arm, initial, fake_evaluator, make_streams(7), recorder)
    recorder.complete(summary)
    records = children(tmp_path / arm)
    assert len(records) == cfg.population_size * (1 + cfg.de_generations) <= cfg.budget
    with (tmp_path / arm / "generations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    # One-to-one replacement: population size is constant and no slot ever gets worse.
    adults = np.load(tmp_path / arm / "adults.npz")["adults"]
    assert adults.shape == (cfg.de_generations + 1, cfg.population_size, LENGTH)
    worst = [float(row["worst"]) for row in rows]
    assert all(later <= earlier for earlier, later in zip(worst, worst[1:]))
    trials = [r for r in records if r["kind"] == "de"]
    assert all(r["parent_uid"] not in r["donor_uids"] for r in trials)


@pytest.mark.parametrize("arm", [ARM_NORMALISED, ARM_SIZE_MATCHED, ARM_GAUSSIAN])
def test_new_ea_arms_log_their_step_kind_and_steps(tmp_path, arm):
    cfg = RunConfig(seed=8, step_log_every=2, **SMALL)
    initial = make_initial(cfg, make_streams(8), fake_evaluator, LENGTH)
    recorder = RunRecorder(tmp_path / arm, cfg, arm, FAKE_HASHES)
    recorder.complete(run_arm(cfg, arm, initial, fake_evaluator, make_streams(8), recorder))
    kinds = {r["kind"] for r in children(tmp_path / arm) if r["kind"] != "init"}
    assert kinds <= {arm, KIND_GAUSSIAN}
    steps = np.load(tmp_path / arm / "steps.npz")
    assert set(steps["generation"]) == {2, 4}
    assert steps["delta"].shape == (2 * cfg.children_per_generation, LENGTH)


def test_parse_arms():
    assert run.parse_arms("difference,gaussian") == ("difference", "gaussian")
    for bad in ("", "nope", "gaussian,gaussian"):
        with pytest.raises(ValueError):
            run.parse_arms(bad)


def test_every_arm_runs_on_spider_in_flat_world(tmp_path):
    args = ["--out", str(tmp_path), "--seeds", "905", "--workers", "1", "--arms", ",".join(ALL_ARMS),
            "--body", "spider_8", "--world", "flat", "--generations", "2", "--population", "4",
            "--duration", "0.2", "--heartbeat", "0"]
    assert run.main(args) == 0
    for arm in ALL_ARMS:
        assert (tmp_path / "seed_905" / arm / "COMPLETE").exists()
    config = json.loads((tmp_path / "seed_905" / ARM_GAUSSIAN / "config.json").read_text())
    assert (config["config"]["body"], config["config"]["world"]) == ("spider_8", "flat")


@pytest.mark.parametrize("body", ["np", "does_not_exist"])
def test_unknown_body_is_rejected(body):
    from mutation_ab.world import build_model

    with pytest.raises(ValueError, match="unknown John Set body"):
        build_model(RunConfig(seed=1, body=body))


def test_de_budget_and_generations_per_arm():
    cfg = RunConfig(seed=1, generations=80)
    assert cfg.budget_for(ARM_GAUSSIAN) == cfg.budget
    assert cfg.budget_for(ARM_DE) == 12 * (1 + cfg.de_generations) == 888
    assert cfg.generations_for(ARM_DE) == cfg.de_generations
    assert cfg.generations_for(ARM_GAUSSIAN) == 80


def test_heartbeat_total_uses_each_arms_budget(tmp_path, capsys):
    from mutation_ab.progress import Heartbeat

    with Heartbeat(tmp_path, [1, 2], (ARM_GAUSSIAN, ARM_DE), {ARM_GAUSSIAN: 892, ARM_DE: 888}, interval_s=3600):
        pass
    assert "(0/3,560 evals)" in capsys.readouterr().out


def test_gaussian_sd_option_reaches_the_config(tmp_path):
    args = ["--out", str(tmp_path), "--seeds", "906", "--workers", "1", "--arms", ARM_GAUSSIAN,
            "--gaussian-sd", "0.3", "--generations", "1", "--population", "4", "--duration", "0.2",
            "--heartbeat", "0"]
    assert run.main(args) == 0
    config = json.loads((tmp_path / "seed_906" / ARM_GAUSSIAN / "config.json").read_text())
    assert config["config"]["gaussian_sd"] == 0.3
