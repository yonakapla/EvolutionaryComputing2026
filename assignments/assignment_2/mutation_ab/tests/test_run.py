"""Complete runs through the command line (population 4, 2 generations, 0.2 s
episodes)."""

import json

import pytest

from mutation_ab import run
from mutation_ab.config import ALL_ARMS
from mutation_ab.simulation import UnstableSimulation

TINY = ["--generations", "2", "--population", "4", "--duration", "0.2"]


def records(root, seed, arm):
    lines = (root / f"seed_{seed}" / arm / "children.jsonl").read_text().splitlines()
    return [
        {k: v for k, v in json.loads(line).items() if k != "wall_s"} for line in lines
    ]


def test_existing_results_are_not_overwritten(tmp_path):
    existing = tmp_path / "seed_900"
    existing.mkdir()
    (existing / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        run.main(["--out", str(tmp_path), "--seeds", "900,901", *TINY])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed_900"]


def test_parallel_workers_give_the_same_runs(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    seeds = ["--seeds", "900,901"]
    assert run.main(["--out", str(serial), *seeds, "--workers", "1", *TINY]) == 0
    assert run.main(["--out", str(parallel), *seeds, "--workers", "2", *TINY]) == 0
    for seed in (900, 901):
        for arm in ALL_ARMS:
            assert records(serial, seed, arm) == records(parallel, seed, arm)


def test_a_diverged_arm_fails_alone(tmp_path, monkeypatch):
    real = run.evaluate
    calls = {"n": 0}

    def diverges_once(genome, model, cfg):
        calls["n"] += 1
        if calls["n"] == 6:
            raise UnstableSimulation("diverged")
        return real(genome, model, cfg)

    monkeypatch.setattr(run, "evaluate", diverges_once)
    args = ["--out", str(tmp_path), "--seeds", "902", "--workers", "1", *TINY]
    assert run.main(args) == 1
    seed_dir = tmp_path / "seed_902"
    failed = [arm for arm in ALL_ARMS if (seed_dir / arm / "FAILED.json").exists()]
    complete = [arm for arm in ALL_ARMS if (seed_dir / arm / "COMPLETE").exists()]
    assert failed == ["difference"]
    assert complete == [arm for arm in ALL_ARMS if arm != "difference"]
