import json

import pytest

from mutation_ab import run
from mutation_ab.config import ARMS
from mutation_ab.evaluate import UnstableSimulation

TINY = ["--generations", "2", "--population", "4", "--duration", "0.2"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("700-702", [700, 701, 702]), ("5,3", [5, 3]), ("900", [900])],
)
def test_parse_seeds(text, expected):
    assert run.parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["700-705,703", "705-700", "abc", "", "-1"])
def test_parse_seeds_rejects_bad_input(text):
    with pytest.raises(ValueError):
        run.parse_seeds(text)


def test_refuses_existing_seed_directory_without_running(tmp_path):
    existing = tmp_path / "seed_900"
    existing.mkdir()
    (existing / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        run.main(["--out", str(tmp_path), "--seeds", "900,901", *TINY])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed_900"]
    assert (existing / "keep.txt").read_text() == "x"


def _records(root, seed, arm):
    lines = (root / f"seed_{seed}" / arm / "children.jsonl").read_text().splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "wall_s"} for line in lines]


def test_serial_and_parallel_runs_are_identical(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    assert run.main(["--out", str(serial), "--seeds", "900,901", "--workers", "1", *TINY]) == 0
    assert run.main(["--out", str(parallel), "--seeds", "900,901", "--workers", "2", *TINY]) == 0
    for seed in (900, 901):
        for arm in ARMS:
            assert (parallel / f"seed_{seed}" / arm / "COMPLETE").exists()
            assert _records(serial, seed, arm) == _records(parallel, seed, arm)


def test_unstable_arm_is_marked_failed_and_others_finish(tmp_path, monkeypatch):
    real = run.evaluate
    calls = {"n": 0}

    def flaky(genome, model, cfg):
        calls["n"] += 1
        if calls["n"] == 6:
            raise UnstableSimulation("blew up")
        return real(genome, model, cfg)

    monkeypatch.setattr(run, "evaluate", flaky)
    assert run.main(["--out", str(tmp_path), "--seeds", "902", "--workers", "1", *TINY]) == 1
    seed_dir = tmp_path / "seed_902"
    failed = [arm for arm in ARMS if (seed_dir / arm / "FAILED.json").exists()]
    assert failed == ["difference"]
    assert not (seed_dir / "difference" / "COMPLETE").exists()
    assert (seed_dir / "mixture" / "COMPLETE").exists()
    assert (seed_dir / "random" / "COMPLETE").exists()
