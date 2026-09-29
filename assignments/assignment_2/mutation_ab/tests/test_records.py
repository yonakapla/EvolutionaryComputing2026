import json

import numpy as np
import pytest

from mutation_ab.config import RunConfig
from mutation_ab.records import GENERATION_FIELDS, RunRecorder, genome_sha1

HASHES = {"terrain_sha1": "t", "model_sha1": "m"}


def row(generation):
    return {field: 0 for field in GENERATION_FIELDS} | {"generation": generation}


def test_reports_progress_every_ten_generations_and_on_finish(tmp_path, capsys):
    recorder = RunRecorder(tmp_path / "run", RunConfig(seed=7, generations=20), "mixture", HASHES)
    for generation in range(21):
        recorder.generation(row(generation) | {"best_so_far": 1.5, "evaluations": 12 + 11 * generation})
    recorder.complete({"evaluations": 232})
    lines = capsys.readouterr().out.splitlines()
    generations = [line.split(" gen ")[1].split()[0] for line in lines if " gen " in line]
    assert generations == ["0/20", "10/20", "20/20"]
    assert all(line.startswith("[seed 7 mixture]") for line in lines)
    assert "done" in lines[-1]


def test_reports_failure(tmp_path, capsys):
    recorder = RunRecorder(tmp_path / "run", RunConfig(seed=7), "difference", HASHES)
    recorder.fail(RuntimeError("boom"))
    assert "FAILED" in capsys.readouterr().out


def test_refuses_existing_directory(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        RunRecorder(target, RunConfig(seed=1), "difference", HASHES)
    assert (target / "keep.txt").read_text() == "x"


def test_complete_run_writes_all_artifacts(tmp_path):
    recorder = RunRecorder(tmp_path / "run", RunConfig(seed=1), "difference", HASHES)
    recorder.child({"uid": 0, "distance": 1.5})
    recorder.child({"uid": 1, "distance": 1.4})
    recorder.generation(row(0))
    recorder.adults(np.zeros((4, 222)))
    recorder.adults(np.ones((4, 222)))
    recorder.complete({"evaluations": 2})

    run = tmp_path / "run"
    config = json.loads((run / "config.json").read_text())
    assert config["arm"] == "difference"
    assert config["config_hash"] == RunConfig(seed=1).config_hash()
    assert config["hashes"] == HASHES
    assert "commit" in config["git"]
    assert config["versions"]["ariel"]
    assert len((run / "children.jsonl").read_text().splitlines()) == 2
    assert (run / "generations.csv").read_text().splitlines()[0].split(",") == list(GENERATION_FIELDS)
    assert np.load(run / "adults.npz")["adults"].shape == (2, 4, 222)
    assert json.loads((run / "COMPLETE").read_text()) == {"evaluations": 2}


def test_failed_run_has_no_complete_marker(tmp_path):
    recorder = RunRecorder(tmp_path / "run", RunConfig(seed=1), "mixture", HASHES)
    recorder.fail(RuntimeError("boom"))
    assert not (tmp_path / "run" / "COMPLETE").exists()
    assert "boom" in json.loads((tmp_path / "run" / "FAILED.json").read_text())["error"]


def test_genome_hash_distinguishes_genomes():
    assert genome_sha1([0.0, 1.0]) == genome_sha1(np.array([0.0, 1.0]))
    assert genome_sha1([0.0, 1.0]) != genome_sha1([1.0, 0.0])
