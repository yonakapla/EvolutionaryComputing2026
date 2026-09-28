import csv
import json
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import make_fake_root

from mutation_ab import analysis
from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE

SMALL = {"generations": 6, "population_size": 6}


@pytest.fixture
def root(tmp_path):
    return make_fake_root(tmp_path, [1, 2, 3], **SMALL)


def test_refuses_root_with_incomplete_arm(root):
    (root / "seed_2" / ARM_MIXTURE / "COMPLETE").unlink()
    with pytest.raises(analysis.IncompleteRuns):
        analysis.load_runs(root)


def test_refuses_empty_root(tmp_path):
    with pytest.raises(analysis.IncompleteRuns):
        analysis.load_runs(tmp_path)


def test_refuses_mixed_configurations(tmp_path):
    make_fake_root(tmp_path, [1], **SMALL)
    make_fake_root(tmp_path, [2], generations=7, population_size=6)
    with pytest.raises(ValueError):
        analysis.load_runs(tmp_path)


def test_refuses_mismatched_hashes(tmp_path):
    make_fake_root(tmp_path, [1, 2], **SMALL)
    config_path = tmp_path / "seed_2" / ARM_DIFFERENCE / "config.json"
    meta = json.loads(config_path.read_text())
    meta["hashes"] = {"terrain_sha1": "different", "model_sha1": "different"}
    config_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hashes"):
        analysis.load_runs(tmp_path)


def test_exclude_leaves_seed_out_and_skips_its_incomplete_arm(tmp_path):
    root = make_fake_root(tmp_path, [1, 2, 3], **SMALL)
    (root / "seed_2" / ARM_MIXTURE / "COMPLETE").unlink()
    runs = analysis.load_runs(root, exclude={2})
    assert set(runs) == {1, 3}


def test_main_records_excluded_seeds(root):
    (root / "seed_2" / ARM_MIXTURE / "COMPLETE").unlink()
    assert analysis.main([str(root), "--exclude", "2"]) == 0
    report = json.loads((root / "analysis" / "analysis.json").read_text())
    assert report["excluded_seeds"] == [2]
    assert report["seeds"] == [1, 3]


def test_zero_probability_mixture_has_identical_metrics(tmp_path):
    root = make_fake_root(tmp_path, [4], replacement_probability=0.0, **SMALL)
    runs = analysis.load_runs(root)[4]
    diff = analysis.seed_metrics(runs[ARM_DIFFERENCE])
    mix = analysis.seed_metrics(runs[ARM_MIXTURE])
    assert diff == mix


def test_all_gaussian_generations_do_not_break_step_metrics(tmp_path):
    root = make_fake_root(tmp_path, [5], replacement_probability=1.0, **SMALL)
    metrics = analysis.seed_metrics(analysis.load_runs(root)[5][ARM_MIXTURE])
    assert np.isnan(metrics["step_mean"])
    assert metrics["improve_rate_difference"] is None


def test_holm_known_values():
    adjusted = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_paired_handles_zero_differences():
    result = analysis.paired(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert result["p"] == 1.0
    result = analysis.paired(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.5, 3.5, 4.5]))
    assert 0.0 < result["p"] <= 1.0
    assert result["mean_diff"] == pytest.approx(0.375)


def test_plateau_generation():
    flat_after_50 = np.concatenate([np.linspace(3.0, 1.0, 51), np.full(30, 1.0)])
    assert analysis.plateau_generation([flat_after_50, flat_after_50]) == 70
    still_falling = np.linspace(3.0, 1.0, 81)
    assert analysis.plateau_generation([flat_after_50, still_falling]) is None


def _force_step_collapse(root, seeds) -> None:
    for seed in seeds:
        path = root / f"seed_{seed}" / analysis.ARM_DIFFERENCE / "generations.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows[1:]:
            row["diff_proposal_rms"] = "0.0"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def test_figures_succeed_and_stay_readable_when_a_step_collapses_to_zero(tmp_path):
    root = make_fake_root(tmp_path, [1, 2], **SMALL)
    _force_step_collapse(root, [1, 2])
    runs = analysis.load_runs(root)
    out = root / "analysis"
    out.mkdir()

    analysis.figures(runs, out)
    assert (out / "fig2.png").exists()

    fig, (left, _right) = plt.subplots(1, 2)
    twin = left.twinx()
    gen1_medians = [analysis._step_panel(left, twin, runs, arm) for arm in (ARM_DIFFERENCE, ARM_MIXTURE)]
    plt.close(fig)
    assert analysis._step_panel_ylim_bottom(gen1_medians) > 1e-6


def test_go_no_go_reports_every_criterion(root):
    report = analysis.go_no_go(
        analysis.load_runs(root),
        final_seeds=10,
        workers=10,
        deadline=date(2026, 10, 8),
        now=datetime(2026, 9, 29, 12, 0),
    )
    assert set(report["criteria"]) == {"speed", "a_collapses", "b_stays_alive", "h4_measurable"}
    for criterion in report["criteria"].values():
        assert set(criterion) == {"passed", "value", "rule"}
    assert "g_final" in report


def test_go_no_go_speed_uses_seed_batches_not_raw_evaluation_count(root):
    runs = analysis.load_runs(root)
    report = analysis.go_no_go(
        runs, final_seeds=10, workers=4, deadline=date(2026, 10, 8), now=datetime(2026, 9, 29, 12, 0)
    )
    seeds = sorted(runs)
    config = runs[seeds[0]][ARM_DIFFERENCE].config
    walls = [
        c["wall_s"]
        for s in seeds
        for arm in analysis.ARMS
        for c in runs[s][arm].children
        if c["kind"] != "init"
    ]
    per_seed = config["population_size"] + (config["population_size"] - 1) * report["g_final"] * len(analysis.ARMS)
    expected_hours = math.ceil(10 / 4) * per_seed * float(np.mean(walls)) / 3600
    assert report["criteria"]["speed"]["value"]["hours"] == pytest.approx(expected_hours)


def test_main_writes_outputs(root):
    assert analysis.main([str(root), "--poc"]) == 0
    out = root / "analysis"
    for name in ("analysis.json", "seed_metrics.csv", "h4.csv", "h4_summary.csv", "fig1.pdf", "fig1.png", "fig2.pdf", "fig2.png"):
        assert (out / name).exists()
    report = json.loads((out / "analysis.json").read_text())
    assert "tests" in report
    assert report["exploratory"] is True
    assert "excluded_seeds" in report

    with (out / "h4_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["arm"] for row in rows} == {ARM_DIFFERENCE, ARM_MIXTURE}
    assert "improve_rate_difference_mean" in rows[0]
    assert "improve_rate_difference_sd" in rows[0]
