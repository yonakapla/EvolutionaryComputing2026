from datetime import datetime

from mutation_ab.progress import Heartbeat, count_progress, progress_line

ARMS = ("difference", "mixture", "random")


def write_run(root, seed, arm, lines, marker=None):
    run = root / f"seed_{seed}" / arm
    run.mkdir(parents=True)
    (run / "children.jsonl").write_text("{}\n" * lines)
    if marker:
        (run / marker).write_text("{}")


def test_count_progress_sums_evaluations_and_finished_runs(tmp_path):
    write_run(tmp_path, 800, "difference", 100, "COMPLETE")
    write_run(tmp_path, 800, "mixture", 40)
    write_run(tmp_path, 801, "difference", 100, "FAILED.json")
    assert count_progress(tmp_path, [800, 801], ARMS) == (240, 2)


def test_count_progress_handles_runs_not_started(tmp_path):
    assert count_progress(tmp_path, [800], ARMS) == (0, 0)


def test_progress_line_reports_percent_rate_runs_and_eta():
    line = progress_line(
        evals=63_910,
        total=99_360,
        finished=10,
        runs=30,
        elapsed_s=1_917.3,
        now=datetime(2026, 9, 28, 15, 5, 55),
    )
    assert line == "15:05:55  64% of 99,360 evaluations, 10/30 runs done, ETA 15:23"


def test_progress_line_before_any_evaluation():
    line = progress_line(
        evals=0,
        total=100,
        finished=0,
        runs=3,
        elapsed_s=5.0,
        now=datetime(2026, 9, 28, 15, 0, 0),
    )
    assert line == "15:00:00  0% of 100 evaluations, 0/3 runs done, ETA --:--"


def test_heartbeat_prints_a_final_line_on_exit(tmp_path, capsys):
    write_run(tmp_path, 800, "difference", 10, "COMPLETE")
    with Heartbeat(tmp_path, [800], ARMS, dict.fromkeys(ARMS, 10), interval_s=3600):
        pass
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert "33% of 30 evaluations, 1/3 runs done" in last


def test_heartbeat_disabled_prints_nothing(tmp_path, capsys):
    with Heartbeat(tmp_path, [800], ARMS, dict.fromkeys(ARMS, 10), interval_s=0):
        pass
    assert capsys.readouterr().out == ""


def test_heartbeat_total_uses_each_arms_budget(tmp_path, capsys):
    budgets = {"gaussian": 892, "de_rand_1_bin": 888}
    with Heartbeat(tmp_path, [1, 2], tuple(budgets), budgets, interval_s=3600):
        pass
    assert "of 3,560 evaluations" in capsys.readouterr().out
