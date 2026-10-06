"""Replaying each run's best controller in a fresh simulation."""

from mutation_ab import replay, run

ARMS = "difference,normalised,size_matched,gaussian,de_rand_1_bin,random"


def test_replay_reproduces_every_recorded_best(tmp_path):
    args = ["--out", str(tmp_path), "--seeds", "908,909", "--arms", ARMS]
    tiny = ["--generations", "2", "--population", "4", "--duration", "0.3"]
    assert run.main([*args, *tiny]) == 0
    rows = replay.replay(tmp_path)
    assert len(rows) == 12
    assert all(row["difference"] == 0 for row in rows)
