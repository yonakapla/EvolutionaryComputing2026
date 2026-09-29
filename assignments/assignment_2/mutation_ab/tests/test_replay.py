from mutation_ab import replay, run

ARMS = "difference,normalised,size_matched,gaussian,de_rand_1_bin,random"


def test_replay_reproduces_every_recorded_best(tmp_path):
    args = ["--out", str(tmp_path), "--seeds", "908,909", "--workers", "1", "--arms", ARMS, "--body", "spider_8",
            "--generations", "2", "--population", "4", "--duration", "0.3", "--heartbeat", "0"]
    assert run.main(args) == 0
    rows = replay.replay(tmp_path)
    assert len(rows) == 12
    assert all(row["difference"] == 0 for row in rows)
    assert replay.size_contrast(rows) is not None


def test_main_writes_csv(tmp_path):
    args = ["--out", str(tmp_path), "--seeds", "910", "--workers", "1", "--arms", "gaussian,random", "--body",
            "spider_8", "--generations", "1", "--population", "4", "--duration", "0.2",
            "--heartbeat", "0"]
    assert run.main(args) == 0
    assert replay.main([str(tmp_path), "--out", str(tmp_path / "replay.csv")]) == 0
    assert (tmp_path / "replay.csv").read_text().startswith("arm,difference,recorded,replayed,seed")
