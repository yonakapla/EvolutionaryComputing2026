"""Replay each run's best controller and compare it with the recorded distance.

    uv run python -m mutation_ab.replay mutation_ab/results/final_spider --out replay.csv
"""

import argparse
from pathlib import Path

import numpy as np

from mutation_ab.analysis import Run, load, write_csv
from mutation_ab.config import FACTORIAL_ARMS, RunConfig
from mutation_ab.controller import genome_length, n_inputs
from mutation_ab.evaluate import evaluate
from mutation_ab.records import genome_sha1
from mutation_ab.streams import make_streams
from mutation_ab.world import build_model


def config_from(stored: dict) -> RunConfig:
    return RunConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in stored.items()})


def best_genome(run: Run, length: int) -> tuple[np.ndarray, float]:
    """The genome of the run's best evaluation and its recorded distance. EA and DE runs keep it in
    their stored populations; random search samples are regenerated from the seeded streams."""
    best = min(run.children, key=lambda child: child["distance"])
    adults = run.directory / "adults.npz"
    candidates = list(np.load(adults)["adults"].reshape(-1, length)) if adults.exists() else []
    if not candidates:
        cfg = config_from(run.config)
        streams = make_streams(cfg.seed)
        candidates = list(streams.init.normal(0.0, cfg.init_sd, (cfg.population_size, length)))
        candidates += [streams.random_search.normal(0.0, cfg.init_sd, length) for _ in run.children]
    for genome in candidates:
        if genome_sha1(genome) == best["genome_sha1"]:
            return genome, best["distance"]
    raise ValueError(f"best genome of {run.directory} not found")


def replay(root: Path) -> list[dict]:
    runs = load(root)
    cfg = config_from(next(iter(next(iter(runs.values())).values())).config)
    model, _ = build_model(cfg)
    length = genome_length(n_inputs(model), cfg.hidden_size, model.nu)
    rows = []
    for arm, by_seed in runs.items():
        for seed, run in by_seed.items():
            genome, recorded = best_genome(run, length)
            replayed = evaluate(genome, model, config_from(run.config)).distance
            rows.append({"seed": seed, "arm": arm, "recorded": recorded, "replayed": replayed,
                         "difference": replayed - recorded})
    return rows


def size_contrast(rows: list[dict]) -> np.ndarray | None:
    """Per-seed size contrast (A + C)/2 - (B + D)/2 from the replayed distances."""
    by_arm = {arm: {r["seed"]: r["replayed"] for r in rows if r["arm"] == arm} for arm in FACTORIAL_ARMS}
    if not all(by_arm.values()):
        return None
    a, b, c, d = (np.array([by_arm[arm][s] for s in sorted(by_arm[arm])]) for arm in FACTORIAL_ARMS)
    return (a + c) / 2 - (b + d) / 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay each run's best controller")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, help="CSV output (default: <root>/analysis/replay.csv)")
    args = parser.parse_args(argv)
    rows = replay(args.root)
    out = args.out or args.root / "analysis" / "replay.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    gaps = np.abs([r["difference"] for r in rows])
    print(f"replayed {len(rows)} best controllers: {int(np.sum(gaps == 0))} exact, "
          f"{int(np.sum(gaps < 1e-3))} within 1 mm, largest difference {gaps.max():.3g} m")
    contrast = size_contrast(rows)
    if contrast is not None:
        print(f"size contrast from replays: {contrast.mean():+.3f} m, positive in {int(np.sum(contrast > 0))}/{len(contrast)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
