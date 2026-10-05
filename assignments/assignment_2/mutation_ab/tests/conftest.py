import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM, RunConfig
from mutation_ab.evaluate import EvalResult
from mutation_ab.initial import make_initial
from mutation_ab.records import RunRecorder
from mutation_ab.run import run_one
from mutation_ab.streams import make_streams

LENGTH = 222  # genome length used with the fake evaluator
TEST_ARMS = (ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM)
FAKE_HASHES = {"model_sha1": "fake"}


def fake_evaluator(genome: np.ndarray) -> EvalResult:
    xy = (float(genome[0]), float(genome[1]))
    return EvalResult(
        final_xy=xy, distance=float(np.hypot(xy[0] - 2.0, xy[1])), warnings=0
    )


def population(seed=0, size=12):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, (size, LENGTH)), rng.random(size)


def children(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "children.jsonl").read_text().splitlines()
    ]


def make_fake_root(root: Path, seeds, arms=TEST_ARMS, **overrides) -> Path:
    for seed in seeds:
        cfg = RunConfig(seed=seed, **overrides)
        initial = make_initial(cfg, make_streams(seed), fake_evaluator, LENGTH)
        for arm in arms:
            recorder = RunRecorder(root / f"seed_{seed}" / arm, cfg, arm, FAKE_HASHES)
            recorder.complete(run_one(cfg, arm, initial, fake_evaluator, recorder))
    return root
