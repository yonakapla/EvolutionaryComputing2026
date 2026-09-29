import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mutation_ab.config import ARM_RANDOM, ARMS, RunConfig
from mutation_ab.evaluate import EvalResult
from mutation_ab.streams import make_streams

FAKE_HASHES = {"terrain_sha1": "fake", "model_sha1": "fake"}


def fake_evaluator(genome: np.ndarray) -> EvalResult:
    xy = (float(genome[0]), float(genome[1]))
    return EvalResult(final_xy=xy, distance=float(np.hypot(xy[0] - 2.0, xy[1])), warnings=0)


def make_fake_root(root: Path, seeds, **overrides) -> Path:
    from mutation_ab.ea_arm import run_arm
    from mutation_ab.initial import make_initial
    from mutation_ab.random_search import run_random
    from mutation_ab.records import RunRecorder

    for seed in seeds:
        cfg = RunConfig(seed=seed, **overrides)
        initial = make_initial(cfg, make_streams(seed), fake_evaluator, 222)
        for arm in ARMS:
            recorder = RunRecorder(root / f"seed_{seed}" / arm, cfg, arm, FAKE_HASHES)
            if arm == ARM_RANDOM:
                summary = run_random(cfg, initial, fake_evaluator, make_streams(seed), recorder)
            else:
                summary = run_arm(cfg, arm, initial, fake_evaluator, make_streams(seed), recorder)
            recorder.complete(summary)
    return root
