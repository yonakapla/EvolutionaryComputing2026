import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mutation_ab.config import RunConfig
from mutation_ab.evaluate import EvalResult
from mutation_ab.records import RunRecorder, genome_sha1
from mutation_ab.streams import Streams

Evaluator = Callable[[np.ndarray], EvalResult]


@dataclass(frozen=True)
class InitialPopulation:
    genomes: np.ndarray
    results: tuple[EvalResult, ...]
    wall_s: tuple[float, ...]


def make_initial(
    cfg: RunConfig, streams: Streams, evaluator: Evaluator, length: int
) -> InitialPopulation:
    genomes = streams.init.normal(0.0, cfg.init_sd, (cfg.population_size, length))
    results, wall = [], []
    for genome in genomes:
        started = time.perf_counter()
        results.append(evaluator(genome))
        wall.append(time.perf_counter() - started)
    return InitialPopulation(genomes=genomes, results=tuple(results), wall_s=tuple(wall))


def record_founders(initial: InitialPopulation, recorder: RunRecorder) -> None:
    for uid, (genome, result, wall) in enumerate(
        zip(initial.genomes, initial.results, initial.wall_s, strict=True)
    ):
        recorder.child(
            {
                "uid": uid,
                "generation": 0,
                "kind": "init",
                "parent_uid": None,
                "donor_uids": [],
                "proposal_rms": None,
                "change_rms": None,
                "parent_distance": None,
                "parent_xy": None,
                "final_xy": list(result.final_xy),
                "warnings": result.warnings,
                "genome_sha1": genome_sha1(genome),
                "distance": result.distance,
                "wall_s": wall,
            }
        )
