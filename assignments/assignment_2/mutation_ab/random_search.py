import time

import numpy as np

from mutation_ab.config import RunConfig
from mutation_ab.initial import Evaluator, InitialPopulation, record_founders
from mutation_ab.metrics import genotype_diversity
from mutation_ab.records import RunRecorder, evaluation_record
from mutation_ab.streams import Streams


def _row(
    generation: int, batch: list[float], best_so_far: float, evaluations: int
) -> dict:
    return {
        "generation": generation,
        "best": min(batch),
        "mean": float(np.mean(batch)),
        "worst": max(batch),
        "best_so_far": best_so_far,
        "unique": len(batch),
        "diversity": float("nan"),
        "diff_proposal_rms": float("nan"),
        "difference_rms": float("nan"),
        "n_difference": 0,
        "n_gaussian": 0,
        "evaluations": evaluations,
    }


def run_random(
    cfg: RunConfig,
    initial: InitialPopulation,
    evaluator: Evaluator,
    streams: Streams,
    recorder: RunRecorder,
) -> dict:
    record_founders(initial, recorder)
    founder_scores = [result.distance for result in initial.results]
    best_so_far = min(founder_scores)
    evaluations = len(founder_scores)
    row = _row(0, founder_scores, best_so_far, evaluations)
    row["diversity"] = genotype_diversity(initial.genomes)
    recorder.generation(row)

    uid = len(founder_scores)
    length = initial.genomes.shape[1]
    for generation in range(1, cfg.generations + 1):
        batch = []
        for _ in range(cfg.children_per_generation):
            genome = streams.random_search.normal(0.0, cfg.init_sd, length)
            started = time.perf_counter()
            result = evaluator(genome)
            evaluations += 1
            batch.append(result.distance)
            recorder.child(
                evaluation_record(
                    uid,
                    generation,
                    "random",
                    genome,
                    result,
                    time.perf_counter() - started,
                )
            )
            uid += 1
        best_so_far = min(best_so_far, min(batch))
        recorder.generation(_row(generation, batch, best_so_far, evaluations))
    return {"evaluations": evaluations, "best_so_far": best_so_far}
