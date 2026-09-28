from dataclasses import dataclass

import numpy as np

from mutation_ab.config import RunConfig
from mutation_ab.metrics import rms
from mutation_ab.streams import Streams

KIND_DIFFERENCE = "difference"
KIND_GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class Proposal:
    child: np.ndarray
    parent: int
    donors: tuple[int, int]
    kind: str
    proposal_rms: float
    change_rms: float


def tournament(fitness: np.ndarray, size: int, rng: np.random.Generator) -> int:
    entrants = rng.choice(len(fitness), size=size, replace=False)
    return int(entrants[np.argmin(fitness[entrants])])


def draw_donors(n: int, parent: int, rng: np.random.Generator) -> tuple[int, int]:
    candidates = np.delete(np.arange(n), parent)
    b, c = rng.choice(candidates, size=2, replace=False)
    return int(b), int(c)


def binomial_mask(length: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(length) < rate
    mask[rng.integers(length)] = True
    return mask


def propose_child(
    genomes: np.ndarray,
    fitness: np.ndarray,
    cfg: RunConfig,
    replacement_probability: float,
    streams: Streams,
) -> Proposal:
    length = genomes.shape[1]
    parent = tournament(fitness, cfg.tournament_size, streams.selection)
    b, c = draw_donors(len(genomes), parent, streams.selection)
    # Both draws happen in every arm so that A and B consume their streams identically.
    replace = streams.replacement.random() < replacement_probability
    noise = streams.gaussian.normal(0.0, cfg.gaussian_sd, length)
    delta = noise if replace else cfg.scale_f * (genomes[b] - genomes[c])
    mask = binomial_mask(length, cfg.crossover_rate, streams.mask)
    child = np.where(mask, genomes[parent] + delta, genomes[parent])
    return Proposal(
        child=child,
        parent=parent,
        donors=(b, c),
        kind=KIND_GAUSSIAN if replace else KIND_DIFFERENCE,
        proposal_rms=rms(delta),
        change_rms=rms(child - genomes[parent]),
    )


def elite_index(fitness: np.ndarray, uids: np.ndarray) -> int:
    return int(np.lexsort((uids, fitness))[0])
