from dataclasses import dataclass

import numpy as np

from mutation_ab.config import STEP_DIFFERENCE, STEP_NORMALISED, STEP_SIZE_MATCHED, RunConfig
from mutation_ab.metrics import rms
from mutation_ab.streams import Streams

KIND_DIFFERENCE = "difference"
KIND_GAUSSIAN = "gaussian"
KIND_NORMALISED = "normalised"
KIND_SIZE_MATCHED = "size_matched"
KIND_DE = "de"


@dataclass(frozen=True)
class Proposal:
    child: np.ndarray
    parent: int
    donors: tuple[int, ...]
    kind: str
    proposal_rms: float
    change_rms: float
    difference_rms: float


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


def shaped_step(difference: np.ndarray, noise: np.ndarray, step: str, target_rms: float) -> tuple[np.ndarray, str]:
    """Turn F(b - c) into the arm's step.

    Sizes are RMS over all coordinates, before the crossover mask. The normalised step therefore
    has the Gaussian step's overall size, but keeps the difference's shape: when parents differ in
    only a few weights, those few weights get large changes. A zero difference has no direction,
    so the normalised arm then falls back to the Gaussian draw (logged as kind "gaussian").
    """
    if step == STEP_DIFFERENCE:
        return difference, KIND_DIFFERENCE
    size = rms(difference)
    if step == STEP_SIZE_MATCHED:
        return noise / rms(noise) * size, KIND_SIZE_MATCHED
    if step == STEP_NORMALISED:
        if size == 0.0:
            return noise, KIND_GAUSSIAN
        return difference / size * target_rms, KIND_NORMALISED
    raise ValueError(f"unknown step {step!r}")


def propose_child(
    genomes: np.ndarray,
    fitness: np.ndarray,
    cfg: RunConfig,
    replacement_probability: float,
    streams: Streams,
    step: str = STEP_DIFFERENCE,
) -> Proposal:
    length = genomes.shape[1]
    parent = tournament(fitness, cfg.tournament_size, streams.selection)
    b, c = draw_donors(len(genomes), parent, streams.selection)
    # Every draw happens in every arm so that all arms consume their streams identically.
    replace = streams.replacement.random() < replacement_probability
    noise = streams.gaussian.normal(0.0, cfg.gaussian_sd, length)
    difference = cfg.scale_f * (genomes[b] - genomes[c])
    if replace:
        delta, kind = noise, KIND_GAUSSIAN
    else:
        delta, kind = shaped_step(difference, noise, step, cfg.gaussian_sd)
    mask = binomial_mask(length, cfg.crossover_rate, streams.mask)
    child = np.where(mask, genomes[parent] + delta, genomes[parent])
    return Proposal(
        child=child,
        parent=parent,
        donors=(b, c),
        kind=kind,
        proposal_rms=rms(delta),
        change_rms=rms(child - genomes[parent]),
        difference_rms=rms(difference),
    )


def de_trial(
    genomes: np.ndarray, target: int, scale_f: float, crossover_rate: float, streams: Streams
) -> Proposal:
    """Canonical DE/rand/1/bin trial for `target`: base and donors distinct and != target."""
    length = genomes.shape[1]
    candidates = np.delete(np.arange(len(genomes)), target)
    a, b, c = (int(i) for i in streams.selection.choice(candidates, size=3, replace=False))
    difference = scale_f * (genomes[b] - genomes[c])
    mask = binomial_mask(length, crossover_rate, streams.mask)
    trial = np.where(mask, genomes[a] + difference, genomes[target])
    return Proposal(
        child=trial,
        parent=target,
        donors=(a, b, c),
        kind=KIND_DE,
        proposal_rms=rms(difference),
        change_rms=rms(trial - genomes[target]),
        difference_rms=rms(difference),
    )


def elite_index(fitness: np.ndarray, uids: np.ndarray) -> int:
    return int(np.lexsort((uids, fitness))[0])
