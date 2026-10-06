"""Every setting of a run, the arms, and the random-number streams of a seed."""

import math
from dataclasses import asdict, dataclass

import numpy as np

ARM_DIFFERENCE = "difference"
ARM_MIXTURE = "mixture"
ARM_RANDOM = "random"

# Step size x step direction factorial (all inside the same generational EA frame).
ARM_NORMALISED = "normalised"  # population direction, fixed size
ARM_SIZE_MATCHED = "size_matched"  # random direction, population (shrinking) size
ARM_GAUSSIAN = "gaussian"  # random direction, fixed size
FACTORIAL_ARMS = (ARM_DIFFERENCE, ARM_NORMALISED, ARM_SIZE_MATCHED, ARM_GAUSSIAN)

# Reference: canonical DE/rand/1/bin with one-to-one survivor selection.
ARM_DE = "de_rand_1_bin"  # textbook F and Cr
ARM_DE_MATCHED = "de_rand_1_bin_matched"  # the EA arms' F and Cr
DE_ARMS = (ARM_DE, ARM_DE_MATCHED)

ALL_ARMS = (*FACTORIAL_ARMS, ARM_MIXTURE, *DE_ARMS, ARM_RANDOM)

# Order in tables and figures: shrinking-size cells, fixed-size cells,
# references, baseline.
REFERENCES = (ARM_MIXTURE, *DE_ARMS)
ARM_ORDER = (
    ARM_DIFFERENCE,
    ARM_SIZE_MATCHED,
    ARM_NORMALISED,
    ARM_GAUSSIAN,
    *REFERENCES,
    ARM_RANDOM,
)


@dataclass(frozen=True)
class RunConfig:
    seed: int
    generations: int = 800
    population_size: int = 12
    tournament_size: int = 3
    duration: float = 15.0
    spawn: tuple[float, float, float] = (0.0, 0.0, 0.1)
    target_xy: tuple[float, float] = (2.0, 0.0)
    hidden_size: int = 6
    phase_hz: float = 1.0
    init_sd: float = 0.5
    gaussian_sd: float = 0.15
    # F(b - c) of two initial genomes has SD 0.15, the default Gaussian step.
    # --gaussian-sd leaves F alone.
    scale_f: float = 0.15 / (math.sqrt(2) * 0.5)
    crossover_rate: float = 0.2
    replacement_probability: float = 0.10
    body: str = "spider_8"
    de_scale_f: float = 0.5
    de_crossover_rate: float = 0.9
    step_log_every: int = 10

    def __post_init__(self) -> None:
        if self.population_size < max(3, self.tournament_size):
            raise ValueError(
                "population_size must allow a parent, two distinct donors "
                "and the tournament"
            )
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        if not 0.0 <= self.replacement_probability <= 1.0:
            raise ValueError("replacement_probability must be in [0, 1]")
        if not 0.0 < self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in (0, 1]")
        if not 0.0 < self.de_crossover_rate <= 1.0:
            raise ValueError("de_crossover_rate must be in (0, 1]")
        if self.step_log_every < 1:
            raise ValueError("step_log_every must be >= 1")

    @property
    def children_per_generation(self) -> int:
        return self.population_size - 1

    @property
    def budget(self) -> int:
        return self.population_size + self.children_per_generation * self.generations

    @property
    def de_generations(self) -> int:
        """Canonical DE generations (population_size trials each) in the EA budget."""
        return (self.budget - self.population_size) // self.population_size

    def budget_for(self, arm: str) -> int:
        if arm in DE_ARMS:
            return self.population_size * (1 + self.de_generations)
        return self.budget

    def generations_for(self, arm: str) -> int:
        return self.de_generations if arm in DE_ARMS else self.generations

    def replacement_probability_for(self, arm: str) -> float:
        if arm in (ARM_DIFFERENCE, ARM_NORMALISED, ARM_SIZE_MATCHED):
            return 0.0
        if arm == ARM_MIXTURE:
            return self.replacement_probability
        if arm == ARM_GAUSSIAN:
            return 1.0
        raise ValueError(f"arm {arm!r} has no mutation operator")

    def de_parameters_for(self, arm: str) -> tuple[float, float]:
        if arm == ARM_DE:
            return self.de_scale_f, self.de_crossover_rate
        if arm == ARM_DE_MATCHED:
            return self.scale_f, self.crossover_rate
        raise ValueError(f"arm {arm!r} is not a DE arm")

    def to_dict(self) -> dict:
        return asdict(self)


STREAM_NAMES = ("init", "selection", "mask", "gaussian", "replacement", "random_search")


@dataclass(frozen=True)
class Streams:
    init: np.random.Generator
    selection: np.random.Generator
    mask: np.random.Generator
    gaussian: np.random.Generator
    replacement: np.random.Generator
    random_search: np.random.Generator


def make_streams(seed: int) -> Streams:
    """Independent generators derived from one seed.

    Keeping each kind of draw on its own stream means that one arm drawing, say,
    an extra Gaussian number cannot shift the parents another arm selects.
    """
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return Streams(*(np.random.default_rng(child) for child in children))
