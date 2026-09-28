import hashlib
import json
import math
from dataclasses import asdict, dataclass

ARM_DIFFERENCE = "difference"
ARM_MIXTURE = "mixture"
ARM_RANDOM = "random"
ARMS = (ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM)


@dataclass(frozen=True)
class RunConfig:
    seed: int
    generations: int = 80
    population_size: int = 12
    tournament_size: int = 3
    duration: float = 15.0
    spawn: tuple[float, float, float] = (0.0, 0.0, 0.1)
    target_xy: tuple[float, float] = (2.0, 0.0)
    hidden_size: int = 6
    phase_hz: float = 1.0
    init_sd: float = 0.5
    gaussian_sd: float = 0.15
    scale_f: float = 0.15 / (math.sqrt(2) * 0.5)
    crossover_rate: float = 0.2
    replacement_probability: float = 0.10
    terrain_seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size < max(3, self.tournament_size):
            msg = "population_size must allow a parent, two distinct donors and the tournament"
            raise ValueError(msg)
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        if not 0.0 <= self.replacement_probability <= 1.0:
            raise ValueError("replacement_probability must be in [0, 1]")
        if not 0.0 < self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in (0, 1]")

    @property
    def children_per_generation(self) -> int:
        return self.population_size - 1

    @property
    def budget(self) -> int:
        return self.population_size + self.children_per_generation * self.generations

    def replacement_probability_for(self, arm: str) -> float:
        if arm == ARM_DIFFERENCE:
            return 0.0
        if arm == ARM_MIXTURE:
            return self.replacement_probability
        raise ValueError(f"arm {arm!r} has no mutation operator")

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(encoded).hexdigest()[:12]
