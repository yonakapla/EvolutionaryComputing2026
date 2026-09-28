# A2 Mutation A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested runner and analysis for the Assignment 2 A/B study: differential mutation (A) vs 10% Gaussian replacement (B), plus random search, on the john_set gecko in a seeded OlympicArena. Then run the POC (seeds 700–705, 80 generations) and report go/no-go.

**Architecture:** A self-contained Python package `mutation_ab` in `assignments/assignment_2/`. The EA arms run on the `ariel.ec` engine (`EA`, `EAOperation`, `Individual`, `Population`, SQLite database). Operators are pure numpy functions with explicit random streams. The evaluator is injected, so EA logic is unit-tested with a fake evaluator and the real MuJoCo evaluator is only used in smoke tests. Seeds run in parallel processes, and each (seed, arm) writes an immutable run directory.

**Tech Stack:** Python 3.12 (uv), numpy, scipy 1.16, matplotlib 3.10, mujoco, ARIEL (`src/ariel`, read-only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-a2-mutation-ab-design.md`

## Global Constraints

- Never modify anything under `src/ariel` (the assignment treats this as fraud). Never call `BaseWorld.store_to_xml` or `compile_terrains.py`. Always build the arena with `load_precompiled=False`.
- Body `john_set.gecko()`; world `OlympicArena` with `PerlinNoise(seed=42)` for the rugged heightmap; spawn `(0, 0, 0.1)` with `correct_collision_with_floor=True`; target XY `(2, 0)`; episode 15 s; direct control `data.ctrl[:] = actions`.
- Controller 29 → 6 tanh → 6 tanh × π/2, with biases; genome length 222; inputs qpos(13), qvel(12), target − core XY(2), sin/cos(2π·1 Hz·t)(2).
- Initialisation N(0, 0.5²); population 12; 11 children per generation; tournament of 3 distinct adults (min distance wins); one elite plus 11 children survive.
- δ = F·(b − c) with F = 0.15/(√2·0.5); donors distinct from each other and from the parent; mask Cr = 0.2 plus one forced coordinate; B replaces δ with N(0, 0.15²) with probability 0.10. The replacement coin and the Gaussian vector are drawn for every child in both arms.
- Budget per arm per seed: 12 + 11·G evaluations, including the shared initial 12. Random search: the shared 12, then 11·G fresh N(0, 0.5²) genomes.
- Fitness: final core XY Euclidean distance to the target; lower is better.
- No `from __future__ import annotations` in modules that define `EAOperation` functions: `ariel.ec` checks the runtime annotation `Population`.
- Comments only where strictly necessary (they explain *why*). No references to AI tools anywhere in code, commits or docs.
- All commands run from the worktree root unless stated. Tests: `uv run pytest assignments/assignment_2/mutation_ab/tests -q`.

## Review Focus

1. An existing output directory for any requested seed: the CLI must refuse before running anything and leave the directory untouched (test in Task 6).
2. Malformed or duplicate `--seeds` (for example `700-705,703`, `705-700`, `abc`): rejected with a clear error (test in Task 6).
3. A simulation blowing up mid-run: that (seed, arm) gets `FAILED.json` and no `COMPLETE`, the other arms and seeds still finish, the CLI exits non-zero, and the analysis refuses the root (tests in Tasks 5, 6 and 7).
4. Fitness ties after collapse (many identical genomes): the elite and tournament choices must be deterministic for a given seed (tests in Task 2).
5. A B generation in which every child was a Gaussian replacement: the step metric must be NaN-safe and the analysis must not crash (test in Task 7).

---

## File structure

```
assignments/assignment_2/mutation_ab/
  __init__.py          empty
  .gitignore           results/
  config.py            RunConfig and arm names
  streams.py           named per-purpose RNG streams from one seed
  metrics.py           rms, genotype diversity, unique-genome count
  operators.py         tournament, donors, mask, proposal/child, elite choice
  world.py             SeededOlympicArena, build_model, terrain fingerprint
  controller.py        genome layout, observation, forward pass
  evaluate.py          one headless episode → EvalResult; UnstableSimulation
  initial.py           shared initial population (drawn + evaluated once)
  records.py           RunRecorder: config.json, children.jsonl, generations.csv, adults.npz, COMPLETE/FAILED
  ea_arm.py            A/B arms on the ariel.ec EA engine
  random_search.py     equal-budget random search
  run.py               CLI: seeds × arms, parallel processes
  analysis.py          metrics, Wilcoxon + Holm, figures, POC go/no-go
  README.md            exact commands
  tests/
    conftest.py        sys.path setup + fake evaluator + fake-root builder
    test_config.py
    test_operators.py
    test_simulation.py
    test_records.py
    test_arms.py
    test_run.py
    test_analysis.py
```

---

### Task 1: Package scaffold, config, random streams, compliance guard

**Files:**
- Create: `assignments/assignment_2/mutation_ab/__init__.py` (empty)
- Create: `assignments/assignment_2/mutation_ab/.gitignore`
- Create: `assignments/assignment_2/mutation_ab/config.py`
- Create: `assignments/assignment_2/mutation_ab/streams.py`
- Create: `assignments/assignment_2/mutation_ab/tests/conftest.py`
- Test: `assignments/assignment_2/mutation_ab/tests/test_config.py`

**Interfaces:**
- Produces:
  - `RunConfig` (frozen dataclass; fields listed below) with `.children_per_generation -> int`, `.budget -> int`, `.replacement_probability_for(arm: str) -> float`, `.to_dict() -> dict`, `.config_hash() -> str`.
  - Constants `ARM_DIFFERENCE = "difference"`, `ARM_MIXTURE = "mixture"`, `ARM_RANDOM = "random"`, `ARMS`.
  - `Streams` (frozen dataclass of `np.random.Generator`: `init, selection, mask, gaussian, replacement, random_search`) and `make_streams(seed: int) -> Streams`.

- [ ] **Step 1: Create the scaffold files**

`.gitignore`:
```
results/
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_config.py`:
```python
import math
import subprocess
from pathlib import Path

import pytest

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM, RunConfig
from mutation_ab.streams import make_streams

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_budget_counts_shared_initial_population():
    cfg = RunConfig(seed=1, generations=80)
    assert cfg.children_per_generation == 11
    assert cfg.budget == 12 + 11 * 80


def test_scale_factor_matches_gaussian_step_at_initialisation():
    cfg = RunConfig(seed=1)
    assert cfg.scale_f * math.sqrt(2) * cfg.init_sd == pytest.approx(cfg.gaussian_sd)


@pytest.mark.parametrize(
    "overrides",
    [
        {"population_size": 2},
        {"population_size": 3, "tournament_size": 4},
        {"generations": 0},
        {"duration": 0.0},
        {"replacement_probability": 1.5},
        {"crossover_rate": 0.0},
    ],
)
def test_rejects_invalid_settings(overrides):
    with pytest.raises(ValueError):
        RunConfig(seed=1, **overrides)


def test_replacement_probability_per_arm():
    cfg = RunConfig(seed=1)
    assert cfg.replacement_probability_for(ARM_DIFFERENCE) == 0.0
    assert cfg.replacement_probability_for(ARM_MIXTURE) == 0.10
    with pytest.raises(ValueError):
        cfg.replacement_probability_for(ARM_RANDOM)


def test_config_hash_is_stable_and_seed_sensitive():
    assert RunConfig(seed=1).config_hash() == RunConfig(seed=1).config_hash()
    assert RunConfig(seed=1).config_hash() != RunConfig(seed=2).config_hash()


def test_streams_are_reproducible_and_independent():
    first, second = make_streams(7), make_streams(7)
    assert first.mask.random() == second.mask.random()
    assert first.selection.random() != first.gaussian.random()
    assert make_streams(7).init.random() != make_streams(8).init.random()


def test_src_ariel_is_untouched():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/ariel"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_config.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'mutation_ab.config'`.

- [ ] **Step 4: Implement `config.py` and `streams.py`**

`config.py`:
```python
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
```

`streams.py`:
```python
from dataclasses import dataclass

import numpy as np

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
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return Streams(*(np.random.default_rng(child) for child in children))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): add mutation A/B config and random streams"
```

---

### Task 2: Operators and metrics (pure functions)

**Files:**
- Create: `assignments/assignment_2/mutation_ab/metrics.py`
- Create: `assignments/assignment_2/mutation_ab/operators.py`
- Test: `assignments/assignment_2/mutation_ab/tests/test_operators.py`

**Interfaces:**
- Consumes: `RunConfig`, `Streams`, `make_streams` (Task 1).
- Produces:
  - `metrics.rms(values: np.ndarray) -> float`; `metrics.genotype_diversity(genomes: np.ndarray) -> float` (mean pairwise L2 / √L); `metrics.unique_genomes(genomes: np.ndarray) -> int`.
  - `operators.KIND_DIFFERENCE = "difference"`, `operators.KIND_GAUSSIAN = "gaussian"`.
  - `operators.tournament(fitness: np.ndarray, size: int, rng) -> int`
  - `operators.draw_donors(n: int, parent: int, rng) -> tuple[int, int]`
  - `operators.binomial_mask(length: int, rate: float, rng) -> np.ndarray` (bool)
  - `operators.Proposal` (frozen dataclass: `child: np.ndarray, parent: int, donors: tuple[int, int], kind: str, proposal_rms: float, change_rms: float`)
  - `operators.propose_child(genomes: np.ndarray, fitness: np.ndarray, cfg: RunConfig, replacement_probability: float, streams: Streams) -> Proposal`
  - `operators.elite_index(fitness: np.ndarray, uids: np.ndarray) -> int` (lowest fitness, ties broken by lowest uid)

- [ ] **Step 1: Write the failing tests**

`tests/test_operators.py`:
```python
import math

import numpy as np
import pytest

from mutation_ab.config import RunConfig
from mutation_ab.metrics import genotype_diversity, rms, unique_genomes
from mutation_ab.operators import (
    KIND_DIFFERENCE,
    KIND_GAUSSIAN,
    binomial_mask,
    draw_donors,
    elite_index,
    propose_child,
    tournament,
)
from mutation_ab.streams import make_streams

LENGTH = 222


def population(seed=0, size=12):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, (size, LENGTH)), rng.random(size)


def test_rms_and_diversity_known_values():
    assert rms(np.array([3.0, 4.0])) == pytest.approx(math.sqrt(12.5))
    genomes = np.vstack([np.zeros(LENGTH), np.ones(LENGTH)])
    assert genotype_diversity(genomes) == pytest.approx(1.0)
    assert genotype_diversity(np.ones((5, LENGTH))) == 0.0


def test_unique_genomes_counts_exact_duplicates():
    genomes = np.vstack([np.zeros(LENGTH), np.zeros(LENGTH), np.ones(LENGTH)])
    assert unique_genomes(genomes) == 2


def test_tournament_returns_best_of_full_sample():
    fitness = np.array([3.0, 1.0, 2.0])
    assert tournament(fitness, 3, np.random.default_rng(0)) == 1


def test_tournament_ties_are_deterministic_per_seed():
    fitness = np.zeros(12)
    picks_a = [tournament(fitness, 3, rng) for rng in [np.random.default_rng(4)] * 20]
    picks_b = [tournament(fitness, 3, rng) for rng in [np.random.default_rng(4)] * 20]
    assert picks_a == picks_b


def test_donors_are_distinct_and_exclude_parent():
    rng = np.random.default_rng(1)
    for _ in range(500):
        b, c = draw_donors(3, 1, rng)
        assert {b, c} == {0, 2}


@pytest.mark.parametrize(("rate", "expected"), [(1e-12, 1), (1.0, LENGTH)])
def test_mask_forces_one_coordinate(rate, expected):
    mask = binomial_mask(LENGTH, rate, np.random.default_rng(2))
    assert mask.dtype == bool
    assert mask.sum() == expected


def test_identical_population_gives_zero_difference_step():
    genomes = np.tile(np.random.default_rng(3).normal(size=LENGTH), (12, 1))
    proposal = propose_child(genomes, np.zeros(12), RunConfig(seed=1), 0.0, make_streams(1))
    assert proposal.kind == KIND_DIFFERENCE
    assert proposal.proposal_rms == 0.0
    assert proposal.change_rms == 0.0
    np.testing.assert_array_equal(proposal.child, genomes[proposal.parent])


@pytest.mark.parametrize(("probability", "kind"), [(0.0, KIND_DIFFERENCE), (1.0, KIND_GAUSSIAN)])
def test_replacement_probability_extremes(probability, kind):
    genomes, fitness = population()
    streams = make_streams(5)
    kinds = {
        propose_child(genomes, fitness, RunConfig(seed=5), probability, streams).kind
        for _ in range(200)
    }
    assert kinds == {kind}


def test_child_changes_only_masked_coordinates_by_delta():
    genomes, fitness = population()
    cfg = RunConfig(seed=6)
    proposal = propose_child(genomes, fitness, cfg, 0.0, make_streams(6))
    parent = genomes[proposal.parent]
    b, c = proposal.donors
    delta = cfg.scale_f * (genomes[b] - genomes[c])
    changed = proposal.child != parent
    np.testing.assert_allclose(proposal.child[changed], (parent + delta)[changed])
    assert proposal.proposal_rms == pytest.approx(rms(delta))
    assert proposal.change_rms == pytest.approx(rms(proposal.child - parent))


def test_arms_consume_random_streams_identically():
    genomes, fitness = population()
    cfg = RunConfig(seed=9)
    control, treatment = make_streams(9), make_streams(9)
    for _ in range(50):
        propose_child(genomes, fitness, cfg, 0.0, control)
        propose_child(genomes, fitness, cfg, 1.0, treatment)
    for name in ("selection", "mask", "gaussian", "replacement"):
        assert getattr(control, name).random() == getattr(treatment, name).random()


def test_elite_ties_break_on_lowest_uid():
    assert elite_index(np.array([1.0, 0.0, 0.0]), np.array([5, 9, 3])) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_operators.py -q`
Expected: `ModuleNotFoundError: No module named 'mutation_ab.metrics'`.

- [ ] **Step 3: Implement `metrics.py` and `operators.py`**

`metrics.py`:
```python
import math

import numpy as np
from scipy.spatial.distance import pdist


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def genotype_diversity(genomes: np.ndarray) -> float:
    return float(pdist(genomes).mean() / math.sqrt(genomes.shape[1]))


def unique_genomes(genomes: np.ndarray) -> int:
    return len({row.tobytes() for row in np.ascontiguousarray(genomes)})
```

`operators.py`:
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_operators.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): add differential and Gaussian-replacement operators"
```

---

### Task 3: Seeded world, controller and headless evaluation

**Files:**
- Create: `assignments/assignment_2/mutation_ab/world.py`
- Create: `assignments/assignment_2/mutation_ab/controller.py`
- Create: `assignments/assignment_2/mutation_ab/evaluate.py`
- Test: `assignments/assignment_2/mutation_ab/tests/test_simulation.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1).
- Produces:
  - `world.SeededOlympicArena(terrain_seed: int)`
  - `world.build_model(cfg: RunConfig) -> tuple[mj.MjModel, dict[str, str]]` (the dict has keys `terrain_sha1`, `model_sha1`)
  - `world.terrain_fingerprint(terrain_seed: int) -> tuple[str, str]`
  - `controller.N_INPUTS = 29`, `controller.N_OUTPUTS = 6`
  - `controller.genome_length(n_inputs: int, hidden: int, n_outputs: int) -> int`
  - `controller.Layers` (frozen dataclass `w1, b1, w2, b2`)
  - `controller.unpack(genome: np.ndarray, n_inputs: int, hidden: int, n_outputs: int) -> Layers`
  - `controller.observe(data: mj.MjData, target_xy: tuple[float, float], phase_hz: float) -> np.ndarray`
  - `controller.act(layers: Layers, observation: np.ndarray) -> np.ndarray`
  - `evaluate.EvalResult` (frozen dataclass: `final_xy: tuple[float, float], distance: float, warnings: int`)
  - `evaluate.UnstableSimulation(RuntimeError)`
  - `evaluate.evaluate(genome: np.ndarray, model: mj.MjModel, cfg: RunConfig) -> EvalResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_simulation.py`:
```python
import multiprocessing

import mujoco as mj
import numpy as np
import pytest

from mutation_ab.config import RunConfig
from mutation_ab.controller import N_INPUTS, N_OUTPUTS, act, genome_length, observe, unpack
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.world import build_model, terrain_fingerprint

SHORT = RunConfig(seed=1, duration=0.2)


@pytest.fixture(scope="module")
def model():
    return build_model(SHORT)[0]


def test_genome_length_matches_design():
    assert genome_length(N_INPUTS, 6, N_OUTPUTS) == 222


def test_unpack_rejects_wrong_length():
    with pytest.raises(ValueError):
        unpack(np.zeros(221), N_INPUTS, 6, N_OUTPUTS)


def test_model_dimensions_match_controller(model):
    assert model.nu == N_OUTPUTS
    data = mj.MjData(model)
    assert observe(data, SHORT.target_xy, SHORT.phase_hz).shape == (N_INPUTS,)


def test_actions_are_within_hinge_range():
    layers = unpack(np.full(222, 5.0), N_INPUTS, 6, N_OUTPUTS)
    actions = act(layers, np.ones(N_INPUTS))
    assert actions.shape == (N_OUTPUTS,)
    assert np.all(np.abs(actions) <= np.pi / 2)


def test_evaluation_is_deterministic(model):
    genome = np.random.default_rng(0).normal(0, 0.5, 222)
    first, second = evaluate(genome, model, SHORT), evaluate(genome, model, SHORT)
    assert first == second
    assert np.isfinite(first.distance)


def test_zero_controller_stays_near_spawn(model):
    result = evaluate(np.zeros(222), model, SHORT)
    assert result.distance == pytest.approx(2.0, abs=0.1)


def test_nan_genome_raises_instead_of_scoring(model):
    genome = np.zeros(222)
    genome[0] = np.nan
    with pytest.raises(UnstableSimulation):
        evaluate(genome, model, SHORT)


def test_terrain_is_identical_across_processes():
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        prints = pool.map(terrain_fingerprint, [42, 42])
    assert prints[0] == prints[1]
    assert terrain_fingerprint(42)[0] != terrain_fingerprint(43)[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_simulation.py -q`
Expected: `ModuleNotFoundError: No module named 'mutation_ab.controller'`.

- [ ] **Step 3: Implement `world.py`**

The heightmap method mirrors `OlympicArena._generate_heightmap` (`src/ariel/simulation/environments/olympic_arena.py:127-167`) exactly. The only difference is the seeded `PerlinNoise`.

```python
import hashlib

import mujoco as mj
import numpy as np
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import OlympicArena
from ariel.utils.noise_gen import PerlinNoise

from mutation_ab.config import RunConfig


class SeededOlympicArena(OlympicArena):
    # OlympicArena seeds nothing, so its rugged section differs on every build.
    def __init__(self, terrain_seed: int) -> None:
        self.terrain_seed = terrain_seed
        super().__init__(load_precompiled=False)

    def _generate_heightmap(self) -> np.ndarray:
        size = self.rugged_resolution
        edge_width = getattr(self, "edge_width", 0.1)
        noise = PerlinNoise(seed=self.terrain_seed).as_grid(
            size, size, scale=self.rugged_hillyness, normalize=False
        )
        u = np.linspace(0.0, 1.0, size)
        v = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, v, indexing="xy")
        edge_distance = np.minimum.reduce([grid_u, 1.0 - grid_u, grid_v, 1.0 - grid_v])
        t = np.clip(edge_distance / edge_width, 0.1, 1.0)
        return noise * (t * t * (3.0 - 2.0 * t))


def _sha1(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def build_model(cfg: RunConfig) -> tuple[mj.MjModel, dict[str, str]]:
    mj.set_mjcb_control(None)
    world = SeededOlympicArena(cfg.terrain_seed)
    world.spawn(gecko().spec, position=list(cfg.spawn), correct_collision_with_floor=True)
    model = world.spec.compile()
    hashes = {
        "terrain_sha1": _sha1(world.heightmap.tobytes()),
        "model_sha1": _sha1(world.spec.to_xml().encode()),
    }
    return model, hashes


def terrain_fingerprint(terrain_seed: int) -> tuple[str, str]:
    _, hashes = build_model(RunConfig(seed=0, terrain_seed=terrain_seed))
    return hashes["terrain_sha1"], hashes["model_sha1"]
```

If `PerlinNoise(seed=...)` is not accepted as a keyword, check `src/ariel/utils/noise_gen.py` (it is a dataclass with a `seed` field) and adapt the call. Do not edit ARIEL.

- [ ] **Step 4: Implement `controller.py`**

```python
from dataclasses import dataclass

import mujoco as mj
import numpy as np

N_INPUTS = 29
N_OUTPUTS = 6


@dataclass(frozen=True)
class Layers:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def genome_length(n_inputs: int, hidden: int, n_outputs: int) -> int:
    return n_inputs * hidden + hidden + hidden * n_outputs + n_outputs


def unpack(genome: np.ndarray, n_inputs: int, hidden: int, n_outputs: int) -> Layers:
    expected = genome_length(n_inputs, hidden, n_outputs)
    if genome.shape != (expected,):
        raise ValueError(f"genome must have shape ({expected},), got {genome.shape}")
    sizes = [n_inputs * hidden, hidden, hidden * n_outputs, n_outputs]
    w1, b1, w2, b2 = np.split(genome, np.cumsum(sizes)[:-1])
    return Layers(w1.reshape(n_inputs, hidden), b1, w2.reshape(hidden, n_outputs), b2)


def observe(data: mj.MjData, target_xy: tuple[float, float], phase_hz: float) -> np.ndarray:
    phase = 2.0 * np.pi * phase_hz * data.time
    to_target = np.asarray(target_xy) - data.qpos[0:2]
    return np.concatenate([data.qpos, data.qvel, to_target, [np.sin(phase), np.cos(phase)]])


def act(layers: Layers, observation: np.ndarray) -> np.ndarray:
    hidden = np.tanh(observation @ layers.w1 + layers.b1)
    return np.tanh(hidden @ layers.w2 + layers.b2) * (np.pi / 2)
```

- [ ] **Step 5: Implement `evaluate.py`**

It mirrors the template: an `mjcb_control` callback plus `simple_runner`, headless.

```python
from dataclasses import dataclass

import mujoco as mj
import numpy as np
from ariel.utils.runners import simple_runner

from mutation_ab.config import RunConfig
from mutation_ab.controller import N_INPUTS, act, observe, unpack


UNSTABLE_WARNINGS = (
    mj.mjtWarning.mjWARN_BADQACC,
    mj.mjtWarning.mjWARN_BADQPOS,
    mj.mjtWarning.mjWARN_BADQVEL,
    mj.mjtWarning.mjWARN_BADCTRL,
)


class UnstableSimulation(RuntimeError):
    pass


@dataclass(frozen=True)
class EvalResult:
    final_xy: tuple[float, float]
    distance: float
    warnings: int


def evaluate(genome: np.ndarray, model: mj.MjModel, cfg: RunConfig) -> EvalResult:
    layers = unpack(np.asarray(genome, dtype=float), N_INPUTS, cfg.hidden_size, model.nu)
    data = mj.MjData(model)

    def control(m: mj.MjModel, d: mj.MjData) -> None:
        d.ctrl[:] = act(layers, observe(d, cfg.target_xy, cfg.phase_hz))

    mj.set_mjcb_control(control)
    try:
        simple_runner(model, data, duration=cfg.duration)
    finally:
        mj.set_mjcb_control(None)

    state = np.concatenate([data.qpos, data.qvel, data.ctrl])
    # MuJoCo silently zeroes bad controls and resets diverged states, so the endpoint would look valid.
    diverged = any(data.warning[int(warning)].number > 0 for warning in UNSTABLE_WARNINGS)
    if not np.all(np.isfinite(state)) or diverged:
        raise UnstableSimulation(f"unstable episode at t={data.time:.3f}s")

    final_xy = (float(data.qpos[0]), float(data.qpos[1]))
    distance = float(np.hypot(final_xy[0] - cfg.target_xy[0], final_xy[1] - cfg.target_xy[1]))
    warnings = int(sum(stat.number for stat in data.warning))
    return EvalResult(final_xy=final_xy, distance=distance, warnings=warnings)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_simulation.py -q`
Expected: all pass. If `test_nan_genome_raises_instead_of_scoring` still fails, print the non-zero `data.warning[i].number` entries to see which MuJoCo warning fired, and add it to `UNSTABLE_WARNINGS`. If `test_terrain_is_identical_across_processes` fails on `model_sha1` only (the XML may contain process-specific names), hash `model.geom_size.tobytes() + model.body_pos.tobytes() + model.hfield_data.tobytes()` instead of the XML, and keep the terrain hash as is.

- [ ] **Step 7: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): add seeded arena, NN controller and evaluator"
```

---

### Task 4: Run recorder

**Files:**
- Create: `assignments/assignment_2/mutation_ab/records.py`
- Test: `assignments/assignment_2/mutation_ab/tests/test_records.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1).
- Produces:
  - `records.genome_sha1(genome) -> str` (SHA1 of the float64 bytes)
  - `records.git_commit() -> dict` (keys `commit`, `dirty`)
  - `records.GENERATION_FIELDS` (tuple of CSV column names)
  - `records.RunRecorder(directory: Path, cfg: RunConfig, arm: str, hashes: dict[str, str])`, which raises `FileExistsError` if `directory` exists, with methods `.child(record: dict)`, `.generation(row: dict)`, `.adults(genomes: np.ndarray)`, `.complete(summary: dict)`, `.fail(error: BaseException)` and attribute `.directory`.

- [ ] **Step 1: Write the failing tests**

`tests/test_records.py`:
```python
import json

import numpy as np
import pytest

from mutation_ab.config import RunConfig
from mutation_ab.records import GENERATION_FIELDS, RunRecorder, genome_sha1

HASHES = {"terrain_sha1": "t", "model_sha1": "m"}


def row(generation):
    return {field: 0 for field in GENERATION_FIELDS} | {"generation": generation}


def test_refuses_existing_directory(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        RunRecorder(target, RunConfig(seed=1), "difference", HASHES)
    assert (target / "keep.txt").read_text() == "x"


def test_complete_run_writes_all_artifacts(tmp_path):
    recorder = RunRecorder(tmp_path / "run", RunConfig(seed=1), "difference", HASHES)
    recorder.child({"uid": 0, "distance": 1.5})
    recorder.child({"uid": 1, "distance": 1.4})
    recorder.generation(row(0))
    recorder.adults(np.zeros((4, 222)))
    recorder.adults(np.ones((4, 222)))
    recorder.complete({"evaluations": 2})

    run = tmp_path / "run"
    config = json.loads((run / "config.json").read_text())
    assert config["arm"] == "difference"
    assert config["config_hash"] == RunConfig(seed=1).config_hash()
    assert config["hashes"] == HASHES
    assert "commit" in config["git"]
    assert len((run / "children.jsonl").read_text().splitlines()) == 2
    assert (run / "generations.csv").read_text().splitlines()[0].split(",") == list(GENERATION_FIELDS)
    assert np.load(run / "adults.npz")["adults"].shape == (2, 4, 222)
    assert json.loads((run / "COMPLETE").read_text()) == {"evaluations": 2}


def test_failed_run_has_no_complete_marker(tmp_path):
    recorder = RunRecorder(tmp_path / "run", RunConfig(seed=1), "mixture", HASHES)
    recorder.fail(RuntimeError("boom"))
    assert not (tmp_path / "run" / "COMPLETE").exists()
    assert "boom" in json.loads((tmp_path / "run" / "FAILED.json").read_text())["error"]


def test_genome_hash_distinguishes_genomes():
    assert genome_sha1([0.0, 1.0]) == genome_sha1(np.array([0.0, 1.0]))
    assert genome_sha1([0.0, 1.0]) != genome_sha1([1.0, 0.0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_records.py -q`
Expected: `ModuleNotFoundError: No module named 'mutation_ab.records'`.

- [ ] **Step 3: Implement `records.py`**

```python
import csv
import hashlib
import json
import platform
import subprocess
import traceback
from pathlib import Path

import mujoco
import numpy as np

from mutation_ab.config import RunConfig

GENERATION_FIELDS = (
    "generation",
    "best",
    "mean",
    "worst",
    "best_so_far",
    "unique",
    "diversity",
    "diff_proposal_rms",
    "n_difference",
    "n_gaussian",
    "evaluations",
)


def genome_sha1(genome) -> str:
    return hashlib.sha1(np.asarray(genome, dtype=np.float64).tobytes()).hexdigest()


def git_commit() -> dict:
    here = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


class RunRecorder:
    def __init__(self, directory: Path, cfg: RunConfig, arm: str, hashes: dict[str, str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        meta = {
            "arm": arm,
            "config": cfg.to_dict(),
            "config_hash": cfg.config_hash(),
            "hashes": hashes,
            "git": git_commit(),
            "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__},
        }
        (self.directory / "config.json").write_text(json.dumps(meta, indent=2))
        self._children = (self.directory / "children.jsonl").open("w")
        self._generations_file = (self.directory / "generations.csv").open("w", newline="")
        self._generations = csv.DictWriter(self._generations_file, fieldnames=GENERATION_FIELDS)
        self._generations.writeheader()
        self._adults: list[np.ndarray] = []

    def child(self, record: dict) -> None:
        self._children.write(json.dumps(record) + "\n")
        self._children.flush()

    def generation(self, row: dict) -> None:
        self._generations.writerow(row)
        self._generations_file.flush()

    def adults(self, genomes: np.ndarray) -> None:
        self._adults.append(np.array(genomes, dtype=np.float64))

    def complete(self, summary: dict) -> None:
        self._close()
        if self._adults:
            np.savez_compressed(self.directory / "adults.npz", adults=np.stack(self._adults))
        (self.directory / "COMPLETE").write_text(json.dumps(summary))

    def fail(self, error: BaseException) -> None:
        self._close()
        report = {
            "error": repr(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        (self.directory / "FAILED.json").write_text(json.dumps(report, indent=2))

    def _close(self) -> None:
        self._children.close()
        self._generations_file.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_records.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): add immutable run recorder"
```

---

### Task 5: Initial population, EA arms on `ariel.ec`, random search

**Files:**
- Create: `assignments/assignment_2/mutation_ab/initial.py`
- Create: `assignments/assignment_2/mutation_ab/ea_arm.py`
- Create: `assignments/assignment_2/mutation_ab/random_search.py`
- Modify: `assignments/assignment_2/mutation_ab/tests/conftest.py` (add the fake evaluator and fake-root builder)
- Test: `assignments/assignment_2/mutation_ab/tests/test_arms.py`

**Interfaces:**
- Consumes: `RunConfig`, `ARM_*`, `make_streams`, `Streams` (Task 1); `propose_child`, `elite_index`, `KIND_*`, metrics (Task 2); `EvalResult` (Task 3); `RunRecorder`, `genome_sha1` (Task 4).
- Produces:
  - `initial.Evaluator = Callable[[np.ndarray], EvalResult]`
  - `initial.InitialPopulation` (frozen dataclass: `genomes: np.ndarray, results: tuple[EvalResult, ...], wall_s: tuple[float, ...]`)
  - `initial.make_initial(cfg: RunConfig, streams: Streams, evaluator: Evaluator, length: int) -> InitialPopulation`
  - `initial.record_founders(initial: InitialPopulation, recorder: RunRecorder) -> None` (writes the P init child records, uids 0..P-1, `kind="init"`)
  - `ea_arm.run_arm(cfg, arm, initial, evaluator, streams, recorder) -> dict` (summary: `evaluations`, `best_so_far`)
  - `random_search.run_random(cfg, initial, evaluator, streams, recorder) -> dict` (same summary keys)
  - Child record keys: `uid, generation, kind, parent_uid, donor_uids, proposal_rms, change_rms, parent_distance, parent_xy, final_xy, warnings, genome_sha1, distance, wall_s`
  - conftest: `fake_evaluator(genome) -> EvalResult` and `make_fake_root(root: Path, seeds, **cfg_overrides) -> Path`

- [ ] **Step 1: Extend `tests/conftest.py`**

Append:
```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/test_arms.py`:
```python
import csv
import json

import numpy as np
import pytest
from conftest import FAKE_HASHES, fake_evaluator, make_fake_root

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, ARMS, RunConfig
from mutation_ab.ea_arm import run_arm
from mutation_ab.evaluate import UnstableSimulation
from mutation_ab.initial import make_initial
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams

SMALL = {"generations": 5, "population_size": 6}


def children(run_dir):
    return [json.loads(line) for line in (run_dir / "children.jsonl").read_text().splitlines()]


def generations(run_dir):
    with (run_dir / "generations.csv").open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def root(tmp_path):
    return make_fake_root(tmp_path, [3], **SMALL)


def test_every_arm_spends_exactly_the_budget(root):
    budget = RunConfig(seed=3, **SMALL).budget
    for arm in ARMS:
        assert len(children(root / "seed_3" / arm)) == budget


def test_all_arms_share_the_initial_population(root):
    founders = {
        arm: [c["genome_sha1"] for c in children(root / "seed_3" / arm) if c["kind"] == "init"]
        for arm in ARMS
    }
    assert founders["difference"] == founders["mixture"] == founders["random"]


def test_elitism_never_loses_the_best(root):
    for arm in (ARM_DIFFERENCE, ARM_MIXTURE):
        best = [float(r["best"]) for r in generations(root / "seed_3" / arm)]
        assert all(later <= earlier for earlier, later in zip(best, best[1:]))


def test_adult_snapshots_cover_every_generation(root):
    adults = np.load(root / "seed_3" / ARM_DIFFERENCE / "adults.npz")["adults"]
    assert adults.shape == (SMALL["generations"] + 1, SMALL["population_size"], 222)


def test_parents_and_donors_come_from_previous_survivors(root):
    records = children(root / "seed_3" / ARM_MIXTURE)
    uids_by_generation = {}
    for record in records:
        uids_by_generation.setdefault(record["generation"], set()).add(record["uid"])
    alive = set(uids_by_generation[0])
    for generation in range(1, SMALL["generations"] + 1):
        born = [r for r in records if r["generation"] == generation]
        for record in born:
            assert record["parent_uid"] in alive
            assert set(record["donor_uids"]) <= alive
            assert record["parent_uid"] not in record["donor_uids"]
        elite = min((r for r in records if r["uid"] in alive), key=lambda r: (r["distance"], r["uid"]))
        alive = {elite["uid"], *(r["uid"] for r in born)}


def test_zero_probability_mixture_reproduces_difference_arm(tmp_path):
    root = make_fake_root(tmp_path, [4], replacement_probability=0.0, **SMALL)
    hashes = {
        arm: [c["genome_sha1"] for c in children(root / "seed_4" / arm)]
        for arm in (ARM_DIFFERENCE, ARM_MIXTURE)
    }
    assert hashes[ARM_DIFFERENCE] == hashes[ARM_MIXTURE]


def test_full_probability_mixture_uses_only_gaussian_children(tmp_path):
    root = make_fake_root(tmp_path, [5], replacement_probability=1.0, **SMALL)
    kinds = {c["kind"] for c in children(root / "seed_5" / ARM_MIXTURE) if c["kind"] != "init"}
    assert kinds == {"gaussian"}


def test_unstable_evaluation_propagates(tmp_path):
    cfg = RunConfig(seed=6, **SMALL)
    initial = make_initial(cfg, make_streams(6), fake_evaluator, 222)
    calls = {"n": 0}

    def exploding(genome):
        calls["n"] += 1
        if calls["n"] > 3:
            raise UnstableSimulation("blew up")
        return fake_evaluator(genome)

    recorder = RunRecorder(tmp_path / "run", cfg, ARM_DIFFERENCE, FAKE_HASHES)
    with pytest.raises(UnstableSimulation):
        run_arm(cfg, ARM_DIFFERENCE, initial, exploding, make_streams(6), recorder)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_arms.py -q`
Expected: `ModuleNotFoundError: No module named 'mutation_ab.initial'`.

- [ ] **Step 4: Implement `initial.py`**

```python
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
```

- [ ] **Step 5: Implement `ea_arm.py`**

```python
import time
from dataclasses import dataclass

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

from mutation_ab.config import RunConfig
from mutation_ab.initial import Evaluator, InitialPopulation, record_founders
from mutation_ab.metrics import genotype_diversity, unique_genomes
from mutation_ab.operators import KIND_DIFFERENCE, elite_index, propose_child
from mutation_ab.records import RunRecorder, genome_sha1
from mutation_ab.streams import Streams


@dataclass
class ArmContext:
    cfg: RunConfig
    replacement_probability: float
    evaluator: Evaluator
    streams: Streams
    recorder: RunRecorder
    generation: int = 0
    next_uid: int = 0
    evaluations: int = 0
    best_so_far: float = float("inf")


def _by_uid(individuals) -> list[Individual]:
    return sorted(individuals, key=lambda ind: ind.tags["uid"])


def _record_generation(ctx: ArmContext, survivors: list[Individual], newborn: list[Individual]) -> None:
    genomes = np.array([ind.genotype for ind in survivors], dtype=float)
    fitness = np.array([ind.fitness for ind in survivors])
    ctx.best_so_far = min(ctx.best_so_far, float(fitness.min()))
    steps = [ind.tags["proposal_rms"] for ind in newborn if ind.tags["kind"] == KIND_DIFFERENCE]
    ctx.recorder.generation(
        {
            "generation": ctx.generation,
            "best": float(fitness.min()),
            "mean": float(fitness.mean()),
            "worst": float(fitness.max()),
            "best_so_far": ctx.best_so_far,
            "unique": unique_genomes(genomes),
            "diversity": genotype_diversity(genomes),
            "diff_proposal_rms": float(np.mean(steps)) if steps else float("nan"),
            "n_difference": len(steps),
            "n_gaussian": sum(ind.tags["kind"] == "gaussian" for ind in newborn),
            "evaluations": ctx.evaluations,
        }
    )
    ctx.recorder.adults(genomes)


def reproduce(population: Population, ctx: ArmContext) -> Population:
    ctx.generation += 1
    adults = _by_uid(population)
    genomes = np.array([ind.genotype for ind in adults], dtype=float)
    fitness = np.array([ind.fitness for ind in adults])
    for _ in range(ctx.cfg.children_per_generation):
        proposal = propose_child(genomes, fitness, ctx.cfg, ctx.replacement_probability, ctx.streams)
        parent = adults[proposal.parent]
        child = Individual()
        child.genotype = proposal.child.tolist()
        child.tags = {
            "uid": ctx.next_uid,
            "generation": ctx.generation,
            "kind": proposal.kind,
            "parent_uid": parent.tags["uid"],
            "donor_uids": [adults[d].tags["uid"] for d in proposal.donors],
            "proposal_rms": proposal.proposal_rms,
            "change_rms": proposal.change_rms,
            "parent_distance": parent.fitness,
            "parent_xy": parent.tags["final_xy"],
        }
        ctx.next_uid += 1
        population.append(child)
    return population


def evaluate_children(population: Population, ctx: ArmContext) -> Population:
    for child in _by_uid(population.unevaluated):
        started = time.perf_counter()
        result = ctx.evaluator(np.asarray(child.genotype, dtype=float))
        child.fitness = result.distance
        child.tags = {"final_xy": list(result.final_xy), "warnings": result.warnings}
        ctx.evaluations += 1
        ctx.recorder.child(
            {
                **child.tags,
                "genome_sha1": genome_sha1(child.genotype),
                "distance": result.distance,
                "wall_s": time.perf_counter() - started,
            }
        )
    return population


def survive(population: Population, ctx: ArmContext) -> Population:
    members = _by_uid(population)
    adults = [ind for ind in members if ind.tags["generation"] < ctx.generation]
    newborn = [ind for ind in members if ind.tags["generation"] == ctx.generation]
    elite = adults[
        elite_index(
            np.array([ind.fitness for ind in adults]),
            np.array([ind.tags["uid"] for ind in adults]),
        )
    ]
    for adult in adults:
        adult.alive = adult is elite
    _record_generation(ctx, _by_uid([elite, *newborn]), newborn)
    return population


def run_arm(
    cfg: RunConfig,
    arm: str,
    initial: InitialPopulation,
    evaluator: Evaluator,
    streams: Streams,
    recorder: RunRecorder,
) -> dict:
    ctx = ArmContext(cfg, cfg.replacement_probability_for(arm), evaluator, streams, recorder)
    record_founders(initial, recorder)
    founders = []
    for genome, result in zip(initial.genomes, initial.results, strict=True):
        founder = Individual()
        founder.genotype = genome.tolist()
        founder.fitness = result.distance
        founder.tags = {
            "uid": ctx.next_uid,
            "generation": 0,
            "kind": "init",
            "final_xy": list(result.final_xy),
        }
        ctx.next_uid += 1
        ctx.evaluations += 1
        founders.append(founder)
    _record_generation(ctx, founders, founders)

    ea = EA(
        Population(founders),
        [
            EAOperation(reproduce, ctx),
            EAOperation(evaluate_children, ctx),
            EAOperation(survive, ctx),
        ],
        num_steps=cfg.generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=recorder.directory / "ariel.db",
        db_handling="halt",
    )
    ea.run()
    return {"evaluations": ctx.evaluations, "best_so_far": ctx.best_so_far}
```

- [ ] **Step 6: Implement `random_search.py`**

```python
import time

import numpy as np

from mutation_ab.config import RunConfig
from mutation_ab.initial import Evaluator, InitialPopulation, record_founders
from mutation_ab.metrics import genotype_diversity
from mutation_ab.records import RunRecorder, genome_sha1
from mutation_ab.streams import Streams


def _row(generation: int, batch: list[float], best_so_far: float, evaluations: int) -> dict:
    return {
        "generation": generation,
        "best": min(batch),
        "mean": float(np.mean(batch)),
        "worst": max(batch),
        "best_so_far": best_so_far,
        "unique": len(batch),
        "diversity": float("nan"),
        "diff_proposal_rms": float("nan"),
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
                {
                    "uid": uid,
                    "generation": generation,
                    "kind": "random",
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
                    "wall_s": time.perf_counter() - started,
                }
            )
            uid += 1
        best_so_far = min(best_so_far, min(batch))
        recorder.generation(_row(generation, batch, best_so_far, evaluations))
    return {"evaluations": evaluations, "best_so_far": best_so_far}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_arms.py -q`
Expected: all pass. If an `ariel.ec` behaviour differs from what `run_arm` assumes (for example, the fetched population order or tag persistence), fix it in `ea_arm.py` only and record the finding in the commit message. Do not bypass the `EA` engine unless it truly cannot express the scheme. If it can't, report the reason before switching to the spec's fallback.

- [ ] **Step 8: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): run A/B arms on ariel.ec and equal-budget random search"
```

---

### Task 6: CLI runner with parallel seeds

**Files:**
- Create: `assignments/assignment_2/mutation_ab/run.py`
- Test: `assignments/assignment_2/mutation_ab/tests/test_run.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5; `ariel.ec.set_seed`.
- Produces:
  - `run.parse_seeds(text: str) -> list[int]`
  - `run.run_seed(seed: int, out_root: Path, overrides: dict) -> dict` (keys `seed`, `status: dict[arm, "complete"|"failed"]`, `wall_s`)
  - `run.main(argv: list[str] | None = None) -> int` (exit code: 0 if everything is complete, 1 otherwise)
  - CLI: `python -m mutation_ab.run --out PATH --seeds SPEC [--generations 80] [--population 12] [--duration 15] [--workers 1]`, run with cwd `assignments/assignment_2`.

- [ ] **Step 1: Write the failing tests**

`tests/test_run.py`:
```python
import json

import pytest

from mutation_ab import run
from mutation_ab.config import ARMS
from mutation_ab.evaluate import UnstableSimulation

TINY = ["--generations", "2", "--population", "4", "--duration", "0.2"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("700-702", [700, 701, 702]), ("5,3", [5, 3]), ("900", [900])],
)
def test_parse_seeds(text, expected):
    assert run.parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["700-705,703", "705-700", "abc", "", "-1"])
def test_parse_seeds_rejects_bad_input(text):
    with pytest.raises(ValueError):
        run.parse_seeds(text)


def test_refuses_existing_seed_directory_without_running(tmp_path):
    existing = tmp_path / "seed_900"
    existing.mkdir()
    (existing / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        run.main(["--out", str(tmp_path), "--seeds", "900,901", *TINY])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed_900"]
    assert (existing / "keep.txt").read_text() == "x"


def _records(root, seed, arm):
    lines = (root / f"seed_{seed}" / arm / "children.jsonl").read_text().splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "wall_s"} for line in lines]


def test_serial_and_parallel_runs_are_identical(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    assert run.main(["--out", str(serial), "--seeds", "900,901", "--workers", "1", *TINY]) == 0
    assert run.main(["--out", str(parallel), "--seeds", "900,901", "--workers", "2", *TINY]) == 0
    for seed in (900, 901):
        for arm in ARMS:
            assert (parallel / f"seed_{seed}" / arm / "COMPLETE").exists()
            assert _records(serial, seed, arm) == _records(parallel, seed, arm)


def test_unstable_arm_is_marked_failed_and_others_finish(tmp_path, monkeypatch):
    real = run.evaluate
    calls = {"n": 0}

    def flaky(genome, model, cfg):
        calls["n"] += 1
        if calls["n"] == 6:
            raise UnstableSimulation("blew up")
        return real(genome, model, cfg)

    monkeypatch.setattr(run, "evaluate", flaky)
    assert run.main(["--out", str(tmp_path), "--seeds", "902", "--workers", "1", *TINY]) == 1
    seed_dir = tmp_path / "seed_902"
    failed = [arm for arm in ARMS if (seed_dir / arm / "FAILED.json").exists()]
    assert failed == ["difference"]
    assert not (seed_dir / "difference" / "COMPLETE").exists()
    assert (seed_dir / "mixture" / "COMPLETE").exists()
    assert (seed_dir / "random" / "COMPLETE").exists()
```

With population 4 the 4 initial evaluations are calls 1–4, so call 6 falls inside the difference arm's first generation.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_run.py -q`
Expected: `ImportError: cannot import name 'run'`.

- [ ] **Step 3: Implement `run.py`**

```python
import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
from pathlib import Path

from ariel.ec import set_seed

from mutation_ab.config import ARM_RANDOM, ARMS, RunConfig
from mutation_ab.controller import N_INPUTS, genome_length
from mutation_ab.ea_arm import run_arm
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.initial import make_initial
from mutation_ab.random_search import run_random
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams
from mutation_ab.world import build_model


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part[1:]:
            low, high = (int(bound) for bound in part.split("-", 1))
            if low > high:
                raise ValueError(f"descending seed range {part!r}")
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError(f"invalid seeds {text!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {text!r}")
    return seeds


def run_seed(seed: int, out_root: Path, overrides: dict) -> dict:
    started = time.perf_counter()
    cfg = RunConfig(seed=seed, **overrides)
    set_seed(seed)
    model, hashes = build_model(cfg)
    evaluator = partial(evaluate, model=model, cfg=cfg)
    length = genome_length(N_INPUTS, cfg.hidden_size, model.nu)
    seed_dir = out_root / f"seed_{seed}"
    status: dict[str, str] = {}
    try:
        initial = make_initial(cfg, make_streams(seed), evaluator, length)
    except UnstableSimulation as error:
        for arm in ARMS:
            RunRecorder(seed_dir / arm, cfg, arm, hashes).fail(error)
            status[arm] = "failed"
        return {"seed": seed, "status": status, "wall_s": time.perf_counter() - started}

    for arm in ARMS:
        recorder = RunRecorder(seed_dir / arm, cfg, arm, hashes)
        try:
            if arm == ARM_RANDOM:
                summary = run_random(cfg, initial, evaluator, make_streams(seed), recorder)
            else:
                summary = run_arm(cfg, arm, initial, evaluator, make_streams(seed), recorder)
        except UnstableSimulation as error:
            recorder.fail(error)
            status[arm] = "failed"
            continue
        recorder.complete(summary)
        status[arm] = "complete"
    return {"seed": seed, "status": status, "wall_s": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Differential vs Gaussian-replacement mutation A/B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="e.g. 700-705 or 800,801")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    existing = [seed for seed in seeds if (args.out / f"seed_{seed}").exists()]
    if existing:
        parser.error(f"output already exists for seeds {existing}; choose a new --out")
    overrides = {
        "generations": args.generations,
        "population_size": args.population,
        "duration": args.duration,
    }
    RunConfig(seed=0, **overrides)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.workers == 1:
        results = [run_seed(seed, args.out, overrides) for seed in seeds]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            results = list(pool.map(run_seed, seeds, repeat(args.out), repeat(overrides)))

    print(json.dumps(results, indent=2))
    complete = all(state == "complete" for r in results for state in r["status"].values())
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_run.py -q`
Expected: all pass. The serial/parallel test runs real MuJoCo episodes (0.2 s each), so it takes a few seconds.

- [ ] **Step 5: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): add parallel seed runner CLI"
```

---

### Task 7: Analysis, statistics, figures and POC go/no-go

**Files:**
- Create: `assignments/assignment_2/mutation_ab/analysis.py`
- Test: `assignments/assignment_2/mutation_ab/tests/test_analysis.py`

**Interfaces:**
- Consumes: run directory layout (Tasks 4–6); `ARM_*`, `ARMS` (Task 1); conftest `make_fake_root` (Task 5).
- Produces:
  - `analysis.IncompleteRuns(RuntimeError)`
  - `analysis.ArmRun` (frozen dataclass: `seed: int, arm: str, generations: dict[str, np.ndarray], children: list[dict], config: dict`)
  - `analysis.load_runs(root: Path) -> dict[int, dict[str, ArmRun]]`
  - `analysis.seed_metrics(run: ArmRun) -> dict[str, float | int | None]`
  - `analysis.paired(a: np.ndarray, b: np.ndarray) -> dict` (keys `mean_a, sd_a, mean_b, sd_b, mean_diff, p`; the difference is b − a)
  - `analysis.holm(pvalues: dict[str, float]) -> dict[str, float]`
  - `analysis.plateau_generation(curves: list[np.ndarray], start: int = 40, every: int = 10, window: int = 15, gain: float = 0.005) -> int | None`
  - `analysis.go_no_go(runs, *, final_seeds: int, workers: int, deadline: date, now: datetime) -> dict`
  - `analysis.main(argv: list[str] | None = None) -> int`; writes `ROOT/analysis/{analysis.json, seed_metrics.csv, h4.csv, fig1.pdf, fig1.png, fig2.pdf, fig2.png}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_analysis.py`:
```python
import json
from datetime import date, datetime

import numpy as np
import pytest
from conftest import make_fake_root

from mutation_ab import analysis
from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE

SMALL = {"generations": 6, "population_size": 6}


@pytest.fixture
def root(tmp_path):
    return make_fake_root(tmp_path, [1, 2, 3], **SMALL)


def test_refuses_root_with_incomplete_arm(root):
    (root / "seed_2" / ARM_MIXTURE / "COMPLETE").unlink()
    with pytest.raises(analysis.IncompleteRuns):
        analysis.load_runs(root)


def test_refuses_empty_root(tmp_path):
    with pytest.raises(analysis.IncompleteRuns):
        analysis.load_runs(tmp_path)


def test_refuses_mixed_configurations(tmp_path):
    make_fake_root(tmp_path, [1], **SMALL)
    make_fake_root(tmp_path, [2], generations=7, population_size=6)
    with pytest.raises(ValueError):
        analysis.load_runs(tmp_path)


def test_zero_probability_mixture_has_identical_metrics(tmp_path):
    root = make_fake_root(tmp_path, [4], replacement_probability=0.0, **SMALL)
    runs = analysis.load_runs(root)[4]
    diff = analysis.seed_metrics(runs[ARM_DIFFERENCE])
    mix = analysis.seed_metrics(runs[ARM_MIXTURE])
    assert diff == mix


def test_all_gaussian_generations_do_not_break_step_metrics(tmp_path):
    root = make_fake_root(tmp_path, [5], replacement_probability=1.0, **SMALL)
    metrics = analysis.seed_metrics(analysis.load_runs(root)[5][ARM_MIXTURE])
    assert np.isnan(metrics["step_mean"])
    assert metrics["improve_rate_difference"] is None


def test_holm_known_values():
    adjusted = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_paired_handles_zero_differences():
    result = analysis.paired(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert result["p"] == 1.0
    result = analysis.paired(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.5, 3.5, 4.5]))
    assert 0.0 < result["p"] <= 1.0
    assert result["mean_diff"] == pytest.approx(0.375)


def test_plateau_generation():
    flat_after_50 = np.concatenate([np.linspace(3.0, 1.0, 51), np.full(30, 1.0)])
    assert analysis.plateau_generation([flat_after_50, flat_after_50]) == 60
    still_falling = np.linspace(3.0, 1.0, 81)
    assert analysis.plateau_generation([flat_after_50, still_falling]) is None


def test_go_no_go_reports_every_criterion(root):
    report = analysis.go_no_go(
        analysis.load_runs(root),
        final_seeds=10,
        workers=10,
        deadline=date(2026, 10, 8),
        now=datetime(2026, 9, 29, 12, 0),
    )
    assert set(report["criteria"]) == {"speed", "a_collapses", "b_stays_alive", "h4_measurable"}
    for criterion in report["criteria"].values():
        assert set(criterion) == {"passed", "value", "rule"}
    assert "g_final" in report


def test_main_writes_outputs(root):
    assert analysis.main([str(root), "--poc"]) == 0
    out = root / "analysis"
    for name in ("analysis.json", "seed_metrics.csv", "h4.csv", "fig1.pdf", "fig1.png", "fig2.pdf", "fig2.png"):
        assert (out / name).exists()
    assert "tests" in json.loads((out / "analysis.json").read_text())
```

In `test_plateau_generation` the curve reaches 1.0 at index 50. At g=50 the drop over the preceding 15 generations (index 35→50) is still 0.6, at g=60 it is 0, so the first checkpoint that passes is 60.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_analysis.py -q`
Expected: `ImportError: cannot import name 'analysis'`.

- [ ] **Step 3: Implement `analysis.py`**

```python
import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mutation_ab.config import ARM_DIFFERENCE, ARM_MIXTURE, ARM_RANDOM, ARMS

COLORS = {ARM_DIFFERENCE: "#1f77b4", ARM_MIXTURE: "#ff7f0e", ARM_RANDOM: "#7f7f7f"}
LABELS = {ARM_DIFFERENCE: "A: differential", ARM_MIXTURE: "B: 10% Gaussian", ARM_RANDOM: "Random search"}
COLLAPSE_FRACTION = 0.01
MIN_IMPROVED = 20
USEFUL_SHIFT_M = 0.10
FALLBACK_GENERATIONS = 120


class IncompleteRuns(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmRun:
    seed: int
    arm: str
    generations: dict[str, np.ndarray]
    children: list[dict]
    config: dict


def _read_generations(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def load_runs(root: Path) -> dict[int, dict[str, ArmRun]]:
    seed_dirs = sorted(Path(root).glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    if not seed_dirs:
        raise IncompleteRuns(f"no seed directories under {root}")
    missing = [str(d / arm) for d in seed_dirs for arm in ARMS if not (d / arm / "COMPLETE").exists()]
    if missing:
        raise IncompleteRuns(f"runs without COMPLETE marker: {missing}")

    runs: dict[int, dict[str, ArmRun]] = {}
    reference = None
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[1])
        runs[seed] = {}
        for arm in ARMS:
            run_dir = seed_dir / arm
            meta = json.loads((run_dir / "config.json").read_text())
            shared = {k: v for k, v in meta["config"].items() if k != "seed"}
            if reference is None:
                reference = shared
            elif shared != reference:
                raise ValueError(f"{run_dir} uses a different configuration")
            children = [json.loads(line) for line in (run_dir / "children.jsonl").read_text().splitlines()]
            runs[seed][arm] = ArmRun(seed, arm, _read_generations(run_dir / "generations.csv"), children, meta["config"])
    return runs


def _last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def _rate(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def seed_metrics(run: ArmRun) -> dict:
    g = run.generations
    metrics: dict = {"best_final": float(g["best_so_far"][-1])}
    if run.arm == ARM_RANDOM:
        return metrics

    diversity = g["diversity"] / g["diversity"][0]
    step = g["diff_proposal_rms"]
    born = [c for c in run.children if c["kind"] not in ("init", "random")]
    improved = [c["distance"] < c["parent_distance"] for c in born]
    shifted = [math.dist(c["final_xy"], c["parent_xy"]) >= USEFUL_SHIFT_M for c in born]
    collapsed = np.flatnonzero(g["unique"] == 1)
    step_after_start = step[1:]
    metrics.update(
        {
            "D_bar": float(np.mean(diversity[1:])),
            "D_final": float(diversity[-1]),
            "step_mean": float(np.nanmean(step_after_start)) if np.isfinite(step_after_start).any() else float("nan"),
            "step_first": float(step[1]),
            "step_final": _last_finite(step),
            "collapse_generation": int(g["generation"][collapsed[0]]) if collapsed.size else None,
            "improved_count": int(sum(improved)),
            "improve_rate": _rate(improved),
            "useful_rate": _rate([i and s for i, s in zip(improved, shifted, strict=True)]),
            "mean_gain": float(np.mean([c["parent_distance"] - c["distance"] for c in born])),
            "improve_rate_difference": _rate([i for c, i in zip(born, improved, strict=True) if c["kind"] == "difference"]),
            "improve_rate_gaussian": _rate([i for c, i in zip(born, improved, strict=True) if c["kind"] == "gaussian"]),
        }
    )
    return metrics


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if np.allclose(diffs, 0.0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(diffs, zero_method="wilcox", method="exact").pvalue)
    return {
        "mean_a": float(np.mean(a)),
        "sd_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "sd_b": float(np.std(b, ddof=1)),
        "mean_diff": float(np.mean(diffs)),
        "p": p,
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def plateau_generation(
    curves: list[np.ndarray], start: int = 40, every: int = 10, window: int = 15, gain: float = 0.005
) -> int | None:
    last = min(len(curve) for curve in curves) - 1
    for generation in range(start, last + 1, every):
        if all(curve[generation - window] - curve[generation] < gain for curve in curves):
            return generation
    return None


def _metric_array(runs, arm: str, key: str) -> np.ndarray:
    return np.array([seed_metrics(runs[seed][arm])[key] for seed in sorted(runs)], dtype=float)


def statistical_tests(runs) -> dict:
    families = {
        "diversity": {k: (ARM_DIFFERENCE, ARM_MIXTURE, k) for k in ("D_bar", "D_final")},
        "step": {k: (ARM_DIFFERENCE, ARM_MIXTURE, k) for k in ("step_mean", "step_final")},
        "fitness": {
            "B_vs_A": (ARM_DIFFERENCE, ARM_MIXTURE, "best_final"),
            "A_vs_random": (ARM_RANDOM, ARM_DIFFERENCE, "best_final"),
            "B_vs_random": (ARM_RANDOM, ARM_MIXTURE, "best_final"),
        },
    }
    report = {}
    for family, contrasts in families.items():
        results = {
            name: paired(_metric_array(runs, a, key), _metric_array(runs, b, key))
            for name, (a, b, key) in contrasts.items()
        }
        adjusted = holm({name: r["p"] for name, r in results.items()})
        for name, result in results.items():
            result["p_holm"] = adjusted[name]
        report[family] = results
    return report


def _mean_curve(runs, arm: str, key: str) -> np.ndarray:
    return np.mean([runs[seed][arm].generations[key] for seed in runs], axis=0)


def go_no_go(runs, *, final_seeds: int, workers: int, deadline: date, now: datetime) -> dict:
    seeds = sorted(runs)
    needed = math.ceil(2 * len(seeds) / 3)
    config = runs[seeds[0]][ARM_DIFFERENCE].config
    plateau = plateau_generation([_mean_curve(runs, arm, "best_so_far") for arm in (ARM_DIFFERENCE, ARM_MIXTURE)])
    g_final = plateau if plateau is not None else FALLBACK_GENERATIONS

    walls = [c["wall_s"] for s in seeds for arm in ARMS for c in runs[s][arm].children if c["kind"] != "init"]
    per_seed = config["population_size"] + (config["population_size"] - 1) * g_final * len(ARMS)
    hours = final_seeds * per_seed * float(np.mean(walls)) / workers / 3600
    finish = now + timedelta(hours=hours)

    diff = {s: seed_metrics(runs[s][ARM_DIFFERENCE]) for s in seeds}
    mix = {s: seed_metrics(runs[s][ARM_MIXTURE]) for s in seeds}
    collapsed = sum(m["step_final"] < COLLAPSE_FRACTION * m["step_first"] for m in diff.values())
    alive = sum(m["step_final"] >= COLLAPSE_FRACTION * m["step_first"] for m in mix.values())
    fewest_improved = min(m["improved_count"] for m in (*diff.values(), *mix.values()))

    criteria = {
        "speed": {
            "passed": finish.date() <= deadline,
            "value": {"hours": hours, "finish": finish.isoformat(timespec="minutes"), "mean_eval_s": float(np.mean(walls))},
            "rule": f"{final_seeds} seeds x {per_seed} evals on {workers} workers finish by {deadline}",
        },
        "a_collapses": {
            "passed": collapsed >= needed,
            "value": collapsed,
            "rule": f"A final step < {COLLAPSE_FRACTION:.0%} of generation-1 step in >= {needed}/{len(seeds)} seeds",
        },
        "b_stays_alive": {
            "passed": alive >= needed,
            "value": alive,
            "rule": f"B final step >= {COLLAPSE_FRACTION:.0%} of generation-1 step in >= {needed}/{len(seeds)} seeds",
        },
        "h4_measurable": {
            "passed": fewest_improved >= MIN_IMPROVED,
            "value": fewest_improved,
            "rule": f">= {MIN_IMPROVED} parent-improving children per arm per seed",
        },
    }
    return {"criteria": criteria, "g_final": g_final, "plateau_found": plateau is not None}


def _band(ax, runs, arm: str, key: str, normalise: bool = False, log: bool = False) -> None:
    curves = np.array([runs[s][arm].generations[key] for s in sorted(runs)])
    if normalise:
        curves = curves / curves[:, :1]
    mean = np.nanmean(curves, axis=0)
    sd = np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    x = runs[sorted(runs)[0]][arm].generations["generation"]
    lower = np.clip(mean - sd, 1e-12, None) if log else mean - sd
    ax.plot(x, mean, color=COLORS[arm], label=LABELS[arm])
    ax.fill_between(x, lower, mean + sd, color=COLORS[arm], alpha=0.2, linewidth=0)
    if log:
        ax.set_yscale("log")


def figures(runs, out: Path) -> None:
    ea_arms = (ARM_DIFFERENCE, ARM_MIXTURE)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for arm in ea_arms:
        _band(left, runs, arm, "diversity", normalise=True)
    for arm in ARMS:
        _band(right, runs, arm, "best_so_far")
    left.set(xlabel="generation", ylabel="genotype diversity / initial")
    right.set(xlabel="generation", ylabel="best distance to target (m)")
    right.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig1.{suffix}", dpi=200)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for arm in ea_arms:
        _band(left, runs, arm, "diff_proposal_rms", log=True)
        _band(right, runs, arm, "unique")
    left.set(xlabel="generation", ylabel="differential step RMS")
    right.set(xlabel="generation", ylabel="unique genomes")
    right.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig2.{suffix}", dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a mutation A/B results root")
    parser.add_argument("root", type=Path)
    parser.add_argument("--poc", action="store_true", help="also evaluate the POC go/no-go criteria")
    parser.add_argument("--final-seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--deadline", type=date.fromisoformat, default=date(2026, 10, 8))
    args = parser.parse_args(argv)

    runs = load_runs(args.root)
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)

    per_seed = [
        {"seed": seed, "arm": arm, **seed_metrics(runs[seed][arm])} for seed in sorted(runs) for arm in ARMS
    ]
    _write_csv(out / "seed_metrics.csv", per_seed)
    h4 = [
        {"seed": row["seed"], "arm": row["arm"], **{k: row[k] for k in ("improve_rate_difference", "improve_rate_gaussian", "improved_count", "useful_rate", "mean_gain")}}
        for row in per_seed
        if row["arm"] != ARM_RANDOM
    ]
    _write_csv(out / "h4.csv", h4)
    report = {"seeds": sorted(runs), "tests": statistical_tests(runs)}
    if args.poc:
        report["go_no_go"] = go_no_go(
            runs, final_seeds=args.final_seeds, workers=args.workers, deadline=args.deadline, now=datetime.now()
        )
    (out / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
    figures(runs, out)

    print(json.dumps(report["tests"], indent=2))
    if args.poc:
        for name, criterion in report["go_no_go"]["criteria"].items():
            verdict = "PASS" if criterion["passed"] else "FAIL"
            print(f"{verdict}  {name}: {criterion['value']}  ({criterion['rule']})")
        g = report["go_no_go"]
        print(f"G_final = {g['g_final']} ({'plateau' if g['plateau_found'] else 'no plateau by POC end; use cap'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests/test_analysis.py -q`
Expected: all pass. If `stats.wilcoxon(..., method="exact")` raises when some differences are zero, keep `zero_method="wilcox"` and switch to `method="auto"` only for that case, with a one-line comment saying why. The zero-difference test pins this behaviour.

- [ ] **Step 5: Commit**

```bash
git add assignments/assignment_2/mutation_ab
git commit -m "feat(assignment-2): add A/B analysis, figures and POC go/no-go"
```

---

### Task 8: README, full verification, smoke run and speed benchmark

**Files:**
- Create: `assignments/assignment_2/mutation_ab/README.md`

**Interfaces:**
- Consumes: the CLIs from Tasks 6 and 7.
- Produces: verified commands and a measured seconds-per-evaluation figure used for the POC launch.

- [ ] **Step 1: Write `README.md`**

````markdown
# Mutation A/B: differential mutation vs 10% Gaussian replacement

Assignment 2 experiment. Design: `docs/superpowers/specs/2026-09-28-a2-mutation-ab-design.md`.

Run everything from `assignments/assignment_2` with the repository environment (`uv sync` at the repository root).

## Tests

```bash
uv run pytest mutation_ab/tests -q
```

## Smoke run (seconds)

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/smoke --seeds 900 --generations 2 --population 4 --duration 0.2
uv run python -m mutation_ab.analysis mutation_ab/results/smoke
```

## POC / calibration (seeds 700-705, never reused)

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/poc --seeds 700-705 --generations 80 --workers 6
uv run python -m mutation_ab.analysis mutation_ab/results/poc --poc --final-seeds 10 --workers 10
```

## Final run (G from the POC's G_final)

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/final --seeds 800-809 --generations G_FINAL --workers 10
uv run python -m mutation_ab.analysis mutation_ab/results/final
```

## Layout

Each `results/<name>/seed_<s>/<arm>/` holds `config.json` (configuration, hashes, git commit), `children.jsonl` (every evaluation), `generations.csv`, `adults.npz`, `ariel.db` (EA arms) and `COMPLETE`, or `FAILED.json` if a simulation diverged. Existing directories are never overwritten.
````

- [ ] **Step 2: Run the full package test suite**

Run: `uv run pytest assignments/assignment_2/mutation_ab/tests -q`
Expected: all tests pass (show the summary line).

- [ ] **Step 3: Lint the package**

Run: `uv run ruff check assignments/assignment_2/mutation_ab`
Expected: no errors. Fix real findings. Where the repository's `ruff.toml` flags rules that conflict with this plan's style (for example, docstring requirements on small private helpers), add a per-directory ignore in the package via `# ruff: noqa` only if necessary, and state which rule and why in the commit message.

- [ ] **Step 4: Run the repository tests and the ARIEL-untouched check**

Run: `uv run pytest tests -q && git status --porcelain -- src/ariel`
Expected: 437 passed and empty `git status` output.

- [ ] **Step 5: Smoke-run the CLI end to end**

Run (from `assignments/assignment_2`): the two smoke commands from the README.
Expected: exit code 0, `mutation_ab/results/smoke/analysis/fig1.png` exists.

- [ ] **Step 6: Benchmark 15 s episodes under parallel load**

Run (from `assignments/assignment_2`):
```bash
uv run python -m mutation_ab.run --out mutation_ab/results/bench --seeds 990-995 --generations 1 --workers 6
```
Then:
```bash
uv run python -c "import json,glob,statistics as s; w=[json.loads(l)['wall_s'] for f in glob.glob('mutation_ab/results/bench/seed_*/*/children.jsonl') for l in open(f)]; print(len(w), round(s.mean(w),3))"
```
Record the mean seconds per evaluation. POC wall-clock estimate = 6 seeds × (12 + 3·11·80) evaluations × mean ÷ 6 workers. Report this estimate to the user before launching the POC.

- [ ] **Step 7: Commit**

```bash
git add assignments/assignment_2/mutation_ab/README.md
git commit -m "docs(assignment-2): document mutation A/B commands"
```

---

### Task 9: Launch the POC and report go/no-go

**Files:** none (run outputs are gitignored under `mutation_ab/results/`).

- [ ] **Step 1: Launch the POC in the background** (from `assignments/assignment_2`)

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/poc --seeds 700-705 --generations 80 --workers 6 > mutation_ab/results/poc.log 2>&1
```
Expected: runs for roughly the Task 8 estimate. Progress is visible through the growing line counts of each `children.jsonl` (the final count per arm is 892).

- [ ] **Step 2: Confirm completion**

Run: `ls mutation_ab/results/poc/seed_*/*/COMPLETE | wc -l` → expected `18`. If there are `FAILED.json` files, stop and report them with the recorded traceback.

- [ ] **Step 3: Analyse**

```bash
uv run python -m mutation_ab.analysis mutation_ab/results/poc --poc --final-seeds 10 --workers 10
```
Expected: PASS/FAIL lines for `speed`, `a_collapses`, `b_stays_alive`, `h4_measurable`, and `G_final`.

- [ ] **Step 4: Report to the user**

Summarise each criterion with its value, G_final, and `fig1.png`/`fig2.png`. Label POC p-values exploratory. Recommend the final-run command (seeds 800–809, `--generations G_final`, or fewer seeds if speed fails). Do not start the final run without the user's go-ahead.
