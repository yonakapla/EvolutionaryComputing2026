# A2 mutation A/B: design

## Goal

A two-arm A/B study for Assignment 2 (targeted locomotion by neuroevolution), in the style of the Assignment 1 report: one manipulated component, paired seeds, an equal-budget random-search baseline, and a mechanism analysis inside the arms instead of extra arms.

- **A:** differential mutation (population-difference proposals) with binomial crossover.
- **B:** identical, except that with probability 0.10 a child's whole proposal is replaced by Gaussian noise.

**Research question.** In small-population neuroevolution for targeted locomotion, does replacing 10% of differential mutation proposals with Gaussian noise prevent loss of variation, and does the retained variation translate into better target-directed search?

**Motivation (course framing).** Differential mutation takes its step sizes from the spread of the population, so the steps shrink as the population converges (ch06 part 2, pp. 6 and 39). Small panmictic populations drift to one peak and converge prematurely (ch05, pp. 22, 25, 29). With a population of 12 these two properties predict that the steps collapse to zero. Mutation can supply new material and restore diversity (ch06 part 1, p. 39). The course's diversity-preservation methods act on population structure; B acts on the variation operator.

The code is built to final quality. The POC run doubles as the calibration run. The final run uses the same code with fresh seeds and a frozen generation count.

## Fixed setup (identical in both arms)

| Component | Setting |
|---|---|
| Body | `ariel...prebuilt_robots.john_set.gecko()`; nq=13, nv=12, nu=6 |
| World | `OlympicArena` with the rugged heightmap generated from `PerlinNoise(seed=42)` through an experiment-local subclass; `src/ariel` untouched |
| Spawn / target | (0, 0, 0.1) with floor-collision correction; target XY (2, 0) |
| Episode | 15 s simulated, headless, direct control `data.ctrl[:] = actions` |
| Controller | Inputs: qpos (13), qvel (12), target − core XY (2), sin/cos(2π·1 Hz·t) (2), giving 29. Hidden layer: 6 tanh units. Output: 6 tanh units × π/2. Biases on both layers; 222 parameters |
| Genome | Flat float vector of length 222; dimensions are validated against the compiled model |
| Initialisation | N(0, 0.5²) per weight |
| Fitness | Final core ground-plane (XY) Euclidean distance to the target; minimised |
| Population | 12 adults; each generation produces 11 children |
| Parent selection | Tournament of 3 distinct adults; lowest distance wins |
| Survivor selection | Generational with one elite: the best adult plus the 11 children |
| Differential proposal | δ = F·(b − c), F = 0.15/(√2·0.5) ≈ 0.2121; donors b and c are distinct adults, both different from the parent |
| Mask (binomial crossover) | Each coordinate takes the mutant value with probability Cr = 0.2, plus one forced coordinate; child = p + mask·δ |
| B only | With probability 0.10 per child, δ is replaced by an independent N(0, 0.15²) vector before masking |

We call A "differential mutation inside a generational EA" and not canonical DE/rand/1/bin, because the parent is the base vector and there is no parent-versus-child replacement. The 0.15 step is fixed (not self-adaptive). The forced coordinate guarantees one mutated position in the mask, not a numerical change when δ = 0.

## Architecture

The code lives in `assignments/assignment_2/mutation_ab/`. Each module has one job:

- `config.py`: a frozen `RunConfig` dataclass holding every constant above plus arm, seed, generations and output root. It is serialised to `config.json` and hashed.
- `world.py`: `SeededOlympicArena(BaseWorld subclass)` and `build_model() -> (MjModel, hashes)`. It records the terrain heightmap hash and the compiled model hash.
- `controller.py`: `observe(model, data) -> (29,)`, `unpack(genome) -> layers`, `act(layers, obs) -> (6,)`.
- `evaluate.py`: `evaluate(genome, model) -> EvalResult(final_xy, distance, warnings)`. It raises `UnstableSimulation` on non-finite qpos, qvel or ctrl.
- `operators.py`: pure functions that take an explicit `np.random.Generator` (tournament, donor draw, differential proposal, Gaussian replacement, mask, child construction, elite survival), plus metrics helpers (pairwise genotype diversity, RMS).
- `run.py`: the CLI. It runs A, B and random search for each seed, with seeds in parallel `ProcessPoolExecutor` workers. The EA steps are `ariel.ec` `EAOperation`s acting on `Individual`/`Population`, with ARIEL's SQLite database per (seed, arm). Per-child metadata goes in `Individual.tags`, mirrored to JSONL.
- `analysis.py`: H1–H4 metrics, statistics, figures and the POC go/no-go summary.
- `tests/`: unit tests plus a short real-simulation smoke test.

If the `ariel.ec` `EA` engine cannot support the paired random streams or the one-elite generational scheme, fall back to the `Individual`/`Population` data model driven by our own loop. This is decided and recorded in the first implementation task.

## Randomness and pairing

Each seed s spawns independent `np.random.Generator` streams from `SeedSequence(s)` for: initial population, parent selection and donors, mask, Gaussian vectors, replacement decisions, and random search.

A and B share the initial population. The replacement coin and the Gaussian vector are drawn for every child in both arms, and A discards them, so the two arms use their streams identically. `ariel.ec.set_seed(s)` is also called. Serial and parallel execution must give identical results.

The 12 initial genomes are evaluated once per seed and reused by A, B and random search, because the simulation is deterministic. They count toward every arm's budget.

## Budget and random search

Each arm uses 12 + 11·G evaluations per seed. Random search evaluates the 12 shared initial genomes and then 11·G fresh N(0, 0.5²) genomes. Its best-so-far is reported after each block of 11, which aligns it with the EA generations.

## Logging (per seed and arm directory)

- `config.json`: config, config hash, git commit, terrain and model hashes, ARIEL and MuJoCo versions.
- `children.jsonl`: one record per evaluated individual with generation, id, parent_id, donor_ids, proposal type (`init`/`difference`/`gaussian`/`random`), pre-mask δ RMS, realised change RMS, genome SHA1, final XY, distance, the parent's XY and distance, warning count and wall time.
- `generations.csv`: per generation, the best, mean and worst distance, the best-so-far, the unique-genome count, D_g and the mean difference-proposal RMS.
- `adults.npz`: the 12×222 adult matrix for every generation.
- The ARIEL SQLite database.
- `COMPLETE`: written last. The analysis refuses directories without it.

The runner refuses an existing output directory. Nothing is ever overwritten or deleted.

## Failure handling

`UnstableSimulation` aborts that (seed, arm) run, writes `FAILED.json` with the child record and traceback, and leaves no `COMPLETE` marker. Other seeds continue. No fitness is ever substituted for a failed episode. MuJoCo warnings are counted, not treated as errors.

## Metrics and hypotheses

Each seed contributes one value per metric; g = 1…G.

| ID | Metric | Definition | Expected |
|---|---|---|---|
| H1 | Genotype diversity | D_g = mean over the 66 adult pairs of ‖wᵢ − wⱼ‖₂/√222, divided by D₀ of the seed. Summarised as run mean D̄ and final D_G | B > A |
| H2 | Differential step | Per-generation mean pre-mask RMS of *difference* proposals (B's Gaussian children excluded). Summarised as run mean and final value | B > A |
| H3 | Best distance | Best-so-far at generation G: A vs B, and each vs random search | B ≤ A; both < random |
| H4 | Attribution | Within B, per seed: improvement rate (child distance < parent distance) of difference children vs Gaussian children | Descriptive |
| Diagnostic | Collapse | First generation with one unique genome; unique-genome trajectory | |
| Diagnostic | Useful change | Improve on the parent *and* move the endpoint ≥ 0.10 m; mean improvement over the parent | |

## Statistics and figures

- Report mean (sample SD) and the mean paired difference B − A.
- Exact two-sided Wilcoxon signed-rank tests over seeds (SciPy). Holm correction within families: diversity {D̄, D_G}, step {run mean, final}, fitness {A vs B, A vs random, B vs random}.
- POC p-values are labelled exploratory. With 6 seeds the minimum p is 0.031.
- Fig. 1, A1 layout: diversity per generation (left); best-so-far per generation with random search (right). Lines are means, bands are sample SD.
- Fig. 2: differential-step RMS (log y) and unique genomes per generation.
- Table: H4 improvement rates per arm and proposal type.

## POC protocol and go/no-go

- **POC:** seeds 700–705, G = 80, 15 s episodes, A + B + random search. These seeds are never reused in the final run.
- **Speed:** the measured mean wall time per episode, projected to the final run (3 × 10 seeds × (12 + 11·G) episodes on the available workers), must finish by 8 October 2026.
- **A collapses:** in ≥ 4/6 seeds, A's differential-step RMS drops below 1% of its generation-1 value by generation 80.
- **B stays alive:** in ≥ 4/6 seeds, B's differential-step RMS stays above 1% of its generation-1 value at generation 80.
- **H4 is measurable:** ≥ 20 parent-improving children per arm per seed.
- **G_final:** the smallest multiple of 10 (at least 40) at which both arms' mean best-so-far improved by less than 0.005 m over the preceding 15 generations. The assignment asks for the plateau, not a fixed generation count, to be the stopping criterion. If no plateau appears by generation 80, the final run uses G = 120, and the same plateau check is applied post hoc to the final curves and reported.
- **Fallbacks:** if speed fails, reduce the final seeds (minimum 5) before shortening episodes. If A does not collapse, report to the team before the final run and reframe "collapse" as "step shrinkage".
- **Final run:** seeds 800–809 (fewer only if speed requires), G = G_final, otherwise identical configuration.

`analysis.py --poc` prints each criterion with pass/fail and the value behind it.

## Testing

Unit tests:
- a zero δ when b = c
- replacement probability 0 and 1
- donors distinct from each other and from the parent
- mask count ≥ 1 and the forced coordinate
- one elite survives
- equal evaluation counts across arms
- genome length and unpacking round-trip
- observation length 29
- terrain hash identical across processes
- serial vs parallel equality on a tiny configuration
- the analysis refuses incomplete runs

Smoke test: seed 900, population 4, G = 2, 0.2 s episodes, run end to end, then the analysis. A 15 s benchmark of 12 episodes measures the wall time per episode before the POC.

## Assignment 2 compliance (Assignment2.html, sections 5–8)

| Requirement | How this design meets it |
|---|---|
| EA built on `ariel.ec`; own search design, no black-box optimiser | `Individual`/`Population`/`EAOperation` from `ariel.ec`; all operators are written in `operators.py`; no nevergrad or CMA-ES library |
| No changes to `src/ariel` (fraud otherwise) | All code lives in `assignments/assignment_2/mutation_ab/`. A test asserts that `git diff --quiet -- src/ariel` passes. Terrain seeding is done by subclassing in our own code. We never call `BaseWorld.store_to_xml` or `compile_terrains.py` (both write into `src/ariel/.../pre_compiled`), and we pass `load_precompiled=False` so a stray precompiled XML cannot silently replace the seeded terrain |
| Fixed John Set body | `john_set.gecko()` for the whole assignment |
| Fixed permitted world, not SimpleTiltedWorld | `OlympicArena`. Unseeded, its rugged heightmap differs on every build, so the Perlin seed is pinned to 42 and the heightmap and model hashes are checked per run |
| NN controller, not CPG | A 29-6-6 tanh MLP. The sin/cos clock is an *input* to the network (the assignment points out that a controller needs a rhythm signal); there are no oscillator dynamics |
| Fitness: final ground-plane Euclidean distance to a fixed target, lower is better | Core `qpos[0:2]` at the end of the episode vs (2, 0), as in `A2_template_2026.fitness_function` |
| At least 5 independent repeats with mean and spread | 10 final seeds (minimum 5 under the fallback); mean (sample SD) reported |
| Baseline at the same evaluation budget | Random search with exactly 12 + 11·G evaluations per seed |
| Line plot across generations showing average and std over runs | Fig. 1 right: mean ± sample SD best-so-far per generation, for both arms and random search |
| Tip: plateau, not a fixed generation count, as the stopping criterion | G_final is taken from the plateau observed in the POC and checked post hoc in the final run (see G_final) |
| Tip: test small first | Smoke test (population 4, G = 2, 0.2 s) and a 12-episode 15 s benchmark before the POC |
| Tip: seed parametrised, one invocation per configuration | `run.py --seeds 800-809 --generations G --workers N`; the seed never appears in code |
| Tip: organised outputs | One directory per (seed, arm), immutable, with a `COMPLETE` marker and config/commit hashes |
| Submission: `groupnumber.zip` containing the report PDF and code | The runner and analysis are self-contained in one directory, with a README giving exact commands, so they can be copied into the zip unchanged |
| Report: at most 6 pages, GECCO19 | Out of scope for the code; the figures are sized for a two-column layout |

## Out of scope

Pure Gaussian arm, step-floor arm, crossover variants, plateau-synchronised stopping, tuning of the 0.10 probability, videos (optional diagnostics later), and the earlier exploratory seed series (500–635) as evidence.
