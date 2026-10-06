# Assignment 2: step size × step direction in small-population neuroevolution

**Research question.** How much of the performance gap between
population-difference and Gaussian mutation in a small-population
neuroevolutionary EA is due to step size, and how much to step direction?

A population of 12 controllers for the `spider_8` body learns to walk to a
target 2 m away. Four arms form a 2×2: the step either follows the population
(F(b − c), from two other members) or a random direction, and its size either
shrinks with the population or stays fixed. Everything else is the same in
every arm: the network (260 weights), the initial population, tournament
selection, the crossover mask, generational replacement with one elite, and the
budget of 8,812 evaluations.

| Arm | Step direction | Step size | Role |
|---|---|---|---|
| A `difference` | population, F(b − c) | shrinks with the population | DE-style mutation |
| B `normalised` | population, F(b − c) | fixed, RMS 0.15 | |
| C `size_matched` | random | the size of F(b − c) | |
| D `gaussian` | random | fixed, SD 0.15 | Gaussian mutation |
| `mixture` | 90% F(b − c), 10% Gaussian | mixed | reference: does occasional injected variation prevent collapse? |
| `de_rand_1_bin` | canonical DE, F = 0.5, Cr = 0.9 | population | reference: DE (0.5, 0.9) |
| `de_rand_1_bin_matched` | canonical DE with the EA arms' F and Cr | population | reference: DE matched |
| `random` | fresh random genomes | none | baseline |

Over ten paired seeds, fixing the step size brings the robot 0.96 m closer to
the target (95% interval 0.78 to 1.15 m, better in every seed), while no effect
of the direction source is detected (−0.05 m, interval −0.17 to +0.06 m). All
numbers are in `analysis_results/final_spider/report.txt`.

Run all commands from **`assignments/assignment_2`**.

## Reading the code

The experiment itself is in four files; read them in this order:

1. `config.py`: the arms, every setting of a run (`RunConfig`), and the
   separate random-number stream for each kind of random choice.
2. `operators.py`: how a child is made: tournament parent, two donors, the
   arm's step (`shaped_step`), and the crossover mask. The whole 2×2 lives in
   `shaped_step` and `propose_child`.
3. `ea.py`: the shared initial population, the generational EA with one elite
   (`run_ea`), and canonical DE (`run_de`), both built on `ariel.ec.EA`.
4. `simulation.py`: the robot in its world, the controller a genome encodes,
   and the episode that scores it.

The rest runs the experiment (`run.py`, `random_search.py`, `records.py`) or
analyses it (`analysis.py`, `stats.py`, `plots.py`, `conditions.py`,
`replay.py`, `metrics.py`).

### The 2×2 on one step

Take one population difference F(b − c) = [0.3, 0, 0, −0.4], of size (RMS)
0.25, and one Gaussian draw [0.15, −0.15, 0.15, −0.15], of size 0.15:

| Arm | Step | Size | Direction |
|---|---|---|---|
| A difference | [0.3, 0, 0, −0.4] | 0.25 | population |
| B normalised | [0.18, 0, 0, −0.24] | 0.15 | population |
| C size-matched | [0.25, −0.25, 0.25, −0.25] | 0.25 | random |
| D Gaussian | [0.15, −0.15, 0.15, −0.15] | 0.15 | random |

Once the population has converged to one genome, every difference is zero: A
and C stop moving for good, while B and D keep taking steps of 0.15. (B then
has no direction to keep and falls back to the Gaussian draw.)
`tests/test_steps.py` checks exactly these numbers.

### Where each part of the report comes from

| Report | Code | Output |
|---|---|---|
| Sect. 2.1, task, controller and fitness | `simulation.py` | |
| Sect. 2.2, parent, donors, mask and the proposal rules of Table 1 | `operators.py`: `propose_child`, `shaped_step`, `binomial_mask` | |
| Sect. 2.2, generational EA with one elite | `ea.py`: `reproduce`, `survive`, `run_ea` | |
| Sect. 2.2, reference conditions | canonical DE: `ea.run_de`, `operators.de_trial`; mixture: `RunConfig.replacement_probability_for`; random search: `random_search.py` | |
| Table 2, parameter settings | `config.RunConfig` | `config.json` of each run |
| Sect. 2.4, paired runs and failed runs | `ea.make_initial`, `config.make_streams`, `simulation.UnstableSimulation`, `RunRecorder.fail` | `FAILED.json` (none occurred) |
| Sect. 2.4, plateau rule | `stats.plateau_generation` | `plateau.csv` |
| Sect. 2.4, clones, success and realised steps | `analysis.seed_metrics`, `analysis.step_shape` | `per_seed.csv`, `summary.csv`, `step_shape.csv` |
| Sect. 2.4, contrasts S, Q, I, Holm correction and bootstrap intervals | `stats.statistical_tests` | `stats.csv` |
| Figure 1 | `plots.fig_fitness` | `fig_fitness` |
| Figure 2 | `plots.fig_mechanism` | `fig_mechanism` |
| Table 3 | `analysis.summary_rows`, `stats.plateaus` | `summary.csv`, `plateau.csv` |
| Table 4 | `stats.statistical_tests` | `stats.csv` |
| Sect. 3.5, population 48 and twice the budget | `run.py` options | `analysis_results/supp_pop48/`, `analysis_results/supp_long/` |
| Sect. 3.6, step scale σ = 0.05 and 0.3 | `run.py --gaussian-sd`, `analysis.py --sigma-free-arms-from` | `analysis_results/supp_sigma_*/` |
| Sect. 2.4 and 3.6, replays | `replay.py` | `replay.csv` |

The outputs are in `analysis_results/final_spider/` unless named otherwise.
`fig_seeds` and `step_span.csv` are not used in the report.

## Setup

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv venv
uv sync
```

The root [`README.md`](../../../README.md) documents the ariel framework and its
MuJoCo requirement. To check the setup:

```bash
uv run pytest mutation_ab/tests
```

## Files

| File | What it does |
|---|---|
| `A2_template_2026.py`, `Assignment2.html` (one folder up) | Course template and assignment text (reference only) |
| `config.py` | The arms, every setting of a run, and the random-number streams |
| `operators.py` | Parent and donor selection, the step of each arm, the crossover mask |
| `ea.py` | The initial population, the EA of the 2×2 arms and the mixture, and canonical DE/rand/1/bin, built on `ariel.ec.EA` |
| `simulation.py` | The world with the robot in it, the neural network a genome encodes, and one episode |
| `random_search.py` | Random search with the same number of evaluations |
| `metrics.py` | Step size and population diversity |
| `records.py` | Writing the result files, and reading a folder of runs back |
| `run.py` | Runs every arm on every seed, optionally in parallel |
| `analysis.py` | Makes the tables, figures and report, using the three files below |
| `conditions.py` | Names, colours and line styles of the arms, and which arms the figures show |
| `stats.py` | The hypothesis tests and the plateau rule |
| `plots.py` | The three figures |
| `replay.py` | Replays each run's best controller and compares the distance |
| `analysis_results/` | The analysis of each experiment, in git |
| `tests/` | 42 tests (49 with their parameter cases) for the code above |
| `lint.toml` | Ruff settings for this folder (see below) |

## Reproducing the results

The raw runs take about 400 MB per experiment and stay out of git (`results/` is
ignored). What git holds is each experiment's analysis in
`analysis_results/<experiment>/`.

### A short run on its own

Give test runs their own output folder so they never mix with the final
results. This takes a few seconds:

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/smoke --seeds 900,901 --arms difference,gaussian,random --generations 2 --population 4 --duration 0.2
uv run python -m mutation_ab.analysis mutation_ab/results/smoke
```

The run prints one line per finished run:

```
2 seeds x 3 arms, 10 evaluations per run, 1 worker(s) -> mutation_ab/results/smoke
difference seed 900: best 1.990 m (0s)
gaussian seed 900: best 1.991 m (0s)
...
6 runs complete, 0 failed
```

### Everything, from scratch

```bash
# 0. Tests                                                     (16 seconds)
uv run pytest mutation_ab/tests

# 1. Final experiment: 10 seeds × 8 arms                       (about 7 hours on 10 workers)
uv run python -m mutation_ab.run --out mutation_ab/results/final_spider --seeds 1000-1009 --workers 10

# 2. Tables, figures and report                                (25 seconds)
uv run python -m mutation_ab.analysis mutation_ab/results/final_spider --out mutation_ab/analysis_results/final_spider

# 3. Replay the best controller of each run                    (35 seconds)
uv run python -m mutation_ab.replay mutation_ab/results/final_spider --out mutation_ab/analysis_results/final_spider/replay.csv
```

An evaluation takes about 0.26 s of simulation, so one run of 8,812
evaluations takes about 40 minutes on one worker. Each worker runs one seed
(all its arms, one after the other).

The analysis output has no timestamps, so rerunning step 2 on the same runs
gives byte-identical files and git shows a change only when a result changes.
The one exception is the last digit or two of `step_span.csv`, which comes from
an SVD and depends on the machine's linear algebra library.

### Supplementary runs

Added after the final run, as checks rather than tests of the hypotheses. They
use the same setup apart from the change listed.

| Run | Arms | Change | Seeds | Question |
|---|---|---|---|---|
| `supp_pop48` | A, D | population 48, 186 generations (8,790 evaluations) | 1000–1009 | Does the collapse depend on the small population? |
| `supp_long` | B, D, `de_rand_1_bin` | 1,600 generations | 1000–1009 | Do B, D and canonical DE plateau, and do the results hold? |
| `supp_sigma_0.05`, `supp_sigma_0.3` | B, D | Gaussian step size 0.05 or 0.3 | 1000–1004 | Does the size effect depend on σ = 0.15? |

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/supp_pop48 --seeds 1000-1009 --arms difference,gaussian --population 48 --generations 186 --workers 10
uv run python -m mutation_ab.run --out mutation_ab/results/supp_long --seeds 1000-1009 --arms normalised,gaussian,de_rand_1_bin --generations 1600 --workers 10
uv run python -m mutation_ab.run --out mutation_ab/results/supp_sigma_0.05 --seeds 1000-1004 --arms normalised,gaussian --gaussian-sd 0.05 --workers 5
uv run python -m mutation_ab.run --out mutation_ab/results/supp_sigma_0.3 --seeds 1000-1004 --arms normalised,gaussian --gaussian-sd 0.3 --workers 5
```

Analyse each into `analysis_results/<run>/` as in step 2. The σ runs contain
only B and D, so their analysis takes A, C and random search from the final run
of the same seeds, which is valid because none of those three uses σ:

```bash
uv run python -m mutation_ab.analysis mutation_ab/results/supp_sigma_0.05 --out mutation_ab/analysis_results/supp_sigma_0.05 --sigma-free-arms-from mutation_ab/results/final_spider
uv run python -m mutation_ab.analysis mutation_ab/results/supp_sigma_0.3 --out mutation_ab/analysis_results/supp_sigma_0.3 --sigma-free-arms-from mutation_ab/results/final_spider
```

The first 800 generations of `supp_long` are identical to `final_spider`,
because both use the same seeds.

| Script | Options |
|---|---|
| `run` | `--out` (required, must not exist yet), `--seeds` (required, e.g. `1000-1009` or `900,901`), `--arms` (default all eight), `--generations 800`, `--population 12`, `--duration 15`, `--body spider_8`, `--gaussian-sd 0.15`, `--workers 1` |
| `analysis` | the results folder, `--out` (default `<folder>/analysis`), `--sigma-free-arms-from` |
| `replay` | the results folder, `--out` |

## The protocol

### Hypotheses

| Hypothesis | Where the result is |
|---|---|
| H1 mechanism: A and C collapse to a single genome and stop moving; B and D never do | `summary.csv`, columns `collapsed_seeds` and `collapse_generation_range`; `fig_mechanism` |
| H2 size (primary): (A + C)/2 − (B + D)/2 > 0 | `stats.csv`, family `factorial`, test `size: …` |
| H3 direction: (A + B)/2 − (C + D)/2 ≠ 0 | `stats.csv`, family `factorial`, tests `direction: …` and `interaction: …`; family `direction within size` |
| H4 baseline: B and D beat random search, A and C do not | `stats.csv`, family `vs random search` |
| References, compared descriptively | `stats.csv`, family `references (unadjusted)` |

### Fixed before the final run

The design, the hypotheses and the analysis below were written down and
committed on 28 September 2026 (commit `1b14f71`), before any final-run result
existed:

- **Primary outcome:** the best-so-far distance at the end of the budget.
- **Tests:** paired by seed, exact two-sided Wilcoxon signed-rank, Holm
  correction within the family {H2, H3, interaction} and within the baseline
  comparisons, bootstrap 95% intervals for the effects. The references are
  compared descriptively.
- **Plateau rule:** an arm has plateaued at the first generation from 40 on
  (checked every 10) at which its mean best-so-far improved by less than 5 mm
  over the previous 15 generations. An arm that never plateaus is compared at
  the full budget.
- **Failures:** a run whose simulation diverges is recorded as failed and
  reported, never silently rerun.
- **Seeds 1000 to 1009.** The pilots that motivated the design (the gecko in
  OlympicArena, seeds 700–705 and 910–914; the spider on flat ground, seeds
  910–914 and 920–924) were exploratory and are not reused.
- **Mechanism analyses (secondary):** step size, population difference size,
  distinct genotypes and diversity per generation; the share of clones and the
  success rate without them; how much of each step lies in the space the
  population spans; and the behaviour of each arm's best controller (path,
  speed, falls), which is not part of this code.

The supplementary runs were added on 29 September, after seeing the final
results, and are reported as checks rather than tests of these hypotheses.
Canonical DE was added to `supp_long` before that run started, because it was
still improving at 800 generations and was the best arm in the final run.

## What each result file contains

```
results/final_spider/
  seed_1000/
    difference/  config.json  children.jsonl  generations.csv  adults.npz  steps.npz  ariel.db  COMPLETE
    normalised/  (same files)
    ...
    random/      config.json  children.jsonl  generations.csv  COMPLETE
analysis_results/final_spider/
  report.txt  summary.csv  per_seed.csv  stats.csv  plateau.csv  step_span.csv  step_shape.csv  replay.csv
  fig_fitness  fig_mechanism  fig_seeds  (.pdf and .png)
```

| File | Contents | In git |
|---|---|---|
| `config.json` | The arm, every setting of the run, and the Python, MuJoCo and ariel versions | no |
| `children.jsonl` | One line per evaluation (see the fields below) | no |
| `generations.csv` | One row per generation (see the columns below) | no |
| `adults.npz` | The surviving genomes of every generation, shape (generations + 1, 12, 260) | no |
| `steps.npz` | The step of every child in every 10th generation, with its kind: the data behind `step_span.csv` and `step_shape.csv` | no |
| `ariel.db` | ariel's own record of every individual. The analysis does not use it | no |
| `COMPLETE` | Marks a finished run; holds its evaluations and best distance | no |
| `FAILED.json` | Instead of `COMPLETE` when the simulation diverged: the error, traceback and offending genome | no |
| `report.txt` | All results below in readable form | yes |
| `summary.csv`, `per_seed.csv` | Final distance, collapse and step statistics, per arm and per run | yes |
| `stats.csv` | One row per hypothesis test | yes |
| `plateau.csv` | The plateau rule on each arm's mean curve | yes |
| `step_span.csv`, `step_shape.csv` | Step direction and shape (see the columns below) | yes |
| `replay.csv` | Recorded and replayed distance of each run's best controller | yes |
| `fig_*.pdf`, `fig_*.png` | The three figures, `fig_fitness` and `fig_mechanism` as in the report. PDF for LaTeX, PNG for quick viewing | yes |

### Columns of `generations.csv`

| Column | Meaning |
|---|---|
| `generation`, `evaluations` | Generation 0 is the initial population; evaluations count from the start of the run |
| `best`, `mean`, `worst` | Distance to the target (m) in the current population. For random search: in the current batch |
| `best_so_far` | Best distance found so far; the outcome of the experiment |
| `unique` | Distinct genomes in the population; 1 means it has collapsed |
| `diversity` | Mean pairwise distance between genomes, divided by √260 |
| `diff_proposal_rms`, `n_difference` | Mean size and number of the generation's non-Gaussian steps |
| `difference_rms` | Mean size of F(b − c), in every arm, whether or not the arm used it |
| `n_gaussian` | Children made with a Gaussian step |

### Fields of `children.jsonl`

`uid`, `generation`, `kind` (`init`, `difference`, `normalised`,
`size_matched`, `gaussian`, `de` or `random`), `parent_uid` and `donor_uids`,
`proposal_rms` (step size before the crossover mask), `change_rms` (what the
child actually changed; 0 for a clone), `difference_rms`, `parent_distance` and
`parent_xy`, `final_xy`, `distance`, `warnings` (MuJoCo warnings), `genome_sha1`
and `wall_s`.

### Columns of the analysis tables

- **`per_seed.csv`**: `best_final`; `collapse_generation` (first generation
  with one distinct genome, empty if never); `unique_final`;
  `frozen_genes_final` (weights equal across the final population);
  `gaussian_step_share`; `clone_share` (children identical to their parent);
  `success_rate` (children closer to the target than their parent), also
  `success_rate_excluding_clones`; `improving_children`.
- **`summary.csv`**: `best_mean`, `best_sd`, `best_median`, `best_min`,
  `best_max` over seeds, and the per-seed columns above averaged.
- **`stats.csv`**: `family`, `test`, `mean` of the per-seed difference (positive
  means the first-named arm ends further from the target), its bootstrap 95%
  interval `ci_low`–`ci_high`, `positive` (seeds where it is positive) out of
  `n`, the exact Wilcoxon `p` and `p_holm` (Holm-corrected within the family,
  empty for the references).
- **`plateau.csv`**: `plateau_generation` and `plateau_evaluations`, empty if
  the arm never plateaued, and the gain over the last 50 and 100 generations.
- **`step_span.csv`**: per logged generation, `in_span_share`, the share of each
  step's squared length inside the space the parents span, its `span_rank`, and
  `isotropic_share`, what a random direction would give (rank / 260).
- **`step_shape.csv`**: per arm, `zero_step_share`, and for the other steps
  `weights_changed_median`, `step_rms_median` and
  `change_per_changed_weight_median`.

All tests are paired by seed: every arm starts from the same initial population,
so the difference between two arms is taken per seed and tested against zero
with a two-sided Wilcoxon signed-rank test. With ten seeds the p-value is exact,
enumerating all 2<sup>10</sup> sign patterns, so the smallest value it can
report is 0.002 before correction.

## Linting and formatting

The settings live in `lint.toml` rather than `ruff.toml` so that they apply only
when passed explicitly, and never rewrite this folder as a side effect:

```bash
uv run ruff check --no-fix --config mutation_ab/lint.toml mutation_ab
uv run ruff format --config mutation_ab/lint.toml mutation_ab
```

## Design choices and known limits

- **Population 12, tournament of 3, 11 children per generation.** A small
  population is where difference mutation is known to struggle, and it is the
  setting the question is about. `supp_pop48` checks a population of 48.
- **F = 0.15 / (√2 · 0.5) ≈ 0.212.** Two initial genomes, drawn from N(0, 0.5²),
  differ by N(0, 2 · 0.5²) per weight, so this F makes F(b − c) start out the
  size of a Gaussian step of 0.15. A and D therefore begin equal in size and
  differ only in how the size develops. `--gaussian-sd` leaves F unchanged.
- **Sizes are RMS over all 260 weights, before the crossover mask.** Population
  differences are sparse, so B's steps change few weights by a lot (median 3
  weights, 0.40 each in `final_spider`), while D's change about a fifth of the
  weights by 0.15.
- **Crossover rate 0.2 plus one forced weight.** A child equals its parent
  only if the step is zero on every weight the mask selects: always for a
  collapsed A or C, and often for B, whose sparse steps change only a few
  weights (21% of B's children are clones).
- **B falls back to the Gaussian draw when F(b − c) is zero.** With identical
  donors there is no direction to normalise. This happened for 3.3% of B's
  steps.
- **Canonical DE spends 12 evaluations per generation, the EA arms 11.** DE
  runs 733 generations (8,808 evaluations) and is plotted at the EA generation
  with the same number of evaluations.
- **Steps are logged every 10th generation** to keep `steps.npz` small.
- **A diverged simulation fails its run and is never rerun silently.** MuJoCo
  resets a diverged state without error, which would otherwise look like a
  valid endpoint. No final run diverged.

## Reproducibility notes

- Each kind of random choice (initial genomes, selection, mask, Gaussian draw,
  mixture replacement, random search) has its own stream derived from the seed,
  and every generational EA arm draws from every stream in the same order. Two
  EA arms on the same seed therefore differ only in how they turn the same
  numbers into a step (there is a test for this).
- Final runs use seeds 1000 to 1009; the σ runs use 1000 to 1004.
- Running seeds in parallel gives the same runs as running them one after the
  other (there is a test for it).
- `run` refuses an output folder that already exists, so a run never overwrites
  another.
- Every run records in its `config.json` the git commit it was made with, and
  whether the code had uncommitted changes.
- **Replays are exact only on the machine that ran the experiment.** There,
  every replayed distance in `replay.csv` matches the recorded one. On a
  different machine small floating-point differences grow over a 15-second
  episode, and single replays differ by up
  to 1.1 m. The conclusion holds: the size contrast from those replays is
  +0.95 m, positive on all 10 seeds, against +0.96 m recorded.
- Library versions are pinned by `uv.lock`.
- We make no changes to the ariel framework (`src/ariel`).
