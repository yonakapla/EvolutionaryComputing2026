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
| `mixture` | 90% F(b − c), 10% Gaussian | mixed | reference |
| `de_rand_1_bin` | canonical DE, F = 0.5, Cr = 0.9 | population | reference: DE (0.5, 0.9) |
| `de_rand_1_bin_matched` | canonical DE with the EA arms' F and Cr | population | reference: DE matched |
| `random` | fresh random genomes | none | baseline |

All numbers in the report are in `analysis_results/final_spider/report.txt`.

### Hypotheses

| Hypothesis | Where the result is |
|---|---|
| H1 mechanism: A and C collapse to a single genome; B and D never do | `summary.csv`, columns `collapsed_seeds`, `collapse_generation_range`; `fig_mechanism` |
| H2 size (primary): (A + C)/2 − (B + D)/2 > 0 | `stats.csv`, family `factorial`, test `size: …` |
| H3 direction: (A + B)/2 − (C + D)/2 ≠ 0 | `stats.csv`, family `factorial` (`direction: …`, `interaction: …`) and family `direction within size` |
| H4 baseline: B and D beat random search, A and C do not | `stats.csv`, family `vs random search` |
| References, compared descriptively | `stats.csv`, family `references (unadjusted)` |

Run all commands from **`assignments/assignment_2`**.

## Setup

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv venv    # create the virtual environment
uv sync    # install ariel and every dependency, at the versions in uv.lock
```

The root [`README.md`](../../../README.md) documents the ariel framework and its
MuJoCo requirement.

## Reproducing the results

**What is in git:** the analysis of every experiment, in
`analysis_results/<experiment>/` (tables, `report.txt`, figures, replays).
**What is not:** the raw runs (0.6 to 4 GB per experiment, in the ignored
`results/` folder). So the report's numbers can be read straight from
`analysis_results/`, but recomputing them means rerunning the experiment.

### Quick check (about 20 seconds)

```bash
uv run pytest mutation_ab/tests
uv run python -m mutation_ab.run --out mutation_ab/results/smoke --seeds 900,901 --arms difference,gaussian,random --generations 2 --population 4 --duration 0.2
uv run python -m mutation_ab.analysis mutation_ab/results/smoke
```

The run prints one line per finished run and ends with `6 runs complete, 0
failed`; the analysis writes into `mutation_ab/results/smoke/analysis/`.

### The final experiment (about 7 hours on 10 workers)

```bash
# 1. 10 seeds × 8 arms
uv run python -m mutation_ab.run --out mutation_ab/results/final_spider --seeds 1000-1009 --workers 10

# 2. Tables, figures and report                         (25 seconds)
uv run python -m mutation_ab.analysis mutation_ab/results/final_spider --out mutation_ab/analysis_results/final_spider

# 3. Replay the best controller of each run             (35 seconds)
uv run python -m mutation_ab.replay mutation_ab/results/final_spider --out mutation_ab/analysis_results/final_spider/replay.csv
```

One run of 8,812 evaluations takes about 40 minutes on one worker. Each worker
runs one seed, with its arms one after the other. Rerunning step 2 on the same
runs gives byte-identical files.

### Supplementary runs

Same setup apart from the change listed.

| Run | Arms | Change | Seeds | Question |
|---|---|---|---|---|
| `supp_pop48` | A, D | population 48, 186 generations (8,790 evaluations) | 1000–1009 | Does the collapse depend on the small population? |
| `supp_long` | B, D, `de_rand_1_bin` | 1,600 generations | 1000–1009 | Do the results hold with twice the budget? |
| `supp_sigma_0.05`, `supp_sigma_0.3` | B, D | Gaussian step size 0.05 or 0.3 | 1000–1004 | Does the result depend on σ = 0.15? |

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/supp_pop48 --seeds 1000-1009 --arms difference,gaussian --population 48 --generations 186 --workers 10
uv run python -m mutation_ab.run --out mutation_ab/results/supp_long --seeds 1000-1009 --arms normalised,gaussian,de_rand_1_bin --generations 1600 --workers 10
uv run python -m mutation_ab.run --out mutation_ab/results/supp_sigma_0.05 --seeds 1000-1004 --arms normalised,gaussian --gaussian-sd 0.05 --workers 5
uv run python -m mutation_ab.run --out mutation_ab/results/supp_sigma_0.3 --seeds 1000-1004 --arms normalised,gaussian --gaussian-sd 0.3 --workers 5
```

Analyse and replay each one as in steps 2 and 3, with its own folder names. The
σ runs contain only B and D, so their analysis also takes A, C and random search
from the final run (none of them uses σ):

```bash
uv run python -m mutation_ab.analysis mutation_ab/results/supp_sigma_0.05 --out mutation_ab/analysis_results/supp_sigma_0.05 --sigma-free-arms-from mutation_ab/results/final_spider
```

### Options

| Script | Options |
|---|---|
| `run` | `--out` (required, must not exist yet), `--seeds` (required, e.g. `1000-1009` or `900,901`), `--arms` (default all eight), `--generations 800`, `--population 12`, `--duration 15`, `--body spider_8`, `--gaussian-sd 0.15`, `--workers 1` |
| `analysis` | the results folder, `--out` (default `<folder>/analysis`), `--sigma-free-arms-from` |
| `replay` | the results folder, `--out` |

## Files

The experiment itself is in the first four; read them in this order.

| File | What it does |
|---|---|
| `config.py` | The arms, every setting of a run (`RunConfig`), and a separate random-number stream for each kind of random choice |
| `operators.py` | How a child is made: tournament parent, two donors, the arm's step (`shaped_step`), the crossover mask |
| `ea.py` | The shared initial population, the generational EA with one elite (`run_ea`) and canonical DE (`run_de`), built on `ariel.ec.EA` |
| `simulation.py` | The robot in its world, the neural network a genome encodes, and one 15 s episode |
| `random_search.py` | Random search with the same number of evaluations |
| `run.py` | Runs every arm on every seed |
| `records.py` | Writes the result files and reads a folder of runs back |
| `analysis.py` | Makes the tables, figures and report, using the three files below |
| `stats.py` | The hypothesis tests and the plateau rule |
| `plots.py` | The figures |
| `conditions.py` | Names, colours and line styles of the arms |
| `metrics.py` | Step size and population diversity |
| `replay.py` | Replays each run's best controller and compares the distance |
| `tests/` | 43 tests (50 with their parameter cases); `test_steps.py` works one step through all four arms |
| `analysis_results/` | The analysis of each experiment, in git |
| `lint.toml` | Ruff settings for this folder |

## What each result file contains

```
results/final_spider/seed_1000/
  difference/  config.json  children.jsonl  generations.csv  adults.npz  steps.npz  ariel.db  COMPLETE
  ...          (one folder per arm; random search has no adults.npz or steps.npz)
analysis_results/final_spider/
  report.txt  summary.csv  per_seed.csv  stats.csv  plateau.csv  step_span.csv  step_shape.csv  replay.csv
  fig_fitness  fig_mechanism  fig_seeds  (.pdf and .png)
```

| File | Contents | In git |
|---|---|---|
| `config.json` | The arm, every setting, the git commit and the library versions | no |
| `children.jsonl` | One line per evaluation: parent, donors, step size, final position, distance | no |
| `generations.csv` | One row per generation: best, mean and best-so-far distance, distinct genomes, size of F(b − c) | no |
| `adults.npz`, `steps.npz` | The surviving genomes of every generation; the steps of every 10th generation | no |
| `COMPLETE` or `FAILED.json` | A finished run, or a diverged simulation with its error and genome (none occurred) | no |
| `ariel.db` | ariel's own record of every individual; not used by the analysis | no |
| `report.txt` | All results in readable form | yes |
| `per_seed.csv`, `summary.csv` | Final distance, collapse, clones, success and Gaussian-step share, per run and per arm | yes |
| `stats.csv` | One row per test: mean difference (positive = first-named arm ends further away), 95% interval, seeds positive, `p` and `p_holm` | yes |
| `plateau.csv`, `step_shape.csv`, `step_span.csv` | Plateau generation and late gains; realised step shape; step direction | yes |
| `replay.csv` | Recorded and replayed distance of each run's best controller | yes |
| `fig_*.pdf`, `fig_*.png` | The figures; PDF for LaTeX, PNG for viewing | yes |

## Design choices and reproducibility notes

- **F = 0.15 / (√2 · 0.5) ≈ 0.212**, so that F(b − c) of two initial genomes
  starts the size of a Gaussian step of 0.15. `--gaussian-sd` leaves F unchanged.
- **Step sizes are RMS over all 260 weights, before the crossover mask**
  (rate 0.2 plus one forced weight).
- **B falls back to the Gaussian draw when F(b − c) is zero** (3.3% of its steps).
- **Canonical DE uses 12 evaluations per generation, the EA arms 11**, so DE runs
  733 generations (8,808 evaluations) and is plotted at the EA generation with
  the same number of evaluations.
- **Paired runs.** Every arm of a seed starts from the same initial population,
  and each kind of random choice has its own stream, so two EA arms on one seed
  differ only in how they turn the same numbers into a step (tested).
- **Seeds:** 1000–1009 for the final runs, 1000–1004 for the σ runs. The pilots
  (seeds 700–705, 910–914 and 920–924) were exploratory and are not reused.
- **Failures are never rerun silently.** A diverged simulation fails its run;
  none did.
- **Each run records its git commit** in `config.json`, and `run` refuses an
  existing output folder, so no run overwrites another.
- **Replays are exact on the machine that ran the experiment.** On a different
  machine, small floating-point differences grow over a 15 s episode and single
  replays differ by up to 1.1 m; the size contrast stays at +0.95 m (all ten
  seeds) against +0.96 m recorded.
- Library versions are pinned by `uv.lock`. We make no changes to the ariel
  framework (`src/ariel`).

## Linting and formatting

The settings live in `lint.toml` so that they apply only when passed explicitly:

```bash
uv run ruff check --no-fix --config mutation_ab/lint.toml mutation_ab
uv run ruff format --config mutation_ab/lint.toml mutation_ab
```
