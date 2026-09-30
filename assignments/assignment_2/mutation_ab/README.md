# Step size × step direction in small-population neuroevolution

Assignment 2 experiment: why does DE-style difference mutation fall behind Gaussian mutation when
evolving a spider_8 controller to walk to a target? Design, hypotheses and supplementary runs are in
`PROTOCOL.md`.

Run everything from `assignments/assignment_2` with the repository environment (`uv sync` at the
repository root).

## Tests

```bash
uv run pytest mutation_ab/tests -q
```

## Smoke run (seconds)

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/smoke --seeds 900,901 --arms difference,gaussian,random \
  --generations 2 --population 4 --duration 0.2
uv run python -m mutation_ab.analysis mutation_ab/results/smoke
```

## Final run

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/final_spider --seeds 1000-1009 --workers 10
uv run python -m mutation_ab.analysis mutation_ab/results/final_spider
```

Arms: the 2×2 `difference`, `normalised`, `size_matched`, `gaussian`; references `mixture`,
`de_rand_1_bin`, `de_rand_1_bin_matched`; baseline `random`. The defaults are the final experiment:
all eight arms, spider_8 in `SimpleFlatWorld`, 800 generations; `--arms` and `--generations` select less.

The analysis writes `analysis/` inside the results folder: `report.txt`, the tables `summary.csv`,
`per_seed.csv`, `stats.csv`, `plateau.csv`, `step_shape.csv`, `step_span.csv`, and the figures
`fig_fitness`, `fig_mechanism`, `fig_seeds` (PNG and PDF).

## Analysis results in git

Raw runs stay out of git (`results/` is ignored); the analysis of each experiment is tracked in
`analysis_results/<experiment>/`. After changing the code, rerun the experiment and refresh its analysis:

```bash
uv run python -m mutation_ab.analysis mutation_ab/results/final_spider --out mutation_ab/analysis_results/final_spider
uv run python -m mutation_ab.analysis mutation_ab/results/supp_pop48 --out mutation_ab/analysis_results/supp_pop48
uv run python -m mutation_ab.analysis mutation_ab/results/supp_sigma_0.05 --out mutation_ab/analysis_results/supp_sigma_0.05 \
  --sigma-free-arms-from mutation_ab/results/final_spider
uv run python -m mutation_ab.analysis mutation_ab/results/supp_long --out mutation_ab/analysis_results/supp_long
```

The outputs have no timestamps, so git shows a change only when a result changes. The σ runs only
contain B and D; `--sigma-free-arms-from` adds A, C and random search from the final run for the same
seeds, which is valid because none of them uses the Gaussian step size.

## Supplementary runs

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/supp_pop48 --seeds 1000-1009 --arms difference,gaussian \
  --population 48 --generations 186 --workers 10
uv run python -m mutation_ab.run --out mutation_ab/results/supp_sigma_0.05 --seeds 1000-1004 --arms normalised,gaussian \
  --gaussian-sd 0.05 --workers 5
uv run python -m mutation_ab.run --out mutation_ab/results/supp_long --seeds 1000-1009 \
  --arms normalised,gaussian,de_rand_1_bin --generations 1600 --workers 10
```

The first 800 generations of `supp_long` are identical to `final_spider`, since both use the same seeds.

## Replay

Replays the best controller of every run and compares it with the recorded distance; on the machine
that produced the runs every replay is exact.

```bash
uv run python -m mutation_ab.replay mutation_ab/results/final_spider --out mutation_ab/analysis_results/final_spider/replay.csv
```

## Progress output

While running, the main process prints an overall line every 60 s (change with `--heartbeat SECONDS`,
`0` turns it off), for example:

```
15:05:55  overall 64% (63,910/99,360 evals) · 230 s/10k evals · runs done 10/30 · ETA 15:21
```

Each (seed, arm) run also prints its best distance every 10 generations and a `done` or `FAILED` line.

## Layout

Each `results/<name>/seed_<s>/<arm>/` holds `config.json` (configuration, hashes, git commit),
`children.jsonl` (every evaluation), `generations.csv`, `adults.npz`, `steps.npz` (the steps of every
10th generation), `ariel.db` and `COMPLETE`, or `FAILED.json` if a simulation diverged. Existing
directories are never overwritten.

## Notes on the logged quantities

- `diff_proposal_rms` and `n_difference` in `generations.csv` cover every non-Gaussian step
  (difference, normalised, size-matched and DE); `difference_rms` is the size of F(b − c) in every arm.
- The normalised arm fixes the step's RMS over all weights before the crossover mask. Population
  differences are sparse, so its steps change few weights by a lot (`step_shape.csv`); Gaussian steps
  change about a fifth of the weights by small amounts.
- A canonical DE generation costs `population_size` evaluations, an EA generation `population_size - 1`.
  Fitness is plotted at equal evaluations; other per-generation quantities use each arm's own generations.
