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

## Progress output

While running, the main process prints an overall line every 60 s (change with `--heartbeat SECONDS`, `0` turns it off), for example:

```
15:05:55  overall 64% (63,910/99,360 evals) · 230 s/10k evals · runs done 10/30 · ETA 15:21
```

Each (seed, arm) run also prints its best distance every 10 generations and a `done` or `FAILED` line.

## Layout

Each `results/<name>/seed_<s>/<arm>/` holds `config.json` (configuration, hashes, git commit), `children.jsonl` (every evaluation), `generations.csv`, `adults.npz`, `ariel.db` (EA arms) and `COMPLETE`, or `FAILED.json` if a simulation diverged. Existing directories are never overwritten.

## Step size × step direction (final experiment)

Protocol: `PROTOCOL.md`. Arms: `difference`, `normalised`, `size_matched`, `gaussian` (the 2×2),
plus references `mixture`, `de_rand_1_bin`, `de_rand_1_bin_matched` and baseline `random`.

```bash
uv run python -m mutation_ab.run --out mutation_ab/results/final_spider --seeds 1000-1009 \
  --arms difference,normalised,size_matched,gaussian,mixture,de_rand_1_bin,de_rand_1_bin_matched,random \
  --body spider_8 --world flat --generations 800 --workers 10
```

Each EA and DE run also writes `steps.npz` (every 10th generation's steps, for the PCA).
The original A/B commands above still work unchanged (default arms, body and world).

Analysis (2×2 contrasts with Holm and bootstrap CIs, plateau per arm, clone share, step-span PCA, figures):

```bash
uv run python -m mutation_ab.factorial_analysis mutation_ab/results/final_spider
```

Writes `analysis/` inside the results folder: `report.txt`, `summary.csv`, `per_seed.csv`, `stats.csv`,
`plateau.csv`, `step_span.csv` and `fig_fitness`, `fig_mechanism`, `fig_seeds` (PNG and PDF).
