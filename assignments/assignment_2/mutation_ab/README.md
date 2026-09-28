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
