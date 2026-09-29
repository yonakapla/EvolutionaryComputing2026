# Final experiment protocol: step size × step direction

Written and committed on 2026-09-28, before any final-run result existed. The pilots that
motivated it (gecko/OlympicArena seeds 700–705 and 910–914, spider/flat seeds 910–914 and
920–924) are exploratory and are not reused.

## Research question

Why does population-difference (DE-style) mutation underperform Gaussian mutation in
small-population neuroevolution for targeted locomotion, and how much of the gap is due
to its step size versus its step direction?

## Fixed setup (identical in every arm)

- Body `john_set.spider_8`, world `SimpleFlatWorld`, spawn (0, 0, 0.1), target (2, 0), 15 s episodes.
- Controller: NN with inputs qpos, qvel, vector to target, 1 Hz sin/cos clock (33 for spider_8),
  6 tanh hidden units, tanh outputs × π/2 (8 hinges); 260 weights, initialised N(0, 0.5²).
- Fitness: final planar distance from the core to the target (lower is better).
- EA frame (all EA arms): population 12, tournament of 3, 11 children per generation,
  generational replacement with one elite, binomial mask Cr = 0.2 plus one forced coordinate.
- The same 12 initial genomes per seed in every arm; all arms consume the random streams identically.

## Arms

| Arm | Step direction | Step size | Role |
| --- | --- | --- | --- |
| `difference` | F(b − c), population | shrinks with the population | 2×2 cell A (DE-style mutation) |
| `normalised` | F(b − c), population | fixed, RMS 0.15 | 2×2 cell B |
| `size_matched` | isotropic Gaussian | RMS of F(b − c) | 2×2 cell C |
| `gaussian` | isotropic Gaussian | fixed, SD 0.15 | 2×2 cell D |
| `mixture` | 90% F(b − c), 10% Gaussian | mixed | reference (original A/B treatment) |
| `de_rand_1_bin` | canonical DE, F = 0.5, Cr = 0.9, one-to-one replacement | population | reference |
| `de_rand_1_bin_matched` | canonical DE, F = 0.212, Cr = 0.2 | population | reference |
| `random` | fresh N(0, 0.5²) genomes | none | required baseline |

F = 0.15 / (√2 · 0.5) makes F(b − c) match the Gaussian step size at initialisation, so
σ = 0.15 is the fixed step size of B and D. In `normalised`, a zero difference (identical
donors) falls back to the Gaussian draw and is logged as kind `gaussian`.

## Budget, seeds and stopping

- Seeds 1000–1009 (10 independent runs per arm).
- 800 generations: 12 + 11 · 800 = 8,812 evaluations per EA arm and for random search;
  canonical DE runs 733 generations = 8,808 evaluations.
- Plateau rule, fixed in advance: an arm has plateaued at the first generation g ≥ 40
  (checked every 10) at which its mean best-so-far improved by less than 0.005 m over the
  previous 15 generations — the rule already implemented in `analysis.plateau_generation`.
  We report each arm's plateau generation; if an arm has not plateaued by 800 we say so and
  compare at the full budget.
- A run whose simulation diverges is recorded as FAILED and reported, never silently rerun.

## Hypotheses and tests

Primary outcome: best-so-far distance at the end of the budget.

1. H1 (mechanism): `difference` and `size_matched` reach a single genome with exactly zero
   steps; `normalised` and `gaussian` never do. Descriptive (collapse generation per seed).
2. H2 (size, primary): per-seed contrast (A + C)/2 − (B + D)/2 > 0.
3. H3 (direction): per-seed contrast (A + B)/2 − (C + D)/2 ≠ 0 (two-sided).
4. H4 (baseline): B and D beat random search; A and C do not.

Tests are paired by seed: exact two-sided Wilcoxon signed-rank, Holm correction within the
family {H2, H3, interaction} and within the family of baseline comparisons; effect sizes with
bootstrap 95% intervals. Reference arms are compared descriptively.

## Mechanism analyses (secondary)

- Step RMS, population difference RMS, unique genomes and genotype diversity per generation.
- Share of exact clones among children; success rate (child closer than its parent)
  excluding clones, by arm and step size.
- PCA of logged steps (`steps.npz`, every 10th generation): how much of the step variance
  lies in the ≤ 11 dimensions spanned by the population, per arm.
- Behaviour of each arm's best controller: path, speed, falls.

## Addendum (29 Sep 2026, after the final run): supplementary runs

Added after seeing the final results, to answer reviewer-style questions. They are reported as
supplementary checks, not as tests of the pre-registered hypotheses, and use the same setup unless
stated.

| Run | Arms | Change | Seeds | Question |
| --- | --- | --- | --- | --- |
| `supp_pop48` | `difference`, `gaussian` | population 48, 186 generations (8,790 evaluations) | 1000–1009 | Does the collapse depend on the small population? |
| `supp_long` | `normalised`, `gaussian` | 1,600 generations | 1000–1009 | Do B and D plateau, and does the direction result hold? |
| `supp_sigma` | `normalised`, `gaussian` | `--gaussian-sd` 0.05 and 0.3 | 1000–1004 | Does the size effect depend on σ = 0.15? |
