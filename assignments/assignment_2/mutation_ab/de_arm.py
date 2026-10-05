"""Reference arm: canonical DE/rand/1/bin (Storn & Price 1997).

One-to-one replacement: each adult's trial replaces it only if the trial is at
least as good.
"""

import numpy as np
from ariel.ec import EAOperation, Population

from mutation_ab.config import RunConfig
from mutation_ab.ea_arm import (
    ArmContext,
    by_uid,
    evaluate_children,
    evolve,
    log_step,
    make_child,
    record_generation,
)
from mutation_ab.initial import Evaluator, InitialPopulation
from mutation_ab.operators import de_trial
from mutation_ab.records import RunRecorder
from mutation_ab.streams import Streams


def make_trials(
    population: Population, ctx: ArmContext, scale_f: float, crossover_rate: float
) -> Population:
    ctx.generation += 1
    targets = by_uid(population)
    genomes = np.array([ind.genotype for ind in targets], dtype=float)
    for index in range(len(targets)):
        proposal = de_trial(genomes, index, scale_f, crossover_rate, ctx.streams)
        population.append(make_child(ctx, proposal, targets))
        log_step(ctx, proposal, genomes)
    return population


def select_one_to_one(population: Population, ctx: ArmContext) -> Population:
    """Each trial competes only with the adult it was made for."""
    members = by_uid(population)
    adults = {
        ind.tags["uid"]: ind
        for ind in members
        if ind.tags["generation"] < ctx.generation
    }
    trials = [ind for ind in members if ind.tags["generation"] == ctx.generation]
    for trial in trials:
        target = adults[trial.tags["parent_uid"]]
        if trial.fitness <= target.fitness:
            target.alive = False
        else:
            trial.alive = False
    survivors = [ind for ind in members if ind.alive]
    record_generation(ctx, survivors, trials)
    return population


def run_de(
    cfg: RunConfig,
    arm: str,
    initial: InitialPopulation,
    evaluator: Evaluator,
    streams: Streams,
    recorder: RunRecorder,
) -> dict:
    if cfg.population_size < 4:
        raise ValueError("DE/rand/1 needs a target plus three distinct other members")
    scale_f, crossover_rate = cfg.de_parameters_for(arm)
    ctx = ArmContext(cfg, arm, evaluator, streams, recorder)
    operations = [
        EAOperation(make_trials, ctx, scale_f, crossover_rate),
        EAOperation(evaluate_children, ctx),
        EAOperation(select_one_to_one, ctx),
    ]
    return evolve(ctx, initial, operations, cfg.de_generations)
