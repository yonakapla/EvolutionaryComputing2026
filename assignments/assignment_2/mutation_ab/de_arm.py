"""Reference arm: canonical DE/rand/1/bin (Storn & Price 1997) on ariel.ec.

Unlike the generational arms, every adult is a target once per generation and its
trial replaces it only if the trial is at least as close to the target (one-to-one
survivor selection). The population size is constant; the shared initial population
and the evaluation budget match the other arms.
"""

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

from mutation_ab.config import RunConfig
from mutation_ab.ea_arm import ArmContext, _by_uid, _record_generation, evaluate_children, log_step, make_founders
from mutation_ab.initial import Evaluator, InitialPopulation
from mutation_ab.operators import de_trial
from mutation_ab.records import RunRecorder
from mutation_ab.streams import Streams


def make_trials(population: Population, ctx: ArmContext, scale_f: float, crossover_rate: float) -> Population:
    ctx.generation += 1
    targets = _by_uid(population)
    genomes = np.array([ind.genotype for ind in targets], dtype=float)
    for index, target in enumerate(targets):
        proposal = de_trial(genomes, index, scale_f, crossover_rate, ctx.streams)
        trial = Individual()
        trial.genotype = proposal.child.tolist()
        trial.tags = {
            "uid": ctx.next_uid,
            "generation": ctx.generation,
            "kind": proposal.kind,
            "parent_uid": target.tags["uid"],
            "donor_uids": [targets[d].tags["uid"] for d in proposal.donors],
            "proposal_rms": proposal.proposal_rms,
            "change_rms": proposal.change_rms,
            "difference_rms": proposal.difference_rms,
            "parent_distance": target.fitness,
            "parent_xy": target.tags["final_xy"],
        }
        ctx.next_uid += 1
        population.append(trial)
        log_step(ctx, proposal, genomes)
    return population


def select_one_to_one(population: Population, ctx: ArmContext) -> Population:
    members = _by_uid(population)
    adults = {ind.tags["uid"]: ind for ind in members if ind.tags["generation"] < ctx.generation}
    trials = [ind for ind in members if ind.tags["generation"] == ctx.generation]
    for trial in trials:
        target = adults[trial.tags["parent_uid"]]
        if trial.fitness <= target.fitness:
            target.alive = False
        else:
            trial.alive = False
    survivors = [ind for ind in members if ind.alive]
    _record_generation(ctx, survivors, trials)
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
    ctx = ArmContext(cfg, 0.0, evaluator, streams, recorder)
    founders = make_founders(ctx, initial)
    ea = EA(
        Population(founders),
        [
            EAOperation(make_trials, ctx, scale_f, crossover_rate),
            EAOperation(evaluate_children, ctx),
            EAOperation(select_one_to_one, ctx),
        ],
        num_steps=cfg.de_generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=recorder.directory / "ariel.db",
        db_handling="halt",
    )
    ea.run()
    return {"evaluations": ctx.evaluations, "best_so_far": ctx.best_so_far}
