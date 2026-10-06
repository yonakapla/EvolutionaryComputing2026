"""The algorithms, built on ariel's EA: the generational EA shared by the four
2x2 arms and the mixture, and canonical DE/rand/1/bin (Storn & Price 1997).

Every arm of a seed starts from the same initial population, drawn and
evaluated once.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

from mutation_ab.config import RunConfig, Streams
from mutation_ab.metrics import genotype_diversity, unique_genomes
from mutation_ab.operators import (
    KIND_GAUSSIAN,
    Proposal,
    de_trial,
    elite_index,
    propose_child,
)
from mutation_ab.records import RunRecorder, evaluation_record, genome_sha1
from mutation_ab.simulation import EvalResult

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
    return InitialPopulation(
        genomes=genomes, results=tuple(results), wall_s=tuple(wall)
    )


def record_founders(initial: InitialPopulation, recorder: RunRecorder) -> None:
    for uid, (genome, result, wall) in enumerate(
        zip(initial.genomes, initial.results, initial.wall_s, strict=True)
    ):
        recorder.child(evaluation_record(uid, 0, "init", genome, result, wall))


@dataclass
class ArmContext:
    cfg: RunConfig
    arm: str
    evaluator: Evaluator
    streams: Streams
    recorder: RunRecorder
    generation: int = 0
    next_uid: int = 0
    evaluations: int = 0
    best_so_far: float = float("inf")


def by_uid(individuals) -> list[Individual]:
    return sorted(individuals, key=lambda ind: ind.tags["uid"])


def record_generation(
    ctx: ArmContext, survivors: list[Individual], newborn: list[Individual]
) -> None:
    genomes = np.array([ind.genotype for ind in survivors], dtype=float)
    fitness = np.array([ind.fitness for ind in survivors])
    ctx.best_so_far = min(ctx.best_so_far, float(fitness.min()))
    bred = [ind for ind in newborn if ind.tags["kind"] != "init"]
    # Every non-Gaussian step counts as "difference" here;
    # difference_rms is F(b - c) in every arm.
    steps = [
        ind.tags["proposal_rms"] for ind in bred if ind.tags["kind"] != KIND_GAUSSIAN
    ]
    spreads = [ind.tags["difference_rms"] for ind in bred]
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
            "difference_rms": float(np.mean(spreads)) if spreads else float("nan"),
            "n_difference": len(steps),
            "n_gaussian": sum(ind.tags["kind"] == KIND_GAUSSIAN for ind in newborn),
            "evaluations": ctx.evaluations,
        }
    )
    ctx.recorder.adults(genomes)


def log_step(ctx: ArmContext, proposal: Proposal, genomes: np.ndarray) -> None:
    """Keep every `step_log_every`-th generation's steps for the direction analysis."""
    if ctx.generation % ctx.cfg.step_log_every == 0:
        ctx.recorder.step(
            ctx.generation, proposal.kind, proposal.child - genomes[proposal.parent]
        )


def make_child(
    ctx: ArmContext, proposal: Proposal, adults: list[Individual]
) -> Individual:
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
        "difference_rms": proposal.difference_rms,
        "parent_distance": parent.fitness,
        "parent_xy": parent.tags["final_xy"],
    }
    ctx.next_uid += 1
    return child


def evaluate_children(population: Population, ctx: ArmContext) -> Population:
    for child in by_uid(population.unevaluated):
        started = time.perf_counter()
        result = ctx.evaluator(np.asarray(child.genotype, dtype=float))
        child.fitness = result.distance
        # ariel's tags setter merges, so the tags set in make_child are kept.
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


def make_founders(ctx: ArmContext, initial: InitialPopulation) -> list[Individual]:
    """Turn the shared, evaluated initial population into generation-0 individuals."""
    record_founders(initial, ctx.recorder)
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
    record_generation(ctx, founders, founders)
    return founders


def evolve(
    ctx: ArmContext,
    initial: InitialPopulation,
    operations: list[EAOperation],
    generations: int,
) -> dict:
    """Run an ariel EA from the shared founders.

    The EA and DE arms differ only in the operations they pass in.
    """
    ea = EA(
        Population(make_founders(ctx, initial)),
        operations,
        num_steps=generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=ctx.recorder.directory / "ariel.db",
        db_handling="halt",
    )
    ea.run()
    return {"evaluations": ctx.evaluations, "best_so_far": ctx.best_so_far}


def reproduce(population: Population, ctx: ArmContext) -> Population:
    ctx.generation += 1
    adults = by_uid(population)
    genomes = np.array([ind.genotype for ind in adults], dtype=float)
    fitness = np.array([ind.fitness for ind in adults])
    for _ in range(ctx.cfg.children_per_generation):
        proposal = propose_child(genomes, fitness, ctx.cfg, ctx.arm, ctx.streams)
        population.append(make_child(ctx, proposal, adults))
        log_step(ctx, proposal, genomes)
    return population


def survive(population: Population, ctx: ArmContext) -> Population:
    """Generational replacement with one elite: the children replace every
    adult except the best."""
    members = by_uid(population)
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
    record_generation(ctx, by_uid([elite, *newborn]), newborn)
    return population


def run_ea(
    cfg: RunConfig,
    arm: str,
    initial: InitialPopulation,
    evaluator: Evaluator,
    streams: Streams,
    recorder: RunRecorder,
) -> dict:
    """The generational EA: tournament parent, the arm's step, one elite."""
    ctx = ArmContext(cfg, arm, evaluator, streams, recorder)
    operations = [
        EAOperation(reproduce, ctx),
        EAOperation(evaluate_children, ctx),
        EAOperation(survive, ctx),
    ]
    return evolve(ctx, initial, operations, cfg.generations)


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
    """Canonical DE: one trial per adult, which replaces it only if at least as
    good."""
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
