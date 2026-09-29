import time
from dataclasses import dataclass

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

from mutation_ab.config import STEP_DIFFERENCE, RunConfig
from mutation_ab.initial import Evaluator, InitialPopulation, record_founders
from mutation_ab.metrics import genotype_diversity, unique_genomes
from mutation_ab.operators import KIND_GAUSSIAN, Proposal, elite_index, propose_child
from mutation_ab.records import RunRecorder, genome_sha1
from mutation_ab.streams import Streams


@dataclass
class ArmContext:
    cfg: RunConfig
    replacement_probability: float
    evaluator: Evaluator
    streams: Streams
    recorder: RunRecorder
    generation: int = 0
    next_uid: int = 0
    evaluations: int = 0
    best_so_far: float = float("inf")
    step: str = STEP_DIFFERENCE


def by_uid(individuals) -> list[Individual]:
    return sorted(individuals, key=lambda ind: ind.tags["uid"])


def record_generation(ctx: ArmContext, survivors: list[Individual], newborn: list[Individual]) -> None:
    genomes = np.array([ind.genotype for ind in survivors], dtype=float)
    fitness = np.array([ind.fitness for ind in survivors])
    ctx.best_so_far = min(ctx.best_so_far, float(fitness.min()))
    bred = [ind for ind in newborn if ind.tags["kind"] != "init"]
    # Every non-Gaussian step counts as "difference" here; difference_rms is F(b - c) in every arm.
    steps = [ind.tags["proposal_rms"] for ind in bred if ind.tags["kind"] != KIND_GAUSSIAN]
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
    """Keep the steps of every `step_log_every`-th generation for the step-direction analysis."""
    if ctx.generation % ctx.cfg.step_log_every == 0:
        ctx.recorder.step(ctx.generation, proposal.kind, proposal.child - genomes[proposal.parent])


def reproduce(population: Population, ctx: ArmContext) -> Population:
    ctx.generation += 1
    adults = by_uid(population)
    genomes = np.array([ind.genotype for ind in adults], dtype=float)
    fitness = np.array([ind.fitness for ind in adults])
    for _ in range(ctx.cfg.children_per_generation):
        proposal = propose_child(genomes, fitness, ctx.cfg, ctx.replacement_probability, ctx.streams, ctx.step)
        population.append(make_child(ctx, proposal, adults))
        log_step(ctx, proposal, genomes)
    return population


def make_child(ctx: ArmContext, proposal: Proposal, adults: list[Individual]) -> Individual:
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


def survive(population: Population, ctx: ArmContext) -> Population:
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


def make_founders(ctx: ArmContext, initial: InitialPopulation) -> list[Individual]:
    """Turn the shared, already evaluated initial population into generation-0 individuals."""
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


def run_arm(
    cfg: RunConfig,
    arm: str,
    initial: InitialPopulation,
    evaluator: Evaluator,
    streams: Streams,
    recorder: RunRecorder,
) -> dict:
    ctx = ArmContext(
        cfg, cfg.replacement_probability_for(arm), evaluator, streams, recorder, step=cfg.step_for(arm)
    )
    founders = make_founders(ctx, initial)

    ea = EA(
        Population(founders),
        [
            EAOperation(reproduce, ctx),
            EAOperation(evaluate_children, ctx),
            EAOperation(survive, ctx),
        ],
        num_steps=cfg.generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=recorder.directory / "ariel.db",
        db_handling="halt",
    )
    ea.run()
    return {"evaluations": ctx.evaluations, "best_so_far": ctx.best_so_far}
