import time
from dataclasses import dataclass

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

from mutation_ab.config import RunConfig
from mutation_ab.initial import Evaluator, InitialPopulation, record_founders
from mutation_ab.metrics import genotype_diversity, unique_genomes
from mutation_ab.operators import KIND_DIFFERENCE, elite_index, propose_child
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


def _by_uid(individuals) -> list[Individual]:
    return sorted(individuals, key=lambda ind: ind.tags["uid"])


def _record_generation(ctx: ArmContext, survivors: list[Individual], newborn: list[Individual]) -> None:
    genomes = np.array([ind.genotype for ind in survivors], dtype=float)
    fitness = np.array([ind.fitness for ind in survivors])
    ctx.best_so_far = min(ctx.best_so_far, float(fitness.min()))
    steps = [ind.tags["proposal_rms"] for ind in newborn if ind.tags["kind"] == KIND_DIFFERENCE]
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
            "n_difference": len(steps),
            "n_gaussian": sum(ind.tags["kind"] == "gaussian" for ind in newborn),
            "evaluations": ctx.evaluations,
        }
    )
    ctx.recorder.adults(genomes)


def reproduce(population: Population, ctx: ArmContext) -> Population:
    ctx.generation += 1
    adults = _by_uid(population)
    genomes = np.array([ind.genotype for ind in adults], dtype=float)
    fitness = np.array([ind.fitness for ind in adults])
    for _ in range(ctx.cfg.children_per_generation):
        proposal = propose_child(genomes, fitness, ctx.cfg, ctx.replacement_probability, ctx.streams)
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
            "parent_distance": parent.fitness,
            "parent_xy": parent.tags["final_xy"],
        }
        ctx.next_uid += 1
        population.append(child)
    return population


def evaluate_children(population: Population, ctx: ArmContext) -> Population:
    for child in _by_uid(population.unevaluated):
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
    members = _by_uid(population)
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
    _record_generation(ctx, _by_uid([elite, *newborn]), newborn)
    return population


def run_arm(
    cfg: RunConfig,
    arm: str,
    initial: InitialPopulation,
    evaluator: Evaluator,
    streams: Streams,
    recorder: RunRecorder,
) -> dict:
    ctx = ArmContext(cfg, cfg.replacement_probability_for(arm), evaluator, streams, recorder)
    record_founders(initial, recorder)
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
    _record_generation(ctx, founders, founders)

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
