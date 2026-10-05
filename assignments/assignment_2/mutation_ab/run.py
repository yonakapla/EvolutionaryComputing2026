import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
from pathlib import Path

from ariel.ec import set_seed

from mutation_ab.config import ALL_ARMS, ARM_RANDOM, DE_ARMS, RunConfig
from mutation_ab.controller import genome_length, n_inputs
from mutation_ab.de_arm import run_de
from mutation_ab.ea_arm import run_ea
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.initial import Evaluator, InitialPopulation, make_initial
from mutation_ab.progress import Heartbeat
from mutation_ab.random_search import run_random
from mutation_ab.records import RunRecorder
from mutation_ab.streams import make_streams
from mutation_ab.world import build_model


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part[1:]:
            low, high = (int(bound) for bound in part.split("-", 1))
            if low > high:
                raise ValueError(f"descending seed range {part!r}")
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError(f"invalid seeds {text!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {text!r}")
    return seeds


def parse_arms(text: str) -> tuple[str, ...]:
    arms = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [arm for arm in arms if arm not in ALL_ARMS]
    if not arms or unknown or len(set(arms)) != len(arms):
        raise ValueError(f"invalid arms {text!r}; choose from {', '.join(ALL_ARMS)}")
    return arms


def run_one(
    cfg: RunConfig,
    arm: str,
    initial: InitialPopulation,
    evaluator: Evaluator,
    recorder: RunRecorder,
) -> dict:
    streams = make_streams(cfg.seed)
    if arm == ARM_RANDOM:
        return run_random(cfg, initial, evaluator, streams, recorder)
    if arm in DE_ARMS:
        return run_de(cfg, arm, initial, evaluator, streams, recorder)
    return run_ea(cfg, arm, initial, evaluator, streams, recorder)


def run_seed(
    seed: int, out_root: Path, overrides: dict, arms: tuple[str, ...] = ALL_ARMS
) -> dict[str, str]:
    """Run every arm of one seed from the same initial population.

    Returns each arm's status, "complete" or "failed". A diverged simulation
    fails only its own arm, unless it happens in the shared initial population.
    """
    cfg = RunConfig(seed=seed, **overrides)
    set_seed(seed)
    model, hashes = build_model(cfg)
    evaluator = partial(evaluate, model=model, cfg=cfg)
    length = genome_length(n_inputs(model), cfg.hidden_size, model.nu)
    seed_dir = out_root / f"seed_{seed}"
    status: dict[str, str] = {}
    try:
        initial = make_initial(cfg, make_streams(seed), evaluator, length)
    except UnstableSimulation as error:
        for arm in arms:
            RunRecorder(seed_dir / arm, cfg, arm, hashes).fail(error)
            status[arm] = "failed"
        return status

    for arm in arms:
        recorder = RunRecorder(seed_dir / arm, cfg, arm, hashes)
        try:
            summary = run_one(cfg, arm, initial, evaluator, recorder)
        except UnstableSimulation as error:
            recorder.fail(error)
            status[arm] = "failed"
            continue
        recorder.complete(summary)
        status[arm] = "complete"
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step size x step direction experiment (see PROTOCOL.md)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="e.g. 700-705 or 800,801")
    parser.add_argument(
        "--generations", type=int, default=RunConfig(seed=0).generations
    )
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--arms",
        default=",".join(ALL_ARMS),
        help=f"comma-separated; any of {', '.join(ALL_ARMS)}",
    )
    parser.add_argument(
        "--body", default=RunConfig(seed=0).body, help="a John Set body"
    )
    parser.add_argument(
        "--gaussian-sd",
        type=float,
        default=RunConfig(seed=0).gaussian_sd,
        help="Gaussian step SD, also the normalised arm's fixed step size "
        "(F stays tied to 0.15)",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=60.0,
        help="seconds between progress lines; 0 disables",
    )
    args = parser.parse_args(argv)

    try:
        seeds = parse_seeds(args.seeds)
        arms = parse_arms(args.arms)
    except ValueError as error:
        parser.error(str(error))
    existing = [seed for seed in seeds if (args.out / f"seed_{seed}").exists()]
    if existing:
        parser.error(f"output already exists for seeds {existing}; choose a new --out")
    overrides = {
        "generations": args.generations,
        "population_size": args.population,
        "duration": args.duration,
        "body": args.body,
        "gaussian_sd": args.gaussian_sd,
    }
    reference = RunConfig(seed=0, **overrides)
    budgets = {arm: reference.budget_for(arm) for arm in arms}
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(seeds)} seeds x {len(arms)} arms, {reference.budget} evaluations "
        f"per run, {args.workers} worker(s) -> {args.out}",
        flush=True,
    )

    with Heartbeat(args.out, seeds, arms, budgets, args.heartbeat):
        if args.workers == 1:
            results = [run_seed(seed, args.out, overrides, arms) for seed in seeds]
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context
            ) as pool:
                results = list(
                    pool.map(
                        run_seed,
                        seeds,
                        repeat(args.out),
                        repeat(overrides),
                        repeat(arms),
                    )
                )

    failed = [
        f"{arm} seed {seed}"
        for seed, status in zip(seeds, results, strict=True)
        for arm, state in status.items()
        if state != "complete"
    ]
    print(f"{len(seeds) * len(arms) - len(failed)} runs complete, {len(failed)} failed")
    for run in failed:
        print(f"  failed: {run}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
