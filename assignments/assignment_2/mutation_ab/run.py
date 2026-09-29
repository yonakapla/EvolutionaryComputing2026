import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
from pathlib import Path

from ariel.ec import set_seed

from mutation_ab.config import ALL_ARMS, ARM_RANDOM, ARMS, DE_ARMS, WORLDS, RunConfig
from mutation_ab.controller import genome_length, n_inputs
from mutation_ab.de_arm import run_de
from mutation_ab.ea_arm import run_arm
from mutation_ab.evaluate import UnstableSimulation, evaluate
from mutation_ab.initial import make_initial
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


def run_seed(seed: int, out_root: Path, overrides: dict, arms: tuple[str, ...] = ARMS) -> dict:
    started = time.perf_counter()
    cfg = RunConfig(seed=seed, **overrides)
    set_seed(seed)
    model, hashes = build_model(cfg)
    evaluator = partial(evaluate, model=model, cfg=cfg)
    length = genome_length(n_inputs(model), cfg.hidden_size, model.nu)
    seed_dir = out_root / f"seed_{seed}"
    status: dict[str, str] = {}
    print(f"[seed {seed}] started: evaluating the shared initial population", flush=True)
    try:
        initial = make_initial(cfg, make_streams(seed), evaluator, length)
    except UnstableSimulation as error:
        for arm in arms:
            RunRecorder(seed_dir / arm, cfg, arm, hashes).fail(error)
            status[arm] = "failed"
        return {"seed": seed, "status": status, "wall_s": time.perf_counter() - started}

    for arm in arms:
        recorder = RunRecorder(seed_dir / arm, cfg, arm, hashes, cfg.generations_for(arm))
        try:
            if arm == ARM_RANDOM:
                summary = run_random(cfg, initial, evaluator, make_streams(seed), recorder)
            elif arm in DE_ARMS:
                summary = run_de(cfg, arm, initial, evaluator, make_streams(seed), recorder)
            else:
                summary = run_arm(cfg, arm, initial, evaluator, make_streams(seed), recorder)
        except UnstableSimulation as error:
            recorder.fail(error)
            status[arm] = "failed"
            continue
        recorder.complete(summary)
        status[arm] = "complete"
    return {"seed": seed, "status": status, "wall_s": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Differential vs Gaussian-replacement mutation A/B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="e.g. 700-705 or 800,801")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated; any of {', '.join(ALL_ARMS)}")
    parser.add_argument("--body", default="gecko", help="a John Set body, e.g. gecko or spider_8")
    parser.add_argument("--world", default="olympic", choices=WORLDS)
    parser.add_argument(
        "--gaussian-sd",
        type=float,
        default=RunConfig(seed=0).gaussian_sd,
        help="Gaussian step SD, also the normalised arm's fixed step size (F stays tied to 0.15)",
    )
    parser.add_argument("--heartbeat", type=float, default=60.0, help="seconds between overall progress lines; 0 disables")
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
        "world": args.world,
        "gaussian_sd": args.gaussian_sd,
    }
    reference = RunConfig(seed=0, **overrides)
    budgets = {arm: reference.budget_for(arm) for arm in arms}
    de_note = (f" ({reference.budget_for(DE_ARMS[0])} for canonical DE)"
               if any(arm in DE_ARMS for arm in arms) else "")
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"Running seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds) x arms {', '.join(arms)}; "
        f"{args.generations} generations, {reference.budget} evaluations per arm{de_note}, "
        f"{args.workers} worker(s); "
        f"output in {args.out}. Overall progress every {args.heartbeat:g}s; "
        f"per-run progress every 10 generations.",
        flush=True,
    )

    with Heartbeat(args.out, seeds, arms, budgets, args.heartbeat):
        if args.workers == 1:
            results = [run_seed(seed, args.out, overrides, arms) for seed in seeds]
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                results = list(pool.map(run_seed, seeds, repeat(args.out), repeat(overrides), repeat(arms)))

    print(json.dumps(results, indent=2))
    complete = all(state == "complete" for r in results for state in r["status"].values())
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
