"""Writing one run's result files, and reading a folder of runs back."""

import csv
import hashlib
import json
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

from mutation_ab.conditions import ARM_ORDER
from mutation_ab.config import ARM_DIFFERENCE, ARM_RANDOM, ARM_SIZE_MATCHED, RunConfig

GENERATION_FIELDS = (
    "generation",
    "best",
    "mean",
    "worst",
    "best_so_far",
    "unique",
    "diversity",
    "diff_proposal_rms",
    "difference_rms",
    "n_difference",
    "n_gaussian",
    "evaluations",
)


def read_generations(path: Path) -> dict[str, np.ndarray]:
    """generations.csv as one array per column."""
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def genome_sha1(genome) -> str:
    return hashlib.sha1(np.asarray(genome, dtype=np.float64).tobytes()).hexdigest()


def evaluation_record(
    uid: int, generation: int, kind: str, genome, result, wall_s: float
) -> dict:
    """children.jsonl record of a parentless evaluation (founder or random sample)."""
    return {
        "uid": uid,
        "generation": generation,
        "kind": kind,
        "parent_uid": None,
        "donor_uids": [],
        "proposal_rms": None,
        "change_rms": None,
        "parent_distance": None,
        "parent_xy": None,
        "final_xy": list(result.final_xy),
        "warnings": result.warnings,
        "genome_sha1": genome_sha1(genome),
        "distance": result.distance,
        "wall_s": wall_s,
    }


def git_commit() -> dict:
    here = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


class RunRecorder:
    def __init__(
        self,
        directory: Path,
        cfg: RunConfig,
        arm: str,
        hashes: dict[str, str],
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        meta = {
            "arm": arm,
            "config": cfg.to_dict(),
            "config_hash": cfg.config_hash(),
            "hashes": hashes,
            "git": git_commit(),
            "versions": {
                "python": platform.python_version(),
                "mujoco": mujoco.__version__,
                "ariel": version("ariel"),
            },
        }
        (self.directory / "config.json").write_text(json.dumps(meta, indent=2))
        self._children = (self.directory / "children.jsonl").open("w")
        self._generations_file = (self.directory / "generations.csv").open(
            "w", newline=""
        )
        self._generations = csv.DictWriter(
            self._generations_file, fieldnames=GENERATION_FIELDS
        )
        self._generations.writeheader()
        self._adults: list[np.ndarray] = []
        self._steps: list[tuple[int, str, np.ndarray]] = []
        self._label = f"{arm} seed {cfg.seed}"
        self._started = time.perf_counter()

    def child(self, record: dict) -> None:
        self._children.write(json.dumps(record) + "\n")
        self._children.flush()

    def generation(self, row: dict) -> None:
        self._generations.writerow(row)
        self._generations_file.flush()

    def adults(self, genomes: np.ndarray) -> None:
        self._adults.append(np.array(genomes, dtype=np.float64))

    def step(self, generation: int, kind: str, delta: np.ndarray) -> None:
        self._steps.append((generation, kind, np.asarray(delta, dtype=np.float32)))

    def complete(self, summary: dict) -> None:
        self._close()
        if self._adults:
            np.savez_compressed(
                self.directory / "adults.npz", adults=np.stack(self._adults)
            )
        if self._steps:
            generation, kind, delta = zip(*self._steps, strict=True)
            np.savez_compressed(
                self.directory / "steps.npz",
                generation=np.array(generation),
                kind=np.array(kind),
                delta=np.stack(delta),
            )
        (self.directory / "COMPLETE").write_text(json.dumps(summary))
        seconds = time.perf_counter() - self._started
        print(
            f"{self._label}: best {summary['best_so_far']:.3f} m ({seconds:.0f}s)",
            flush=True,
        )

    def fail(self, error: BaseException) -> None:
        self._close()
        report = {
            "error": repr(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        genome = getattr(error, "genome", None)
        if genome is not None:
            genome_list = np.asarray(genome, dtype=np.float64).tolist()
            report["genome"] = genome_list
            report["genome_sha1"] = genome_sha1(genome_list)
        (self.directory / "FAILED.json").write_text(json.dumps(report, indent=2))
        print(
            f"{self._label}: FAILED, see {self.directory / 'FAILED.json'}", flush=True
        )

    def _close(self) -> None:
        self._children.close()
        self._generations_file.close()


@dataclass(frozen=True)
class Run:
    seed: int
    arm: str
    directory: Path
    generations: dict[str, np.ndarray]
    children: list[dict]
    config: dict


def load(root: Path) -> dict[str, dict[int, Run]]:
    """arm -> seed -> Run for every completed run under `root`.

    Runs must share one setup and every arm must cover the same seeds, or the
    paired tests would compare different things.
    """
    runs: dict[str, dict[int, Run]] = {}
    setups: dict[str, list[str]] = {}
    for seed_dir in sorted(
        Path(root).glob("seed_*"), key=lambda p: int(p.name.split("_")[1])
    ):
        seed = int(seed_dir.name.split("_")[1])
        for arm_dir in sorted(seed_dir.iterdir()):
            if not (arm_dir / "COMPLETE").exists():
                continue
            meta = json.loads((arm_dir / "config.json").read_text())
            setup = json.dumps(
                {
                    "config": {k: v for k, v in meta["config"].items() if k != "seed"},
                    "hashes": meta["hashes"],
                },
                sort_keys=True,
            )
            setups.setdefault(setup, []).append(str(arm_dir))
            children = [
                json.loads(line)
                for line in (arm_dir / "children.jsonl").read_text().splitlines()
            ]
            runs.setdefault(arm_dir.name, {})[seed] = Run(
                seed,
                arm_dir.name,
                arm_dir,
                read_generations(arm_dir / "generations.csv"),
                children,
                meta["config"],
            )
    if len(setups) > 1:
        examples = [dirs[0] for dirs in setups.values()]
        raise ValueError(
            f"runs under {root} use {len(setups)} different setups, e.g. {examples}"
        )
    seed_sets = {arm: tuple(sorted(by_seed)) for arm, by_seed in runs.items()}
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"arms cover different seeds: {seed_sets}")
    return {arm: runs[arm] for arm in ARM_ORDER if arm in runs}


SIGMA_FREE_ARMS = (ARM_DIFFERENCE, ARM_SIZE_MATCHED, ARM_RANDOM)


def borrow_sigma_free_arms(runs, reference_root: Path) -> dict[str, dict[int, Run]]:
    """Add A, C and random search from another run of the same seeds.

    None of them uses the Gaussian step size, so a run that only changes
    gaussian_sd can reuse them for the full 2x2 contrasts.
    """
    reference = load(reference_root)
    seeds = sorted(next(iter(runs.values())))
    own = next(iter(next(iter(runs.values())).values())).config
    for arm in SIGMA_FREE_ARMS:
        borrowed = {seed: reference[arm][seed] for seed in seeds}
        for run in borrowed.values():
            differing = {
                k
                for k in own
                if k not in ("seed", "gaussian_sd") and own[k] != run.config.get(k)
            }
            if differing:
                raise ValueError(f"{run.directory} differs in {sorted(differing)}")
        runs[arm] = borrowed
    return {arm: runs[arm] for arm in ARM_ORDER if arm in runs}


def finals(runs, arm: str) -> np.ndarray:
    """Each seed's best distance at the end of the budget, in seed order."""
    return np.array(
        [runs[arm][seed].generations["best_so_far"][-1] for seed in sorted(runs[arm])]
    )


def mean_curve(runs, arm: str, key: str = "best_so_far") -> np.ndarray:
    return np.mean([run.generations[key] for run in runs[arm].values()], axis=0)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
