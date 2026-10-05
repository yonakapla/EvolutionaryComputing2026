import csv
import hashlib
import json
import platform
import subprocess
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

from mutation_ab.config import RunConfig

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
        total_generations: int | None = None,
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
        self._label = f"[seed {cfg.seed} {arm}]"
        self._total_generations = total_generations or cfg.generations
        self._started = time.perf_counter()

    def _report(self, message: str) -> None:
        elapsed = time.perf_counter() - self._started
        print(f"{self._label} {message} ({elapsed:.0f}s)", flush=True)

    def child(self, record: dict) -> None:
        self._children.write(json.dumps(record) + "\n")
        self._children.flush()

    def generation(self, row: dict) -> None:
        self._generations.writerow(row)
        self._generations_file.flush()
        generation = int(row["generation"])
        if generation % 10 == 0 or generation == self._total_generations:
            self._report(
                f"gen {generation}/{self._total_generations}"
                f"  best {float(row['best_so_far']):.3f} m  evals {row['evaluations']}"
            )

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
        self._report("done")

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
        self._report(f"FAILED: {error!r}; details in {self.directory / 'FAILED.json'}")

    def _close(self) -> None:
        self._children.close()
        self._generations_file.close()
