import csv
import hashlib
import json
import platform
import subprocess
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
    "n_difference",
    "n_gaussian",
    "evaluations",
)


def genome_sha1(genome) -> str:
    return hashlib.sha1(np.asarray(genome, dtype=np.float64).tobytes()).hexdigest()


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
    def __init__(self, directory: Path, cfg: RunConfig, arm: str, hashes: dict[str, str]) -> None:
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
        self._generations_file = (self.directory / "generations.csv").open("w", newline="")
        self._generations = csv.DictWriter(self._generations_file, fieldnames=GENERATION_FIELDS)
        self._generations.writeheader()
        self._adults: list[np.ndarray] = []

    def child(self, record: dict) -> None:
        self._children.write(json.dumps(record) + "\n")
        self._children.flush()

    def generation(self, row: dict) -> None:
        self._generations.writerow(row)
        self._generations_file.flush()

    def adults(self, genomes: np.ndarray) -> None:
        self._adults.append(np.array(genomes, dtype=np.float64))

    def complete(self, summary: dict) -> None:
        self._close()
        if self._adults:
            np.savez_compressed(self.directory / "adults.npz", adults=np.stack(self._adults))
        (self.directory / "COMPLETE").write_text(json.dumps(summary))

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

    def _close(self) -> None:
        self._children.close()
        self._generations_file.close()
