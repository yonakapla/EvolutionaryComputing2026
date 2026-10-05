import threading
import time
from datetime import datetime, timedelta
from pathlib import Path


def count_progress(out: Path, seeds, arms) -> tuple[int, int]:
    evaluations = finished = 0
    for seed in seeds:
        for arm in arms:
            run = Path(out) / f"seed_{seed}" / arm
            children = run / "children.jsonl"
            if children.exists():
                with children.open("rb") as handle:
                    evaluations += sum(1 for _ in handle)
            if (run / "COMPLETE").exists() or (run / "FAILED.json").exists():
                finished += 1
    return evaluations, finished


def progress_line(
    evals: int, total: int, finished: int, runs: int, elapsed_s: float, now: datetime
) -> str:
    percent = 100 * evals / total if total else 0.0
    if evals and elapsed_s > 0:
        seconds_per_eval = elapsed_s / evals
        rate = f"{seconds_per_eval * 10_000:.0f} s/10k evals"
        eta = (now + timedelta(seconds=seconds_per_eval * (total - evals))).strftime(
            "%H:%M"
        )
    else:
        rate, eta = "-- s/10k evals", "--:--"
    return (
        f"{now:%H:%M:%S}  overall {percent:.0f}% ({evals:,}/{total:,} evals) · {rate} · "
        f"runs done {finished}/{runs} · ETA {eta}"
    )


class Heartbeat:
    """Prints overall progress, read from the run directories, while seeds run in worker processes."""

    def __init__(
        self, out: Path, seeds, arms, budgets: dict[str, int], interval_s: float
    ) -> None:
        self.out = Path(out)
        self.seeds = list(seeds)
        self.arms = tuple(arms)
        self.total = len(self.seeds) * sum(budgets[arm] for arm in self.arms)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._started = time.perf_counter()

    def __enter__(self) -> "Heartbeat":
        if self.interval_s > 0:
            self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self.interval_s > 0:
            self._stop.set()
            self._thread.join()
            self.report()

    def report(self) -> None:
        evaluations, finished = count_progress(self.out, self.seeds, self.arms)
        line = progress_line(
            evaluations,
            self.total,
            finished,
            len(self.seeds) * len(self.arms),
            time.perf_counter() - self._started,
            datetime.now(),
        )
        print(line, flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.report()
