"""One episode (15 s by default): how far the robot ends from the target."""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from ariel.utils.runners import simple_runner

from mutation_ab.config import RunConfig
from mutation_ab.controller import act, n_inputs, observe, unpack

UNSTABLE_WARNINGS = (
    mj.mjtWarning.mjWARN_BADQACC,
    mj.mjtWarning.mjWARN_BADQPOS,
    mj.mjtWarning.mjWARN_BADQVEL,
    mj.mjtWarning.mjWARN_BADCTRL,
)


class UnstableSimulation(RuntimeError):
    def __init__(self, message: str, genome: np.ndarray | None = None) -> None:
        super().__init__(message)
        self.genome = genome


@dataclass(frozen=True)
class EvalResult:
    final_xy: tuple[float, float]
    distance: float
    warnings: int


def evaluate(genome: np.ndarray, model: mj.MjModel, cfg: RunConfig) -> EvalResult:
    """Run one episode and return the final distance to the target.

    Raises UnstableSimulation if the physics diverged, rather than scoring an
    endpoint that MuJoCo has silently reset.
    """
    layers = unpack(
        np.asarray(genome, dtype=float), n_inputs(model), cfg.hidden_size, model.nu
    )
    data = mj.MjData(model)

    def control(_model: mj.MjModel, d: mj.MjData) -> None:
        d.ctrl[:] = act(layers, observe(d, cfg.target_xy, cfg.phase_hz))

    mj.set_mjcb_control(control)
    try:
        simple_runner(model, data, duration=cfg.duration)
    finally:
        mj.set_mjcb_control(None)

    state = np.concatenate([data.qpos, data.qvel, data.ctrl])
    # MuJoCo silently zeroes bad controls and resets diverged states, so the
    # endpoint would look valid.
    diverged = any(
        data.warning[int(warning)].number > 0 for warning in UNSTABLE_WARNINGS
    )
    if not np.all(np.isfinite(state)) or diverged:
        raise UnstableSimulation(
            f"unstable episode at t={data.time:.3f}s",
            genome=np.asarray(genome, dtype=float),
        )

    final_xy = (float(data.qpos[0]), float(data.qpos[1]))
    distance = float(
        np.hypot(final_xy[0] - cfg.target_xy[0], final_xy[1] - cfg.target_xy[1])
    )
    warnings = int(sum(stat.number for stat in data.warning))
    return EvalResult(final_xy=final_xy, distance=distance, warnings=warnings)
