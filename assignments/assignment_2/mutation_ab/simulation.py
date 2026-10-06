"""The robot in its world, the neural-network controller a genome encodes, and
one episode (15 s by default) that scores it by its final distance to the target.
"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner

from mutation_ab.config import RunConfig


def build_model(cfg: RunConfig) -> mj.MjModel:
    """The compiled MuJoCo model of the flat world with the robot body in it."""
    mj.set_mjcb_control(None)
    body = getattr(john_set, cfg.body, None)
    if not callable(body) or getattr(body, "__module__", None) != john_set.__name__:
        raise ValueError(f"unknown John Set body {cfg.body!r}")
    world = SimpleFlatWorld()
    world.spawn(
        body().spec, position=list(cfg.spawn), correct_collision_with_floor=True
    )
    return world.spec.compile()


@dataclass(frozen=True)
class Layers:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def n_inputs(model: mj.MjModel) -> int:
    """qpos + qvel + target vector (2) + sin/cos clock (2); 33 for the spider."""
    return model.nq + model.nv + 4


def genome_length(n_inputs: int, hidden: int, n_outputs: int) -> int:
    return n_inputs * hidden + hidden + hidden * n_outputs + n_outputs


def unpack(genome: np.ndarray, n_inputs: int, hidden: int, n_outputs: int) -> Layers:
    expected = genome_length(n_inputs, hidden, n_outputs)
    if genome.shape != (expected,):
        raise ValueError(f"genome must have shape ({expected},), got {genome.shape}")
    sizes = [n_inputs * hidden, hidden, hidden * n_outputs, n_outputs]
    w1, b1, w2, b2 = np.split(genome, np.cumsum(sizes)[:-1])
    return Layers(w1.reshape(n_inputs, hidden), b1, w2.reshape(hidden, n_outputs), b2)


def observe(
    data: mj.MjData, target_xy: tuple[float, float], phase_hz: float
) -> np.ndarray:
    phase = 2.0 * np.pi * phase_hz * data.time
    to_target = np.asarray(target_xy) - data.qpos[0:2]
    return np.concatenate(
        [data.qpos, data.qvel, to_target, [np.sin(phase), np.cos(phase)]]
    )


def act(layers: Layers, observation: np.ndarray) -> np.ndarray:
    hidden = np.tanh(observation @ layers.w1 + layers.b1)
    return np.tanh(hidden @ layers.w2 + layers.b2) * (np.pi / 2)


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
