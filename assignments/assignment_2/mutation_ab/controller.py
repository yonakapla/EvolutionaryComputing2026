from dataclasses import dataclass

import mujoco as mj
import numpy as np

@dataclass(frozen=True)
class Layers:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def n_inputs(model: mj.MjModel) -> int:
    """qpos + qvel + target vector (2) + sin/cos clock (2); 29 for the gecko."""
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


def observe(data: mj.MjData, target_xy: tuple[float, float], phase_hz: float) -> np.ndarray:
    phase = 2.0 * np.pi * phase_hz * data.time
    to_target = np.asarray(target_xy) - data.qpos[0:2]
    return np.concatenate([data.qpos, data.qvel, to_target, [np.sin(phase), np.cos(phase)]])


def act(layers: Layers, observation: np.ndarray) -> np.ndarray:
    hidden = np.tanh(observation @ layers.w1 + layers.b1)
    return np.tanh(hidden @ layers.w2 + layers.b2) * (np.pi / 2)
