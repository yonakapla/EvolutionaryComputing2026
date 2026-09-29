import hashlib

import mujoco as mj
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import SimpleFlatWorld

from mutation_ab.config import RunConfig


def build_model(cfg: RunConfig) -> tuple[mj.MjModel, dict[str, str]]:
    mj.set_mjcb_control(None)
    body = getattr(john_set, cfg.body, None)
    if not callable(body) or getattr(body, "__module__", None) != john_set.__name__:
        raise ValueError(f"unknown John Set body {cfg.body!r}")
    world = SimpleFlatWorld()
    world.spawn(body().spec, position=list(cfg.spawn), correct_collision_with_floor=True)
    model = world.spec.compile()
    return model, {"model_sha1": hashlib.sha1(world.spec.to_xml().encode()).hexdigest()}
