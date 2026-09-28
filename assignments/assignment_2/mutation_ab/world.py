import hashlib

import mujoco as mj
import numpy as np
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import OlympicArena
from ariel.utils.noise_gen import PerlinNoise

from mutation_ab.config import RunConfig


class SeededOlympicArena(OlympicArena):
    # OlympicArena seeds nothing, so its rugged section differs on every build.
    def __init__(self, terrain_seed: int) -> None:
        self.terrain_seed = terrain_seed
        super().__init__(load_precompiled=False)

    def _generate_heightmap(self) -> np.ndarray:
        size = self.rugged_resolution
        edge_width = getattr(self, "edge_width", 0.1)
        noise = PerlinNoise(seed=self.terrain_seed).as_grid(
            size, size, scale=self.rugged_hillyness, normalize=False
        )
        u = np.linspace(0.0, 1.0, size)
        v = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, v, indexing="xy")
        edge_distance = np.minimum.reduce([grid_u, 1.0 - grid_u, grid_v, 1.0 - grid_v])
        t = np.clip(edge_distance / edge_width, 0.1, 1.0)
        return noise * (t * t * (3.0 - 2.0 * t))


def _sha1(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def build_model(cfg: RunConfig) -> tuple[mj.MjModel, dict[str, str]]:
    mj.set_mjcb_control(None)
    world = SeededOlympicArena(cfg.terrain_seed)
    world.spawn(gecko().spec, position=list(cfg.spawn), correct_collision_with_floor=True)
    model = world.spec.compile()
    hashes = {
        "terrain_sha1": _sha1(world.heightmap.tobytes()),
        "model_sha1": _sha1(world.spec.to_xml().encode()),
    }
    return model, hashes


def terrain_fingerprint(terrain_seed: int) -> tuple[str, str]:
    _, hashes = build_model(RunConfig(seed=0, terrain_seed=terrain_seed))
    return hashes["terrain_sha1"], hashes["model_sha1"]
