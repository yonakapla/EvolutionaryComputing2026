from dataclasses import dataclass

import numpy as np

STREAM_NAMES = ("init", "selection", "mask", "gaussian", "replacement", "random_search")


@dataclass(frozen=True)
class Streams:
    init: np.random.Generator
    selection: np.random.Generator
    mask: np.random.Generator
    gaussian: np.random.Generator
    replacement: np.random.Generator
    random_search: np.random.Generator


def make_streams(seed: int) -> Streams:
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return Streams(*(np.random.default_rng(child) for child in children))
