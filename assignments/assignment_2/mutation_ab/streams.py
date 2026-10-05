"""Separate random-number streams for each kind of random choice."""

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
    """Independent generators derived from one seed.

    Keeping each kind of draw on its own stream means that one arm drawing, say,
    an extra Gaussian number cannot shift the parents another arm selects.
    """
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return Streams(*(np.random.default_rng(child) for child in children))
