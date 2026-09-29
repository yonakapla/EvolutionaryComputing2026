import math

import numpy as np
from scipy.spatial.distance import pdist


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def genotype_diversity(genomes: np.ndarray) -> float:
    return float(pdist(genomes).mean() / math.sqrt(genomes.shape[1]))


def unique_genomes(genomes: np.ndarray) -> int:
    return len({row.tobytes() for row in np.ascontiguousarray(genomes)})
