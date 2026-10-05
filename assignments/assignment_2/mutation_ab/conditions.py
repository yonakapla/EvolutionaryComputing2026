"""The order, names and drawing style of each arm in the tables and figures.

Colour shows the step direction (population blue, random orange); line style
and marker fill show the step size (dashed and hollow for shrinking, solid and
filled for fixed).
"""

from mutation_ab.config import (
    ARM_DE,
    ARM_DE_MATCHED,
    ARM_DIFFERENCE,
    ARM_GAUSSIAN,
    ARM_MIXTURE,
    ARM_NORMALISED,
    ARM_RANDOM,
    ARM_SIZE_MATCHED,
)

REFERENCES = (ARM_MIXTURE, ARM_DE, ARM_DE_MATCHED)
ARM_ORDER = (
    ARM_DIFFERENCE,
    ARM_SIZE_MATCHED,
    ARM_NORMALISED,
    ARM_GAUSSIAN,
    *REFERENCES,
    ARM_RANDOM,
)
SHRINKING_SIZE = (ARM_DIFFERENCE, ARM_SIZE_MATCHED)
FIXED_SIZE = (ARM_NORMALISED, ARM_GAUSSIAN)

NAMES = {
    ARM_DIFFERENCE: "A difference",
    ARM_NORMALISED: "B normalised",
    ARM_SIZE_MATCHED: "C size-matched",
    ARM_GAUSSIAN: "D Gaussian",
    ARM_MIXTURE: "Mixture",
    ARM_DE: "DE (0.5, 0.9)",
    ARM_DE_MATCHED: "DE matched",
    ARM_RANDOM: "Random search",
}

POPULATION_DIRECTION, RANDOM_DIRECTION = "#1f5fbf", "#e07000"
MUTED, LIGHT = "#555555", "#d0d0d0"
COLORS = {
    ARM_DIFFERENCE: POPULATION_DIRECTION,
    ARM_NORMALISED: POPULATION_DIRECTION,
    ARM_SIZE_MATCHED: RANDOM_DIRECTION,
    ARM_GAUSSIAN: RANDOM_DIRECTION,
    ARM_MIXTURE: "#b8479a",
    ARM_DE: "#1a8a3a",
    ARM_DE_MATCHED: "#7a5cc4",
    ARM_RANDOM: MUTED,
}

DASHED, DASH_DOT = (0, (4, 2)), (0, (5, 1.5, 1, 1.5))
STYLES = {
    ARM_DIFFERENCE: DASHED,
    ARM_SIZE_MATCHED: DASHED,
    ARM_NORMALISED: "-",
    ARM_GAUSSIAN: "-",
    ARM_MIXTURE: (0, (1, 1.5)),
    ARM_DE: DASH_DOT,
    ARM_DE_MATCHED: DASH_DOT,
    ARM_RANDOM: "-",
}
