import numpy as np
#使用hatree单位制
me = 1
e = 1
h = 1
pi_epi_4 = 1
TWO_PI = 2.0 * np.pi

# 单位转换
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


def length_to_bohr(value: float, unit: str) -> float:
    """Convert a length expressed in a supported input unit to Bohr."""
    normalized_unit = unit.lower()
    if normalized_unit == "bohr":
        return float(value)
    if normalized_unit == "angstrom":
        return float(value) * ANGSTROM_TO_BOHR
    raise ValueError(f"unknown length unit: {unit!r}")


def energy_to_hartree(value: float, unit: str) -> float:
    """Convert an energy expressed in a supported input unit to Hartree."""
    normalized_unit = unit.lower()
    if normalized_unit == "hartree":
        return float(value)
    if normalized_unit == "ev":
        return float(value) * EV_TO_HARTREE
    raise ValueError(f"unknown energy unit: {unit!r}")
