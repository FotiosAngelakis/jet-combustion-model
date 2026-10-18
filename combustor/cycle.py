"""Compressor outlet state (combustor inlet air)."""
from __future__ import annotations

import math

from scipy.optimize import brentq

from .thermo import RU, SPECIES

N2_PER_O2 = 3.76  # air model: O2 + 3.76 N2


def air_h(T):
    """Molar enthalpy of air [J/kmol]."""
    return (SPECIES["O2"].h(T) + N2_PER_O2 * SPECIES["N2"].h(T)) / (1.0 + N2_PER_O2)


def air_s(T):
    """Standard-state molar entropy of air [J/(kmol K)] (fixed composition)."""
    return (SPECIES["O2"].s(T) + N2_PER_O2 * SPECIES["N2"].s(T)) / (1.0 + N2_PER_O2)


def compressor_outlet_temperature(
    t_in: float,
    pressure_ratio: float,
    eta: float,
    model: str = "gamma",
    gamma: float = 1.4,
) -> float:
    """Compressor outlet temperature [K].

    model="gamma": calorically perfect air with constant gamma (textbook approach).
    model="nasa":  variable cp; isentropic state from s°(T2s) = s°(T1) + Ru ln(PR),
                   then h2 = h1 + (h2s - h1) / eta.
    """
    if pressure_ratio <= 1.0 or not 0.0 < eta <= 1.0:
        raise ValueError("pressure_ratio must be > 1 and 0 < eta <= 1")
    if model == "gamma":
        t_2s = t_in * pressure_ratio ** ((gamma - 1.0) / gamma)
        return t_in + (t_2s - t_in) / eta
    if model == "nasa":
        s_target = air_s(t_in) + RU * math.log(pressure_ratio)
        t_2s = brentq(lambda T: air_s(T) - s_target, t_in, 3000.0)
        h_2 = air_h(t_in) + (air_h(t_2s) - air_h(t_in)) / eta
        return brentq(lambda T: air_h(T) - h_2, t_in, 3000.0)
    raise ValueError(f"unknown compressor model {model!r}")
