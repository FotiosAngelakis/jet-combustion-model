"""Cylindrical combustor dimensions.

    m_fuel = P / LHV,   m_air = λ AFR_st m_fuel,   m = m_fuel + m_air
    L = u τ                      (constant through-flow velocity)
    A = m / (ρ u),  D = sqrt(4A/π)

With u fixed, ρ u A = const means a truly constant-velocity duct would change
area as the gas heats up. The diameter is therefore reported at both ends; the
exit value (lowest density) is the conservative one for a straight cylinder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .combustion import AFR_STOICH, NU_O2
from .thermo import MW_C6H6, RU, SPECIES, T_REF


def lower_heating_value(h_fuel: float) -> float:
    """LHV of benzene [J/kg] (products CO2 + H2O vapour at 298 K)."""
    dh = (
        h_fuel
        + NU_O2 * SPECIES["O2"].h(T_REF)
        - 6.0 * SPECIES["CO2"].h(T_REF)
        - 3.0 * SPECIES["H2O"].h(T_REF)
    )
    return dh / MW_C6H6


@dataclass(frozen=True)
class CombustorSize:
    lhv: float  # J/kg
    m_fuel: float  # kg/s
    m_air: float  # kg/s
    m_total: float  # kg/s
    length: float  # m
    rho_inlet: float  # kg/m3
    rho_exit: float  # kg/m3
    d_inlet: float  # m
    d_exit: float  # m
    volume: float  # m3, based on exit diameter
    intensity: float  # W/m3 (heat-release rate per unit volume)


def size_combustor(
    power_w: float,
    lam: float,
    h_fuel: float,
    tau: float,
    u: float,
    p_pa: float,
    T_inlet: float,
    mw_inlet: float,
    T_exit: float,
    mw_exit: float,
) -> CombustorSize:
    lhv = lower_heating_value(h_fuel)
    m_fuel = power_w / lhv
    m_air = lam * AFR_STOICH * m_fuel
    m_total = m_fuel + m_air
    rho_in = p_pa * mw_inlet / (RU * T_inlet)
    rho_out = p_pa * mw_exit / (RU * T_exit)

    def diameter(rho: float) -> float:
        return math.sqrt(4.0 * m_total / (rho * u) / math.pi)

    length = u * tau
    d_out = diameter(rho_out)
    volume = math.pi * d_out**2 / 4.0 * length
    return CombustorSize(
        lhv, m_fuel, m_air, m_total, length, rho_in, rho_out, diameter(rho_in), d_out, volume,
        power_w / volume if volume > 0 else float("nan"),
    )
