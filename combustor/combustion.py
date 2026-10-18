"""Complete combustion of benzene with excess air.

    C6H6 + 7.5 λ (O2 + 3.76 N2) -> 6 CO2 + 3 H2O + 7.5 (λ - 1) O2 + 28.2 λ N2

Adiabatic, constant pressure:  H_reactants(T_air, T_fuel) = H_products(TET).
The balance is linear in λ, so λ(TET) has a closed form; TET(λ) is found with a
root solve.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq

from .cycle import N2_PER_O2
from .thermo import MW_C6H6, SPECIES, mixture_enthalpy, mole_fractions

NU_O2 = 7.5  # stoichiometric kmol O2 per kmol C6H6
_O2, _N2, _CO2, _H2O = (SPECIES[k] for k in ("O2", "N2", "CO2", "H2O"))
AIR_MW_PER_O2 = SPECIES["O2"].mw + N2_PER_O2 * SPECIES["N2"].mw  # kg air per kmol O2
AFR_STOICH = NU_O2 * AIR_MW_PER_O2 / MW_C6H6  # kg air / kg fuel


def air_moles(lam: float) -> tuple[float, float]:
    """kmol O2 and N2 supplied per kmol fuel."""
    a = NU_O2 * lam
    return a, N2_PER_O2 * a


def reactant_enthalpy(lam: float, t_air: float, h_fuel: float) -> float:
    """Reactant enthalpy per kmol of fuel [J]."""
    n_o2, n_n2 = air_moles(lam)
    return h_fuel + n_o2 * _O2.h(t_air) + n_n2 * _N2.h(t_air)


def complete_products(lam: float) -> dict[str, float]:
    if lam < 1.0:
        raise ValueError("complete-combustion model needs lean mixtures (λ >= 1)")
    n_o2, n_n2 = air_moles(lam)
    return {"CO2": 6.0, "H2O": 3.0, "O2": n_o2 - NU_O2, "N2": n_n2}


def lambda_for_tet(tet, t_air: float, h_fuel: float):
    """Air-fuel equivalence ratio λ that gives products at `tet` (scalar or array)."""
    num = 6.0 * _CO2.h(tet) + 3.0 * _H2O.h(tet) - NU_O2 * _O2.h(tet) - h_fuel
    den = NU_O2 * ((_O2.h(t_air) - _O2.h(tet)) + N2_PER_O2 * (_N2.h(t_air) - _N2.h(tet)))
    return num / den


def tet_for_lambda(lam: float, t_air: float, h_fuel: float) -> float:
    """Adiabatic temperature of complete-combustion products [K]."""
    h_r = reactant_enthalpy(lam, t_air, h_fuel)
    prods = complete_products(lam)
    return brentq(lambda T: mixture_enthalpy(prods, T) - h_r, 200.0, 6000.0)


@dataclass(frozen=True)
class DesignPoint:
    tet: float
    t_air: float
    lam: float
    phi: float
    afr: float  # kg air / kg fuel
    far: float  # kg fuel / kg air
    products: dict  # kmol per kmol fuel
    x_products: dict  # mole fractions


def design_point(tet: float, t_air: float, h_fuel: float) -> DesignPoint:
    if tet <= t_air:
        raise ValueError("TET must be higher than the compressor outlet temperature")
    lam = float(lambda_for_tet(tet, t_air, h_fuel))
    if lam < 1.0:
        raise ValueError(f"TET is above the stoichiometric adiabatic temperature (λ = {lam:.3f} < 1)")
    prods = complete_products(lam)
    afr = lam * AFR_STOICH
    return DesignPoint(tet, t_air, lam, 1.0 / lam, afr, 1.0 / afr, prods, mole_fractions(prods))
