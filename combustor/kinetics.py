"""CO burnout and thermal-NO formation versus residence time.

Reactor: adiabatic, constant-pressure, homogeneous (equivalent to a plug-flow
combustor where t = x / u). The fuel is assumed to break down instantly into
CO and H2:

    C6H6 + 7.5 λ (O2 + 3.76 N2) -> 6 CO + 3 H2 + (7.5 λ - 3) O2 + 28.2 λ N2

after which three processes run in parallel:

1. Global CO oxidation
       CO + 1/2 O2 <-> CO2
       R_CO = kf [CO][H2O]^0.5 [O2]^0.25 - kb [CO2]
       kf = 2.24e12 exp(-1.674e8 / Ru T)   (m3/kmol)^0.75 / s
       kb = 5.0e8   exp(-1.674e8 / Ru T)   1/s

2. Water formation
       H + OH + M <-> H2O + M,  kf = A T^n exp(-E/RuT),  A = 45.6e22 cm6/(mol2 s), n = -2
   kb comes from the equilibrium constant. H and OH are in partial equilibrium
   with H2 and O2 (1/2 H2 <-> H, 1/2 H2 + 1/2 O2 <-> OH). H2 consumption
   equals H2O production; O2 is consumed 1/2 per H2O so that
   oxygen atoms are conserved (net: H2 + 1/2 O2 -> H2O).

3. Thermal NO, extended Zeldovich mechanism with N atoms in steady state
       O  + N2 <-> NO + N
       N  + O2 <-> NO + O
       N  + OH <-> NO + H
   d[NO]/dt = 2 k1f [O][N2] (1 - [NO]^2 / (K [O2][N2])) / (1 + k1r [NO] / (k2f [O2] + k3f [OH]))
   with equilibrium O atoms (1/2 O2 <-> O). Rate constants: Hanson & Salimian,
   as tabulated in Turns, "An Introduction to Combustion".

State vector: [T, n_CO, n_CO2, n_H2, n_H2O, n_O2, n_N2, n_NO], n in kmol per kg
of mixture. With this choice mass is conserved automatically and the
constant-pressure equations are simply
    dn_i/dt = ω_i / ρ,      dT/dt = -Σ h_i ω_i / Σ [X_i] cp_i
The system is stiff, so it is integrated with an implicit BDF solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .combustion import air_moles, reactant_enthalpy
from .thermo import MW_C6H6, P_REF, RU, SPECIES

STATE_SPECIES = ("CO", "CO2", "H2", "H2O", "O2", "N2", "NO")
_SP = tuple(SPECIES[k] for k in STATE_SPECIES)
_H2, _O2, _H, _OH, _O, _H2O = (SPECIES[k] for k in ("H2", "O2", "H", "OH", "O", "H2O"))
_CO, _CO2 = SPECIES["CO"], SPECIES["CO2"]
_IDX = {k: i for i, k in enumerate(STATE_SPECIES)}


@dataclass(frozen=True)
class KineticParameters:
    co_af: float = 2.24e12  # (m3/kmol)^0.75 / s
    co_ab: float = 5.0e8  # 1/s
    co_ea: float = 1.674e8  # J/kmol
    h2o_a_cgs: float = 45.6e22  # cm6 / (mol2 s)
    h2o_n: float = -2.0
    h2o_ea: float = 0.0  # J/kmol
    # "global": kb = 5e8 exp(-E/RuT). Its implied CO/CO2 equilibrium is not
    #           thermodynamic and leaves ~10^2 ppm CO at turbine-entry temperatures.
    # "equilibrium": reverse rate from Kc of CO + 1/2 O2 <-> CO2 (thermodynamically consistent).
    co_reverse: str = "global"


def radicals(T: float, c_h2: float, c_o2: float) -> tuple[float, float, float]:
    """Partial-equilibrium [H], [OH] and equilibrium [O] in kmol/m3."""
    rt = RU * T
    sqrt_cref = math.sqrt(P_REF / rt)
    g_h2, g_o2 = _H2.g(T), _O2.g(T)
    c_h = math.exp(-(_H.g(T) - 0.5 * g_h2) / rt) * sqrt_cref * math.sqrt(c_h2)
    c_oh = math.exp(-(_OH.g(T) - 0.5 * g_h2 - 0.5 * g_o2) / rt) * math.sqrt(c_h2 * c_o2)
    c_o = math.exp(-(_O.g(T) - 0.5 * g_o2) / rt) * sqrt_cref * math.sqrt(c_o2)
    return c_h, c_oh, c_o


def zeldovich_constants(T: float) -> dict[str, float]:
    """Extended Zeldovich rate constants in m3/(kmol s) (cm3/(mol s) x 1e-3)."""
    return {
        "k1f": 1.8e11 * math.exp(-38370.0 / T),
        "k1r": 3.8e10 * math.exp(-425.0 / T),
        "k2f": 1.8e7 * T * math.exp(-4680.0 / T),
        "k2r": 3.8e6 * T * math.exp(-20820.0 / T),
        "k3f": 7.1e10 * math.exp(-450.0 / T),
        "k3r": 1.7e11 * math.exp(-24560.0 / T),
    }


def reaction_rates(T: float, c: np.ndarray, c_tot: float, p: KineticParameters):
    """Net rates [kmol/(m3 s)] of the three processes and the radical pool."""
    c_co, c_co2, c_h2, c_h2o, c_o2, c_n2, c_no = (float(v) for v in c)
    rt = RU * T
    g_o2 = _O2.g(T)
    c_h, c_oh, c_o = radicals(T, c_h2, c_o2)

    # 1. global CO oxidation
    arr = math.exp(-p.co_ea / rt)
    kf_co = p.co_af * arr * math.sqrt(c_h2o) * c_o2**0.25
    if p.co_reverse == "equilibrium":
        # Kc = [CO2]/([CO][O2]^0.5) = Kp (P°/RuT)^(-1/2)
        kc_co = math.exp(-(_CO2.g(T) - _CO.g(T) - 0.5 * g_o2) / rt) / math.sqrt(P_REF / rt)
        r_co = kf_co * (c_co - c_co2 / (kc_co * math.sqrt(c_o2) + 1e-300))
    else:
        r_co = kf_co * c_co - p.co_ab * arr * c_co2

    # 2. H + OH + M <-> H2O + M  (all third-body efficiencies = 1 -> [M] = c_tot)
    kf_w = p.h2o_a_cgs * 1e-6 * T**p.h2o_n * math.exp(-p.h2o_ea / rt)  # m6/(kmol2 s)
    kc_w = math.exp(-(_H2O.g(T) - _H.g(T) - _OH.g(T)) / rt) * rt / P_REF  # m3/kmol
    r_w = kf_w * c_tot * (c_h * c_oh - c_h2o / kc_w)

    # 3. thermal NO (extended Zeldovich, steady-state N)
    k = zeldovich_constants(T)
    k_eq = (k["k1f"] / k["k1r"]) * (k["k2f"] / k["k2r"])
    reverse = c_no**2 / max(k_eq * c_o2 * c_n2, 1e-300)
    r_no = 2.0 * k["k1f"] * c_o * c_n2 * (1.0 - reverse) / (
        1.0 + k["k1r"] * c_no / (k["k2f"] * c_o2 + k["k3f"] * c_oh + 1e-300)
    )
    return r_co, r_w, r_no, (c_h, c_oh, c_o)


def _concentrations(T: float, n: np.ndarray, p_pa: float):
    n = np.maximum(n, 0.0)
    n_tot = n.sum()
    c_tot = p_pa / (RU * T)
    return n * (c_tot / n_tot), c_tot, c_tot / n_tot  # [X_i], C, rho


def _rhs(t, y, p_pa: float, params: KineticParameters):
    T = y[0]
    c, c_tot, rho = _concentrations(T, y[1:], p_pa)
    r_co, r_w, r_no, _ = reaction_rates(T, c, c_tot, params)
    w = np.array(
        [-r_co, r_co, -r_w, r_w, -0.5 * (r_co + r_w + r_no), -0.5 * r_no, r_no]
    )
    h = np.array([sp.h(T) for sp in _SP])
    cp = np.array([sp.cp(T) for sp in _SP])
    dT = -np.dot(h, w) / np.dot(c, cp)
    return np.concatenate(([dT], w / rho))


def instant_burn_moles(lam: float) -> dict[str, float]:
    """Composition (kmol per kmol fuel) after C6H6 -> 6 CO + 3 H2."""
    n_o2, n_n2 = air_moles(lam)
    return {"CO": 6.0, "CO2": 0.0, "H2": 3.0, "H2O": 0.0, "O2": n_o2 - 3.0, "N2": n_n2, "NO": 0.0}


def initial_state(lam: float, t_air: float, h_fuel: float) -> tuple[float, np.ndarray]:
    """Temperature [K] and composition [kmol/kg] right after the instant burn."""
    moles = instant_burn_moles(lam)
    n_o2, n_n2 = air_moles(lam)
    mass = MW_C6H6 + n_o2 * SPECIES["O2"].mw + n_n2 * SPECIES["N2"].mw  # kg per kmol fuel
    h_r = reactant_enthalpy(lam, t_air, h_fuel)
    T0 = brentq(lambda T: sum(n * SPECIES[k].h(T) for k, n in moles.items()) - h_r, 200.0, 5000.0)
    return T0, np.array([moles[k] for k in STATE_SPECIES]) / mass


def ppm(n: np.ndarray, species: str, basis: str = "wet") -> np.ndarray:
    """Mole-fraction ppm of `species` from an (n_species, n_times) array."""
    n = np.maximum(n, 0.0)
    total = n.sum(axis=0)
    if basis == "dry":
        total = total - n[_IDX["H2O"]]
    return n[_IDX[species]] / total * 1e6


@dataclass
class ReactorResult:
    t: np.ndarray  # s
    T: np.ndarray  # K
    n: np.ndarray  # kmol/kg, shape (7, len(t))
    co_ppm: np.ndarray
    no_ppm: np.ndarray
    basis: str
    T0: float
    pressure: float  # Pa
    co_target: float  # ppm
    t_target: float | None  # first time CO <= target [s]
    energy_error_K: float  # enthalpy drift expressed as a temperature

    @property
    def mw(self) -> np.ndarray:
        """Mixture molar mass [kg/kmol] versus time."""
        return 1.0 / np.maximum(self.n, 0.0).sum(axis=0)

    def mole_fraction(self, species: str) -> np.ndarray:
        n = np.maximum(self.n, 0.0)
        return n[_IDX[species]] / n.sum(axis=0)

    def at(self, tau: float) -> dict[str, float]:
        return {
            "T": float(np.interp(tau, self.t, self.T)),
            "co_ppm": float(np.interp(tau, self.t, self.co_ppm)),
            "no_ppm": float(np.interp(tau, self.t, self.no_ppm)),
            "mw": float(np.interp(tau, self.t, self.mw)),
        }

    def fallback_tau(self, tolerance: float = 0.05) -> float:
        """First time CO is within `tolerance` of its final (equilibrium-limited) value."""
        idx = np.argmax(self.co_ppm <= (1.0 + tolerance) * self.co_ppm[-1])
        return float(self.t[idx])


def simulate(
    lam: float,
    t_air: float,
    h_fuel: float,
    p_pa: float,
    t_end: float = 0.05,
    co_target_ppm: float = 550.0,
    basis: str = "wet",
    params: KineticParameters = KineticParameters(),
) -> ReactorResult:
    T0, n0 = initial_state(lam, t_air, h_fuel)
    y0 = np.concatenate(([T0], n0))
    i_co, i_h2o = _IDX["CO"], _IDX["H2O"]

    def co_event(t, y, *_):  # solve_ivp also passes `args` to event functions
        n = np.maximum(y[1:], 0.0)
        total = n.sum() - (n[i_h2o] if basis == "dry" else 0.0)
        return n[i_co] / total * 1e6 - co_target_ppm

    co_event.direction = -1
    co_event.terminal = False

    t_eval = np.unique(
        np.concatenate(([0.0], np.geomspace(1e-7, t_end, 600), np.linspace(0.0, t_end, 3000)))
    )
    atol = np.full(y0.size, 1e-15)
    atol[0] = 1e-6
    sol = solve_ivp(
        _rhs,
        (0.0, t_end),
        y0,
        method="BDF",
        t_eval=t_eval,
        events=co_event,
        args=(p_pa, params),
        rtol=1e-7,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t_out, y_out = sol.t, sol.y
    if sol.t_events[0].size:
        # insert the exact event state so interpolation at τ is exact (CO falls steeply there)
        t_e, y_e = sol.t_events[0][0], sol.y_events[0][0]
        k = int(np.searchsorted(t_out, t_e))
        t_out = np.insert(t_out, k, t_e)
        y_out = np.insert(y_out, k, y_e, axis=1)
    T, n = y_out[0], y_out[1:]
    h_mix = np.array([sum(n[i, j] * _SP[i].h(T[j]) for i in range(len(_SP))) for j in (0, -1)])
    cp_mix = sum(n[i, -1] * _SP[i].cp(T[-1]) for i in range(len(_SP)))
    t_target = float(sol.t_events[0][0]) if sol.t_events[0].size else None

    return ReactorResult(
        t=t_out,
        T=T,
        n=n,
        co_ppm=ppm(n, "CO", basis),
        no_ppm=ppm(n, "NO", basis),
        basis=basis,
        T0=T0,
        pressure=p_pa,
        co_target=co_target_ppm,
        t_target=t_target,
        energy_error_K=float(abs(h_mix[1] - h_mix[0]) / cp_mix),
    )
