"""Ideal-gas thermodynamic properties from NASA 7-coefficient polynomials.

Coefficients are the GRI-Mech 3.0 set (Smith et al., 1999), the same data that
Cantera ships in ``gri30.yaml``, so every number here can be cross-checked
against Cantera.

Units (kmol based, SI):
    h   [J/kmol]      absolute enthalpy (includes enthalpy of formation)
    cp  [J/(kmol K)]
    s   [J/(kmol K)]  at the standard pressure P_REF
    g   [J/kmol]      g = h - T s
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

RU = 8314.462618  # universal gas constant [J/(kmol K)]
P_ATM = 101325.0  # [Pa]
P_REF = P_ATM  # standard-state pressure of the polynomials [Pa]
T_REF = 298.15  # [K]

# Benzene: only its enthalpy of formation is needed (it is a reactant at 298 K).
MW_C6H6 = 78.114  # [kg/kmol]
HF_C6H6_GAS = 82.93e6  # [J/kmol]  NIST
HF_C6H6_LIQUID = 49.08e6  # [J/kmol] NIST


def _elementwise(fun):
    """Allow a scalar property function to accept floats or numpy arrays."""

    @functools.wraps(fun)
    def wrapper(self, T):
        if np.ndim(T) == 0:
            return fun(self, float(T))
        arr = np.asarray(T, dtype=float)
        out = np.fromiter((fun(self, t) for t in arr.ravel()), dtype=float, count=arr.size)
        return out.reshape(arr.shape)

    return wrapper


@dataclass(frozen=True)
class Species:
    name: str
    mw: float  # [kg/kmol]
    low: tuple  # a1..a7 for T < t_mid
    high: tuple  # a1..a7 for T >= t_mid
    t_mid: float = 1000.0

    def _a(self, T: float) -> tuple:
        return self.high if T >= self.t_mid else self.low

    @_elementwise
    def cp(self, T: float) -> float:
        a = self._a(T)
        return RU * (a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4]))))

    @_elementwise
    def h(self, T: float) -> float:
        a = self._a(T)
        return RU * (T * (a[0] + T * (a[1] / 2 + T * (a[2] / 3 + T * (a[3] / 4 + T * a[4] / 5)))) + a[5])

    @_elementwise
    def s(self, T: float) -> float:
        a = self._a(T)
        return RU * (a[0] * math.log(T) + T * (a[1] + T * (a[2] / 2 + T * (a[3] / 3 + T * a[4] / 4))) + a[6])

    def g(self, T):
        return self.h(T) - T * self.s(T)


# fmt: off
SPECIES: dict[str, Species] = {
    "O2": Species("O2", 31.998,
        low=(3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1.06394356e+03, 3.65767573),
        high=(3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1.08845772e+03, 5.45323129)),
    "N2": Species("N2", 28.014,
        low=(3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12, -1.0208999e+03, 3.950372),
        high=(2.92664, 1.4879768e-03, -5.68476e-07, 1.0097038e-10, -6.753351e-15, -9.227977e+02, 5.980528)),
    "CO": Species("CO", 28.010,
        low=(3.57953347, -6.1035368e-04, 1.01681433e-06, 9.07005884e-10, -9.04424499e-13, -1.4344086e+04, 3.50840928),
        high=(2.71518561, 2.06252743e-03, -9.98825771e-07, 2.30053008e-10, -2.03647716e-14, -1.41518724e+04, 7.81868772)),
    "CO2": Species("CO2", 44.009,
        low=(2.35677352, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13, -4.83719697e+04, 9.90105222),
        high=(3.85746029, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14, -4.8759166e+04, 2.27163806)),
    "H2": Species("H2", 2.016,
        low=(2.34433112, 7.98052075e-03, -1.9478151e-05, 2.01572094e-08, -7.37611761e-12, -9.17935173e+02, 6.83010238e-01),
        high=(3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14, -9.50158922e+02, -3.20502331)),
    "H2O": Species("H2O", 18.015,
        low=(4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -3.02937267e+04, -8.49032208e-01),
        high=(3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14, -3.00042971e+04, 4.9667701)),
    "H": Species("H", 1.008,
        low=(2.5, 7.05332819e-13, -1.99591964e-15, 2.30081632e-18, -9.27732332e-22, 2.54736599e+04, -4.46682853e-01),
        high=(2.50000001, -2.30842973e-11, 1.61561948e-14, -4.73515235e-18, 4.98197357e-22, 2.54736599e+04, -4.46682914e-01)),
    "OH": Species("OH", 17.007,
        low=(3.99201543, -2.40131752e-03, 4.61793841e-06, -3.88113333e-09, 1.3641147e-12, 3.61508056e+03, -1.03925458e-01),
        high=(3.09288767, 5.48429716e-04, 1.26505228e-07, -8.79461556e-11, 1.17412376e-14, 3.858657e+03, 4.4766961)),
    "O": Species("O", 15.999,
        low=(3.1682671, -3.27931884e-03, 6.64306396e-06, -6.12806624e-09, 2.11265971e-12, 2.91222592e+04, 2.05193346),
        high=(2.56942078, -8.59741137e-05, 4.19484589e-08, -1.00177799e-11, 1.22833691e-15, 2.92175791e+04, 4.78433864)),
    "NO": Species("NO", 30.006,
        low=(4.2184763, -4.638976e-03, 1.1041022e-05, -9.3361354e-09, 2.803577e-12, 9.8446230e+03, 2.2808464),
        high=(3.2606056, 1.1911043e-03, -4.2917048e-07, 6.9457669e-11, -4.0336099e-15, 9.9209746e+03, 6.3693027)),
}
# fmt: on


def mixture_enthalpy(moles: dict[str, float], T) -> float:
    """Total enthalpy [J] of a mixture given kmol of each species."""
    return sum(n * SPECIES[k].h(T) for k, n in moles.items())


def mole_fractions(moles: dict[str, float]) -> dict[str, float]:
    total = sum(moles.values())
    return {k: v / total for k, v in moles.items()}
