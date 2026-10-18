import pytest

from combustor.thermo import SPECIES, T_REF

# Enthalpies of formation at 298.15 K [J/kmol] (JANAF / GRI-Mech 3.0)
HF = {
    "O2": 0.0,
    "N2": 0.0,
    "H2": 0.0,
    "CO": -110.53e6,
    "CO2": -393.51e6,
    "H2O": -241.83e6,
    "H": 217.99e6,
    "O": 249.17e6,
}


@pytest.mark.parametrize("name,hf", HF.items())
def test_formation_enthalpy(name, hf):
    assert SPECIES[name].h(T_REF) == pytest.approx(hf, abs=0.5e6)


@pytest.mark.parametrize("name", SPECIES)
def test_continuity_at_1000K(name):
    sp = SPECIES[name]
    lo, hi = 1000.0 - 1e-6, 1000.0
    assert sp.cp(lo) == pytest.approx(sp.cp(hi), rel=5e-3)
    assert sp.h(lo) == pytest.approx(sp.h(hi), rel=5e-3, abs=2e5)
    assert sp.s(lo) == pytest.approx(sp.s(hi), rel=5e-3)


def test_vectorised_matches_scalar():
    import numpy as np

    T = np.array([500.0, 1500.0])
    co2 = SPECIES["CO2"]
    assert np.allclose(co2.h(T), [co2.h(500.0), co2.h(1500.0)])
