import numpy as np
import pytest

from combustor import combustion, cycle, kinetics, sizing
from combustor.thermo import HF_C6H6_GAS, P_ATM

T2 = cycle.compressor_outlet_temperature(300.0, 25.0, 0.9)
LAM = combustion.lambda_for_tet(1860.0, T2, HF_C6H6_GAS)
P = 25.0 * P_ATM


@pytest.fixture(scope="module")
def result():
    return kinetics.simulate(LAM, T2, HF_C6H6_GAS, P, t_end=0.05)


def elements(n):
    co, co2, h2, h2o, o2, n2, no = n
    return np.array([co + co2, 2 * h2 + 2 * h2o, co + 2 * co2 + 2 * o2 + h2o + no, 2 * n2 + no])


def test_elements_conserved(result):
    e0, e1 = elements(result.n[:, 0]), elements(result.n[:, -1])
    assert np.allclose(e0, e1, rtol=1e-6)


def test_energy_conserved(result):
    assert result.energy_error_K < 1.0


def test_final_temperature_close_to_tet(result):
    # Small residual CO/H2 keep the final T slightly below the complete-combustion TET.
    assert 1800.0 < result.T[-1] < 1861.0


def test_co_decreases_and_no_increases(result):
    assert result.co_ppm[-1] < 0.05 * result.co_ppm[0]
    assert result.no_ppm[-1] > result.no_ppm[len(result.t) // 2] > 0.0
    assert np.all(np.isfinite(result.T))


def test_start_temperature_between_air_and_tet(result):
    assert T2 < result.T0 < 1860.0


def test_global_kinetics_plateau_above_550ppm(result):
    # The global kb implies CO_eq ~ 575 ppm (wet) at TET 1860 K -> 550 ppm is unreachable.
    assert result.t_target is None
    assert result.co_ppm[-1] > 550.0


def test_consistent_reverse_rate_reaches_limit():
    params = kinetics.KineticParameters(co_reverse="equilibrium")
    res = kinetics.simulate(LAM, T2, HF_C6H6_GAS, P, t_end=0.1, params=params)
    assert res.t_target is not None
    assert res.at(res.t_target)["co_ppm"] == pytest.approx(550.0, rel=0.02)


def test_sizing_positive(result):
    tau = result.t_target or result.fallback_tau()
    s = result.at(tau)
    size = sizing.size_combustor(50e6, LAM, HF_C6H6_GAS, tau, 100.0, P,
                                 result.T0, float(result.mw[0]), s["T"], s["mw"])
    assert size.length == pytest.approx(100.0 * tau)
    assert size.d_exit > size.d_inlet > 0.0
    assert sizing.lower_heating_value(HF_C6H6_GAS) == pytest.approx(40.6e6, rel=0.01)
