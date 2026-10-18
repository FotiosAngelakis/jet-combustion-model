import pytest

from combustor import combustion, cycle
from combustor.thermo import HF_C6H6_GAS


def test_compressor_constant_gamma():
    # 300 K, PR 25, eta 0.9 -> 802.8 K
    assert cycle.compressor_outlet_temperature(300.0, 25.0, 0.9) == pytest.approx(802.8, abs=0.1)


def test_compressor_variable_cp_is_close():
    t_gamma = cycle.compressor_outlet_temperature(300.0, 25.0, 0.9, model="gamma")
    t_nasa = cycle.compressor_outlet_temperature(300.0, 25.0, 0.9, model="nasa")
    assert abs(t_nasa - t_gamma) < 40.0  # real-gas cp lowers T2 slightly


def test_lambda_regression():
    # Reference design case: lambda ~ 2.216 at TET = 1860 K
    t2 = cycle.compressor_outlet_temperature(300.0, 25.0, 0.9)
    lam = combustion.lambda_for_tet(1860.0, t2, HF_C6H6_GAS)
    assert lam == pytest.approx(2.216, abs=0.02)


@pytest.mark.parametrize("tet", [1200.0, 1600.0, 1860.0, 2200.0])
def test_lambda_and_tet_are_inverse(tet):
    t2 = 802.8
    lam = combustion.lambda_for_tet(tet, t2, HF_C6H6_GAS)
    assert combustion.tet_for_lambda(lam, t2, HF_C6H6_GAS) == pytest.approx(tet, abs=1e-6)


def test_rich_tet_rejected():
    with pytest.raises(ValueError):
        combustion.design_point(4000.0, 802.8, HF_C6H6_GAS)
