"""Streamlit front end: jet-engine combustor design (λ, CO burnout, NO, sizing)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from combustor import combustion, cycle, kinetics, sizing
from combustor.thermo import HF_C6H6_GAS, HF_C6H6_LIQUID, P_ATM

st.set_page_config(page_title="Jet Combustor Design", page_icon="🔥", layout="wide")


# ---------------------------------------------------------------- cached model
@st.cache_data(show_spinner=False)
def run_part1(t_in, pr, eta, comp_model, tet, h_fuel):
    t2 = cycle.compressor_outlet_temperature(t_in, pr, eta, model=comp_model)
    dp = combustion.design_point(tet, t2, h_fuel)
    lam_grid = np.linspace(1.0, 6.0, 151)
    tet_grid = np.array([combustion.tet_for_lambda(l, t2, h_fuel) for l in lam_grid])
    return t2, dp, lam_grid, tet_grid


@st.cache_data(show_spinner=False)
def run_reactor(lam, t2, h_fuel, p_pa, t_end, target, basis, a_h2o, co_reverse):
    params = kinetics.KineticParameters(h2o_a_cgs=a_h2o, co_reverse=co_reverse)
    return kinetics.simulate(lam, t2, h_fuel, p_pa, t_end, target, basis, params)


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("Inputs")
sb.subheader("Compressor")
t_in = sb.number_input("Inlet temperature T_in [K]", 200.0, 400.0, 300.0, 5.0)
p_in = sb.number_input("Inlet pressure [atm]", 0.2, 2.0, 1.0, 0.1)
pr = sb.number_input("Pressure ratio [-]", 2.0, 60.0, 25.0, 1.0)
eta = sb.number_input("Isentropic efficiency [-]", 0.5, 1.0, 0.9, 0.01)
comp_label = sb.radio("Compressor model", ["Constant γ = 1.4", "Variable cp (NASA)"])
comp_model = "gamma" if comp_label.startswith("Constant") else "nasa"

sb.subheader("Combustor")
tet = sb.number_input("Turbine entry temperature TET [K]", 1000.0, 2400.0, 1860.0, 5.0)
power_mw = sb.number_input("Thermal power P [MW]", 1.0, 500.0, 50.0, 5.0)
u = sb.number_input("Through-flow velocity u [m/s]", 5.0, 300.0, 100.0, 5.0)
fuel_state = sb.radio("Benzene enters as", ["Gas (h_f = 82.93 MJ/kmol)", "Liquid (h_f = 49.08 MJ/kmol)"])
h_fuel = HF_C6H6_GAS if fuel_state.startswith("Gas") else HF_C6H6_LIQUID

sb.subheader("Emissions & residence time")
co_target = sb.number_input("CO limit [ppm]", 10.0, 50000.0, 550.0, 10.0)
basis = sb.radio("ppm basis", ["wet", "dry"], horizontal=True)
t_end_ms = sb.number_input("Simulated time [ms]", 1.0, 1000.0, 50.0, 5.0)
tau_mode = sb.radio("Residence time τ", ["From CO limit", "Manual"])
tau_manual_ms = sb.number_input(
    "Manual τ [ms]", 0.1, float(t_end_ms), min(10.0, float(t_end_ms)), 0.5, disabled=tau_mode != "Manual"
)

with sb.expander("Advanced kinetics"):
    reverse_label = st.radio(
        "CO reverse rate",
        ["Global rate (k_b = 5·10⁸ e^(−E/RT))", "Thermodynamically consistent (from K_c)"],
        help=(
            "The global k_b implies a CO/CO₂ equilibrium that is not thermodynamic: at TET it "
            "leaves several hundred ppm of CO, so a 550 ppm limit may be unreachable. The second "
            "option keeps k_f and takes the reverse rate from the true equilibrium constant."
        ),
    )
    co_reverse = "global" if reverse_label.startswith("Global") else "equilibrium"
    a_h2o = st.number_input(
        "A for H + OH + M → H₂O + M [cm⁶/mol²s]", value=45.6e22, format="%.3e",
        help="Pre-exponential factor of kf = A·T^n, n = −2.",
    )

p_pa = pr * p_in * P_ATM

# ---------------------------------------------------------------- run
st.title("Jet-engine combustor design")
st.caption(
    "Benzene-fuelled gas-turbine combustor: air–fuel ratio for a TET limit, CO burnout, "
    "thermal NO (extended Zeldovich) and combustor sizing."
)

try:
    t2, dp, lam_grid, tet_grid = run_part1(t_in, pr, eta, comp_model, tet, h_fuel)
except ValueError as err:
    st.error(str(err))
    st.stop()

with st.spinner("Integrating the reactor equations…"):
    try:
        res = run_reactor(dp.lam, t2, h_fuel, p_pa, t_end_ms / 1e3, co_target, basis, a_h2o, co_reverse)
    except (RuntimeError, ValueError) as err:
        st.error(str(err))
        st.stop()

if tau_mode == "Manual":
    tau, tau_source = tau_manual_ms / 1e3, "manual input"
elif res.t_target is not None:
    tau, tau_source = res.t_target, f"CO reaches {co_target:g} ppm"
else:
    tau, tau_source = res.fallback_tau(), "CO within 5 % of its final value (limit not reachable)"
state = res.at(tau)
size = sizing.size_combustor(
    power_mw * 1e6, dp.lam, h_fuel, tau, u, p_pa,
    res.T0, float(res.mw[0]), state["T"], state["mw"],
)

tabs = st.tabs(["1 · Air–fuel ratio", "2 · CO burnout", "3 · NO formation", "4 · Combustor size", "Model"])

# ---------------------------------------------------------------- tab 1
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Compressor outlet T₂", f"{t2:.1f} K")
    c[1].metric("Combustor pressure", f"{p_pa / 1e5:.2f} bar")
    c[2].metric("λ (air excess ratio)", f"{dp.lam:.3f}")
    c[3].metric("Equivalence ratio φ", f"{dp.phi:.3f}")
    c[4].metric("AFR [kg air / kg fuel]", f"{dp.afr:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lam_grid, y=tet_grid, name="Adiabatic T (complete combustion)"))
    fig.add_hline(y=tet, line_dash="dash", annotation_text=f"TET limit {tet:.0f} K")
    fig.add_trace(go.Scatter(x=[dp.lam], y=[tet], mode="markers", marker_size=11, name="Design point"))
    fig.update_layout(xaxis_title="λ [-]", yaxis_title="Turbine entry temperature [K]", height=430,
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Products of complete combustion** (per kmol C₆H₆)")
    st.dataframe(
        pd.DataFrame({"kmol": dp.products, "mole fraction": dp.x_products}).style.format("{:.4f}")
    )

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    c = st.columns(4)
    c[0].metric("Start temperature (after CO + H₂ burn)", f"{res.T0:.0f} K")
    c[1].metric("Residence time τ", f"{tau * 1e3:.3f} ms", help=tau_source)
    c[2].metric(f"CO at τ ({basis})", f"{state['co_ppm']:.0f} ppm")
    c[3].metric("Gas temperature at τ", f"{state['T']:.0f} K")

    if tau_mode != "Manual" and res.t_target is None:
        st.warning(
            f"CO never drops to {co_target:g} ppm within {t_end_ms:g} ms. It levels off at "
            f"**{res.co_ppm[-1]:.0f} ppm** — the CO ⇌ CO₂ equilibrium limit of the global "
            "mechanism at this λ and pressure. Switch *Advanced kinetics → CO reverse rate* to the "
            "thermodynamically consistent option, or change TET / the limit. τ was taken where CO "
            "is within 5 % of its final value."
        )

    log_t = st.toggle("Logarithmic time axis", value=True, key="log_t_co")
    mask = res.t > 0 if log_t else slice(None)
    t_ms = res.t * 1e3
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t_ms[mask], y=res.co_ppm[mask], name=f"CO [ppm, {basis}]"), secondary_y=False)
    fig.add_trace(go.Scatter(x=t_ms[mask], y=res.T[mask], name="T [K]", line_dash="dot"), secondary_y=True)
    fig.add_hline(y=co_target, line_dash="dash")
    fig.add_annotation(x=0, xref="x domain", y=np.log10(co_target), yref="y", text=f"{co_target:g} ppm limit",
                       showarrow=False, xanchor="left", yanchor="top")
    # shapes take data values on log axes, annotations take log10 values (see the hline above)
    fig.add_vline(x=tau * 1e3, line_dash="dash", line_color="grey")
    fig.add_annotation(x=np.log10(tau * 1e3) if log_t else tau * 1e3, xref="x", y=1, yref="y domain",
                       text="τ", showarrow=False, xanchor="left", yanchor="bottom")
    fig.update_xaxes(type="log" if log_t else "linear", title_text="Time in combustor [ms]")
    fig.update_yaxes(type="log", dtick=1, title_text="CO [ppm]", secondary_y=False)
    fig.update_yaxes(title_text="Temperature [K]", showgrid=False, tickmode="auto", secondary_y=True)
    fig.update_layout(height=460, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Energy-conservation check: enthalpy drift ≡ {res.energy_error_K:.2e} K.")

# ---------------------------------------------------------------- tab 3
with tabs[2]:
    c = st.columns(3)
    c[0].metric(f"NO at τ ({basis})", f"{state['no_ppm']:.2f} ppm")
    c[1].metric(f"NO at {t_end_ms:g} ms", f"{res.no_ppm[-1]:.2f} ppm")
    c[2].metric("Peak temperature", f"{res.T.max():.0f} K")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=res.t * 1e3, y=res.no_ppm, name=f"NO [ppm, {basis}]"), secondary_y=False)
    fig.add_trace(go.Scatter(x=res.t * 1e3, y=res.T, name="T [K]", line_dash="dot"), secondary_y=True)
    fig.add_vline(x=tau * 1e3, line_dash="dash", line_color="grey", annotation_text="τ")
    fig.update_yaxes(title_text="NO [ppm]", secondary_y=False)
    fig.update_yaxes(title_text="Temperature [K]", showgrid=False, tickmode="auto", secondary_y=True)
    fig.update_layout(xaxis_title="Time in combustor [ms]", height=460, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Thermal NO is negligible while CO and H₂ are still burning at low temperature. "
        "Once the mixture approaches TET it grows almost linearly with time, because the "
        "O + N₂ step (E/R ≈ 38 400 K) is far from equilibrium. Every extra millisecond of "
        "residence time beyond what CO burnout needs therefore adds NO — the classic CO–NOx trade-off."
    )

# ---------------------------------------------------------------- tab 4
with tabs[3]:
    c = st.columns(4)
    c[0].metric("Length L = u·τ", f"{size.length:.3f} m")
    c[1].metric("Diameter (exit ρ)", f"{size.d_exit:.3f} m")
    c[2].metric("Diameter (start-of-burn ρ)", f"{size.d_inlet:.3f} m",
                help="Density just after the instantaneous C6H6 → CO + H2 step (reactor start).")
    c[3].metric("Volume", f"{size.volume:.3f} m³")
    st.dataframe(
        pd.DataFrame(
            {
                "value": [size.lhv / 1e6, size.m_fuel, size.m_air, size.m_total,
                          size.rho_inlet, size.rho_exit, size.intensity / 1e6],
                "unit": ["MJ/kg", "kg/s", "kg/s", "kg/s", "kg/m³", "kg/m³", "MW/m³"],
            },
            index=["Lower heating value", "Fuel flow", "Air flow", "Total flow",
                   "Density at reactor start", "Density at exit", "Heat-release intensity"],
        ).style.format({"value": "{:.4g}"})
    )
    st.caption(
        "With a fixed velocity the flow area must grow as the gas heats up (ρuA = const). "
        "For a straight cylinder the exit-density diameter is the conservative choice."
    )

# ---------------------------------------------------------------- tab 5
with tabs[4]:
    st.markdown((Path(__file__).parent / "MODEL.md").read_text(encoding="utf-8"))
