# 🔥 Jet-Engine Combustor Design Tool

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B)
![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive tool for preliminary design of a gas-turbine combustor. Set the engine cycle and a turbine-entry-temperature (TET) limit, and the app works out the air–fuel ratio, how fast CO burns out, how much NO forms, and how large the combustor needs to be. Everything updates live in the browser.

**▶ Live demo:** *add your Streamlit link here*


---

## What it does

| Step | Question it answers | Method |
|---|---|---|
| **1 · Air–fuel ratio** | How much excess air keeps the products at the TET limit? | Adiabatic energy balance with NASA-polynomial thermodynamics, plus the TET–λ curve |
| **2 · CO burnout** | How long must the gas stay in the combustor to bring CO below a limit (e.g. 550 ppm)? | Stiff, constant-pressure reactor model with global CO kinetics and H₂/H₂O chemistry |
| **3 · Thermal NO** | How much NO forms over that residence time? | Extended Zeldovich mechanism (steady-state N atoms, reverse reactions included) |
| **4 · Combustor sizing** | What length and diameter deliver the required residence time at a given power? | Mass flows from the heating value, L = u·τ, D from continuity |

The app shows the **CO–NOx trade-off** directly. A longer residence time burns out CO but keeps forming thermal NO once the gas is near TET.

## Example result

Design case: benzene fuel, 300 K / 1 atm inlet, pressure ratio 25, η_c = 0.9, TET = 1860 K, 50 MW, u = 100 m/s.

| Output | Value |
|---|---|
| Compressor outlet temperature | 802.8 K |
| Air excess ratio λ (φ) | 2.215 (0.452) |
| Residence time for CO ≤ 550 ppm | 4.34 ms |
| Combustor length × diameter | 0.43 m × 0.30 m |
| Air / fuel mass flow | 36.0 / 1.23 kg/s |

## Validation

The thermodynamics are cross-checked against [Cantera](https://cantera.org) with GRI-Mech 3.0 (`validation/cantera_check.py`):

| Quantity | This model | Cantera |
|---|---|---|
| Adiabatic T of complete-combustion products, λ = 2.2148 | 1860.0 K | 1860.0 K |
| Full-equilibrium temperature (with dissociation) | — | 1850.3 K |
| Late-time CO plateau (thermodynamically consistent kinetics) | ≈ 15 ppm | 15.2 ppm (equilibrium) |

The automated test suite checks:

- formation enthalpies against JANAF data
- continuity of the thermodynamic polynomials
- the round trip λ ↔ TET
- conservation of C, H, O and N atoms and of enthalpy in the reactor
- sizing consistency

## Quick start

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

To run the tests and the optional Cantera check:

```bash
pip install -r requirements-dev.txt
python -m pytest
python validation/cantera_check.py
```

## Project structure

```
combustor/
  thermo.py       NASA 7-coefficient thermodynamics (GRI-Mech 3.0 data)
  cycle.py        compressor outlet state (constant-γ or variable-cp)
  combustion.py   complete combustion: λ(TET) and TET(λ)
  kinetics.py     reactor model: CO burnout, H2O formation, thermal NO
  sizing.py       mass flows and combustor dimensions
app.py            Streamlit user interface
tests/            pytest suite
validation/       Cantera cross-check
MODEL.md          equations, assumptions and limitations
```

## Modelling highlights

- **Thermodynamics:** NASA polynomials with the GRI-Mech 3.0 data that Cantera also uses.
- **Initial state:** the fuel breaks down instantly to CO + H₂, and the start temperature follows from the energy balance.
- **Radicals:** H, OH and O are assumed to be in partial equilibrium. Every source term conserves atoms exactly.
- **Solver:** implicit BDF integration. An event function locates the exact time CO reaches the limit.
- **CO reverse rate:** two options. The global rate holds CO at about 575 ppm at TET, which is a kinetic artefact. The thermodynamically consistent rate relaxes CO to the true equilibrium and agrees with Cantera.

Full equations and limitations are in [MODEL.md](MODEL.md).

## References

- S. R. Turns, *An Introduction to Combustion: Concepts and Applications*, McGraw-Hill.
- G. P. Smith et al., *GRI-Mech 3.0*, <http://combustion.berkeley.edu/gri-mech/>.
- R. K. Hanson, S. Salimian, "Survey of rate constants in the N/H/O system", in *Combustion Chemistry*, Springer, 1984.

## License

[MIT](LICENSE) © 2026 Fotios Angelakis
