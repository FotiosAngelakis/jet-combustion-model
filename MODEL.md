### Model summary

**1 · Air–fuel ratio.** Air leaves the compressor at T₂. For constant γ:
T₂ = T₁ + T₁(PR^((γ−1)/γ) − 1)/η_c. For variable cp, an isentropic NASA-polynomial calculation is used instead. Complete lean combustion

C₆H₆ + 7.5λ(O₂ + 3.76N₂) → 6CO₂ + 3H₂O + 7.5(λ−1)O₂ + 28.2λN₂

is adiabatic at constant pressure. The energy balance is linear in λ, so λ(TET) is explicit. TET(λ) is found by root solving.

**2 · CO burnout.** The fuel is assumed to break down instantly to 6 CO + 3 H₂, which gives the start temperature. An adiabatic, constant-pressure reactor (equivalent to plug flow, with t = x/u) then integrates:

- CO + ½O₂ ⇌ CO₂ with R = k_f[CO][H₂O]^0.5[O₂]^0.25 − k_b[CO₂] (global rate)
- H + OH + M ⇌ H₂O + M, with k_f = 4.56·10²³ T⁻² cm⁶/mol²s and k_b from the equilibrium constant
- H and OH in partial equilibrium with H₂ and O₂. H₂ consumption = H₂O production, and ½ O₂ is consumed per H₂O so that O atoms are conserved.

Energy: dT/dt = −Σhᵢωᵢ / Σ[Xᵢ]cpᵢ. The system is stiff and is solved with an implicit BDF method (SciPy), with an event that locates the time when CO reaches its limit.

**3 · Thermal NO.** Extended Zeldovich mechanism (O + N₂, N + O₂, N + OH), with N atoms in steady state and equilibrium O atoms. Reverse reactions are included. Rate constants are from Hanson & Salimian (as tabulated in Turns).

**4 · Size.** ṁ_fuel = P/LHV, ṁ_air = λ·AFR_st·ṁ_fuel, L = u·τ, D = √(4ṁ/(πρu)).

**Thermodynamics.** NASA 7-coefficient polynomials, GRI-Mech 3.0 data (identical to Cantera's `gri30.yaml`).

### CO reverse-rate options

- **Global rate:** k_b = 5·10⁸ e^(−E/RT). The CO ⇌ CO₂ equilibrium this implies is not thermodynamic. At TET = 1860 K, λ ≈ 2.2 and 25 atm, CO levels off at about 575 ppm (wet).
- **Thermodynamically consistent:** k_f is unchanged and the reverse rate comes from K_c of CO + ½O₂ ⇌ CO₂. CO then relaxes to the true equilibrium of about 15 ppm, in agreement with Cantera.

### Limitations

- 0-D/plug-flow model: no mixing, spray evaporation, recirculation zone or dilution-air staging.
- Prompt and N₂O-route NO are not modelled. Radicals (H, OH, O) are assumed equilibrated, so super-equilibrium O atoms, which would raise NO, are ignored.
- When τ is set by the CO limit, the gas may leave the reactor below TET. Use a manual τ to see the extra NO cost of reaching TET.
