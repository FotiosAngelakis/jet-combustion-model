"""Cross-check the in-house thermodynamics against Cantera (GRI-Mech 3.0).

Run:  pip install cantera  &&  python validation/cantera_check.py

1. Frozen products: Cantera's HP temperature for the complete-combustion
   products should equal the TET used to compute λ (same NASA data).
2. Full chemical equilibrium (HP): shows the effect of dissociation and gives
   the equilibrium CO and NO levels as reference points for the kinetic model.
"""
import sys
from pathlib import Path

import cantera as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combustor import combustion, cycle  # noqa: E402
from combustor.combustion import air_moles  # noqa: E402
from combustor.thermo import HF_C6H6_GAS, MW_C6H6, P_ATM, SPECIES  # noqa: E402

TET, PR, ETA = 1860.0, 25.0, 0.9
P = PR * P_ATM

t2 = cycle.compressor_outlet_temperature(300.0, PR, ETA)
lam = combustion.lambda_for_tet(TET, t2, HF_C6H6_GAS)
n_o2, n_n2 = air_moles(lam)
mass = MW_C6H6 + n_o2 * SPECIES["O2"].mw + n_n2 * SPECIES["N2"].mw  # kg per kmol fuel
h_mass = combustion.reactant_enthalpy(lam, t2, HF_C6H6_GAS) / mass  # J/kg

gas = ct.Solution("gri30.yaml")
gas.TPX = TET, P, combustion.complete_products(lam)
gas.HP = h_mass, P
print(f"lambda = {lam:.4f}, T2 = {t2:.1f} K")
print(f"Frozen complete-combustion T: model {TET:.1f} K | Cantera {gas.T:.1f} K")

gas.equilibrate("HP")
x = dict(zip(gas.species_names, gas.X))
print(f"Equilibrium T: {gas.T:.1f} K")
print(f"Equilibrium CO: {x['CO'] * 1e6:.1f} ppm, NO: {x['NO'] * 1e6:.1f} ppm (wet)")
