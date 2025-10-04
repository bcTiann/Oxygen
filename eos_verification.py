"""Small utilities to inspect saha_eos.ns_from_P_T outputs.

Steps performed:
1. Call ns_from_P_T for a baseline (P,T) pair.
2. Compare its electron density to simple n_e = P_e / (k_B T).
3. Check that the returned rho/mu reproduces the input gas pressure.
4. Vary oxygen abundance to see how the electron/oxygen populations respond.
"""
import numpy as np
from astropy import units as u
from astropy.constants import k_B
import saha_eos as eos

# Ensure numpy does not abort on benign divisions inside EOS code.
np.seterr(all='warn')

# Tabulated atmospheric data (same as structure.py)
log_tau_table = np.array([
    -5.00, -4.80, -4.60, -4.40, -4.20, -4.00, -3.80, -3.60, -3.40, -3.20,
    -3.00, -2.80, -2.60, -2.40, -2.20, -2.00, -1.80, -1.60, -1.40, -1.20,
    -1.00, -0.80, -0.60, -0.40, -0.20,  0.00,  0.20,  0.40,  0.60,  0.80,
     1.00,  1.20,  1.40,  1.60,  1.80,  2.00
])

temp_table = np.array([
    4143, 4169, 4198, 4230, 4263, 4297, 4332, 4368, 4402, 4435,
    4467, 4500, 4535, 4571, 4612, 4658, 4711, 4773, 4849, 4944,
    5066, 5221, 5420, 5676, 6000, 6412, 6919, 7500, 8084, 8619,
    9086, 9478, 9797, 10060, 10290, 10493
])

pressure_pa_table = np.array([
    2.34E+01, 3.10E+01, 4.10E+01, 5.41E+01, 7.13E+01, 9.34E+01, 1.22E+02, 1.58E+02, 2.04E+02, 2.63E+02,
    3.38E+02, 4.33E+02, 5.55E+02, 7.10E+02, 9.08E+02, 1.16E+03, 1.49E+03, 1.92E+03, 2.46E+03, 3.14E+03,
    4.01E+03, 5.08E+03, 6.40E+03, 7.97E+03, 9.72E+03, 1.14E+04, 1.28E+04, 1.39E+04, 1.49E+04, 1.58E+04,
    1.67E+04, 1.76E+04, 1.86E+04, 1.97E+04, 2.09E+04, 2.25E+04
])

pe_pa_table = np.array([
    2.09E-03, 2.73E-03, 3.58E-03, 4.69E-03, 6.11E-03, 7.94E-03, 1.03E-02, 1.32E-02, 1.69E-02, 2.16E-02,
    2.74E-02, 3.48E-02, 4.41E-02, 5.59E-02, 7.12E-02, 9.12E-02, 1.17E-01, 1.53E-01, 2.00E-01, 2.67E-01,
    3.61E-01, 5.02E-01, 7.44E-01, 1.27E+00, 2.64E+00, 6.71E+00, 1.98E+01, 5.69E+01, 1.33E+02, 2.52E+02,
    4.08E+02, 5.90E+02, 7.82E+02, 9.84E+02, 1.21E+03, 1.46E+03
])

# Pick the layer closest to log tau = 0 for testing
layer_idx = np.where(log_tau_table == 0.0)[0][0]
log_tau = log_tau_table[layer_idx]
T_layer = temp_table[layer_idx] * u.K
P_layer = pressure_pa_table[layer_idx] * u.Pa
Pe_layer = pe_pa_table[layer_idx] * u.Pa

print(f"Using layer with log tau = {log_tau:.2f}")
print(f"  Tabulated T  = {T_layer.value:.1f} K")
print(f"  Tabulated P  = {P_layer.value:.3e} Pa")
print(f"  Tabulated Pe = {Pe_layer.value:.3e} Pa")

# === 1. Call ns_from_P_T ===
ne, ns, mu, Ui, rho = eos.ns_from_P_T(P_layer, T_layer)
print("\nns_from_P_T results for default A_O_log=8.69:")
print("  n_e  =", ne)
print("  rho  =", rho)
print("  mu   =", mu)
print("  ns[H^0, H^+, H^-, He^0, He^+, He^2+] =", ns[:6])

# === 2. Compare with simple n_e = P_e / (k_B T) ===
Pe_cgs = Pe_layer.to(u.dyne/u.cm**2).value
kT_cgs = (k_B * T_layer).to(u.erg).value
n_e_from_table = Pe_cgs / kT_cgs
print("\nElectron density from table (Pe/kT):")
print(f"  n_e_table = {n_e_from_table:.3e} cm^-3")
print(f"  n_e_eos   = {ne.cgs.value:.3e} cm^-3")

ratio = ne.cgs.value / n_e_from_table
print(f"  ratio (EOS / table) = {ratio:.3f}")

# === 3. Verify ideal gas consistency ===
rho_q = rho * u.g / u.cm**3
P_pred = (rho_q / (mu * u.u) * k_B * T_layer).to(u.dyne/u.cm**2).value

print("\nIdeal gas check:")
print(f"  Input gas pressure      = {P_layer.to(u.dyne/u.cm**2).value:.3e} dyn/cm^2")
print(f"  Reconstructed pressure  = {P_pred:.3e} dyn/cm^2")

# === 4. Vary oxygen abundance ===
O_index = 4
start = 3 * O_index
print("\nVarying oxygen abundance:")
for A_O_log in [8.39, 8.69, 8.99]:
    ne_var, ns_var, mu_var, Ui_var, rho_var = eos.ns_from_P_T(P_layer, T_layer, A_O_log=A_O_log)
    O_slice = ns_var[start:start+3].cgs.value
    print(f"  A_O_log = {A_O_log:.2f}")
    print(f"    n_e  = {ne_var.cgs.value:.3e} cm^-3")
    print(f"    rho  = {rho_var:.3e} g/cm^3")
    print(f"    mu   = {mu_var:.4f}")
    print(f"    O neutral/ion1/ion2 = {O_slice}")
