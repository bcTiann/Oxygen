import saha_eos as eos
import astropy.units as u

# Define the physical conditions we want to test
# Condition 1: Cool star photosphere (like a red dwarf)
temp_cool = 4000 * u.K
rho_cool = 1e-7 * u.g / u.cm**3

# Condition 2: Hot star photosphere (like the one in the script's example)
temp_hot = 15000 * u.K
rho_hot = 1e-7 * u.g / u.cm**3 # Using the same density for comparison


# --- Calculations ---
# Use the function we studied to get the number densities
n_e_cool, ns_cool, mu_cool, Ui_cool = eos.ns_from_rho_T(rho_cool, temp_cool)
n_e_hot, ns_hot, mu_hot, Ui_hot = eos.ns_from_rho_T(rho_hot, temp_hot)


# --- Print the results in a readable way ---
# The total number of hydrogen nuclei is the sum of neutral (ns[0]) and ionized (ns[1])
total_H_cool = ns_cool[0] + ns_cool[1]
total_H_hot = ns_hot[0] + ns_hot[1]

# Calculate the ionization fraction for hydrogen
ion_frac_cool = ns_cool[1] / total_H_cool
ion_frac_hot = ns_hot[1] / total_H_hot


print(f"--- At a cool temperature of {temp_cool} ---")
print(f"Hydrogen Ionization Fraction: {ion_frac_cool:.4f}")
print(f"Electron density: {n_e_cool:.2e}")
print("\n" + "="*40 + "\n")
print(f"--- At a hot temperature of {temp_hot} ---")
print(f"Hydrogen Ionization Fraction: {ion_frac_hot:.4f}")
print(f"Electron density: {n_e_hot:.2e}")
