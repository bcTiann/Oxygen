# Let's compute the structure of an atmosphere, using the folowing modules and assumptions
# 1) A grey atmosphere and hydrostatic equilibrium (analytic)
# 2) An equation of state using the Saha equation: tabulated in rho_Ui_mu_ns_ne.fits
# 3) Opacities computed using the methods in opac.py: tabulated in Ross_Planck_opac.fits
#
# For speed, units are:
# - Length: cm
# - Mass: g
# - Time: s
# - Temperature: K
# - Frequency: Hz
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
import astropy.units as u
import astropy.constants as const
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import opac
from scipy.special import expn
import saha_eos as eos
from scipy.interpolate import interp1d
from lambda_matrix import lambda_matrix
import Collision_rate as cr


OI_77719 = {"E_l": 9.1460911*u.eV, "E_u": 10.7409314*u.eV, "f": 4.68e-1, "J_l": 2, "J_u": 3, "A_ul": 3.69e7/u.s}
OI_TRIPLET = [
    {"E_l": 9.1460911*u.eV, "E_u": 10.7409314*u.eV, "f": 4.68e-1, "J_l": 2, "J_u": 3, "A_ul": 3.69e7/u.s},
    {"E_l": 9.1460911*u.eV, "E_u": 10.7404756*u.eV, "f": 3.35e-1, "J_l": 2, "J_u": 2, "A_ul": 3.69e7/u.s},
    {"E_l": 9.1460911*u.eV, "E_u": 10.7402250*u.eV, "f": 2.01e-1, "J_l": 2, "J_u": 1, "A_ul": 3.69e7/u.s},
]

# trial_abundances = np.array([8.00, 8.69, 8.95, 9.55, 10.55])
trial_abundances = np.array([6.0, 7.39, 8.39, 8.69, 9.1, 9.5, 10.0, 10.55, 11.0, 11.5,  12.0, 13.0])
# trial_abundances = np.array([8.95])
results_list = []


# Load the opacity table for Rosseland mean.
f_opac = pyfits.open('Ross_Planck_opac.fits')
kappa_bar_Ross = f_opac['kappa_Ross [cm**2/g]'].data
#plt.loglog(tau_grid, f_kappa_bar_Ross((np.log10(Ps), Ts)))

# Construct the log(P) and T vectors. 
h = f_opac[0].header
T_grid = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
Ps_log10 = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']


# Build an interpolator that retrieves the Rosseland mean opacity for arbitrary (log10 P, T).
f_kappa_bar_Ross = RegularGridInterpolator((Ps_log10, T_grid), kappa_bar_Ross)



def planck_nu(T, wavelength_nm):
    """Return the Planck function B_nu for temperature array T (K) at wavelength_nm (nm)."""
    h_val = 6.626e-34  # J*s, Planck constant
    c_val = 3.0e8      # m/s, speed of light
    k_B_val = 1.38e-23 # J/K, Boltzmann constant

    # Convert wavelength from nm to frequency in Hz
    # This handles T being an array and wavelength_nm being a scalar
    nu = c_val / (wavelength_nm * 1e-9)
    T_arr = np.asarray(T) # Ensure T is an array
    exponent = h_val * nu / (k_B_val * T_arr)

    # Create a result array of zeros first.
    B_nu = np.zeros_like(T_arr, dtype=float)

    # Only calculate for exponents that won't overflow
    valid_indices = exponent < 700 

    # Perform calculation only on the valid elements
    B_nu[valid_indices] = (2 * h_val * nu**3 / c_val**2) / (np.exp(exponent[valid_indices]) - 1)

    # If the original T was a single number, return a single number
    if B_nu.ndim == 0:
        return B_nu.item()
    return B_nu


# Solar T, P, and density profiles digitised from Asplund et al. (2004) Table 1.
log_tau_table = np.array([
    -5.00, -4.80, -4.60, -4.40, -4.20, -4.00, -3.80, -3.60, -3.40, -3.20, 
    -3.00, -2.80, -2.60, -2.40, -2.20, -2.00, -1.80, -1.60, -1.40, -1.20, 
    -1.00, -0.80, -0.60, -0.40, -0.20, 0.00, 0.20, 0.40, 0.60, 0.80, 
    1.00, 1.20, 1.40, 1.60, 1.80, 2.00
])

temp_table = np.array([
    4143, 4169, 4198, 4230, 4263, 4297, 4332, 4368, 4402, 4435, 
    4467, 4500, 4535, 4571, 4612, 4658, 4711, 4773, 4849, 4944, 
    5066, 5221, 5420, 5676, 6000, 6412, 6919, 7500, 8084, 8619, 
    9086, 9478, 9797, 10060, 10290, 10493
])

# Gas pressure P_gas in Pascal [Pa] from the paper 
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


# Convert electron pressure to cgs units (dyn/cm^2)
pe_cgs_table = pe_pa_table * 10.0

rho_kg_m3_table = np.array([
    8.28E-13, 1.09E-12, 1.42E-12, 1.88E-12, 2.47E-12, 3.21E-12, 4.16E-12, 5.36E-12, 6.89E-12, 8.82E-12,
    1.13E-11, 1.44E-11, 1.82E-11, 2.21E-11, 2.95E-11, 3.75E-11, 4.77E-11, 6.05E-11, 7.65E-11, 9.61E-11,
    1.20E-10, 1.47E-10, 1.79E-10, 2.13E-10, 2.46E-10, 2.70E-10, 2.82E-10, 2.85E-10, 2.83E-10, 2.81E-10,
    2.79E-10, 2.79E-10, 2.82E-10, 2.87E-10, 2.96E-10, 3.09E-10
])
# 1 kg/m^3 = 1000 g / (100 cm)^3 = 1000 / 1,000,000 g/cm^3 = 0.001 g/cm^3
rho_cgs_table = rho_kg_m3_table * 0.001

# Pa convert to dyn/cm^2
pressure_cgs_table = pressure_pa_table * 10.0

# Create interpolation functions that map log10 optical depth to the tabulated quantities.
f_temp_from_tau = interp1d(log_tau_table, temp_table, kind='cubic')
f_pressure_from_tau = interp1d(log_tau_table, pressure_cgs_table, kind='cubic')
f_rho_from_tau = interp1d(log_tau_table, rho_cgs_table, kind='cubic')

tau_grid = np.concatenate((np.arange(3)/3*1e-3,np.logspace(-3,1.3,30)))
tau_grid = tau_grid[tau_grid > 0]
log_tau_grid = np.log10(tau_grid)
valid_indices = (log_tau_grid >= log_tau_table.min()) & (log_tau_grid <= log_tau_table.max())
tau_grid = tau_grid[valid_indices]
Ts = f_temp_from_tau(log_tau_grid[valid_indices])
Ps = f_pressure_from_tau(log_tau_grid[valid_indices])




# Interpolate onto the tau grid
kappa_bars = f_kappa_bar_Ross((np.log10(Ps), Ts))

rhos = f_rho_from_tau(log_tau_grid[valid_indices])

# First, lets plot a continuum spectrum
# wave = np.linspace(775.0, 780.0, 10000) * u.nm  # Wavelength in nm
# flux = np.zeros_like(wave)  # Initialize flux array
nu0 = const.c.to('cm/s').value / (1000 * 1e-7)  # Start Freq @ 1000 nm
dlnu = 1e-6  # Logarithmic frequency step, very high resolution
Nnu = 400000 # Number of frequency points, covers a large range

# Compute the frequency and wavelength arrays based on the grid settings
nu = nu0 * np.exp(dlnu * np.arange(Nnu)) # in Hz
wave_nm = (const.c / (nu * u.Hz)).to(u.nm).value

# Create the main array to store the final opacities
# It has dimensions (Number of Frequencies, Number of Atmospheric Layers)
kappa_nu_bars = np.empty((Nnu, len(tau_grid)))

print(f"Spectral grid setup complete. Calculating for {Nnu} frequency points.")
print(f"This will cover a wavelength range from ~{wave_nm.min():.1f} nm to ~{wave_nm.max():.1f} nm.")

# Just like in grey_flux.py, but in frequency 
planck_C1 = (2*const.h*const.c**2/(1*u.um)**5).si.value
planck_C2 = (const.h*const.c/(1*u.um)/const.k_B/(1*u.K)).si.value

# Planck function, like in grey_flux.py
def Blambda_SI(wave_um, T):
    """
    Planck function in cgs units.
    """
    return planck_C1/wave_um**5/(np.exp(planck_C2/wave_um/T)-1)

def compute_H(wave, Ts, tau_grid, kappa_nu_bars, kappa_bars):
    Hlambda = np.zeros(len(wave))  # Initialize H array
    # Compute the flux for each wavelength
    for i, w in enumerate(wave):
        # Now we need S(tau_nu), i.e. B(tau_nu(tau))
        tau_nu  = cumulative_trapezoid(kappa_nu_bars[i]/kappa_bars, x=tau_grid, initial=0)
        wave_um = w.to(u.um).value
        Slambda = Blambda_SI(wave_um, Ts)
        Hlambda[i] = 0.5*(Slambda[0]*expn(3,0) + \
		np.sum((Slambda[1:]-Slambda[:-1])/(tau_nu[1:]-tau_nu[:-1])*\
			(expn(4,tau_nu[:-1])-expn(4,tau_nu[1:]))))
    return Hlambda

# S_matrix has shape (num_wavelengths, num_layers).
def compute_H_NLTE(wave, Ts, tau_grid, kappa_nu_bars, kappa_bars, S_matrix):
    Hlambda = np.zeros(len(wave))
    for i, w in enumerate(wave):
        tau_nu  = cumulative_trapezoid(kappa_nu_bars[i]/kappa_bars, x=tau_grid, initial=0)
        
        # Unlike the LTE solver, use the NLTE source function provided in S_matrix instead of B_nu.
        Slambda = S_matrix[i, :] 
        
        Hlambda[i] = 0.5*(Slambda[0]*expn(3,0) + \
		np.sum((Slambda[1:]-Slambda[:-1])/(tau_nu[1:]-tau_nu[:-1])*\
			(expn(4,tau_nu[:-1])-expn(4,tau_nu[1:]))))
    return Hlambda


# Reference equivalent widths (pm) from Asplund et al. (2004) Table 3 for the O I triplet.
table_W_lambda_list_pm = [7.12, 6.18, 4.88]
all_normalized_flux_lte = {}
all_normalized_flux_nlte = {}

Lambda_mat = lambda_matrix(tau_grid)
identity_mat = np.identity(len(tau_grid))

# Each iteration recomputes the full opacity and emergent spectrum for one oxygen abundance.
for A_O_log in trial_abundances:
    print(f"{'='*60}")
    print(f"--- Running model for log A(O) = {A_O_log:.2f} ---")
    print(f"{'='*60}")

    print("\n--- Step 1: Calculating the physically correct combined spectrum ---")
    # kappa_total_nu = np.empty((len(wave), len(tau_grid)))
    # nu = (const.c / wave).to(u.Hz).value

    epsilon_profile = np.zeros(len(tau_grid))

    for i, (T, P, _) in enumerate(zip(Ts, Ps, rhos)):

        try:
            # We must pass astropy units to this function.
            P_with_units = P * u.dyne / u.cm**2
            T_with_units = T * u.K

            # This is the new core of our EOS calculation. It's a solver.
            n_e, ns, _mu, _Ui, rho_check = eos.ns_from_P_T(P_with_units, T_with_units, A_O_log=A_O_log)

            # For later use, get the unitless values
            n_e_val = n_e.cgs.value
            ns_val = ns.cgs.value
            n_H_val = ns_val[0] + ns_val[1] + ns_val[2]

            n_e_q = n_e_val * u.cm**-3
            n_H_q = n_H_val * u.cm**-3

            weigthed_eps_sum = 0.0
            f_sum = 0.0

            for comp in OI_TRIPLET:
                C_ul_val = cr.derive_C_ul(
                        T_with_units,
                        n_e_q,
                        n_H_q,
                        comp["E_l"],
                        comp["E_u"],
                        comp["f"],
                        comp["J_l"],
                        comp["J_u"]
                ) * u.s**-1

                epsilon_k = (C_ul_val / (comp["A_ul"] + C_ul_val)).value
                weigthed_eps_sum += comp["f"] * epsilon_k
                f_sum += comp["f"]

            epsilon_profile[i] = weigthed_eps_sum / f_sum
            

            rho_check_val = rho_check # rho_check is already a float in g/cm^3


            # a) Calculate Continuum Opacity
            kappa_continuum = opac.kappa_cont(nu, T, nHI=ns_val[0], nHII=ns_val[1], nHm=ns_val[2], ne=n_e_val)

            # b) Calculate Opacity from all Weak Lines
            # We pass a microturbulence value, a key piece of the new physics.
            kappa_weak_lines = opac.weak_line_kappa(nu0, dlnu, Nnu, T, ns_vector=ns_val, microturb=1.5)

            # c) Calculate Opacity from all Strong Lines
            # This function includes detailed broadening physics (microturbulence, Stark, etc.)
            kappa_strong_lines = opac.strong_line_kappa(nu0, dlnu, Nnu, T, n_e=n_e_val, ns_vector=ns_val, microturb=1.5)

            # d) Sum all opacity sources (they are all in units of cm^-1)
            total_kappa_per_volume = kappa_continuum + kappa_weak_lines + kappa_strong_lines

            # e) Convert to mass absorption coefficient (cm^2/g) using the SELF-CONSISTENT rho_check
            # This is the final, physically accurate opacity for this atmospheric layer.
            kappa_nu_bars[:, i] = total_kappa_per_volume / rho_check_val

            # --- CHECKPOINT: Print values to verify ---
            # We print every 5 layers to avoid too much output.
            if i % 5 == 0:

                current_log_tau = log_tau_grid[valid_indices][i]
                print(f"log(tau)={current_log_tau:.2f} | T={T:.0f}K | log(P)={np.log10(P):.2f} -> Solved n_e={n_e_val:.2e} cm^-3, Solved rho={rho_check_val:.2e} g/cm^3")

        except Exception as e:
            current_log_tau = log_tau_grid[valid_indices][i]
            raise RuntimeError(f'layer failure at log(tau)={current_log_tau:.2f}, T={T:.0f}K, P={P:.2e}: {e}')




    # Restrict the NLTE solve to 776-779 nm around the O I triplet.
    nlte_indices = np.where((wave_nm >= 776) & (wave_nm <= 779))[0]
    nlte_wave_nm = wave_nm[nlte_indices]

    epsilon_vec = np.clip(epsilon_profile, 1e-6, 1.0)
    print("="*40)
    print(f"epsilon_vec = {epsilon_vec}")
    print("="*40)
    epsilon_mat = np.diag(epsilon_vec)
    
    # S_matrix stores the depth-dependent NLTE source function for each wavelength.
    S_matrix = np.empty((len(nlte_wave_nm), len(tau_grid)))

    # Solve the simplified Lambda-operator problem for every wavelength in the window.
    for i in range(len(nlte_wave_nm)):
        current_wave = nlte_wave_nm[i]
        
        # Adopt a constant photon destruction probability epsilon to mimic scattering dominance in the core.
        # epsilon_val = 0.01
        # epsilon_vec = np.full_like(tau_grid, epsilon_val)
        # epsilon_mat = np.diag(epsilon_vec)
        
        B_vector = planck_nu(Ts, current_wave)

        # Assemble A = I - (1-epsilon)Lambda and b = epsilon*B and solve for S.
        A_mat = identity_mat - np.dot((identity_mat - epsilon_mat), Lambda_mat)
        b_vector = np.dot(epsilon_mat, B_vector)
        
        S_vector = np.linalg.solve(A_mat, b_vector)
        
        S_matrix[i, :] = S_vector

    print("NLTE Source Function calculation complete!")

    H_nlte = compute_H_NLTE(
        nlte_wave_nm * u.nm, 
        Ts, 
        tau_grid, 
        kappa_nu_bars[nlte_indices, :], 
        kappa_bars, 
        S_matrix
    )

    print("\nStarting NLTE spectrum analysis...")

    print("Starting LTE spectrum analysis...")
    H_lte = compute_H(
        nlte_wave_nm * u.nm,
        Ts,
        tau_grid,
        kappa_nu_bars[nlte_indices, :],
        kappa_bars
    )



    print("\nOpacity calculation for all layers complete. Now computing the emergent spectrum...")
    print("This may take a moment...")

    print("Spectrum calculation complete!")

    # Apply macroturbulent, rotational, and instrumental broadening.
    print("Applying final broadening (macroturbulence, rotation, instrumental)...")

    # Define broadening parameters in km/s. These are typical values for the Sun.
    macro_v = 2.0  # Macroturbulence velocity
    rot_v = 1.9    # Rotational velocity of the Sun
    inst_v = 2.5   # Instrumental broadening (simulating a spectrograph's resolution)

    # Total broadening width (added in quadrature)
    width_v = np.sqrt(macro_v**2 + rot_v**2 + inst_v**2)

    # Convert the velocity width to a Gaussian width in nm at 777 nm.
    width_nm = width_v / (const.c.to(u.km/u.s).value) * 777.0

    # The wavelength grid is logarithmic, so compute the local spacing near 777 nm.
    idx_777 = np.argmin(np.abs(nlte_wave_nm - 777.0))
    dx_nm_local = np.abs(nlte_wave_nm[idx_777] - nlte_wave_nm[idx_777 - 1])
    width_pixels = width_nm / dx_nm_local


    x_kernel_range = int(5 * width_pixels)
    x_kernel = np.arange(-x_kernel_range, x_kernel_range + 1)
    gaussian_kernel = np.exp(-x_kernel**2 / (2 * width_pixels**2))
    gaussian_kernel /= np.sum(gaussian_kernel) # Normalize the kernel
    



    # Convolve the high-resolution spectrum with the kernel to get the final broadened spectrum
    H_broadened_nlte = np.convolve(H_nlte, gaussian_kernel, mode='same')
    H_broadened_lte = np.convolve(H_lte, gaussian_kernel, mode='same')

    ########## ########## ########## ########## ########## ########## ##########
    # --- Raw (un-normalized) flux comparison in the NLTE window ---
    line_centers_nm = np.array([777.1944, 777.4166, 777.5388])

    # Indices nearest to each line center
    center_ix = [np.argmin(np.abs(nlte_wave_nm - lc)) for lc in line_centers_nm]

    # Global minima in the plotted window (raw, broadened)
    nlte_min_flux = H_broadened_nlte.min()
    lte_min_flux  = H_broadened_lte.min()
    nlte_min_wave = nlte_wave_nm[H_broadened_nlte.argmin()]
    lte_min_wave  = nlte_wave_nm[H_broadened_lte.argmin()]

    print("\n--- Raw broadened flux (no normalization) ---")
    print(f"Global min NLTE: {nlte_min_flux:.6f} at {nlte_min_wave:.4f} nm")
    print(f"Global min  LTE: {lte_min_flux:.6f} at {lte_min_wave:.4f} nm")

    # Values at the nominal line centers
    for lc, ix in zip(line_centers_nm, center_ix):
        print(f"{lc:.4f} nm -> NLTE {H_broadened_nlte[ix]:.6f} | LTE {H_broadened_lte[ix]:.6f}")

    # Plot raw fluxes to visually compare depth without any continuum fitting
    plt.figure(figsize=(10, 5))
    plt.plot(nlte_wave_nm, H_broadened_nlte, label=f'NLTE raw (log A(O)={A_O_log:.2f})', color='tab:purple')
    plt.plot(nlte_wave_nm, H_broadened_lte,  label=f'LTE  raw (log A(O)={A_O_log:.2f})', color='tab:orange', alpha=0.85)

    for lc in line_centers_nm:
        plt.axvline(lc, color='k', ls=':', alpha=0.3)

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('H (raw, broadened)')
    plt.title('Raw broadened flux (no normalization)')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'raw_broadened_flux_{A_O_log:.2f}.png', dpi=200)
    plt.show()
    # --- end raw flux check ---
 ########## ########## ########## ########## ##########

    print("Broadening complete.")

    # Measure equivalent widths and compare them with observations.
    print("\n--- Final Analysis: Equivalent Width Comparison ---")

    # Step 1: normalise the broadened spectrum using continuum windows on each side of the triplet.
    continuum_indices = np.where(
        ((nlte_wave_nm >= 776.5) & (nlte_wave_nm <= 777.0)) | 
        ((nlte_wave_nm >= 778.0) & (nlte_wave_nm <= 778.5))
    )[0]

    # Fit a straight line to the continuum samples and divide it out to get unit continuum.
    p_fit_nlte  = np.polyfit(nlte_wave_nm[continuum_indices], H_broadened_nlte[continuum_indices], 1)
    continuum_fit_nlte = np.poly1d(p_fit_nlte)
    flux_normalized_nlte = H_broadened_nlte / continuum_fit_nlte(nlte_wave_nm)

    p_fit_lte = np.polyfit(nlte_wave_nm[continuum_indices], H_broadened_lte[continuum_indices], 1)
    continuum_fit_lte = np.poly1d(p_fit_lte)
    flux_normalized_lte = H_broadened_lte / continuum_fit_lte(nlte_wave_nm)



    # Step 2: integrate the equivalent width of each component of the O I triplet.
    # Central wavelengths of the three O I triplet lines in nm
    line_centers_nm = [777.1944, 777.4166, 777.5388]
    # Observed EWs from Asplund's paper in pm (pico-meters)
    table_W_lambda_list_pm = [7.12, 6.18, 4.88]

    # Define integration boundaries for each line (in nm)
    boundaries = [
        (777.05, 777.30),   # Range for the 1st line
        (777.30, 777.48),   # Range for the 2nd line
        (777.48, 777.65)    # Range for the 3rd line
    ]

    # Loop through each line to calculate and compare its EW

    model_EW_results_nlte = []
    model_EW_results_lte = [] 
    for i in range(len(line_centers_nm)):
        table_W = table_W_lambda_list_pm[i]
        min_wave, max_wave = boundaries[i]
        
        # Find the indices of our wavelength array that fall within the boundaries
        line_indices = np.where((nlte_wave_nm >= min_wave) & (nlte_wave_nm < max_wave))[0]
        
        # Compute the local wavelength step (grid is logarithmic).
        d_wave_nm = np.abs(nlte_wave_nm[line_indices][1:] - nlte_wave_nm[line_indices][:-1])

        # Approximate the integral sum[(1 - F) d_lambda] using the left-hand rule.

        model_W_lambda_nm_nlte = np.sum((1.0 - flux_normalized_nlte[line_indices][:-1]) * d_wave_nm)
        model_EW_results_nlte.append(model_W_lambda_nm_nlte * 1000)
        
        # Convert from nanometers (nm) to pico-meters (pm)
        model_W_lambda_pm = model_W_lambda_nm_nlte * 1000
    

        model_W_lambda_nm_lte = np.sum((1.0 - flux_normalized_lte[line_indices][:-1]) * d_wave_nm)
        model_EW_results_lte.append(model_W_lambda_nm_lte * 1000)
        

        # Print the final comparison for this line
        print(f"\nLine {line_centers_nm[i]:.4f} nm (A(O)={A_O_log:.2f}):")
        print(f"  NLTE Model EW: {model_EW_results_nlte[i]:.2f} pm")
        print(f"  LTE Model EW:  {model_EW_results_lte[i]:.2f} pm")
        print(f"  Observed EW:   {table_W_lambda_list_pm[i]:.2f} pm")

        diff = (model_W_lambda_pm - table_W) / table_W * 100
        if diff > 0:
            print(f"  --> Model line is {diff:.1f}% STRONGER than observed.")
        else:
            print(f"  --> Model line is {abs(diff):.1f}% WEAKER than observed.")

        all_normalized_flux_lte[A_O_log] = flux_normalized_lte
        all_normalized_flux_nlte[A_O_log] = flux_normalized_nlte

    # Store the results for this abundance value
    results_list.append({
        'abundance': A_O_log,
        'model_EW_pm_nlte': model_EW_results_nlte,
        'model_EW_pm_lte': model_EW_results_lte
    })

# --- NEW: Final Analysis and Plotting of All Results ---
print(f"{'='*60}")
print("--- Final Analysis of All Abundance Trials ---")
print(f"{'='*60}")

# Observed EWs from Asplund's paper in pm
observed_EW = np.array([7.12, 6.18, 4.88])
line_centers_nm = np.array([777.1944, 777.4166, 777.5388])
best_fit_abundances_nlte = []
best_fit_abundances_lte = []

print("\nRendering LTE and NLTE spectra for every trial abundance...")

plt.figure(figsize=(14, 8))

# Use a colour map so LTE/NLTE pairs share the same base colour.
colors = plt.cm.viridis(np.linspace(0, 1, len(trial_abundances)))

for i, abundance in enumerate(trial_abundances):
    flux_nlte = all_normalized_flux_nlte[abundance]
    flux_lte = all_normalized_flux_lte[abundance]

    plt.plot(nlte_wave_nm, flux_nlte, 
             color=colors[i],
             linestyle='-',
             label=f'NLTE log A(O) = {abundance:.2f}')

    plt.plot(nlte_wave_nm, flux_lte, 
             color=colors[i],
             linestyle='--',
             label=f'LTE  log A(O) = {abundance:.2f}')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Normalized Flux')
plt.title('LTE vs NLTE spectra as a function of oxygen abundance')

plt.xlim(777.1, 777.3)

plt.legend(title="Line Type & Abundance", fontsize='small')

plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('spectrum_all_abundances_LTE_NLTE.png', dpi=300)
print("Saved LTE/NLTE comparison plot to spectrum_all_abundances_LTE_NLTE.png")

plt.show()

plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red']

# For each line, plot its curve of growth and find the best fit
for i in range(len(line_centers_nm)):
    abundances = [res['abundance'] for res in results_list]
    
    model_ews_nlte = [res['model_EW_pm_nlte'][i] for res in results_list]
    model_ews_lte = [res['model_EW_pm_lte'][i] for res in results_list]

    plt.plot(abundances, model_ews_nlte, 'o-', label=f'NLTE Model {line_centers_nm[i]:.2f} nm', color=colors[i])
    plt.plot(abundances, model_ews_lte, 'x--', label=f'LTE Model {line_centers_nm[i]:.2f} nm', color=colors[i], alpha=0.7)

    plt.axhline(y=observed_EW[i], linestyle=':', color=colors[i], label=f'Observed {line_centers_nm[i]:.2f} nm')

    fit_ab_nlte = np.interp(observed_EW[i], model_ews_nlte, abundances)
    best_fit_abundances_nlte.append(fit_ab_nlte)
    
    fit_ab_lte = np.interp(observed_EW[i], model_ews_lte, abundances)
    best_fit_abundances_lte.append(fit_ab_lte)
    
    print(f"Line {line_centers_nm[i]:.2f} nm -> Best-fit NLTE Abundance: {fit_ab_nlte:.3f}, Best-fit LTE Abundance: {fit_ab_lte:.3f}")

# Summarise the abundance inferences.
avg_nlte_abundance_fit = np.mean(best_fit_abundances_nlte)
print(f"\nAverage best-fit NLTE abundance (before correction): {avg_nlte_abundance_fit:.3f}")
nlte_correction = -0.24 # From Asplund et al. (2004) Table 2, for 777.41 nm line
final_nlte_abundance = avg_nlte_abundance_fit
print(f"Final NLTE Oxygen Abundance: log A(O) = {final_nlte_abundance:.3f}")

# Cross-check: infer abundance from the LTE curve of growth and apply the literature NLTE correction.
avg_lte_abundance_fit = np.mean(best_fit_abundances_lte)
print(f"\nAverage best-fit LTE abundance: {avg_lte_abundance_fit:.3f}")
corrected_lte_abundance = avg_lte_abundance_fit + nlte_correction
print(f"LTE abundance + paper's NLTE correction ({nlte_correction:.2f} dex) = {corrected_lte_abundance:.3f}")
print(f"(This should be close to your direct NLTE result of {final_nlte_abundance:.3f})")


plt.title('Equivalent Width vs. Oxygen Abundance')
plt.xlabel('Log Oxygen Abundance [log A(O)]')
plt.ylabel('Equivalent Width [pm]')
plt.grid(True)
plt.legend()
plt.savefig('abundance_fit.png', dpi=300)
print("Curve of growth plot saved as abundance_fit.png")
plt.show()
