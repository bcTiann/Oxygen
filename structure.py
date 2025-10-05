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
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import cumulative_trapezoid
import astropy.units as u
import astropy.constants as const
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import opac
from scipy.special import expn
import saha_eos as eos
from scipy.interpolate import interp1d
from lambda_matrix import lambda_matrix
import Collision_rate as cr
from nlte_branch import spectrum_nlte, H_nlte_at_nu
import matplotlib
matplotlib.use("Agg")
OI_77719 = {"E_l": 9.1460911*u.eV, "E_u": 10.7409314*u.eV, "f": 4.68e-1, "J_l": 2, "J_u": 3, "A_ul": 3.69e7/u.s}
OI_TRIPLET = [
    {"E_l": 9.1460911*u.eV, "E_u": 10.7409314*u.eV, "f": 4.68e-1, "J_l": 2, "J_u": 3, "A_ul": 3.69e7/u.s},
    {"E_l": 9.1460911*u.eV, "E_u": 10.7404756*u.eV, "f": 3.35e-1, "J_l": 2, "J_u": 2, "A_ul": 3.69e7/u.s},
    {"E_l": 9.1460911*u.eV, "E_u": 10.7402250*u.eV, "f": 2.01e-1, "J_l": 2, "J_u": 1, "A_ul": 3.69e7/u.s},
]


DEFAULT_ABUNDANCES = np.array([8.59, 9.50, 10.00])
# trial_abundances = np.array([8.59, 8.69])
# ---- Tunable configuration ----
BROADENING = {
    "macro": 2.0,   # km/s  # Macroturbulence velocity
    "rot": 1.9,     # km/s # Rotational velocity of the Sun
    "inst": 2.5,    # km/s # Instrumental broadening (simulating a spectrograph's resolution)
}

CONTINUUM_WINDOWS = [
    (776.2, 777.0),
    (778.2, 778.6),
]

LINE_BOUNDS = [
    (777.05, 777.30),
    (777.30, 777.48),
    (777.48, 777.65),
]
# --------------------------------




results_list = []


def continuum_indices_from_windows(wave_nm, windows):
    """Return indices of wave_nm that fall inside any (min,max) window.""" 
    mask = np.zeros_like(wave_nm, dtype=bool)
    for w_min, w_max in windows:
        mask |= (wave_nm >= w_min) & (wave_nm <= w_max)
    return np.where(mask)[0]


def normalize_flux(wave_nm, flux, indices):
    """Fit a line to the continuum points and normalize the whole spectrum."""
    fit = np.poly1d(np.polyfit(wave_nm[indices], flux[indices], 1))
    return flux / fit(wave_nm)


def compute_equivalent_widths(wave_nm, flux, bounds, precomputed_indices=None):
    """Return equivalent widths (pm) and the indices used for each line window."""
    ew_list = []
    indices_list = []

    for i, (w_min, w_max) in enumerate(bounds):
        if precomputed_indices is not None:
            line_indices = precomputed_indices[i]
        else:
            line_indices = np.where((wave_nm >= w_min) & (wave_nm < w_max))[0]

        indices_list.append(line_indices)

        wave_segment = wave_nm[line_indices]
        flux_segment = flux[line_indices]

        sort_idx = np.argsort(wave_segment)
        wave_sorted = wave_segment[sort_idx]
        flux_sorted = flux_segment[sort_idx]

        ew_nm = np.trapz(1.0 - flux_sorted, wave_sorted)
        ew_list.append(ew_nm * 1000.0)

    return np.array(ew_list), indices_list


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Synthesize NLTE spectra for the O I triplet.")
    parser.add_argument(
        "--abundances",
        type=float,
        nargs='+',
        help="List of log A(O) abundances to evaluate in a single run",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("tables"),
        help="Directory storing/generated EOS and opacity tables (default: ./tables)",
    )
    parser.add_argument(
        "--regenerate-tables",
        action="store_true",
        help="Force regeneration of EOS/opac tables even if files already exist.",
    )
    parser.add_argument(
        "--show-table-plot",
        action="store_true",
        help="Display diagnostic plots when building opacity tables.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable process-based parallel execution (forces serial mode).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker processes when parallel execution is enabled.",
    )
    parser.add_argument(
        "--epsilon-scale",
        type=float,
        default=1.0,
        help="Global multiplicative factor applied to the epsilon profile (default: 1.0).",
    )
    return parser.parse_args()


def ensure_opacity_tables(abundance, tables_dir, regenerate=False, show_plot=False):
    """Ensure EOS and opacity tables exist for the requested abundance."""
    suffix = f"{abundance:.2f}".replace('.', 'p')
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    eos_path = tables_dir / f"saha_eos_AO{suffix}.fits"
    opacity_path = tables_dir / f"Ross_Planck_opac_AO{suffix}.fits"

    if regenerate or not eos_path.exists():
        print(f"Generating EOS table for log A(O) = {abundance:.2f}...")
        eos.P_T_tables(None, None, savefile=str(eos_path), A_O_log=abundance)

    if regenerate or not opacity_path.exists():
        print(f"Generating opacity table for log A(O) = {abundance:.2f}...")
        opac.generate_opacity_tables(
            eos_filename=str(eos_path),
            outfile=str(opacity_path),
            show_plot=show_plot,
            A_O_log=abundance,
        )

    return opacity_path


def load_kappa_bars(opacity_path, Ps, Ts):
    """Load Rosseland mean opacities and interpolate onto the model grid."""
    with pyfits.open(opacity_path) as f_opac:
        kappa_bar_Ross = f_opac['kappa_Ross [cm**2/g]'].data
        header = f_opac[0].header
        T_grid = header['CRVAL1'] + np.arange(header['NAXIS1']) * header['CDELT1']
        Ps_log10 = header['CRVAL2'] + np.arange(header['NAXIS2']) * header['CDELT2']

    interpolator = RegularGridInterpolator((Ps_log10, T_grid), kappa_bar_Ross)
    return interpolator((np.log10(Ps), Ts))


def run_abundance_case(job):
    """Run the full NLTE/LTE calculation for a single oxygen abundance."""
    A_O_log = job['abundance']
    tau_grid = job['tau_grid']
    Ts = job['Ts']
    Ps = job['Ps']
    rhos = job['rhos']
    nu = job['nu']
    wave_nm = job['wave_nm']
    nlte_indices = job['nlte_indices']
    nlte_wave_nm = job['nlte_wave_nm']
    kappa_bars = job['kappa_bars']
    line_centers_nm = job['line_centers_nm']
    continuum_indices = job['continuum_indices']
    line_bounds = job['line_bounds']
    broadening_cfg = job['broadening']
    nu0 = job['nu0']
    dlnu = job['dlnu']
    Nnu = job['Nnu']
    epsilon_scale = job.get('epsilon_scale', 1.0)

    epsilon_profile = np.zeros(len(tau_grid))
    kappa_nu_bars = np.empty((Nnu, len(tau_grid)))

    for i, (T, P, _) in enumerate(zip(Ts, Ps, rhos)):
        P_with_units = P * u.dyne / u.cm**2
        T_with_units = T * u.K

        n_e, ns, _mu, _Ui, rho_check = eos.ns_from_P_T(P_with_units, T_with_units, A_O_log=A_O_log)

        n_e_val = n_e.cgs.value
        ns_val = ns.cgs.value
        n_H_val = ns_val[0]

        n_e_q = n_e_val * u.cm**-3
        n_H_q = n_H_val * u.cm**-3

        weighted_eps_sum = 0.0
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
            weighted_eps_sum += comp["f"] * epsilon_k
            f_sum += comp["f"]

        epsilon_profile[i] = weighted_eps_sum / f_sum

        rho_val = rho_check

        kappa_continuum = opac.kappa_cont(nu, T, nHI=ns_val[0], nHII=ns_val[1], nHm=ns_val[2], ne=n_e_val)
        kappa_weak_lines = opac.weak_line_kappa(nu0, dlnu, Nnu, T, ns_vector=ns_val, microturb=1.5)
        kappa_strong_lines = opac.strong_line_kappa(nu0, dlnu, Nnu, T, n_e=n_e_val, ns_vector=ns_val, microturb=1.5)

        total_kappa_per_volume = kappa_continuum + kappa_weak_lines + kappa_strong_lines

        kappa_nu_bars[:, i] = total_kappa_per_volume / rho_val

    if epsilon_scale != 1.0:
        print(f"Scaling epsilon_profile by factor {epsilon_scale:.3f}")
    epsilon_profile *= epsilon_scale
    epsilon_profile = np.clip(epsilon_profile, 1e-4, 1.0)

    nlte_wave_nm = wave_nm[nlte_indices]

    H_nlte = spectrum_nlte(
        wave_nm=nlte_wave_nm,
        tau_grid=tau_grid,
        kappa_all_nu_bars=kappa_nu_bars[nlte_indices, :],
        kappa_R_bars=kappa_bars,
        T_of_tau=Ts,
        epsilon_profile=epsilon_profile
    )

    H_lte = spectrum_nlte(
        wave_nm=nlte_wave_nm,
        tau_grid=tau_grid,
        kappa_all_nu_bars=kappa_nu_bars[nlte_indices, :],
        kappa_R_bars=kappa_bars,
        T_of_tau=Ts,
        epsilon_profile=np.ones_like(epsilon_profile)
    )

    macro_v = broadening_cfg["macro"]
    rot_v = broadening_cfg["rot"]
    inst_v = broadening_cfg["inst"]

    width_v = np.sqrt(macro_v**2 + rot_v**2 + inst_v**2)
    width_nm = width_v / (const.c.to(u.km/u.s).value) * 777.0

    idx_777 = np.argmin(np.abs(nlte_wave_nm - 777.0))
    dx_nm_local = np.abs(nlte_wave_nm[idx_777] - nlte_wave_nm[idx_777 - 1])
    width_pixels = width_nm / dx_nm_local

    x_kernel_range = int(5 * width_pixels)
    x_kernel = np.arange(-x_kernel_range, x_kernel_range + 1)
    gaussian_kernel = np.exp(-x_kernel**2 / (2 * width_pixels**2))
    gaussian_kernel /= np.sum(gaussian_kernel)

    H_broadened_nlte = np.convolve(H_nlte, gaussian_kernel, mode='same')
    H_broadened_lte = np.convolve(H_lte, gaussian_kernel, mode='same')

    continuum_idx = continuum_indices
    flux_normalized_nlte = normalize_flux(nlte_wave_nm, H_broadened_nlte, continuum_idx)
    flux_normalized_lte = normalize_flux(nlte_wave_nm, H_broadened_lte, continuum_idx)

    scale = flux_normalized_lte[continuum_idx].mean()
    flux_normalized_nlte /= scale
    flux_normalized_lte /= scale

    boundaries = line_bounds
    model_EW_results_nlte, line_indices_list = compute_equivalent_widths(
        nlte_wave_nm,
        flux_normalized_nlte,
        boundaries
    )

    model_EW_results_lte, _ = compute_equivalent_widths(
        nlte_wave_nm,
        flux_normalized_lte,
        boundaries,
        precomputed_indices=line_indices_list
    )

    S_profiles = []
    contrib_profiles = []
    for lc in line_centers_nm:
        local_idx = np.argmin(np.abs(nlte_wave_nm - lc))
        global_idx = nlte_indices[local_idx]

        nu_val = (const.c / (nlte_wave_nm[local_idx] * u.nm)).to(u.Hz).value
        kappa_ratio = kappa_nu_bars[global_idx, :] / kappa_bars

        _, S_profile, contrib_profile = H_nlte_at_nu(
            nu_val,
            tau_grid,
            kappa_ratio,
            Ts,
            epsilon_profile,
            return_S=True,
            return_contrib=True
        )

        S_profiles.append(np.clip(S_profile, 1e-300, None))
        contrib_profiles.append(contrib_profile)

    result = {
        'abundance': A_O_log,
        'epsilon_profile': epsilon_profile,
        'H_broadened_nlte': H_broadened_nlte,
        'H_broadened_lte': H_broadened_lte,
        'flux_normalized_nlte': flux_normalized_nlte,
        'flux_normalized_lte': flux_normalized_lte,
        'model_EW_pm_nlte': model_EW_results_nlte,
        'model_EW_pm_lte': model_EW_results_lte,
        'line_indices_list': line_indices_list,
        'S_profiles': np.array(S_profiles),
        'contrib_profiles': np.array(contrib_profiles),
    }

    return result


def main():
    args = parse_cli_args()
    if args.abundances:
        abundance_values = np.array(args.abundances, dtype=float)
    else:
        abundance_values = DEFAULT_ABUNDANCES.astype(float)

    tables_dir = args.tables_dir
    regenerate_tables = args.regenerate_tables
    show_table_plot = args.show_table_plot
    epsilon_scale = args.epsilon_scale


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


    tau_grid = np.concatenate((np.arange(3)/3*1e-3, np.logspace(-3,1.3,30)))

    tau_grid = tau_grid[tau_grid > 0]
    print(f"tau_grid = {tau_grid}")

    log_tau_grid = np.log10(tau_grid)
    print(f"log_tau_grid = {log_tau_grid}")

    valid_indices = (log_tau_grid >= log_tau_table.min()) & (log_tau_grid <= log_tau_table.max())
    tau_grid = tau_grid[valid_indices] 
    Ts = f_temp_from_tau(log_tau_grid[valid_indices])
    Ps = f_pressure_from_tau(log_tau_grid[valid_indices])




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
    print(f"Spectral grid setup complete. Calculating for {Nnu} frequency points.")
    print(f"This will cover a wavelength range from ~{wave_nm.min():.1f} nm to ~{wave_nm.max():.1f} nm.")

    line_centers_nm = np.array([777.1944, 777.4166, 777.5388])
    table_W_lambda_list_pm = np.array([7.12, 6.18, 4.88])

    nlte_indices = np.where((wave_nm >= 776) & (wave_nm <= 779))[0]
    nlte_wave_nm = wave_nm[nlte_indices]
    continuum_indices = continuum_indices_from_windows(nlte_wave_nm, CONTINUUM_WINDOWS)

    jobs = []
    for abundance in abundance_values:
        print(f"{'='*60}")
        print(f"Submitting model for log A(O) = {abundance:.2f}")
        print(f"{'='*60}")

        opacity_path = ensure_opacity_tables(
            abundance,
            tables_dir,
            regenerate=regenerate_tables,
            show_plot=show_table_plot,
        )
        kappa_bars = load_kappa_bars(opacity_path, Ps, Ts)
        print(f"Using opacity table: {opacity_path}")

        jobs.append({
            'abundance': abundance,
            'tau_grid': tau_grid,
            'Ts': Ts,
            'Ps': Ps,
            'rhos': rhos,
            'nu': nu,
            'wave_nm': wave_nm,
            'nlte_indices': nlte_indices,
            'nlte_wave_nm': nlte_wave_nm,
            'kappa_bars': kappa_bars,
            'line_centers_nm': line_centers_nm,
            'continuum_indices': continuum_indices,
            'line_bounds': LINE_BOUNDS,
            'broadening': BROADENING,
            'nu0': nu0,
            'dlnu': dlnu,
            'Nnu': Nnu,
            'epsilon_scale': epsilon_scale,
        })

    results = []
    max_workers = args.max_workers if args.max_workers is not None else min(len(jobs), os.cpu_count() or 1)

    parallel_enabled = not args.no_parallel and len(jobs) > 1

    if parallel_enabled:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_abundance = {executor.submit(run_abundance_case, job): job['abundance'] for job in jobs}
                for future in as_completed(future_to_abundance):
                    abundance = future_to_abundance[future]
                    result = future.result()
                    print(f"Completed synthesis for log A(O) = {abundance:.2f}")
                    results.append(result)
        except (PermissionError, OSError) as exc:
            print(f"Process-based parallelism unavailable ({exc}). Falling back to serial execution.")
            parallel_enabled = False

    if not parallel_enabled:
        for job in jobs:
            abundance = job['abundance']
            print(f"Running sequential synthesis for log A(O) = {abundance:.2f}")
            result = run_abundance_case(job)
            print(f"Completed synthesis for log A(O) = {abundance:.2f}")
            results.append(result)

    results.sort(key=lambda item: item['abundance'])

    all_normalized_flux_lte = {}
    all_normalized_flux_nlte = {}
    results_list = []
    log_tau = np.log10(tau_grid)

    for res in results:
        abundance = res['abundance']
        epsilon_profile = res['epsilon_profile']
        H_broadened_nlte = res['H_broadened_nlte']
        H_broadened_lte = res['H_broadened_lte']
        flux_normalized_nlte = res['flux_normalized_nlte']
        flux_normalized_lte = res['flux_normalized_lte']
        line_indices_list = res['line_indices_list']
        model_EW_results_nlte = res['model_EW_pm_nlte']
        model_EW_results_lte = res['model_EW_pm_lte']
        S_profiles = res['S_profiles']

        print(f"{'='*60}")
        print(f"--- Results for log A(O) = {abundance:.2f} ---")
        print(f"{'='*60}")

        center_ix = [np.argmin(np.abs(nlte_wave_nm - lc)) for lc in line_centers_nm]
        nlte_min_flux = H_broadened_nlte.min()
        lte_min_flux = H_broadened_lte.min()
        nlte_min_wave = nlte_wave_nm[H_broadened_nlte.argmin()]
        lte_min_wave = nlte_wave_nm[H_broadened_lte.argmin()]

        print("\n--- Raw broadened flux (no normalization) ---")
        print(f"Global min NLTE: {nlte_min_flux:.6f} at {nlte_min_wave:.4f} nm")
        print(f"Global min  LTE: {lte_min_flux:.6f} at {lte_min_wave:.4f} nm")
        for lc, ix in zip(line_centers_nm, center_ix):
            print(f"{lc:.4f} nm -> NLTE {H_broadened_nlte[ix]:.6f} | LTE {H_broadened_lte[ix]:.6f}")

        plt.figure(figsize=(10, 5))
        plt.plot(nlte_wave_nm, H_broadened_nlte, label=f'NLTE raw (log A(O)={abundance:.2f})', color='tab:purple')
        plt.plot(nlte_wave_nm, H_broadened_lte, label=f'LTE  raw (log A(O)={abundance:.2f})', color='tab:orange', alpha=0.85)
        for lc in line_centers_nm:
            plt.axvline(lc, color='k', ls=':', alpha=0.3)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('H (raw, broadened)')
        plt.title('Raw broadened flux (no normalization)')
        plt.grid(True, ls=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'raw_broadened_flux_{abundance:.2f}.png', dpi=200)
        plt.show()

        print("\n--- Final Analysis: Equivalent Width Comparison ---")
        print("NLTE cont mean:", H_broadened_nlte[continuum_indices].mean())
        print("LTE  cont mean:", H_broadened_lte[continuum_indices].mean())

        for i in range(len(line_centers_nm)):
            table_W = table_W_lambda_list_pm[i]
            line_indices = line_indices_list[i]
            print("LTE normalized max:", flux_normalized_lte[line_indices].max())
            print(f"\nLine {line_centers_nm[i]:.4f} nm (A(O)={abundance:.2f}):")
            print(f"  NLTE Model EW: {model_EW_results_nlte[i]:.2f} pm")
            print(f"  LTE Model EW:  {model_EW_results_lte[i]:.2f} pm")
            print(f"  Observed EW:   {table_W:.2f} pm")
            diff = (model_EW_results_nlte[i] - table_W) / table_W * 100
            if diff > 0:
                print(f"  --> Model line is {diff:.1f}% STRONGER than observed.")
            else:
                print(f"  --> Model line is {abs(diff):.1f}% WEAKER than observed.")

        all_normalized_flux_nlte[abundance] = flux_normalized_nlte
        all_normalized_flux_lte[abundance] = flux_normalized_lte

        results_list.append({
            'abundance': abundance,
            'model_EW_pm_nlte': model_EW_results_nlte,
            'model_EW_pm_lte': model_EW_results_lte
        })

        fig, (ax_eps, ax_state) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax_eps.semilogx(tau_grid, epsilon_profile, marker='o', color='tab:blue')
        ax_eps.set_ylabel('Thermalisation Parameter Epsilon')
        ax_eps.grid(True, linestyle=':', alpha=0.6)

        temp_line, = ax_state.semilogx(tau_grid, Ts, color='tab:red', label='Temperature [K]')
        ax_state.set_ylabel('Temperature [K]')
        ax_state.grid(True, linestyle=':', alpha=0.4)

        ax_press = ax_state.twinx()
        pressure_line, = ax_press.semilogx(tau_grid, Ps, color='tab:green', linestyle='--', label='Pressure [dyn/cm^2]')
        ax_press.set_ylabel('Pressure [dyn/cm^2]')
        ax_press.set_yscale('log')

        ax_state.set_xlabel('Optical Depth Tau')

        lines = [temp_line, pressure_line]
        labels = [line.get_label() for line in lines]
        ax_state.legend(lines, labels, loc='best')

        fig.suptitle(f'Layer Properties vs Tau (log A(O)={abundance:.2f})')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(f'epsilon_vs_tau_{abundance:.2f}.png', dpi=300)
        print(f"Saved epsilon/T/P vs tau plot to epsilon_vs_tau_{abundance:.2f}.png")
        plt.show()

        plt.figure(figsize=(8, 5))
        for lc, S_profile in zip(line_centers_nm, S_profiles):
            plt.plot(log_tau, np.log10(S_profile), label=f'{lc:.3f} nm')
        plt.xlabel(r'$\log_{10}\,\tau$')
        plt.ylabel(r'$\log_{10} S$')
        plt.title(f'$\log S$ vs $\log_{{10}}\tau$ (log A(O)={abundance:.2f})')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(title='Line Center')
        plt.savefig(f'logS_vs_logtau_{abundance:.2f}.png', dpi=300)
        print(f"Saved log S vs log tau plot to logS_vs_logtau_{abundance:.2f}.png")
        plt.show()

        contrib_profiles = res.get('contrib_profiles')
        if contrib_profiles is not None and contrib_profiles.size:
            plt.figure(figsize=(8, 5))
            tau_positive = tau_grid[1:]
            log_tau_positive = np.log10(tau_positive)

            contrib_arrays = []
            for lc, contrib in zip(line_centers_nm, contrib_profiles):
                contrib_arr = np.asarray(contrib)[1:]
                contrib_arrays.append(contrib_arr)
                plt.plot(log_tau_positive, contrib_arr, label=f'{lc:.3f} nm')

            combined = np.sum(contrib_arrays, axis=0) if contrib_arrays else None
            if combined is not None and np.any(combined > 0):
                threshold = 0.5 * np.max(combined)
                mask = combined >= threshold
                if np.any(mask):
                    shaded_x = log_tau_positive[mask]
                    plt.fill_between(shaded_x, 0, combined[mask], color='gray', alpha=0.2,
                                     label='Major contribution')

            plt.xlabel(r'$\log_{10}\,\tau$')
            plt.ylabel(r'Contribution $\Phi(0,\tau)\,S(\tau)$')
            plt.title(f'Contribution vs $\log_{{10}}\tau$ (log A(O)={abundance:.2f})')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(title='Line Center')
            fname = f'contrib_vs_logtau_{abundance:.2f}.png'
            plt.savefig(fname, dpi=300)
            print(f"Saved contribution function plot to {fname}")
            plt.show()

        print("Broadening complete.\n")

    print(f"{'='*60}")
    print("--- Final Analysis of All Abundance Trials ---")
    print(f"{'='*60}")

    observed_EW = table_W_lambda_list_pm
    best_fit_abundances_nlte = []
    best_fit_abundances_lte = []

    print("\nRendering LTE and NLTE spectra for every trial abundance...")

    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(abundance_values)))

    for i, abundance in enumerate(abundance_values):
        flux_nlte = all_normalized_flux_nlte[abundance]
        flux_lte = all_normalized_flux_lte[abundance]

        plt.plot(nlte_wave_nm, flux_nlte, color=colors[i], linestyle='-', label=f'NLTE log A(O) = {abundance:.2f}')
        plt.plot(nlte_wave_nm, flux_lte, color=colors[i], linestyle='--', label=f'LTE  log A(O) = {abundance:.2f}')

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Normalized Flux')
    plt.title('LTE vs NLTE spectra as a function of oxygen abundance')
    plt.xlim(777.1, 777.6)
    plt.ylim(0.7, 1.0)
    plt.legend(title='Line Type & Abundance', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('spectrum_all_abundances_LTE_NLTE.png', dpi=300)
    print("Saved LTE/NLTE comparison plot to spectrum_all_abundances_LTE_NLTE.png")
    plt.show()

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red']
    ax = plt.gca()
    annotation_lines = []
    all_abundances = [res['abundance'] for res in results_list]

    for i in range(len(line_centers_nm)):
        abundances = all_abundances
        model_ews_nlte = [res['model_EW_pm_nlte'][i] for res in results_list]
        model_ews_lte = [res['model_EW_pm_lte'][i] for res in results_list]

        ax.plot(abundances, model_ews_nlte, 'o-', label=f'NLTE Model {line_centers_nm[i]:.2f} nm', color=colors[i])
        ax.plot(abundances, model_ews_lte, 'x--', label=f'LTE Model {line_centers_nm[i]:.2f} nm', color=colors[i], alpha=0.7)
        ax.axhline(y=observed_EW[i], linestyle=':', color=colors[i], label=f'Observed {line_centers_nm[i]:.2f} nm')

        fit_ab_nlte = np.interp(observed_EW[i], model_ews_nlte, abundances)
        best_fit_abundances_nlte.append(fit_ab_nlte)

        fit_ab_lte = np.interp(observed_EW[i], model_ews_lte, abundances)
        best_fit_abundances_lte.append(fit_ab_lte)

        ax.axvline(
            fit_ab_nlte,
            color=colors[i],
            linestyle='-.',
            linewidth=1.2,
            alpha=0.6,
            label='NLTE best-fit abundance' if i == 0 else '_nolegend_'
        )
        ax.axvline(
            fit_ab_lte,
            color=colors[i],
            linestyle=':',
            linewidth=1.2,
            alpha=0.6,
            label='LTE best-fit abundance' if i == 0 else '_nolegend_'
        )
        ax.scatter([fit_ab_nlte], [observed_EW[i]], color=colors[i], marker='o', s=55, edgecolor='k', zorder=5)
        ax.scatter([fit_ab_lte], [observed_EW[i]], color=colors[i], marker='^', s=55, edgecolor='k', zorder=5, alpha=0.85)

        y_offset = 0.3 * (i - 1)
        va_nlte = 'bottom' if y_offset >= 0 else 'top'
        ax.text(
            fit_ab_nlte + 0.02,
            observed_EW[i] + y_offset,
            f'NLTE {fit_ab_nlte:.3f}',
            color=colors[i],
            fontsize=9,
            ha='left',
            va=va_nlte
        )

        y_offset_lte = y_offset - 0.15
        va_lte = 'bottom' if y_offset_lte >= 0 else 'top'
        ax.text(
            fit_ab_lte - 0.02,
            observed_EW[i] + y_offset_lte,
            f'LTE {fit_ab_lte:.3f}',
            color=colors[i],
            fontsize=9,
            ha='right',
            va=va_lte
        )

        annotation_lines.append(f"{line_centers_nm[i]:.2f} nm: NLTE {fit_ab_nlte:.3f}, LTE {fit_ab_lte:.3f}")
        print(f"Line {line_centers_nm[i]:.2f} nm -> Best-fit NLTE Abundance: {fit_ab_nlte:.3f}, Best-fit LTE Abundance: {fit_ab_lte:.3f}")

    if annotation_lines:
        ax.text(
            0.98,
            0.02,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        )

    if all_abundances:
        x_min = min(all_abundances)
        x_max = 9.5
        if x_max <= x_min:
            x_max = max(all_abundances)
        ax.set_xlim(x_min, x_max)

    avg_nlte_abundance_fit = np.mean(best_fit_abundances_nlte)
    print(f"\nAverage best-fit NLTE abundance (before correction): {avg_nlte_abundance_fit:.3f}")
    final_nlte_abundance = avg_nlte_abundance_fit
    print(f"Final NLTE Oxygen Abundance: log A(O) = {final_nlte_abundance:.3f}")

    avg_lte_abundance_fit = np.mean(best_fit_abundances_lte)
    print(f"\nAverage best-fit LTE abundance: {avg_lte_abundance_fit:.3f}")

    ax.set_title('Equivalent Width vs. Oxygen Abundance')
    ax.set_xlabel('Log Oxygen Abundance [log A(O)]')
    ax.set_ylabel('Equivalent Width [pm]')
    ax.grid(True)
    ax.legend()
    plt.savefig('abundance_fit.png', dpi=300)
    print("Curve of growth plot saved as abundance_fit.png")
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    main()
