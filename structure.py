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
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp, cumulative_trapezoid
import astropy.units as u
import astropy.constants as c
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import opac
from scipy.special import expn
import saha_eos as eos
from strontium_barium import *
from scipy.interpolate import interp1d
from scipy.special import voigt_profile


Teff = 5777 # K #4850 is the lowest. 9000 is the highest that makes sense.
g = 27400  # cm/s^2
P0 = 10 # Initial pressure in dyn/cm^2
# --- Atomic Data for the Oxygen I 7774 Angstrom Triplet ---
# For simplicity, we treat the triplet as a single line with its
# weighted-average physical parameters.
oxygen_lambdas_nm = np.array([777.1944, 777.4166, 777.5388]) * u.nm
oxygen_chi_i = 73768.2  # Excitation energy of the lower level in cm^-1
oxygen_g_i = 5.0      # Statistical weight (degeneracy) of the lower level
oxygen_log_gfs = np.array([0.369, 0.223, 0.002])
# The partition function Z is temperature-dependent. For this project, we can
# approximate it with a constant value suitable for the solar photosphere.
# oxygen_Z_T = 9.0


# Set to 1.3 to limit T due to the onset of convection.
# If set to 2.0, there is no effect.
convective_cutoff = 1.3

# Load the opacity table for Rosseland mean.
f_opac = pyfits.open('Ross_Planck_opac.fits')
kappa_bar_Ross = f_opac['kappa_Ross [cm**2/g]'].data
#plt.loglog(tau_grid, f_kappa_bar_Ross((np.log10(Ps), Ts)))

# Construct the log(P) and T vectors. 
h = f_opac[0].header
T_grid = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
Ps_log10 = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']

P0 = np.maximum(10**(Ps_log10[0]), P0)  # Ensure P0 is not less than the minimum pressure in the table

#Create our interpolator functions
f_kappa_bar_Ross = RegularGridInterpolator((Ps_log10, T_grid), kappa_bar_Ross)

def T_tau(tau, Teff):
	"""
	Temperature for a simplified grey atmosphere, with an analytic
    approximation for the Hopf q (feel free to check this!)
	"""
	q = 0.71044 - 0.1*np.exp(-2.0*tau)
	T = (0.75*Teff**4*(tau + q))**.25
	return T

def dPdtau(_, P, T):
	"""
	Compute the derivative of pressure with respect to optical depth.
	"""
	kappa_bar = f_kappa_bar_Ross((np.log10(P), T))
	return g / kappa_bar

##############################################################################################################
############################(Solar T, P, rho from M. Asplund 2004 paper) #####################################
# Data extracted from Table 1 
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
# Pa convert to dyn/cm^2
pressure_cgs_table = pressure_pa_table * 10.0

# use above values, create interplation function

f_pe_from_tau = interp1d(log_tau_table, pe_cgs_table, kind='cubic')
f_temp_from_tau = interp1d(log_tau_table, temp_table, kind='cubic')
f_pressure_from_tau = interp1d(log_tau_table, pressure_cgs_table, kind='cubic')
#######################################################################################################
#######################################################################################################

tau_grid = np.concatenate((np.arange(3)/3*1e-3,np.logspace(-3,1.3,30)))
tau_grid = tau_grid[tau_grid > 0]
log_tau_grid = np.log10(tau_grid)
valid_indices = (log_tau_grid >= log_tau_table.min()) & (log_tau_grid <= log_tau_table.max())
tau_grid = tau_grid[valid_indices]
Ts = f_temp_from_tau(log_tau_grid[valid_indices])
Ps = f_pressure_from_tau(log_tau_grid[valid_indices])


# ##################################### Mike's code #####################################
# # Starting from the lowest value of log(P), integrate P using solve_ivp
# #solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
# tau_grid = np.concatenate((np.arange(3)/3*1e-3,np.logspace(-3,1.3,30)))
# sol = solve_ivp(dPdtau, [0, 20], [P0], args=(Teff,), t_eval=tau_grid, method='RK45')
# Ps = sol.y[0]
# Ts = T_tau(tau_grid, Teff)
# # Artificially cut the deep layer temperature due to convection.
# Ts = np.minimum(Ts,convective_cutoff*Teff)
# ##################################### Mike's code #####################################


# Load the equation of state
f_eos = pyfits.open('rho_Ui_mu_ns_ne.fits')
rho = f_eos['rho [g/cm**3]'].data

# Add interpolation functions for whatever isn't already in opac - just rho.
f_rho = RegularGridInterpolator((Ps_log10, T_grid), rho)

# Interpolate onto the tau grid
kappa_bars = f_kappa_bar_Ross((np.log10(Ps), Ts))
rhos = f_rho((np.log10(Ps), Ts))    

# First, lets plot a continuum spectrum
wave = np.linspace(775.0, 780.0, 10000) * u.nm  # Wavelength in nm
flux = np.zeros_like(wave)  # Initialize flux array

# Just like in grey_flux.py, but in frequency 
planck_C1 = (2*c.h*c.c**2/(1*u.um)**5).si.value
planck_C2 = (c.h*c.c/(1*u.um)/c.k_B/(1*u.K)).si.value

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


# store equivlent width for each line
model_W_lambda_list_pm = []

# equivlent width from Table 3 (M. Asplund 2004 paper) , will be used to verify our result
table_W_lambda_list_pm = [7.12, 6.18, 4.88] # each coresponds to emission line in 777.19, 777.41, 777.53 nm 



# --- Start: New Main Loop for Line Synthesis ---

# print("Computing total opacities (continuum + line)...")
# # Prepare an empty array for the final opacity table
# # It has dimensions (number_of_wavelength_points, number_of_atmospheric_layers)
# kappa_total_nu = np.empty((len(wave), len(tau_grid)))
# nu = (c.c / wave).to(u.Hz).value # Corresponding frequency grid in Hz

# # A constant needed for the line opacity calculation (from opac.py)
# f_const = (np.pi * c.e.gauss**2 / c.m_e / c.c).cgs.value

# # Loop through each layer of our solar atmosphere model
# for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
#     # --- Part A: Calculate the number of absorbers (same as before) ---
#     current_T = T * u.K
#     current_rho = rho * u.g / u.cm**3

#     # given rho and T of a layer, calculate how much  nutral O atoms
#     n_e, ns, mu, Ui = eos.ns_from_rho_T(current_rho, current_T)
#     O_ix = np.where(eos.solarmet()[-1] == 'O')[0][0]
#     n_O_I = ns[3 * O_ix]

#     current_Z_T = eos.get_Z_O_I(T) # T is the layer temp in Kelvin (a plain number)


#     # 2. Use this new, more accurate Z value in the Boltzmann equation
#     # In all nutral O atoms, how much fraction are excited in the specific level where it can absorb 777nm photons
#     frac_excited = eos.calculate_excitation_fraction(
#         current_T, oxygen_g_i, oxygen_chi_i, current_Z_T
#     )
#     # how many O atoms can absorb 777nm photons
#     n_absorbers = n_O_I * frac_excited
#     n_absorbers = (n_O_I * frac_excited).value



#     # --- Part B, C: 计算三条谱线产生的总不透明度 ---
#     kappa_line_total = np.zeros_like(nu) # 为该大气层初始化总谱线不透明度
#     print()
#     # 遍历三联线中的每一条谱线
#     for j in range(len(oxygen_lambdas_nm)):
#         # 获取当前谱线的中心波长和频率
#         line_lambda = oxygen_lambdas_nm[j]
#         line_nu_center = (c.c / line_lambda).to(u.Hz).value

#         # 为当前谱线计算其多普勒展宽和线型
#         doppler_width_nu = np.sqrt(2 * c.k_B * current_T / (16 * c.u)) / c.c * line_nu_center

#         # Inside your loop, change your print statement to this:
#         print(f"{j}th: doppler_width_nu = {doppler_width_nu.to(u.Hz)}")
#         line_profile = (1.0 / (doppler_width_nu * np.sqrt(np.pi))) * \
#                     np.exp(-(nu - line_nu_center)**2 / doppler_width_nu**2)

#         # 为当前谱线计算其不透明度
#         gf_value = 10**oxygen_log_gfs[j]
#         kappa_line_single = (f_const * gf_value / rho) * n_absorbers * line_profile

#         # 将当前谱线的贡献累加到总和中
#         kappa_line_total += kappa_line_single

#     # --- Part D: 计算连续谱不透明度，并与总的谱线不透明度相加 ---
#     kappa_continuum = opac.kappa_cont(nu, np.log10(P), T) / rho
#     kappa_total_nu[:, i] = kappa_continuum + kappa_line_total


# # # With the total opacity calculated, we can now compute the final spectrum
# print("Computing final spectrum...")
# H = compute_H(wave, Ts, tau_grid, kappa_total_nu, kappa_bars)





# --- calculate for each emission line  ---
for line_index in range(len(oxygen_lambdas_nm)):

    current_line_lambda = oxygen_lambdas_nm[line_index]
    print(f"\n--- Calculating for line: {current_line_lambda.to_value(u.nm):.4f} nm ---")

    # an empty list to store kappa
    kappa_total_nu = np.empty((len(wave), len(tau_grid)))
    nu = (c.c / wave).to(u.Hz).value
    f_const = (np.pi * c.e.gauss**2 / c.m_e / c.c).cgs.value

    # interate over each layer of atmosphere
    for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
        # Get the log_tau value for the current layer
        current_log_tau = log_tau_grid[valid_indices][i]
        # Use the new interpolator to get electron pressure from the table
        Pe_cgs = f_pe_from_tau(current_log_tau) # Electron pressure in dyn/cm^2

        # Convert electron pressure (Pe) to electron number density (n_e)
        # using the ideal gas law for electrons: Pe = n_e * k_B * T
        n_e = Pe_cgs / (c.k_B.cgs.value * T) # n_e in cm^-3

        # Now that we have n_e and T, call the core saha function directly
        # This is much faster than the solver ns_from_rho_T
        rho_check, mu, Ui, ns = eos.saha(n_e, T)

        current_T = T * u.K
        current_rho = rho * u.g / u.cm**3
        # n_e, ns, mu, Ui = eos.ns_from_rho_T(current_rho, current_T)
        O_ix = np.where(eos.solarmet()[-1] == 'O')[0][0]
        n_O_I = ns[3 * O_ix]
        current_Z_T = eos.get_Z_O_I(T)
        frac_excited = eos.calculate_excitation_fraction(
            current_T, oxygen_g_i, oxygen_chi_i, current_Z_T
        )
        n_absorbers = (n_O_I * frac_excited).value



        print(f"line_index: {line_index}: n_absorbers: {n_absorbers}")

        line_nu_center = (c.c / current_line_lambda).to(u.Hz).value
        # doppler_width_nu = np.sqrt(2 * c.k_B * current_T / (16 * c.u)) / c.c * line_nu_center
        # line_profile = (1.0 / (doppler_width_nu.value * np.sqrt(np.pi))) * \
        #                np.exp(-(nu - line_nu_center)**2 / doppler_width_nu.value**2)

        # 1. 计算多普勒宽度 (和以前一样)
        doppler_width_nu = np.sqrt(2 * c.k_B * current_T / (16 * c.u)) / c.c * line_nu_center

        # 2. 【新增】计算 Voigt Profile 所需的两个参数
        sigma_gauss = doppler_width_nu.value / np.sqrt(2)
        # 为洛伦兹增宽估算一个合理的阻尼常数 gamma (单位: Hz)
        # 这是一个近似值，但能很好地体现出碰撞增宽的效应
        gamma_lorentz = 3e8 # 这是一个合理的数量级估算

        # 3. 【修改】使用 voigt_profile 函数计算谱线轮廓
        # 注意：voigt_profile 函数本身已经归一化，所以我们不需要再乘以归一化因子
        # 函数的第一个参数是频率点到中心频率的距离
        line_profile = voigt_profile(nu - line_nu_center, sigma_gauss, gamma_lorentz)

        
        gf_value = 10**oxygen_log_gfs[line_index]
        kappa_line = (f_const * gf_value / rho) * n_absorbers * line_profile

        # continuum kappa + single line kappa 
        kappa_continuum = opac.kappa_cont(nu, np.log10(P), T) / rho
        kappa_total_nu[:, i] = kappa_continuum + kappa_line

    #  spectrum for a single line
    H_single_line = compute_H(wave, Ts, tau_grid, kappa_total_nu, kappa_bars)

    plt.figure(figsize=(15, 7))
    plt.plot(wave.to_value(u.nm), H_single_line, label='Synthetic Spectrum', lw=1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Visually Inspecting the Continuum for Normalization')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"{line_index} get_continum_spectrum")
    plt.show()

    wave_nm = wave.to_value(u.nm)
    indices_1 = np.where((wave_nm >= 775.0) & (wave_nm <= 776.5))
    indices_2 = np.where((wave_nm >= 778.5) & (wave_nm <= 780.0))
    continuum_indices = np.concatenate((indices_1[0], indices_2[0]))

    # 从完整光谱中提取出这些锚点的波长和流量值
    continuum_wavelengths = wave_nm[continuum_indices]
    continuum_fluxes = H_single_line[continuum_indices]

    # 用一次多项式（直线）来拟合这些锚点 (如果连续谱是弯的，第二个参数可以改成2)
    p = np.polyfit(continuum_wavelengths, continuum_fluxes, 1)
    continuum_fit_function = np.poly1d(p)

    # 使用拟合函数计算出每一个波长点上的连续谱理论值
    continuum_level_fit = continuum_fit_function(wave_nm)

    # 用这个精确的连续谱来归一化整个光谱
    model_flux_normalized_single = H_single_line / continuum_level_fit

    # --- equivelent width for a single line  ---
    # continuum_level_single = H_single_line[0]
    # model_flux_normalized_single = H_single_line / continuum_level_single

    d_wave_nm = wave[1].to_value(u.nm) - wave[0].to_value(u.nm)
    model_W_lambda_nm = np.sum(1.0 - model_flux_normalized_single) * d_wave_nm
    model_W_lambda_pm = model_W_lambda_nm * 1000


    model_W_lambda_list_pm.append(model_W_lambda_pm)




# --- For plotting spectrum simulation ，recalculate kappa_line_total for all three lines ---
print("\nRecalculating combined spectrum for plotting...")
kappa_total_nu = np.empty((len(wave), len(tau_grid)))


n_absorbers_list = []


for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
    current_T = T * u.K
    current_rho = rho * u.g / u.cm**3
    # n_e, ns, mu, Ui = eos.ns_from_rho_T(current_rho, current_T)
    O_ix = np.where(eos.solarmet()[-1] == 'O')[0][0]
    n_O_I = ns[3 * O_ix]
    current_Z_T = eos.get_Z_O_I(T)
    frac_excited = eos.calculate_excitation_fraction(current_T, oxygen_g_i, oxygen_chi_i, current_Z_T)
    n_absorbers = (n_O_I * frac_excited).value

    n_absorbers_list.append(n_absorbers)

    kappa_line_total = np.zeros_like(nu)
    for j in range(len(oxygen_lambdas_nm)): 
        line_lambda = oxygen_lambdas_nm[j]
        line_nu_center = (c.c / line_lambda).to(u.Hz).value
        doppler_width_nu = np.sqrt(2 * c.k_B * current_T / (16 * c.u)) / c.c * line_nu_center
        line_profile = (1.0 / (doppler_width_nu.value * np.sqrt(np.pi))) * np.exp(-(nu - line_nu_center)**2 / doppler_width_nu.value**2)
        gf_value = 10**oxygen_log_gfs[j]
        kappa_line_single = (f_const * gf_value / rho) * n_absorbers * line_profile
        kappa_line_total += kappa_line_single
    kappa_continuum = opac.kappa_cont(nu, np.log10(P), T) / rho
    kappa_total_nu[:, i] = kappa_continuum + kappa_line_total



# --- Print all results ---
print("\n--- Final Equivalent Width Comparison ---")
for i in range(len(oxygen_lambdas_nm)):
    model_W = model_W_lambda_list_pm[i]
    table_W = table_W_lambda_list_pm[i]
    lambda_nm = oxygen_lambdas_nm[i].to_value(u.nm)
    
    print(f"Line {lambda_nm:.4f} nm:")
    print(f"  Model Equivalent Width:    {model_W:.2f} pm")
    print(f"  Observed Equivalent Width: {table_W:.2f} pm")
    if model_W > table_W:
        print("  --> Model line is STRONGER than observed.")
    elif model_W < table_W:
        print("  --> Model line is WEAKER than observed.")
    else:
        print("  --> Model matches observation well.")

plt.figure(figsize=(8, 6))
# 使用 semilogy 来让 Y 轴以对数形式呈现
plt.semilogy(log_tau_grid[valid_indices], n_absorbers_list)
plt.xlabel('log($\\tau_{500}$)')
plt.ylabel('Number of Absorbers ($N_{abs}$ per cm$^3$)')
plt.title('Calculated Oxygen Absorber Density vs. Depth in Sun')
plt.grid(True)
plt.savefig("n_absorbers_plot")
plt.show()



# ---- ADD THIS ENTIRE BLOCK BACK IN ----
# Recalculate the combined spectrum for plotting
print("\nRecalculating combined spectrum for plotting...")
kappa_total_nu = np.empty((len(wave), len(tau_grid)))
for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
    current_T = T * u.K
    current_rho = rho * u.g / u.cm**3
    # n_e, ns, mu, Ui = eos.ns_from_rho_T(current_rho, current_T)
    O_ix = np.where(eos.solarmet()[-1] == 'O')[0][0]
    n_O_I = ns[3 * O_ix]
    current_Z_T = eos.get_Z_O_I(T)
    frac_excited = eos.calculate_excitation_fraction(current_T, oxygen_g_i, oxygen_chi_i, current_Z_T)
    n_absorbers = (n_O_I * frac_excited).value
    kappa_line_total = np.zeros_like(nu)
    for j in range(len(oxygen_lambdas_nm)): # This loop sums the opacity of all 3 lines
        line_lambda = oxygen_lambdas_nm[j]
        line_nu_center = (c.c / line_lambda).to(u.Hz).value
        doppler_width_nu = np.sqrt(2 * c.k_B * current_T / (16 * c.u)) / c.c * line_nu_center
        line_profile = (1.0 / (doppler_width_nu.value * np.sqrt(np.pi))) * np.exp(-(nu - line_nu_center)**2 / doppler_width_nu.value**2)
        gf_value = 10**oxygen_log_gfs[j]
        kappa_line_single = (f_const * gf_value / rho) * n_absorbers * line_profile
        kappa_line_total += kappa_line_single
    kappa_continuum = opac.kappa_cont(nu, np.log10(P), T) / rho
    kappa_total_nu[:, i] = kappa_continuum + kappa_line_total

H = compute_H(wave, Ts, tau_grid, kappa_total_nu, kappa_bars)
# ----------------------------------------




print("Plotting results...")

plt.figure(figsize=(12, 7)) # 创建一个大小合适的画布

# --- Part 1: Load and Process Observational Data using Pandas ---
try:
    # 使用你建议的 pandas 方法读取原始文件
    obs_data = pd.read_csv('spectrum.txt', header=0, sep='\s*,\s*', usecols=[0, 1], engine='python')
    
    # 自动获取列名
    obs_wave_col = obs_data.columns[0]
    obs_intensity_col = obs_data.columns[1]

    # 从 pandas DataFrame 中获取数据
    obs_wave_aa = obs_data[obs_wave_col].values
    obs_flux_raw = obs_data[obs_intensity_col].values
    print("Successfully loaded spectrum.txt using pandas.")

    # --- Part 2: Process and Plot Data (same logic as before) ---
    
    # a) Process observational data
    obs_wave_nm = obs_wave_aa / 10.0  # Convert Angstroms to nm
    obs_flux_normalized = obs_flux_raw / obs_flux_raw.max() # Normalize

    # b) Process model data
    continuum_level = H[0] # Continuum is the flux at the first wavelength point
    model_flux_normalized = H / continuum_level

    # c) Plot both datasets
    # Plot our theoretical model (Blue line)
    # The 'wave' variable is already in nm from our calculation
    plt.plot(wave.to(u.nm).value, model_flux_normalized, lw=2, label=f'Model (3 lines)')

    # Plot the real observation (Red dashed line)
    plt.plot(obs_wave_nm, obs_flux_normalized, '--r', label='Observed Sun')

except FileNotFoundError:
    print("Error: 'spectrum.txt' not found. Skipping observation plot.")
    # If the observation file isn't found, just plot the model
    plt.plot(wave.to(u.nm).value, H, lw=2, label=f'Model Flux (log A(O) = ?)')


# --- Part 3: Finalize and Show Plot ---
plt.title('Solar Spectrum vs. Model near O I Triplet')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.xlim(777.1, 777.7)  # Zoom in on the line
plt.ylim(0.6, 1.05)   # Zoom in to see the line depth clearly
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.savefig('final_comparison_triplet.png', dpi=300)
print("最终光谱对比图已成功保存为 final_comparison_triplet.png")
plt.show()

