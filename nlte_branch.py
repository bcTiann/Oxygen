# nlte_branch.py
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import cumulative_trapezoid as cumtrapz

from solutions2 import lambda_matrix, phi_matrix, calc_Bnu  # 老师作业的实现

# ==================================================================
#         VVV 請將這兩個新函數加入到 nlte_branch.py VVV
# ==================================================================


def calculate_physical_epsilon_profile(Ts, ns_all_layers, ne_all_layers, h_index, A_ul, stark_gamma_rate, waals_gamma_rate):
    """
    使用物理公式，為大氣的每一層計算 epsilon 值。
    【更清晰的設計】：直接接收 ns 和 n_e 陣列，而不是合併它們。
    
    參數:
    Ts (array): 溫度陣列 (K)。
    ns_all_layers (array): 維度為 (層數, 物種數) 的粒子數密度陣列 (cm^-3)。
    ne_all_layers (array): 維度為 (層數,) 的電子數密度陣列 (cm^-3)。
    h_index (int): 中性氫 (H I) 在 ns_all_layers 的物種維度中的索引。
    A_ul (float): 愛因斯坦 A 係數 (s^-1)。
    stark_gamma_rate (float): Stark 碰撞速率係數 (cm^3 s^-1)。
    waals_gamma_rate (float): van der Waals 碰撞速率係數 (cm^3 s^-1)。
    """
    print("Calculating physical epsilon profile for all atmospheric layers...")
    
    # 根據索引從 ns 陣列中提取中性氫的密度
    n_H_all_layers = ns_all_layers[:, h_index]
    
    # 計算碰撞致衰變率 C_ul
    # C_ul ≈ n_e * γ_e + n_H * γ_H
    C_ul = (ne_all_layers * stark_gamma_rate) + (n_H_all_layers * waals_gamma_rate)
    
    # 計算 epsilon profile
    epsilon_profile = C_ul / (C_ul + A_ul)
        
    return np.clip(epsilon_profile, 0.0, 1.0)

def spectrum_nlte_phys_eps(wave_nm, tau_grid, kappa_all_nu_bars, kappa_R_bars, T_of_tau, epsilon_profile):
    """
    使用物理計算的 epsilon profile 來計算整個 NLTE 光譜。
    """
    print("Calculating full NLTE spectrum using the physical epsilon profile...")
    H_nlte_spectrum = np.zeros_like(wave_nm)
    
    for i, w in enumerate(wave_nm):
        # 這些是可呼叫的核心函數
        nu = (c.c / (w * u.nm)).to(u.Hz).value
        kappa_ratio = kappa_all_nu_bars[i, :] / kappa_R_bars
        
        # 結合物理 epsilon 和線芯判斷
        # 在連續譜 (kappa_ratio < 2), epsilon 強制為 1 (LTE)
        # 在線芯 (kappa_ratio >= 2), 使用我們物理計算的 epsilon
        eps_final = np.ones_like(T_of_tau)
        mask = (kappa_ratio >= 2.0)
        eps_final[mask] = epsilon_profile[mask]

        # 呼叫單一頻率的計算函數 (H_nlte_at_nu 稍作修改以傳入 eps_final)
        tau_nu = tau_nu_from_ratio(tau_grid, kappa_ratio)
        Λ = lambda_matrix(tau_nu)
        Φ = phi_matrix(tau_nu)
        Bnus = calc_Bnu(T_of_tau * u.K, nu * u.Hz)

        I = np.eye(len(tau_grid))
        J = np.linalg.solve(I - Λ @ (I - np.diag(eps_final)), Λ @ (eps_final * Bnus))
        S = (1 - eps_final) * J + eps_final * Bnus
        H = (Φ @ S)[0]   # emergent flux
        H_nlte_spectrum[i] = H.cgs.value
        
    return H_nlte_spectrum


def tau_nu_from_ratio(tau_grid, kappa_ratio):
    tau = np.asarray(tau_grid, float)
    if tau[0] != 0: tau = tau - tau[0]
    tau_nu = np.concatenate(([0.0], cumtrapz(kappa_ratio, tau)))
    return tau_nu

def make_eps_profile(tau_nu, kappa_ratio, eps_line=1e-2):
    eps = np.ones_like(tau_nu)
    # 在 κ_ν/κ_R 较大的“线芯”层给一个小 ε
    mask = (kappa_ratio >= 2.0)
    eps[mask] = eps_line
    return eps

def H_nlte_at_nu(nu_hz, tau_grid, kappa_ratio, T_of_tau, eps_line=1e-2):
    tau_nu = tau_nu_from_ratio(tau_grid, kappa_ratio)
    Λ = lambda_matrix(tau_nu)
    Φ = phi_matrix(tau_nu)
    Bnus = calc_Bnu(T_of_tau * u.K, nu_hz * u.Hz)

    eps = make_eps_profile(tau_nu, kappa_ratio, eps_line)
    I = np.eye(len(tau_grid))
    J = np.linalg.solve(I - Λ @ (I - np.diag(eps)), Λ @ (eps * Bnus))
    S = (1-eps)*J + eps*Bnus
    H = (Φ @ S)[0]   # emergent
    return H.cgs.value

def spectrum_nlte(wave_nm, tau_grid, kappa_all_nu_bars, kappa_R_bars, T_of_tau, eps_line=1e-2):
    H_nlte = np.zeros_like(wave_nm)
    for i, w in enumerate(wave_nm):
        nu = (c.c / (w * u.nm)).to(u.Hz).value
        kappa_ratio = kappa_all_nu_bars[i,:] / kappa_R_bars
        H_nlte[i] = H_nlte_at_nu(nu, tau_grid, kappa_ratio, T_of_tau, eps_line)
    return H_nlte
