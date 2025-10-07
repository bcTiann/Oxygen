# nlte_branch.py
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import cumulative_trapezoid as cumtrapz

from solutions2 import lambda_matrix, phi_matrix, calc_Bnu  # 老师作业的实现


def tau_nu_from_ratio(tau_grid, kappa_ratio):
    tau = np.asarray(tau_grid, float)
    if tau[0] != 0: tau = tau - tau[0]
    tau_nu = np.concatenate(([0.0], cumtrapz(kappa_ratio, tau)))
    return tau_nu


def H_nlte_at_nu(
    nu_hz,
    tau_grid,
    kappa_ratio,
    T_of_tau,
    epsilon_profile,
    return_S=False,
    return_contrib=False,
):

    eps = epsilon_profile
    eps = np.asarray(eps, dtype=float)
    # print(f"eps = {eps}")
    tiny = 1e-12
    eps_clip = np.clip(eps, tiny, 1.0)

    # kappa_ratio_eff = kappa_ratio / eps_clip
    # tau_nu = tau_nu_from_ratio(tau_grid, kappa_ratio_eff)
    
    tau_nu = tau_nu_from_ratio(tau_grid, kappa_ratio)

    Λ = lambda_matrix(tau_nu)
    Φ = phi_matrix(tau_nu)
    Bnus = calc_Bnu(T_of_tau * u.K, nu_hz * u.Hz)

    I = np.eye(len(tau_grid))
    J = np.linalg.solve(I - Λ @ (I - np.diag(eps_clip)),
                        Λ @ (eps_clip * Bnus))
    S = (1 - eps_clip) * J + eps_clip * Bnus
    
    HS = Φ @ S
    H = HS[0]

    try:
        H_val = H.cgs.value
    except Exception:
        H_val = np.asarray(H, float)

    contrib_val = None
    if return_contrib:
        try:
            contrib_val = HS.cgs.value
        except Exception:
            contrib_val = np.asarray(HS, float)

    if not return_S and not return_contrib:
        return H_val

    try:
        S_val = S.cgs.value
    except Exception:
        S_val = np.asarray(S, float)

    outputs = [H_val]
    if return_S:
        outputs.append(S_val)
    if return_contrib:
        outputs.append(contrib_val)

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)
    

def spectrum_nlte(wave_nm, tau_grid, kappa_all_nu_bars, kappa_R_bars,
                  T_of_tau, epsilon_profile):
    H_nlte = np.zeros_like(wave_nm)
    for i, w in enumerate(wave_nm):
        nu = (c.c / (w * u.nm)).to(u.Hz).value
        kappa_ratio = kappa_all_nu_bars[i, :] / kappa_R_bars
        H_nlte[i] = H_nlte_at_nu(nu, tau_grid, kappa_ratio, T_of_tau, epsilon_profile)
    return H_nlte
