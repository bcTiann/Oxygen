import numpy as np
import astropy.units as u
import astropy.constants as c


def derive_C_ul(T, 
                n_e,
                n_H, 
                E_l, 
                E_u, 
                f_ij,
                J_l,
                J_u
                ):
    
    # Upper level: E_u 
    # Lower level: E_l

    n_e_si = n_e.to(u.m**-3)
    n_H_si = n_H.to(u.m**-3)

    E_ij = E_u - E_l

    # E_H = Ionization Energy of Hydrogen 
    E_H = 13.6 * u.eV 
    u0 = E_ij / (c.k_B * T)

    # m_O = 16 * c.m_p # O atom Mass
    # m_H = c.m_p + c.m_e    # H atom Mass
    m_O = 15.994914619 * c.u
    m_H = 1.00782503224 * c.u

    m_A = m_O
    mu = (m_A * m_H) / (m_A + m_H)
    
    ########################################################
    # Equation from lecture 10, page 8
    qe_sqrt_term = np.sqrt(8 * c.k_B * T / (c.m_e * np.pi))
    qH_sqrt_term = np.sqrt(2 * c.k_B * T / (np.pi * mu))


    qe_lu = 14.5 * np.pi * c.a0**2 * qe_sqrt_term * (E_H / E_ij)**2 * f_ij * np.exp(-u0) * 1
    qH_lu = 16 * np.pi * c.a0**2 * qH_sqrt_term * (E_H / E_ij)**2 * f_ij * m_A/m_H * c.m_e/(m_H + c.m_e) * np.exp(-u0) * (1 + 2/u0)
    ########################################################
    # print(f"qe_lu = {qe_lu.decompose():.2e}")
    # print(f"qH_lu = {qH_lu.decompose():.2e}")

    C_lu= n_e_si * qe_lu + n_H_si * qH_lu

    g_l = 2*J_l + 1
    g_u = 2*J_u + 1


    # print(f'C_lu = {C_lu.decompose():.2e}')

    C_ul = C_lu * g_l/g_u * np.exp(E_ij / (c.k_B * T))

    # print(f'C_ul = {C_ul.decompose():.2e}')
    return C_ul.decompose().to_value()




# ## Observed Wavelength: 7774.17  Å
# # Execute Calculation for C_ul
# derive_C_ul(T=6412 * u.K, 
#             P_gas=1.14e4 * u.Pa,
#             P_e=6.71 * u.Pa,
#             rho=2.7e-10 * u.kg/u.m**3,
#             E_l=9.1460911 * u.eV,
#             E_u=10.7409314  * u.eV,
#             f_ij=4.68e-1,
#             J_l=2,
#             J_u=3
#             )


# ## Observed Wavelength: 7774.17  Å
# # Execute Calculation for C_ul
# derive_C_ul(T=6412 * u.K, 
#             P_gas=1.14e4 * u.Pa,
#             P_e=6.71 * u.Pa,
#             rho=2.7e-10 * u.kg/u.m**3,
#             E_l=9.1460911 * u.eV,
#             E_u=10.7404756  * u.eV,
#             f_ij=4.68e-1,
#             J_l=2,
#             J_u=2
#             )

# ## Observed Wavelength: 7775.39  Å 
# # Execute Calculation for C_ul
# derive_C_ul(T=6412 * u.K, 
#             P_gas=1.14e4 * u.Pa,
#             P_e=6.71 * u.Pa,
#             rho=2.7e-10 * u.kg/u.m**3,
#             E_l=9.1460911 * u.eV,
#             E_u=10.7402250  * u.eV,
#             f_ij=4.68e-1,
#             J_l=2,
#             J_u=1
#             )

