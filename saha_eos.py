"""
Here is an extract from some of Mike's equation of state code, focusing on the 
Saha equation.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
import scipy.optimize as op
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from astropy.table import Table
#For fsolve - no idea why this happens at high temperatures...
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
np.seterr(divide='raise')
#Some constants! For speed, not readability.
debroglie_const = (c.h**2/2/np.pi/c.m_e/c.k_B).cgs.value
eV_Boltzmann_const = (u.eV/c.k_B).cgs.value
deybe_const = (c.k_B/8/np.pi/c.e.esu**2).cgs.value
delchi_const = (c.e.esu**2/(1*u.eV)).cgs.value

def solarmet(A_O_log=8.69):
    """Return solar metalicity abundances by number and masses for low mass elements.
    From Asplund et al (2009), up to an abundance of 1e-5 only plus Na, K, Ca. 
    Degeneracy and ionization energies from IA05 in Scholz code"""
    elt_names = np.array(['H', 'He',   'C',   'N',   'O',   'Ne',  'Na',   'Mg',   'Al',  'Si',  'S',   'Cl', 'K',   'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni'])
    n_p =        np.array([1,   2,      6,     7,     8,     10,    11,     12,      13,    14,   16,     17,  19,     20,   22,   24,   24,   26,   27,  28 ])
    masses=      np.array([1.0, 4.0,    12.01, 14.01, 16.00, 18.0,  22.99, 24.31, 26.98, 28.09, 32.06, 35.45, 39.10,40.08, 47.87, 52.0, 54.94, 55.85, 58.93,63.55])
    abund = 10**(np.array([12, 10.93,   8.43,  7.83,  8.69,  7.93,  6.24,  7.60,   6.43,  7.51,  7.12,  5.31,  5.03, 6.30,  4.97, 5.62, 5.42, 7.46, 4.94, 6.20])-12)
    O_ix = np.where(elt_names == 'O')[0][0]
    abund[O_ix] = 10**(A_O_log - 12.0)
    ionI  =      np.array([13.595,24.58,11.26, 14.53, 13.61, 21.56, 5.14,  7.644,  5.98,  8.149,10.36, 13.01, 4.339,6.111, 6.80, 6.74, 6.76, 7.87, 7.88 ,7.63])
    ionII  =     np.array([-0.754,  54.403,  24.376,29.593,35.108,40.96, 47.29, 15.03, 18.82,  16.34, 23.40,23.80, 31.81,11.87, 13.57, 16.49, 15.64,16.18, 17.08, 18.15])

    #Degeneracy of many of these elements are somewhat temperature-dependent,
    #as it is really a partition function. But as this is mostly H/He plus 
    #other elements as a mass reservoir and source of low-T
    #electrons, we're ignoring this. 
    gI =   np.array([2,1,9,4,9,1,2,1,6,9,9,6,2,1,21,7,6,25,25,21])
    gII =  np.array([1,2,6,9,4,6,1,2,1,6,4,9,1,2,28,6,7,30,30,10])

    #A lot of these degeneracies are educated guesses! But we're not worried
    #about most elements in the doubly ionized state. The last 5 are made up.
    gIII = np.array([1,1,1,6,9,9,6,1,2,1,9,9,6,1,25,10,10,10,10,10])

    return abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names
        
        

def saha(n_e, T, A_O_log=8.69):
    """Compute the solution to the Saha equation as a function of electron number
    density and temperature, in CGS units. 
    
    This enables the problem to be a simple Ax=b linear problem.
    Results from this function can be used to solve the Saha equation as e.g. a function 
    of rho and T via e.g. tabulating or solving.
    
    Parameters
    ----------
    n_e: the dimensioned electron number density in cm^-3
    T: Temperature in K.
    
    Returns
    -------
    rho: astropy.units quantity compatible with density
    mu: Mean molecular weight (dimensionless, i.e. to be multiplied by the AMU)
    ns: A vector of number densities of H, H+, H-, He, He+, He++ in cm^{-3}
        (i.e. H- is returned as a special case)
    """
    
    #Input the abundances of the elements
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names = solarmet(A_O_log=A_O_log)
    n_elt = len(n_p)
    
    #Find the Deybe length, and the decrease in the ionization potential in eV, 
    #according to Mihalas 9-178
    deybe_length = np.sqrt(deybe_const*T/n_e)
    z1_delchi = delchi_const/deybe_length
    
    #This will break for very low temperatures. In this case, fix a zero  
    #ionization fraction
    if (T<1000):
        ns = np.zeros(n_elt*3)
        ns[3*np.arange(n_elt)] = abund
        ns = np.maximum(n_e*1e15*ns, 1e-300)
    else:
        #The thermal de Broglie wavelength. See dimensioned version of 
        #this constant above
        debroglie=np.sqrt(debroglie_const/T)
    
        #Hydrogen ionization. We neglect the excited states because
        #they are only important when the series diverges... 
        #!!! h1  = 2./debroglie**3 *gII[0]/gI[0]*np.exp(-(ionI[0] - n_p[0]*z1_delchi)*eV_Boltzmann_const/T)  
    
        #Now construct our matrix of n_elt*3-1 equations defining these number densities.
        A = np.zeros( (3*n_elt,3*n_elt) );
        for i in range(n_elt):
            #Our first equation for this element: If the element is H, 
            #We add in all the electrons that come from all other elements, and get
            #the total number of electrons.
            #For all other elements, the ratio of the sum of element number
            #densities to H number densities is the ratio of abundances.
            if (i==0):
                A[0,:] =np.concatenate(([0,1,-1], np.tile([0,1,2], n_elt-1)))
            else:
                A[3*i,  3*i:3*(i+1)] = [-abund[0],-abund[0],-abund[0]]
                A[3*i,  :3] = [abund[i],abund[i],abund[i]]
            
            #Element single ionization. 
            el1 = 2./debroglie**3 *gII[i]/gI[i]*np.exp(-(ionI[i] - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)
            #The second equation for this element: 
            A[3*i+1,3*i:3*i+2]=[-el1/n_e, 1]
                
            #Element double-ionization, or H-
            if i == 0:
                el2 = 2./debroglie**3 *gI[i]/gIII[i]*np.exp(-(np.abs(ionII[i]) - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)
                A[3*i+2,3*i:3*i+3]  =[1,0,-el2/n_e]   
            else:
                #The same as the single ionization formula! 
                try:
                    el2 = 2./debroglie**3 *gIII[i]/gII[i]*np.exp(-(ionII[i] - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)
                except:
                    import pdb; pdb.set_trace()
                A[3*i+2,3*i+1:3*i+3]  =[-el2/n_e, 1]
        
        #Convert the electron density to a dimensioless value prior to solving.
        b =np.zeros((3*n_elt))
        b[0] = n_e
        ns =np.linalg.solve(A,b)
        
        #Make sure that all number densities are positive.
        ns = np.maximum(ns,1e-300)
    
    #The next lines ensure ionization at high electron pressure, roughly due to nuclei 
    #being separated by less than the size of an atom. 
    #There is also a hack included based on a typical atom size.
    ns_highT = np.zeros(n_elt*3 - 1)
    ns_highT[1 + np.arange(n_elt)*3] = abund
    ns_highT=ns_highT/(abund[0]+2*np.sum(abund[1:]))*n_e
    atom_size = 1e-8 #In cm
    if (n_e*atom_size**3 > 2):
        ns=ns_highT
        print("High T")
    elif (n_e*atom_size**3 > 1):
        frac=((n_e*atom_size **3) - 1)/1.0
        ns = frac*ns_highT + (1-frac)*ns
        
    #For normalization... we need the number density of Hydrogen
    #nuclei, which is the sum of the number densities of H and H+.
    n_h = np.sum(ns[:2])
    
    #Density. Masses should be scalars.
    rho_cgs = n_h*np.sum(abund*masses)*c.u.to(u.g).value
   
    #Fractional "abundance" of electrons.
    f_e = n_e/n_h
    
    #mu is mean "molecular" weight, and we make the approximation that
    #electrons have zero weight.
    mu = np.sum(abund*masses)/(np.sum(abund) + f_e)
    
    #Finally, we should compute the internal energy with respect to neutral gas.
    #This is the internal energy per H atom, divided by the mass in grams per H atom. 
    Ui=(ns[1]*13.6 + ns[3]*24.58 + ns[4]*(54.403+24.58))*u.eV/n_h/np.sum(abund*masses*u.u);
    
    return rho_cgs, mu, Ui, ns
    
def saha_solve(log_n_e_mol_cm3, T, rho_0_in_g_cm3):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed density."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T)
    
    return np.log(rho_0_in_g_cm3/rho)
 
def ns_from_rho_T(rho,T):
    """Compute number densities given a density and temperature
    
    Parameters
    ----------
    rho: density in g/cm^-3
    T: Temperature in K.
    
    Returns
    -------
    Electron number density, element & ion number densities, mean molecular weight
    and internal energy.
    """
    
    rho_in_g_cm3 = rho.to(u.g/u.cm**3).value
    
    #Start with the electron number density equal in mol/cm^3 equal to the density
    #in g/cm^3, or a much lower number at low temperatures. Modify it a little to help
    #starting point at low T
    x0 = np.log(rho_in_g_cm3)
    T_K = np.maximum(T.to(u.K).value,1000)
    x0 += np.log(2/(np.exp(50e3/T_K) + 1))
    
    #The following line is the important one that can't have units associated
    #with it, as it takes too long.
    res = op.fsolve(saha_solve, x0, args=(T.cgs.value, rho_in_g_cm3), xtol=1e-6)
    n_e = np.exp(res[0])*c.N_A.value
    rho_check, mu, Ui, ns = saha(n_e, T.cgs.value)
    if (np.abs(rho_check/rho_in_g_cm3-1) > 0.01):
        raise UserWarning("Density check incorrect!")

    #Return dimensioned quantities
    return n_e*(u.cm**(-3)), ns*(u.cm**(-3)), mu, Ui 

def ns_from_P_T(P, T, A_O_log=8.69):
    """Compute number densities given a pressure and temperature"""
    
    P_cgs = P.to(u.dyne/u.cm**2).value
    
    #Same as ns_from_rho_T, at a mu=1.
    x0 = np.log(P_cgs/((c.k_B*T*u.K/u.u).cgs.value))
    T_K = np.maximum(T.to(u.K).value,1000)
    x0 += np.log(2/(10*np.exp(40e3/T_K) + 1))
    
    #The following line is the important one that can't have units associated
    #with it, as it takes too long.

    res = op.fsolve(saha_solve_P, x0, args=(P_cgs, T.cgs.value, A_O_log), xtol=1e-6)
    n_e = np.exp(res[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T.cgs.value, A_O_log=A_O_log)
    
    #Return dimensioned quantities
    return n_e*(u.cm**(-3)), ns*(u.cm**(-3)), mu, Ui, rho

def saha_solve_P(log_n_e_mol_cm3, P_0_cgs, T, A_O_log):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed density."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T, A_O_log=A_O_log)
    #Ideal gas equation...
    P = rho/mu*(c.k_B*T*u.K/u.u).cgs.value
    return np.log(P_0_cgs/P)


def P_T_tables(Ps, Ts, savefile=''):
    """
    For an array of densities and specific entropies, create tables of
    density (TODO: gammas and entropy).
    
    Pressures are assumed in a logarithmic grid.
    
    Parameters
    ----------
    Ps : array
        input pressure list.
    Ts : array
        input temperature list.

    Returns
    -------
    density, internal energy, number densities tables

    """
    # Some defaults
    if Ps is None:
        Ps = np.logspace(-4,6,41)*u.dyn/u.cm**2
    if Ts is None:
        Ts = (2000 + 500*np.arange(47))*u.K
        
    
    Ps_log = np.log(Ps.to(u.dyne/u.cm**2).value)
    nP = len(Ps)
    nT = len(Ts)
    rho_tab = np.empty((nP, nT))
    mu_tab = np.empty((nP, nT))
    Ui_tab = np.empty((nP,nT))
    Q_tab = np.empty((nP,nT))
    cP_tab = np.empty((nP,nT))
    
    #Find the number of atoms and ions
    n_e, ns, mu, Ui, rho  = ns_from_P_T(Ps[0], Ts[0])
    n_species = len(ns)
    
    ns_tab = np.empty((nP,nT,n_species))
    n_e_tab = np.empty((nP,nT))
    
    #gamma1_tab = np.empty((nT, nP))
    #entropy_tab = np.empty((nT, nP))
    dT = 1*u.K #A small amount of temperature!
    k_b_u_cgs = (c.k_B/c.u).cgs.value
    for i, P in enumerate(Ps):
        for j, T in enumerate(Ts):
            #Compute number densities and densities, and also a single-sided derivative
            n_e, ns, mu, Ui, rho  = ns_from_P_T(P, T)
            _, _, mu_plus, Ui_plus, rho_plus = ns_from_P_T(P, T + dT)
            
            #Fill in the tables not involving derivatives
            rho_tab[i,j] = rho #Already in cgs
            Ui_tab[i,j] = Ui.to(u.erg/u.g).value
            mu_tab[i,j] = mu
            ns_tab[i,j] = ns.cgs.value
            n_e_tab[i,j] = n_e.cgs.value
            
            #Now the tables involving derivatives
            Q_tab[i,j] = 1 - (mu_plus - mu)/dT * T/mu
            cP_tab[i,j] = ((Ui_plus - Ui)/dT).cgs.value + 5/2*k_b_u_cgs*Q_tab[i,j]/mu_tab[i,j] 

        
    if len(savefile)>0:
            hdu1 = pyfits.PrimaryHDU(rho_tab)
            hdu1.header['CRVAL1'] = Ts[0].cgs.value
            hdu1.header['CDELT1'] = (Ts[1]-Ts[0]).cgs.value
            hdu1.header['CTYPE1'] = 'Temperature [K]'
            hdu1.header['CRVAL2'] = (Ps_log[0])/np.log(10)
            hdu1.header['CDELT2'] = (Ps_log[1]-Ps_log[0])/np.log(10)
            hdu1.header['CTYPE2'] = 'log10(pressure) [dyne/cm^2]'
            hdu1.header['EXTNAME'] = 'rho [g/cm**3]'
            hdu2 = pyfits.ImageHDU(Ui_tab)
            hdu2.header['EXTNAME'] = 'Ui [erg/g]'
            hdu3 = pyfits.ImageHDU(mu_tab)
            hdu3.header['EXTNAME'] = 'mu'
            hdu4 = pyfits.ImageHDU(ns_tab)
            hdu4.header['EXTNAME'] = 'ns [cm^-3]'
            hdu5 = pyfits.ImageHDU(n_e_tab)
            hdu5.header['EXTNAME'] = 'n_e [cm^-3]'
            hdu6 = pyfits.ImageHDU(Q_tab)
            hdu6.header['EXTNAME'] = 'Q'
            hdu7 = pyfits.ImageHDU(cP_tab)
            hdu7.header['EXTNAME'] = 'cP [erg/K/g]'
            hdulist = pyfits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5,hdu6,hdu7])
            hdulist.writeto(savefile, overwrite=True)
    return rho_tab, Ui_tab, mu_tab, ns_tab
 

 

def calculate_excitation_fraction(T, g_i, chi_i, Z_T):
    """
    Calculates the fraction of atoms in a specific excitation state 'i'
    using the Boltzmann Equation.

    Parameters
    ----------
    T : astropy.units.Quantity
        Temperature in Kelvin.
    g_i : float
        Statistical weight of the level 'i'.
    chi_i : float
        Excitation energy of the level 'i' in cm^-1.
    Z_T : float
        The partition function at temperature T.

    Returns
    -------
    fraction : float
        The fraction of atoms in state 'i' (n_i / N_total).
    """
    # Convert excitation energy from cm^-1 to Joules
    chi_i_J = chi_i * u.cm**-1 * c.h * c.c
    
    # Calculate kT in Joules
    kT_J = c.k_B * T
    
    # Boltzmann Equation: (g_i / Z) * exp(-chi_i / kT)
    fraction = (g_i / Z_T) * np.exp(-chi_i_J / kT_J)
    
    return fraction




if __name__=='__main__':
	#Saha Test - number density of electrons, and temperature!
    rho, mu, Ui, ns = saha(1e18, 50000)
    
    #Now the calculations
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names  = solarmet()
    np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
    print("Hydrogen Ionisation Fraction: {:.4f}".format(ns[1]/(ns[0] + ns[1])))
    print("Neutral, 1st ionisation, 2nd ionisation fraction of other elements: ")
    for i in range(len(elt_names)-1):
        print("{:2s}:".format(elt_names[i+1]) + str(ns[2+3*i:5+3*i]/np.sum(ns[0:2])/abund[i+1]))
    print("Density (g/cm^3): {:.2e}".format(rho))
        
