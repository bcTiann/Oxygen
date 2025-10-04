import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn
import astropy.units as u
import astropy.constants as c
from scipy.integrate import cumulative_trapezoid as cumtrapz

def lambda_matrix(tau_grid):
	"""
	Compute the Lambda operator as a matrix based on a tau grid.
	
	Parameters
	----------
	tau_grid: (n_tau) numpy.array
		A grid of optical depth points, starting at 0 and monotonically increasing.
	
	Returns
	-------
	lambda_mat: (n_tau) numpy.array
		A matrix defined so that to obtain vector of mean intensities J from a vector 
		of source functions S we simply compute: J=numpy.dot(lambda_mat, S)
	"""
	#Fill our final result matrix with zeros
	lambda_mat = np.zeros( (len(tau_grid), len(tau_grid)) )
	
	#Create a delta-tau grid
	delta_tau = tau_grid[1:] - tau_grid[:-1]
	
	#For simplicity and readability, just go through one layer at a time.
	for j in range(len(tau_grid)):
		#Create E2 and E3 vectors
		E2_vect = expn(2,np.abs(tau_grid - tau_grid[j]))
		E3_vect = expn(3,np.abs(tau_grid - tau_grid[j]))
		
		#Add the contribution from the i'th layer, for upwards going rays
		lambda_mat[j,j:-1] +=  E2_vect[j:-1] - (E3_vect[j:-1] - E3_vect[j+1:])/delta_tau[j:] 
		
		#Add the contribution from the i+1'th layer, for upwards going rays
		lambda_mat[j,j+1:]  += -E2_vect[j+1:]  + (E3_vect[j:-1] - E3_vect[j+1:])/delta_tau[j:]
		
		#Add the contribution from the i'th layer, for downwards going rays
		lambda_mat[j,1:j+1] +=  E2_vect[1:j+1] - (E3_vect[1:j+1] - E3_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the i-1'th layer, for downwards going rays
		lambda_mat[j,:j]  += -E2_vect[:j]  + (E3_vect[1:j+1] - E3_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the lower boundary condition
		lambda_mat[j,-1] +=  E3_vect[-1]/delta_tau[-1] + E2_vect[-1]
		lambda_mat[j,-2] += -E3_vect[-1]/delta_tau[-1]
	return 0.5*lambda_mat
	
def phi_matrix(tau_grid):
	"""
	Compute the phi operator as a matrix based on a tau grid.
	
	This differs from the Lambda matrix because of a sign difference.
	
	Parameters
	----------
	tau_grid: (n_tau) numpy.array
		A grid of optical depth points, starting at 0 and monotonically increasing.
	
	Returns
	-------
	phi_mat: (n_tau) numpy.array
		A matrix defined so that to obtain vector of Eddington fluxes H from a vector 
		of source functions S we simply compute: H=numpy.dot(phi_mat, S)
	"""
	#Fill our final result matrix with zeros
	phi_mat = np.zeros( (len(tau_grid), len(tau_grid)) )
	
	#Create a delta-tau grid
	delta_tau = tau_grid[1:] - tau_grid[:-1]
	
	#For simplicity and readability, just go through one layer at a time.
	for j in range(len(tau_grid)):
		#Create E3 and E4 vectors
		E3_vect = expn(3,np.abs(tau_grid - tau_grid[j]))
		E4_vect = expn(4,np.abs(tau_grid - tau_grid[j]))
		
		#Add the contribution from the i'th layer, for upwards going rays
		phi_mat[j,j:-1] +=  E3_vect[j:-1] - (E4_vect[j:-1] - E4_vect[j+1:])/delta_tau[j:] 
		
		#Add the contribution from the i+1'th layer, for upwards going rays
		phi_mat[j,j+1:]  += -E3_vect[j+1:]  + (E4_vect[j:-1] - E4_vect[j+1:])/delta_tau[j:]
		
		#Add the contribution from the i'th layer, for downwards going rays
		phi_mat[j,1:j+1] -=  E3_vect[1:j+1] - (E4_vect[1:j+1] - E4_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the i-1'th layer, for downwards going rays
		phi_mat[j,:j]  -= -E3_vect[:j]  + (E4_vect[1:j+1] - E4_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the lower boundary condition
		phi_mat[j,-1] +=  E4_vect[-1]/delta_tau[-1] + E3_vect[-1]
		phi_mat[j,-2] += -E4_vect[-1]/delta_tau[-1]
	return 0.5*phi_mat

	
def calc_Bnu(T, nu):
	return ( 2*c.h*nu**3/c.c**2 / (np.exp(c.h*nu/c.k_B/T) - 1) ).to(u.erg/u.cm**2)
	
if __name__=="__main__":
	np.set_printoptions(precision=3)
	#Part (c) and (d)
	#Create a couple of tau grids. The second is random.
	tau_grids = [np.array([0,1,2]), \
		np.concatenate(([0],np.cumsum(np.random.random(5))))]

	#Create the Lambda matrices
	for tau_grid in tau_grids:
		print("\n tau Grid:")
		print(tau_grid)
		ntau = len(tau_grid)
		lambda_mat = lambda_matrix(tau_grid)
		Jconst = np.dot(lambda_mat, np.ones(ntau))
		Jlin = np.dot(lambda_mat, 1 + tau_grid)
		print("J for constant S:")
		print(Jconst)
		print("J for S = 1 + tau:")
		print(Jlin)

	print("\nEddington-Barbier for J (constant S): {:.3f}".format((1 + 0)/2))
	print("Eddington-Barbier for J   (linear S): {:.3f}".format((1 + 1/2)/2))


	#Part (f) Firstly - the continuum opacity
	tau = np.array([0,.5,1,1.5,2,3,5])
	T = 6500*u.K* (3/4 * (tau + 2/3))**0.25
	lambda_mat = lambda_matrix(tau)
	I = np.eye(len(tau))
	for wave in [200*u.nm, 390*u.nm, 2000*u.nm]:
		print("\n--- Wavelength: {:.1f} ---".format(wave))
		nu = c.c/wave
		Bnus = calc_Bnu(T, nu)
		J_LTE = np.dot(lambda_mat, Bnus)
		J_scat = np.linalg.solve(I - np.dot(lambda_mat, 0.8*I), 0.2*np.dot(lambda_mat, Bnus))
		print('LTE intensity at surface : {:.2e}'.format(J_LTE[0]))
		print('With Scattering: {:.2e}'.format(J_scat[0]))
		
	#Part (g)
	#Now for the line opacity. Create a tau grid, that is finely spaced between 0 and 0.1,
	#then coarsely spaced to deeper tau
	#Also create the corresponding T(tau) grid
	tau = np.arange(40)/200
	tau[20:] = 0.1 + np.arange(20)/10
	T = 6500*u.K* (3/4 * (tau + 2/3))**0.25
	nu = c.c/(393*u.nm)
	Bnus = calc_Bnu(T, nu)
	I = np.eye(len(tau))
	
	#This is the ratio of the line scattering to continuum opacity
	line_strength = 100
	
	#Integrate the ratio of kappa_nu/kappa_R to get tau_nu(tau)
	chi_nu_rat = np.ones(40)
	chi_nu_rat[:20] = line_strength + 1
	tau_nu = np.concatenate(([0],cumtrapz(chi_nu_rat, tau)))
	 
	#Create the matrices, and make the computation!
	eps = np.ones_like(tau_nu)
	eps[:20] = 1/(line_strength + 1)
	eps_mat = np.diag(eps)
	lambda_mat = lambda_matrix(tau_nu)
	J_LTE = np.dot(lambda_mat, Bnus)
	J_scat = np.linalg.solve(I - np.dot(lambda_mat, I-eps_mat), np.dot(lambda_mat, eps*Bnus))
	print("\n--- Line Core computation: ---")
	print("LTE line flux: {:.2e}".format(J_LTE[0]))
	print("Scattered line flux: {:.2e}".format(J_scat[0]))

	#Repeat, but for a whole line in order to really see the effect on the line core.
	nnu = 81
	dnu = np.linspace(-20,20,nnu)
	line_strengths = 100/(1 + dnu**2)
	J_LTE = np.empty(nnu)
	J_scat = np.empty(nnu)
	H_LTE = np.empty(nnu)
	H_scat = np.empty(nnu)
	for i, line_strength in enumerate(line_strengths):
		#Integrate the ratio of kappa_nu/kappa_R to get tau_nu(tau)
		chi_nu_rat[:20] = line_strength + 1
		tau_nu = np.concatenate(([0],cumtrapz(chi_nu_rat, tau)))
	 
		#Create the matrices, and make the computation!
		eps[:20] = 1/(line_strength + 1)
		eps_mat = np.diag(eps)
		lambda_mat = lambda_matrix(tau_nu)
		J_LTE[i] = np.dot(lambda_mat, Bnus)[0].cgs.value
		J_all_layers = np.linalg.solve(I - np.dot(lambda_mat, I-eps_mat), np.dot(lambda_mat, eps*Bnus))
		J_scat[i] = J_all_layers[0].cgs.value
		
		#From J, we can get S and then H as well.
		S_all_layers = (1-eps)*J_all_layers + eps * Bnus
		phi_mat = phi_matrix(tau_nu)
		H_LTE[i] = np.dot(phi_mat, Bnus)[0].cgs.value
		H_scat[i] = np.dot(phi_mat, S_all_layers)[0].cgs.value

	#Make a mean intensity line figure
	plt.figure(1)
	plt.clf()
	plt.plot(dnu, J_LTE, label='LTE')
	plt.plot(dnu, J_scat, label='With Scattering')
	plt.xlabel(r"$(\nu - \nu_0)/\Gamma$")
	plt.ylabel("Mean Intensity J")
	plt.legend()

	#Make a flux line figure
	plt.figure(2)
	plt.clf()
	plt.plot(dnu, H_LTE, label='LTE')
	plt.plot(dnu, H_scat, label='With Scattering')
	plt.xlabel(r"$(\nu - \nu_0)/\Gamma$")
	plt.ylabel("Eddington Flux H")
	plt.legend()


	#Here was a another test I did - to check if we can get close to the Hopf q function
	if True:
		tau = np.linspace(0,4,100)**2
		lambda_mat = lambda_matrix(tau)
		eigenvalues, eigenvectors = np.linalg.eig(lambda_mat)
		ix_eval1 = np.argmin(np.abs(eigenvalues-1))
		J = np.dot(lambda_mat,eigenvectors[:,ix_eval1])
		J /= (J[-1]-J[-2])/(tau[-1]-tau[-2])
		q = J - tau
		tau_print = [0,.01,.02,.04,.06,.1,.2,.4,.6,1,1.5,2]
		print('\ntau	q')
		print('----------')
		for t in tau_print:
			print("{:.3f} {:.6f}".format(t, np.interp(t, tau, q)))
		
