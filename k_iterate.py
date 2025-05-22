import numpy as np
import math
from IRAM import iram
from moving_mesh_transport.solver_classes.functions import *
from moving_mesh_transport.solver_classes.make_phi import make_output
import time
from scipy.interpolate import interp1d as interp1d
from scipy import integrate as integrate

def integrate_phi_cell(cs, ws, a, b, M, N_ang):
    cell_volume = 4 * math.pi * (b**3 - a**3)
    psi = np.zeros(N_ang)
    for l in range(N_ang):
        for j in range(M+1):
            psi[l] += cs[l, j] * normTn_intcell(j, a, b)
    res = np.sum(np.multiply(psi,ws))
    return 4 * math.pi * res #* cell_volume

def power_iterate(kguess, transport_parameters, mesh_parameters, run, tol = 1e-12):
    """
    Calls the solver and updates k_eff until desired tolerance between sucessive k_values is achieved

    parameters:
    -------------------------
    - kguess: starting k_eff
    - transport_parameters: name of YAML file for solver parameters
    - mesh_parameters: name of YAML file for mesh parameters
    - run: solver object

    returns:
    -------------------------
    - k_list: list of sucessive dominant k values
    - calc_time_list: list of computation timees

    """

    run.load(transport_parameters, mesh_parameters)
    klist = []
    klist.append(kguess)
    k_old = kguess
    converged = False
    sigma_f = run.parameters['all']['sigma_f']
    nu = run.parameters['all']['nu']
    chi = run.parameters['all']['chi']
    coeffs = run.sol_ob.y[:,-1]
    N_ang = run.parameters['fixed_source']['N_angles'][0]
    if run.parameters['all']['angular_derivative']['diamond'] == True:
        N_ang += 2
    ws = run.ws
    N_groups = run.parameters['all']['N_groups']
    M  = run.parameters['all']['Ms'][0]
    N_space = run.parameters['all']['N_spaces'][0]
    tt = run.parameters['all']['tfinal']
    run.custom_source(randomstart = True, uncollided = 0, moving = 0)
    coeffs_old = np.copy(run.sol_ob.y[:,-1].reshape((N_ang * N_groups, N_space, M+1)))
    uncollided = False
    # geometry = run.parameters['all']['geometry']
    geometry = run.geometry
    uncollided_ob = run.uncollided_ob
    edges = run.edges
    phi_interpolated = interp1d(run.xs, run.phi[:,0])

            # self.sigma_f = np.zeros(self.N_space)
            # self.nu = np.zeros(self.N_space)
    edges2 = edges - run.parameters['fixed_source']['shift']
    sigma_f_array = np.ones(run.xs.size) *sigma_f
    for k in range(run.xs.size):
        if -3.5 < run.xs[k] < 3.5:
            sigma_f_array[k] = 0.0 
    sigma_interp = interp1d(run.xs, sigma_f_array)
    integrand = lambda x:  phi_interpolated(x) * x**2 * 4 * math.pi * sigma_interp  # because nu and sigma_t are constant right now, I don't need them in the integrand
    # normalization = integrate.quad(integrand, run.xs[0], run.xs[-1])[0]
    normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)), edges, ws, N_ang, M, N_space, N_groups)
    n_iters = 0
    normalization_list = []
    calc_time_list = []
    normalization_list.append(normalization)

    while converged == False and n_iters < 15: 
        run.load(transport_parameters, mesh_parameters)
        # scale sigma_f
        # run.parameters['all']['sigma_f'] = sigma_f / k_old
        # normalize fission source
        normalized_source = coeffs_old / normalization
        # run solver    
        t1 = time.time()
        run.custom_source(randomstart = False, sol_coeffs = normalized_source, uncollided = 0, moving = 0)
        coeffs_old = np.copy(run.sol_ob.y[:,-1].reshape((N_ang * N_groups, N_space, M+1)))
        t_calc = time.time() - t1
        # update k
        xs = run.xs
        phi_interpolated_new = interp1d(run.xs, run.phi[:,0])
        integrand = lambda x:  phi_interpolated_new(x) * x**2 * 4 * math.pi * sigma_interp
        integrand_old = lambda x: (phi_interpolated(x)+ 1e-12) * x**2 * 4 * math.pi *sigma_interp
        k_new = k_old * integrate.quad(integrand, xs[0], xs[-1])[0] / integrate.quad(integrand_old, xs[0], xs[-1])[0]
        k_new2 = k_old * normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)), edges, ws, N_ang, M, N_space, N_groups) / normalization
        if k_new <0:
            raise ValueError('negative k_eff')
        # k_new = k_old * normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups,N_space,M+1)), edges, ws, N_ang, M, N_space, N_groups, sigma_f, nu, chi) / normalization
        print(k_new, 'k')
        print(k_new2, 'k2')
        # converged = True
        if abs(k_new - k_old ) <=tol:
            print('power iteration complete')
            print(k_new, 'k effective')
            converged = True
        else:
            k_old = k_new
            phi_interpolated = phi_interpolated_new
            klist.append(k_new)
            n_iters +=1
            normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)), edges, ws, N_ang, M, N_space, N_groups)
            normalization_list.append(normalization)
            calc_time_list.append(t_calc)
    
    return klist, calc_time_list, normalization_list