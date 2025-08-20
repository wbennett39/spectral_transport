import numpy as np
import math
from IRAM import iram
from moving_mesh_transport.solver_classes.functions import *
from moving_mesh_transport.solver_classes.make_phi import make_output
import time
from scipy.interpolate import interp1d as interp1d
from scipy import integrate as integrate
import matplotlib.pyplot as plt

def integrate_phi_cell(cs, ws, a, b, M, N_ang):
    # cell_volume = 4 * math.pi * (b**3 - a**3)
    # normTn_intcell includes the r^2 term in the integrand
    psi = np.zeros(N_ang)
    for l in range(N_ang):
        for j in range(M+1):
            psi[l] += cs[l, j] * normTn_intcell(j, a, b)
    res = np.sum(np.multiply(psi,ws))
    return 4 * math.pi * res #* cell_volume

def wynn_epsilon(S):
        n = S.size
        width = n-1
        # print(width)
        tableau = np.zeros((n + 1, width + 2))
        tableau[:,0] = 0
        tableau[1:,1] = S.copy() 
        for w in range(2,width + 2):
            for r in range(w,n+1):
                #print(r,w)
                # if abs(tableau[r,w-1] - tableau[r-1,w-1]) <= 1e-15:
                #     print('potential working precision issue')
                tableau[r,w] = tableau[r-1,w-2] + 1/(tableau[r,w-1] - tableau[r-1,w-1])
        return tableau

def power_iterate(kguess, transport_parameters, mesh_parameters, run, tol = 1e-12, use_we_accel = False):
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
    # klist.append(kguess)
    # k_old = kguess
    # k_old = kguess
    # klist.append(kguess)
    converged = False
    sigma_f = run.parameters['all']['sigma_f']
    nu = run.parameters['all']['nu']
    chi = run.parameters['all']['chi']
    coeffs = run.sol_ob.y[:,-1]
    N_ang = run.parameters['fixed_source']['N_angles'][0]
    if run.parameters['all']['angular_derivative']['diamond'] == True:
        N_ang += 1
    ws = run.ws
    N_groups = run.parameters['all']['N_groups']
    M  = run.parameters['all']['Ms'][0]
    N_space = run.parameters['all']['N_spaces'][0]
    tt = run.parameters['all']['tfinal']
    

    
    run.custom_source(randomstart = True, uncollided = 0, moving = 0)
    res_coefficients_new = np.copy(run.sol_ob.y[:,-1].reshape((N_ang * N_groups, N_space, M+1)))
    coeffs_old = res_coefficients_new.copy()
    IC = np.copy(run.IC.reshape((N_ang * N_groups, N_space, M+1)))
    initial_fission_source = IC.copy()
    sigma_f_vec = np.ones(N_space) * sigma_f
    nu_vec = np.ones(N_space) * nu
    edges = run.edges
    shift = run.parameters['fixed_source']['shift']
    for space in range(N_space):
                left_edge = edges[space]-shift
                right_edge = edges[space+1]-shift
                if -3.5 <= left_edge <= 3.5 and -3.5 <= right_edge <= 3.5:
                    sigma_f_vec[space] = 0.0   
                    nu_vec[space] = 0.0
    # calculate the fission source from the random IC
    for k in range(N_space):
             res_coefficients_new[:, k, :] *= sigma_f_vec[k] * nu_vec[k]
             initial_fission_source[:, k, :] *= sigma_f_vec[k] * nu_vec[k]
    # S_old = IC / normalize_phi(initial_fission_source, edges, ws, N_ang, M, N_space, N_groups)
    S_old = 1

    S_old = normalize_phi(initial_fission_source, edges, ws, N_ang, M, N_space, N_groups)
    # print(k0, 'k0')
    print(S_old, 'S0')
    # calculate the new fission source
    S_new = normalize_phi(res_coefficients_new, edges, ws, N_ang, M, N_space, N_groups)
    print(S_new, 'S1')
    k_old = kguess * S_new/ S_old 
    
    # Initializing k 
    uncollided = False
    
    sigma_f_array = np.ones(run.xs.size) * sigma_f
    nu_array = np.ones(run.xs.size) * nu
    # # geometry = run.parameters['all']['geometry']
    for k in range(run.xs.size): # build the fission production vector for the Kornreich problem
        if -3.5 <= run.xs[k]-shift <= 3.5:
            sigma_f_array[k] = 0.0 
            nu_array[k] = 0.
    # geometry = run.geometry
    # uncollided_ob = run.uncollided_ob
    
    # phi_interpolated = interp1d(run.xs, run.phi[:,0])

            # self.sigma_f = np.zeros(self.N_space)
            # self.nu = np.zeros(self.N_space)

  
    
    # print(sigma_f_array, 'sigmaf')
    # sigma_interp = interp1d(run.xs, sigma_f_array * nu_array) # interpolated fission rate 
    # integrand = lambda x:  phi_interpolated(x) * x**2 * 4 * math.pi * sigma_interp(x) 
    # S_new = integrate.quad(integrand, run.xs[0], run.xs[-1])[0]
    
    S_old = S_new
    # klist.append(k0)
    # klist.append(k_old)
    n_iters = 1

    # plt.ioff()
    # plt.figure('fission')
    # xtest = np.linspace(edges[0], edges[-1], 100)
    # plt.plot(xtest, sigma_interp(xtest))
    # plt.show()
    # assert 0
    # normalization = integrate.quad(integrand, run.xs[0], run.xs[-1])[0]
    # normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)), edges, ws, N_ang, M, N_space, N_groups) # this is broken. I would need to multiply by sigma nu
    # n_iters = 0
    normalization_list = []
    calc_time_list = []
    # normalization_list.append(normalization)
    plt.close()
    plt.close()
    plt.close()
    k_old = kguess
    klist.append(k_old)
    


    while converged == False and n_iters < 100: 
        run.load(transport_parameters, mesh_parameters) # reset parameters to agree with YAML file
        # the source is actually not normalized
        normalized_source = coeffs_old/ k_old #/ normalization
        # run solver    
        t1 = time.time()
        run.parameters['all']['kold'] = k_old
        run.custom_source(randomstart = False, sol_coeffs = normalized_source, uncollided = 0, moving = 0)
        t_calc = time.time() - t1
        # update k
        xs = run.xs
        # phi_interpolated = interp1d(run.xs, run.phi[:,0])
        # integrand = lambda x:  phi_interpolated(x) * x**2 * 4 * math.pi * sigma_interp(x)  # new fission source 
        # plt.figure(2)
        # plt.plot(xs, sigma_interp(xs), '-')
        # plt.show()
        # integrand_old = lambda x: (phi_interpolated(x)) * x**2 * 4 * math.pi * sigma_interp(x) # old fission source

        # output_ob = make_output(500, N_ang, ws, xs, normalized_source, M, edges, uncollided, geometry, N_groups)
        # phi_old = output_ob.make_phi_no_uncol()
        # phi_interpolated = interp1d(xs, phi_old[:,0])
        # integrand_old = lambda x: (phi_interpolated(x)) * x**2 * 4 * math.pi * sigma_interp(x) # old fission source
        # k_new = k_old *  integrate.quad(integrand, xs[0], xs[-1])[0] / integrate.quad(integrand_old, xs[0], xs[-1])[0]
        # k_new = integrate.quad(integrand, xs[0], xs[-1])[0] 
        # print(k_new, 'k from scipy integration')
        coeffs_old = run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1))
        res_coefficients_new = coeffs_old.copy()
        for k in range(N_space):
             res_coefficients_new[:, k, :] *= sigma_f_vec[k] * nu_vec[k]
        S_new = normalize_phi(res_coefficients_new, edges, ws, N_ang, M, N_space, N_groups) #/ normalization # currently broken
        k_new =  k_old * S_new / S_old # update k 
        print(S_new, 'S new')
        print(S_old, 'S old')
        print(k_old, 'k old')
        print(k_new, 'k from analytic integral')
        if k_new <0:
            raise ValueError('negative k_eff')
        # k_new = k_old * normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups,N_space,M+1)), edges, ws, N_ang, M, N_space, N_groups, sigma_f, nu, chi) / normalization

        if abs(k_new - k_old ) <=tol:
            klist.append(k_new)
            print('power iteration complete')
            print(k_new, 'k effective')
            print(n_iters, 'total iterations required')
            converged = True
        else:
            print(k_old-k_new, 'k difference')
            print('iteration count: ', n_iters)
            k_old = k_new
            S_old = S_new
            # coeffs_old = res_coefficients_new
            # phi_interpolated = lambda x:  phi_interpolated_new(x)
            klist.append(k_new)
            k_wynn_epsilon = wynn_epsilon(np.array(klist))
            if n_iters % 2 == 0:
                iw = n_iters - 1
            else:
                iw = n_iters
            print(k_wynn_epsilon[iw:,iw], 'k accelerated')
            if use_we_accel == True and n_iters > 4:
                k_old = k_wynn_epsilon[iw:, iw][-1]
                print(k_wynn_epsilon, 'full k table')
    
                print(k_old, 'k accelerated')
            n_iters +=1

            # normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)), edges, ws, N_ang, M, N_space, N_groups)
            # normalization_list.append(normalization)
            calc_time_list.append(t_calc)
    
    return klist, calc_time_list, normalization_list, run, sigma_f_array, nu_array, run.phi[:,0]