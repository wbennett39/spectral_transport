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

def build_fission_source(coeffs, fission_vector):
    F = coeffs.copy()
    for k in range(coeffs[0, :, 0].size):
        F[:, k, :] = np.multiply(F[:, k, :], fission_vector[k])
    return F 
    

def boundary_leakage_from_angular_flux(psi, ws, mus,  R):
     leak = 0.0
     for il in range(ws.size):
          if mus[il] > 0:
               leak += 2 * math.pi * R**2 * ws[il] * mus[il] * psi[il, -1]
     return leak

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
    sigma_a = run.parameters['all']['sigma_t'] - run.parameters['all']['sigma_s']
    print(nu, 'nu')

    chi = run.parameters['all']['chi']
    coeffs = run.sol_ob.y[:,-1]
    N_ang = run.parameters['fixed_source']['N_angles'][0]
    if run.parameters['all']['angular_derivative']['diamond'] == True:
        N_ang += 1
    ws = run.ws
    mus = run.mus
    N_groups = run.parameters['all']['N_groups']
    M  = run.parameters['all']['Ms'][0]
    N_space = run.parameters['all']['N_spaces'][0]
    print(N_space, 'spatial cells')
    # run.parameters['all']['rt'] = 1
    # run.parameters['all']['at'] = 1e-5
    # run.parameters['all']['integrator'] = 'Euler'

    run.custom_source(randomstart = True, uncollided = 0, moving = 0)
    

    res_coefficients = np.copy(run.sol_ob.y[:,-1].reshape((N_ang * N_groups, N_space, M+1)))
    initial_condition = run.fission_source
    coeffs_old = res_coefficients.copy()
    sigma_f_vec = np.ones(N_space) * sigma_f
    nu_vec = np.ones(N_space) * nu
    chi_vec = np.ones(N_space) * chi
    edges = run.edges
    shift = run.parameters['fixed_source']['shift']
    sigma_f_array = np.ones(run.xs.size) * sigma_f
    nu_array = np.ones(run.xs.size) * nu
    chi_array = np.ones(run.xs.size) * chi
    sigma_a_vec = np.zeros(N_space) 
    # # geometry = run.parameters['all']['geometry']
    for k in range(run.xs.size): # build the fission production vector for the Kornreich problem
        if -3.5 <= run.xs[k]-shift <= 3.5:
            sigma_f_array[k] = 0.0 
            nu_array[k] = 0.
            chi_array[k] = 0

    for space in range(N_space):
                left_edge = edges[space]-shift
                right_edge = edges[space+1]-shift
                if -3.5 <= left_edge <= 3.5 and -3.5 <= right_edge <= 3.5:
                    sigma_f_vec[space] = 0.0   
                    nu_vec[space] = 0.0
                    chi_vec[space] = 0.0
                if -2.5 <= left_edge <= 2.5 and -2.5 <= right_edge <= 2.5:
                     sigma_a_vec[space] = 0.9
                if (-3.5 <= left_edge <= -2.5 and -3.5 <= right_edge <= -2.5) or (2.5 <= left_edge <= 3.5 and 2.5 <= right_edge <= 3.5):
                     sigma_a_vec[space] = 0.2
    print(nu_vec, 'nu_vec')
    print(sigma_f_vec, 'sigma vec')
    print(sigma_a_vec, 'sigma a')
    # calculate the fission source from the random IC

    initial_fission_source = build_fission_source(initial_condition, sigma_f_vec * nu_vec * chi)
    new_fission_source = build_fission_source(coeffs_old,sigma_f_vec * nu_vec * chi )
    S_new = normalize_phi(new_fission_source, edges, ws, N_ang, M, N_space, N_groups)
    S_old = normalize_phi(initial_fission_source, edges, ws, N_ang, M, N_space, N_groups)
    print(S_old, 'Sold')
    new_fission_source /= S_old  # normalize fission source
    
    # Initializing k 
    normalization_list = []
    normalization_list.append(S_old)
    normalization_list.append(S_new)

    norm = S_old
    k_old = S_new / S_old * kguess
    klist.append(kguess)
    klist.append(k_old)
    n_iters = 1

    
    
    plt.ion()
    plt.figure('fission source')
    plt.plot(run.xs, run.phi[:, -1] * sigma_f_array * nu_array * chi_array, '--', label = f'iteration {0}')
    # plt.plot(run.xs, run.fission_source, '--', label = f'iteration {n_iters -1}')
    
    plt.legend()
    plt.show()

    plt.ion()
    plt.figure('scalar flux')
    plt.plot(run.xs, run.phi[:, -1], '--', label = f'iteration {0}')
    plt.legend()
    plt.show()

    plt.figure('scalar flux difference')
    plt.plot(run.xs, np.abs(run.phi[:, -1] - run.phi[:,0]), '--', label = f'iteration {0}')
    plt.legend()
    plt.show()

   
    
    calc_time_list = []
    # normalization_list.append(normalization)
    plt.close()
    plt.close()
    plt.close()

    


    while converged == False and n_iters < 10: 
        
        # run.load(transport_parameters, mesh_parameters) # reset parameters to agree with YAML file
        # the source is actually not normalized
        # run.parameters['all']['integrator'] = 'Euler'
        plt.ion()
        plt.figure('fission source')
        plt.plot(run.xs, run.phi[:, -1] * sigma_f_array * nu_array, '--', label = f'iteration {n_iters -1}')
        # plt.plot(run.xs, run.fission_source, '--', label = f'iteration {n_iters -1}')
        
        plt.legend()
        plt.show()

        plt.ion()
        plt.figure('scalar flux')
        plt.plot(run.xs, run.phi[:, -1], '--', label = f'iteration {n_iters -1}')
        plt.legend()
        plt.show()

        plt.figure('scalar flux difference')
        plt.plot(run.xs, np.abs(run.phi[:, -1] - run.phi[:,0]), '--', label = f'iteration {n_iters -1}')
        plt.legend()
        plt.show()

        # run solver    
        t1 = time.time()
        run.parameters['all']['kold'] = k_old
        new_fission_source = build_fission_source(coeffs_old, sigma_f_vec * nu_vec) # multiply the scalar flux coefficientes by the fission vector
        P = normalize_phi(new_fission_source, edges, ws, N_ang, M, N_space, N_groups )  # integrate the source over the volume, Chi cancels out the normalized weights
        new_fission_source *= 1/P # normalize fission source
        run.custom_source(randomstart = False, sol_coeffs = new_fission_source / k_old, phi_coeffs = coeffs_old, uncollided = 0, moving = 0) # steady state solve
        plt.figure(f'initial vs final {n_iters}')
        phioutIC, psi_outIC = make_phi_no_uncol(run.xs, N_groups, N_ang, edges, M, coeffs_old, ws)
        plt.plot(run.xs, phioutIC, 'k--', label = 'IC')
        plt.plot(run.xs, run.phi[:, -1], '-', label = 'Final')
        plt.legend()
        plt.show()
        t_calc = time.time() - t1
        coeffs_old = run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)) # update scalar flux
        F = build_fission_source(coeffs_old, sigma_f_vec * nu_vec) # build the new fission source (without chi?)
        k_new = normalize_phi(F, edges, ws, N_ang, M, N_space, N_groups ) *2 # update k. x2 is because the chi is not included
        absorption_term = build_fission_source(coeffs_old, sigma_a_vec) 
        A = normalize_phi(absorption_term, edges, ws, N_ang, M, N_space, N_groups) * 2
        L = boundary_leakage_from_angular_flux(run.psi[:, :, -1], ws, mus,  edges[-1])      # (2π) R^2 * sum_{μ>0} w μ ψ
        print(L, 'leakage')
        print(A, 'absorption')
        FS = build_fission_source(coeffs_old, sigma_f_vec * nu_vec)
        F2 = normalize_phi(FS, edges, ws, N_ang, M, N_space, N_groups)
        print(A + L, F2 / k_new, 'balance term')
        # sigma_interp = interp1d(run.xs, sigma_f_array * nu_array * chi) # interpolated fission rate
        # phi_interpolated = interp1d(run.xs, run.phi[:, -1]) 
        # integrand = lambda x:  phi_interpolated(x) * x**2 * 4 * math.pi * sigma_interp(x) 
        # test_norm = integrate.quad(integrand, run.xs[0], run.xs[-1])[0]
        # print((norm-test_norm) /test_norm, 'norm difference')
        # print(test_norm, 'scipy integral')

        if k_new <0:
            raise ValueError('negative k_eff')
    
        if abs(k_new - k_old ) <=tol and abs(S_new - S_old) / S_old <= tol:
            klist.append(k_new)
            normalization_list.append(S_new)
            print('power iteration complete')
            print(k_new, 'k effective')
            print(n_iters, 'total iterations required')
            converged = True
        else:
            print(k_old-k_new, 'k difference')
            print('iteration count: ', n_iters)
            k_old = k_new
            print(k_old, 'k old')
            print(klist, 'k list')
    

            # S_old = 0.4243163

            # coeffs_old = res_coefficients_new
            # phi_interpolated = lambda x:  phi_interpolated_new(x)
            klist.append(k_new)
            normalization_list.append(P)
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