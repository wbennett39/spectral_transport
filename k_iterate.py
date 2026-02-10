import numpy as np
import math
from IRAM import iram
from moving_mesh_transport.solver_classes.functions import *
from moving_mesh_transport.solver_classes.make_phi import make_output
import time
from scipy.interpolate import interp1d as interp1d
from scipy import integrate as integrate
import matplotlib.pyplot as plt
import yaml


# def integrate_phi_cell(cs, ws, a, b, M, N_ang):
#     # cell_volume = 4 * math.pi * (b**3 - a**3)
#     # normTn_intcell includes the r^2 term in the integrand
#     psi = np.zeros(N_ang)
#     for l in range(N_ang):
#         for j in range(M+1):
#             psi[l] += cs[l, j] * normTn_intcell(j, a, b)
#     res = np.sum(np.multiply(psi,ws)) # the weights were divided by 2
#     return 4 * math.pi * res #* cell_volume

def coeff_for_const_one(a, b, n):
    # c_n so that φ(r)=1 = Σ c_n * normTn(n,·,a,b)
    # Only n=0 is nonzero:
    if n == 0:
        # 1 / norm(a,b) for n=0
        return math.sqrt(math.pi) * math.sqrt(b - a)
    return 0.0
def make_fission_scalar_flux(coeffs_old, edges, ws, N_ang, M, N_space, N_groups, fission_vec):
    phi = np.zeros((edges.size-1, M+1))
    # psi = np.zeros((N_ang, edges.size-1, M+1))
   
    for ik in range(edges.size-1):
        a = edges[ik]
        b = edges[ik+1]
        for ij in range(M+1):
            phi[ ik, ij] += np.sum(np.multiply(ws, coeffs_old[ :, ik,ij ] )) * fission_vec[ik]
    return phi 


# normTn_intcell(j,a,b) must compute ∫_a^b ϕ_j^{(a,b)}(r) r^2 dr  (cell-local normalized basis)

@njit
def integrate_phi_cell(phi_coeffs, a, b, M):
    acc = 0.0
    for j in range(M+1):
        acc += phi_coeffs[j] * normTn_intcell(j, a, b)  # = ∫ φ r^2 dr on this cell
    return acc

@njit
def total_flux(VV_g, edges, M):
    # VV_g shape: (N_cells, M+1) for one energy group
    N = VV_g.shape[0]
    tot = 0.0
    for i in range(N):
        tot += integrate_phi_cell(VV_g[i, :], edges[i], edges[i+1], M)
    return 4.0 * math.pi * tot

@njit
def total_fission_production(VV, edges, M, sigma_f, nu):
    """
    VV shape: (G, N, M+1)
    sigma_f shape: (G, N)   # piecewise-constant per cell, per group
    nu shape: (G,)          # or (G,N) if spatially varying
    """
    G, N, _ = VV.shape
    P = 0.0
    for g in range(G):
        for i in range(N):
            cell_int = 0.0
            for j in range(M+1):
                cell_int += VV[g, i, j] * normTn_intcell(j, edges[i], edges[i+1])
            P += (nu[g] * sigma_f[g, i]) * cell_int
    return 4.0 * math.pi * P

@njit
def renormalize_fission_source(VV, edges, M, sigma_f, nu, target_P):
    """
    Scale all flux coefficients so that total fission production equals target_P.
    Returns (scale, new_P).
    """
    P = total_fission_production(VV, edges, M, sigma_f, nu)
    alpha = target_P / P
    VV *= alpha  # scale all groups & modes consistently
    return alpha, target_P



def check_norm_flux():
     j = 0
     ws = np.zeros(16)
     N_ang = ws.size
     Nlist = [10, 20, 30, 40, 60]
     R = 4.5
     for N in Nlist:
         edges = np.linspace(0, 4.5, N+1)
         VV = np.ones((N, j+1 ))
         for i in range(N):
            VV[i,0] = coeff_for_const_one(edges[i], edges[i+1], 0)
         res =   normalize_phi(VV, edges, ws, N_ang, j, N, 1)
         analytic_val = 4 * math.pi * normTn_intcell(j, 0, 4.5) * np.sum(VV[:,j])
         print(analytic_val, res)
         S_cells = sum(normTn_intcell(0, edges[i], edges[i+1]) for i in range(N))
         S_whole = normTn_intcell(0, 0.0, R)
        #  print(S_cells, S_whole, 'S_cells, S_whole')



     


def test_normTnintcell():
    for n in range(3):
        # print('n', n)
        for N in [10, 20, 30, 40, 50]:
            edges = np.linspace(0, 5, N+1)
            # print(N, 'N')
            for ix in range(edges.size-1):
                xs = np.linspace(edges[ix], edges[ix+1], 100)
                f = normTn(n, xs, edges[ix], edges[ix+1])
                interp_f = interp1d( xs,f)

                # print(edges[ix], edges[ix+1])
                integrand = lambda x: interp_f(x)  * x**2 
                analytic = normTn_intcell(n, edges[ix], edges[ix+1])
                scipy_answer = integrate.quad(integrand, edges[ix], edges[ix+1])[0]
                # print(analytic/ scipy_answer, 'ratio')    
                # print(analytic, 'analytic')
                # print('answer scipy', scipy_answer)


def build_fission_source(coeffs, fission_vector):
    F = coeffs.copy()
    # scalar_flux = np.zeros(coeffs[0, :,0].size)
    # print(scalar_flux.size, 'size of scalar flux')
    # for ik in range(scalar_flux.size):
    #     for j in range(coeffs[0, 0, :].size):
    #         scalar_flux[ik] += coeffs[:, ik, j] * 
        
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




def transfer_coefficients(coeffs_old, M):
    K = coeffs_old.shape[1]
    M_old = coeffs_old.shape[2]
    N_ang = coeffs_old.shape[0]
    coeffs_new = np.zeros((N_ang, K, M+1))
    for k in range(K):
        for im in range(M_old+1):
            coeffs_new[:, k, im] = coeffs_old[:, k, im]
    return coeffs_new


def power_iterate(kguess, transport_parameters, mesh_parameters, run, tol = 1e-12, use_we_accel = False, max_its = 100, coarse_angles = 4, coarse_solve = False, input_phi = np.array([0.0]), input_psi = None, ss_tol = 1e-10, coarse_M=0):
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

    
    if coarse_solve == True:
        with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

    # Use yaml.safe_load() for security when dealing with untrusted input
    # For a trusted config file, you might use yaml.FullLoader
            data = yaml.safe_load(file)
            data['all']['Ms'][0] = coarse_M
            data['fixed_source']['N_angles'][0] = coarse_angles
            with open('moving_mesh_transport/input_scripts/Kornreich_new.yaml', 'w') as file:
    # Use sort_keys=False to maintain a sensible order (optional)
                yaml.dump(data, file, sort_keys=False)
        run.load('Kornreich_new', mesh_parameters)
    else:
        run.load(transport_parameters, mesh_parameters)
    # print(run.ws.size)
    # assert 0
    klist = []
    converged = False
    sigma_f = run.parameters['all']['sigma_f']
    nu = run.parameters['all']['nu'] 
    chi = run.parameters['all']['chi'] 
    sigma_a = run.parameters['all']['sigma_t'] - run.parameters['all']['sigma_s']
    chi = run.parameters['all']['chi']
    N_ang = run.parameters['fixed_source']['N_angles'][0]
    # kold = run.parameters['all']['kold']
    kold = kguess
    klist.append(kold)
    if run.parameters['all']['angular_derivative']['diamond'] == True:
        N_ang += 1
        print(f'{N_ang} angles in k_iterator')


    # assert ws.size == N_ang
    N_groups = run.parameters['all']['N_groups']
    M  = run.parameters['all']['Ms'][0]
    N_space = run.parameters['all']['N_spaces'][0]

    sigma_a_vec = np.zeros(N_space) 
    sigma_f_vec = np.ones(N_space) * sigma_f
    nu_vec = np.ones(N_space) * nu
    shift = run.parameters['fixed_source']['shift']

    
    at = float(run.parameters['all']['at']) 
    rt = float(run.parameters['all']['rt'])
    euler_dt_num = int(run.parameters['all']['Euler_dt_num'])
    atlist = np.logspace(-1, np.log10(at),3)
    rtlist = np.logspace(-1, np.log10(rt), 3)

    if coarse_solve == True:
        run.parameters['all']['rt'] = 1
        run.parameters['all']['at'] = 1e-3
        run.parameters['all']['integrator'] = 'Euler'
        run.parameters['all']['kold'] = kguess
    
    t1 = time.time()
    if coarse_solve == True:
        run.custom_source(randomstart = True, uncollided = 0, moving = 0)
    else:
        if input_phi is not None:
            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

                data = yaml.safe_load(file)
                data['all']['kold'] = kold
                # data['all']['at'] = float(atlist[0])
                # data['all']['rt'] = float(rtlist[0])

                with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
        # Use sort_keys=False to maintain a sensible order (optional)
                    yaml.dump(data, file, sort_keys=False)
            run.load('Kornreich', mesh_parameters)
            run.parameters['all']['kold'] = kold
            new_phi_coeffs = transfer_coefficients(input_phi, M)

            edges = run.edges
            ws = run.ws
            edges = run.edges
            chi_vec = np.ones(N_space) * chi
            for space in range(N_space):
                        left_edge = edges[space]-shift
                        right_edge = edges[space+1]-shift
                        middle = 0.5 * (right_edge + left_edge)
                        if -3.5 <= middle < 3.5:
                            sigma_f_vec[space] = 0.0   
                            nu_vec[space] = 0.0
                            # chi_vec[space] = 0.0
                            if left_edge <-3.5 or right_edge >4.6:
                                print('edge straddle')
                                print(left_edge, right_edge)
                                assert 0
                        if -2.5 <= left_edge <= 2.5 and -2.5 <= right_edge <= 2.5:
                            sigma_a_vec[space] = 0.9
                        if (-3.5 <= left_edge <= -2.5 and -3.5 <= right_edge <= -2.5) or (2.5 <= left_edge <= 3.5 and 2.5 <= right_edge <= 3.5):
                            sigma_a_vec[space] = 0.2
            transfer_fission_source = make_fission_scalar_flux(new_phi_coeffs, edges, ws, N_ang, M, N_space, N_groups, sigma_f_vec * nu_vec)
            print(np.shape(input_phi), 'shape of input phi')
            run.parameters['all']['rt'] = rtlist[0]
            run.parameters['all']['at'] = atlist[0]
            transfer_fission_source = normalize_fission_source(transfer_fission_source ,N_space, 0, 1/kold, edges)
            run.custom_source(randomstart = False, uncollided = 0, moving = 0, input_phi_coeffs = new_phi_coeffs, sol_coeffs = transfer_fission_source )
        else:
            run.custom_source(randomstart = True, uncollided = 0, moving = 0 )
    ws = run.ws
    mus = run.mus
    t_calc = time.time() - t1
    res_coefficients = np.copy(run.sol_ob.y[:,-1].reshape((N_ang * N_groups, N_space, M+1)))
    initial_condition = run.fission_source # I think this is just the initial condition mislabeled 
    coeffs_old = res_coefficients.copy()

    sigma_f_array = np.ones(run.xs.size) * sigma_f
    nu_array = np.ones(run.xs.size) * nu
    chi_array = np.ones(run.xs.size) * chi
    edges = run.edges
    chi_vec = np.ones(N_space) * chi
    for space in range(N_space):
                left_edge = edges[space]-shift
                right_edge = edges[space+1]-shift
                middle = 0.5 * (right_edge + left_edge)
                if -3.5 <= middle < 3.5:
                    sigma_f_vec[space] = 0.0   
                    nu_vec[space] = 0.0
                    # chi_vec[space] = 0.0
                    if left_edge <-3.5 or right_edge >4.6:
                         print('edge straddle')
                         print(left_edge, right_edge)
                         assert 0
                if -2.5 <= left_edge <= 2.5 and -2.5 <= right_edge <= 2.5:
                     sigma_a_vec[space] = 0.9
                if (-3.5 <= left_edge <= -2.5 and -3.5 <= right_edge <= -2.5) or (2.5 <= left_edge <= 3.5 and 2.5 <= right_edge <= 3.5):
                     sigma_a_vec[space] = 0.2

    

    # # geometry = run.parameters['all']['geometry']
    for k in range(run.xs.size): # build the fission production vector for the Kornreich problem
        if -3.5 <= run.xs[k]-shift <= 3.5:
            sigma_f_array[k] = 0.0 
            nu_array[k] = 0.
            chi_array[k] = 0

    
    # calculate the fission source from the random IC
    # initial_fission_source = build_fission_source(initial_condition, sigma_f_vec * nu_vec)
    # new_fission_source = build_fission_source(coeffs_old, sigma_f_vec * nu_vec)
    new_fission_source = make_fission_scalar_flux(coeffs_old, edges, ws, N_ang, M, N_space, N_groups, sigma_f_vec * nu_vec)
    initial_fission_source = make_fission_scalar_flux(initial_condition, edges, ws, N_ang, M, N_space, N_groups, sigma_f_vec * nu_vec)
    S_new = normalize_phi(new_fission_source, edges, ws, N_ang, M, N_space, N_groups) 
    S_old = normalize_phi(initial_fission_source, edges, ws, N_ang, M, N_space, N_groups)
    # print(S_old, 'Sold')
    print(S_new, 'S_new first iteration')
    sigma_interp = interp1d(run.xs, sigma_f_array * nu_array) # interpolated fission rate
    phi_interpolated = interp1d(run.xs, run.phi[:, -1])
    # phioutIC, psi_outIC = make_phi_no_uncol(run.xs, N_groups, N_ang, edges, M, initial_fission_source, ws)
    
    # integrand = lambda x:  phi_interpolated(x) * x**2 * 4 * math.pi * sigma_interp(x) 
    
    # plt.figure('initial fission source')
    # plt.plot(run.xs, phioutIC, label = 'initial condition')
    # plt.plot(run.xs, integrand(run.xs), '-', label = 'after 1 iteration')
    # plt.show()
    # test_norm = integrate.quad(integrand, run.xs[0], run.xs[-1])[0]
    # # print((norm-test_norm) /test_norm, 'norm difference')
    # print(test_norm, 'scipy integral for S_new')

    # Initializing k 
    normalization_list = []
    normalization_list.append(S_old)
    normalization_list.append(S_new)

    # knew = kold * S_new #/ S_old
    knew = S_new 
    klist.append(S_new)
    # print(klist)
    # print(S_new, 'Snew')
    # print(S_old, 'S_OLD')
 
    # if coarse_solve == False:
    #     assert 0
    S_old = S_new
    
    
    # k_old = 1

    n_iters = 1
    # new_fission_source *= 1/S_new 
    # coeffs_old /=knew
    # new_fission_source/= knew
    new_fission_source = normalize_fission_source(new_fission_source,N_space, M, 1/knew, edges)
    old_fission_source = new_fission_source.copy()
    kold = knew


    
    
    plt.ion()
    plt.figure('fission source')
    plt.plot(run.xs, run.phi[:, -1] * sigma_f_array * nu_array * chi_array, '--', label = f'iteration {0}')
    # plt.plot(run.xs, run.fission_source, '--', label = f'iteration {n_iters -1}')
    
    plt.legend()
    plt.show()

    plt.ion()
    plt.figure('k_it scalar flux')
    plt.plot(run.xs, run.phi[:, -1], '--', label = f'iteration {0}')
    plt.legend()
    plt.show()

    plt.figure('scalar flux difference')
    plt.plot(run.xs, np.abs(run.phi[:, -1] - run.phi[:,0]), '--', label = f'iteration {0}')
    plt.legend()
    plt.show()

   
    
    calc_time_list = []
    calc_time_list.append(t_calc)
    # normalization_list.append(normalization)
    plt.close()
    plt.close()
    plt.close()

    
    if coarse_solve == True:
            run.load('Kornreich_new', mesh_parameters)
    else:
            run.load(transport_parameters, mesh_parameters) 
   
    while converged == False and n_iters < max_its: 
        if coarse_solve == True:
            run.load('Kornreich_new', mesh_parameters)
        else:
            run.load(transport_parameters, mesh_parameters) # reset parameters to agree with YAML file
        # the source is actually not normalized
        # run.parameters['all']['integrator'] = 'Euler'
        if n_iters < 3:
            run.parameters['all']['at'] = float(atlist[n_iters])
            run.parameters['all']['rt'] = float(rtlist[n_iters])
        plt.ion()
        plt.figure('fission source')
        plt.plot(run.xs, run.phi[:, -1] * sigma_f_array * nu_array * chi, '--', label = f'iteration {n_iters -1}')
        # plt.plot(run.xs, run.fission_source, '--', label = f'iteration {n_iters -1}')
        
        plt.legend()
        plt.show()


        plt.ion()
        plt.figure('k_it scalar flux')
        plt.plot(run.xs, run.phi[:, -1], '--', label = f'iteration {n_iters -1}')
        plt.legend()
        plt.show()

        plt.figure('scalar flux difference')
        plt.plot(run.xs, np.abs(run.phi[:, -1] - run.phi[:,0]), '--', label = f'iteration {n_iters -1}')
        plt.legend()
        plt.show()

        # run solver    
        t1 = time.time()
        run.parameters['all']['kold'] = kold
        # new_fission_source = build_fission_source(coeffs_old, sigma_f_vec * nu_vec) # multiply the scalar flux coefficientes by the fission vector
        # new_fission_source = F
      
        # np.testing.assert_allclose(normalize_phi(new_fission_source, edges, ws, N_ang, M, N_space, N_groups ),1)

 
        # plt.figure(f'fission source scaled')
        # phioutIC, psi_outIC = make_phi_no_uncol(run.xs, N_groups, N_ang, edges, M, new_fission_source, ws)
        # plt.plot(run.xs, phioutIC, '--', label = 'fission source')
        # plt.legend()
        # plt.show()
         # normalize fission source
        # print(normalize_phi(old_fission_source, edges, ws, N_ang, M, N_space, N_groups), 'should be 1/k')

        # solve with new source
        run.custom_source(randomstart = False, sol_coeffs = old_fission_source , phi_coeffs = coeffs_old, uncollided = 0, moving = 0) # steady state solve
        # plt.figure(f'initial vs final {n_iters}')
        t_calc = time.time() - t1
        with open('moving_mesh_transport/input_scripts/mesh_parameters_Kornreich.yaml', 'r') as file:

        # Use yaml.safe_load() for security when dealing with untrusted input
        # For a trusted config file, you might use yaml.FullLoader
                data = yaml.safe_load(file)
                # data['all']['integrator'] = 'Euler'
                # data['dense'] = True
                # data['eval_times'] =False
                ts = run.sol_ob.t
                first_step = float(ts[1] - ts[0])
                data['first_step'] = first_step
                data['dense'] = True
                data['eval_times'] =False
                # print(run.sol_ob.t[1] - run.sol_ob.t[0], 'first step')
                # assert 0
                with open('moving_mesh_transport/input_scripts/mesh_parameters_Kornreich.yaml', 'w') as file:
        # Use sort_keys=False to maintain a sensible order (optional)
                    yaml.dump(data, file, sort_keys=False)
        # phioutIC, psi_outIC = make_phi_no_uncol(run.xs, N_groups, N_ang, edges, M, coeffs_old, ws)
        # plt.plot(run.xs, phioutIC, 'k--', label = 'IC')
        # plt.plot(run.xs, run.phi[:, -1], '-', label = 'Final')
        # plt.legend()
        # plt.show()
        
        coeffs_new = run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)) # update scalar flux
        Y = run.sol_ob.y
        # if np.max(np.abs(Y[:,-1] - Y[:,-2])) <=ss_tol:
        #     new_tf = float(ts[-2])
        # else: 
        #     new_tf = float(ts[-1]*10)
        #     # euler_dt_num += 1
        # if np.max(np.abs(Y[:,-1] - Y[:,-3])) <= ss_tol:
        #     euler_dt_num -= 1
        #     if euler_dt_num <= 3:
        #         euler_dt_num = 3
        with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

        # Use yaml.safe_load() for security when dealing with untrusted input
        # For a trusted config file, you might use yaml.FullLoader
                data = yaml.safe_load(file)
                # data['all']['integrator'] = 'Euler'
                # data['dense'] = True
                # data['eval_times'] =False
                # data['all']['tfinal'] = new_tf
                # data['all']['Euler_dt_num'] = euler_dt_num
               
                # print(run.sol_ob.t[1] - run.sol_ob.t[0], 'first step')
                # assert 0
                with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
        # Use sort_keys=False to maintain a sensible order (optional)
                    yaml.dump(data, file, sort_keys=False)
        # phioutf, psi_outf = make_phi_no_uncol(run.xs, N_groups, N_ang, edges, M, coeffs_old, ws)
        # plt.figure(f'initial vs final {n_iters}')
        # plt.plot(run.xs, phioutf, 'k--', label = 'Final (calculated from coefficients)')
        # plt.legend()
        # plt.show()    
        new_fission_source = make_fission_scalar_flux(coeffs_new, edges, ws, N_ang, M, N_space, N_groups, sigma_f_vec * nu_vec)

        interp_phi = interp1d(run.xs, run.phi[:,-1])
        integrand_scipy = lambda x: x**2 * interp_phi(x) * sigma_interp(x) * 4 * math.pi
        P_scipy = integrate.quad(integrand_scipy, run.xs[0], run.xs[-1])
        # P = normalize_phi(new_fission_source, edges, ws, N_ang, M, N_space, N_groups ) 
          # integrate the source over the volume
        S_new = normalize_phi(new_fission_source, edges, ws, N_ang, M, N_space, N_groups )
        print(S_new, 'P', P_scipy, "P_scipy")
        print(S_new/P_scipy[0], 'P ratio')
        # knew = S_new * kold# /S_old   # update k. x2 is because the chi is not included
        knew = S_new
        coeffs_old = coeffs_new
        new_fission_source = normalize_fission_source(new_fission_source,N_space, M, 1/knew, edges)
        old_fission_source = new_fission_source
        # new_fission_source *= 1/S_new
        
        # absorption_term = build_fission_source(coeffs_old, sigma_a_vec) 
        # A = normalize_phi(absorption_term, edges, ws, N_ang, M, N_space, N_groups) 
        # L = boundary_leakage_from_angular_flux(run.psi[:, :, -1], ws, mus,  edges[-1])      # (2π) R^2 * sum_{μ>0} w μ ψ
        # print(L, 'leakage')
        # print(A, 'absorption')
        # FS = build_fission_source(coeffs_old, sigma_f_vec * nu_vec)
        # F2 = normalize_phi(FS, edges, ws, N_ang, M, N_space, N_groups)
        # print(A + L, 1 / knew, 'balance term')
        # sigma_interp = interp1d(run.xs, sigma_f_array * nu_array * chi) # interpolated fission rate
        # phi_interpolated = interp1d(run.xs, run.phi[:, -1]) 
        # integrand = lambda x:  phi_interpolated(x) * x**2 * 4 * math.pi * sigma_interp(x) 
        # test_norm = integrate.quad(integrand, run.xs[0], run.xs[-1])[0]
        # print((norm-test_norm) /test_norm, 'norm difference')
        # print(test_norm, 'scipy integral')

        # if knew <0:
        #     raise ValueError('negative k_eff')
    
        if abs(knew - kold ) <=tol:
            klist.append(knew)
            normalization_list.append(S_new)
            print('k iteration complete')
            print(knew, 'k effective')
            print(n_iters, 'total iterations required')
            converged = True
            calc_time_list.append(t_calc)
        else:
            print(kold-knew, 'k difference')
            print('iteration count: ', n_iters)
            kold = knew
            klist.append(knew)
            print(kold, 'k old')
            print(klist, 'k list')
            S_old = S_new
    

            # S_old = 0.4243163

            # coeffs_old = res_coefficients_new
            # phi_interpolated = lambda x:  phi_interpolated_new(x)
            
            normalization_list.append(S_new)
            k_wynn_epsilon = wynn_epsilon(np.array(klist))
            if n_iters % 2 == 0:
                iw = n_iters - 1
            else:
                iw = n_iters
            print(k_wynn_epsilon[iw:,iw], 'k accelerated')
            if use_we_accel == True and n_iters > 4:
                kold = k_wynn_epsilon[iw:, iw][-1]
                print(k_wynn_epsilon, 'full k table')
    
                print(kold, 'k accelerated')
            n_iters +=1

            # normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups, N_space, M+1)), edges, ws, N_ang, M, N_space, N_groups)
            # normalization_list.append(normalization)
            calc_time_list.append(t_calc)
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    return klist, calc_time_list, normalization_list, run, sigma_f_array, nu_array, run.phi[:,-1]