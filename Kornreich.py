# imports functions to run package from terminal 

import sys
import os
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings
from k_iterate import make_fission_scalar_flux
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
                      
# from benchmarks import integrate_greens as intg
from moving_mesh_transport.plots import plotting_script as plotter
from moving_mesh_transport import solver
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigs  # o
import h5py 

from moving_mesh_transport.solver_classes.functions import *

from k_iterate import power_iterate, test_normTnintcell, check_norm_flux
import yaml
import pandas as pd
from scipy.optimize import newton

# from moving_mesh_transport.plots.plot_square_s_times import main as plot_square_s_times
# from moving_mesh_transport.solution_plotter import plot_thin_nonlinear_problems as plot_thin
# from moving_mesh_transport.solution_plotter import plot_thin_nonlinear_problems_s2 as plot_thin_s2
# from moving_mesh_transport.solution_plotter import plot_thick_nonlinear_problems as plot_thick
# from moving_mesh_transport.solution_plotter import plot_thick_nonlinear_problems_s2 as plot_thick_s2
# from moving_mesh_transport.solution_plotter import plot_thick_suolson_problems as plot_sut
# from moving_mesh_transport.solution_plotter import plot_su_olson as plot_su
# from moving_mesh_transport.solution_plotter import plot_su_olson_gaussian as plot_sug
# from moving_mesh_transport.solution_plotter import plot_coeffs_nov28_crc as pca_28
# from moving_mesh_transport.solution_plotter import plot_coeffs_nov23_crc as pca_23
# from moving_mesh_transport.solution_plotter import plot_coeffs_nov31_crc as pca_31
# from moving_mesh_transport.solution_plotter import plot_coeffs_all_local as pca_loc
# from moving_mesh_transport.table_script import make_all_tables as mat
from moving_mesh_transport.solver_classes.functions import test_square_sol
from moving_mesh_transport.solver_classes.functions import test_s2_sol
#from moving_mesh_transport.tests.test_functions import test_interpolate_point_source
# from moving_mesh_transport.mesh_tester import test_square_mesh as test_mesh
# from moving_mesh_transport.solution_plotter import make_tables_su_olson as tab_sus

# from moving_mesh_transport.solver_classes.functions import test_s2_sol
from moving_mesh_transport.loading_and_saving.load_solution import load_sol as load
from moving_mesh_transport.solver_functions.run_functions import run
from moving_mesh_transport.solver_functions.DMD_functions import DMD_func3
import h5py


# def make_table(x0, nu, DMD_alpha, IRAM_alpha, iteration_k):
#     try:
#         df = pd.read_csv('Kornreich_results/table/eigenvalues.csv')
#         print(df)
#         eigenvals = {df['analytic k'], df['Iteration k'], df['analytic alpha'], df['DMD alpha'], df['IRAM alpha']}
#     except:
#         eigenvals = {
#         "analytic k": {
#             4.5: [0.4241317, 0.9896407],
#             4.6: [0.4556758, 1.063244],
#         },
#         "VDMD alpha": {
#             4.5: [0,0],
#             4.6: [0,0],
#         },
#         "IRAM alpha": {
#             4.5: [0,0],
#             4.6: [0,0],
#         }, 
#         "Iteration k": {
#             4.5: [0,0],
#             4.6: [0, 0],
#         },
#         "analytic alpha": {
#             4.5: [-0.3229855,-0.006440766],
#             4.6: [-0.2932468, 0.03759991],
#         },

        
#     }


#     if nu == 1.5:
#         index = 0
#     elif nu == 3.5:
#             index = 1


#     eigenvals['VDMD alpha'][x0][index] = DMD_alpha
#     eigenvals['IRAM alpha'][x0][index] = IRAM_alpha
#     eigenvals['Iteration k'][x0][index] = iteration_k




#     # rows = []
#     # x0s = [4.5, 4.6]
#     # for method, resolutions in eigenvals.items():
#     #     for N, (x01, x02) in x0s.items():
#     #         rows.append({
#     #             "Method": method,
#     #             "Resolution (particles)": int(N),
#     #             "Eigenvalue 1": np.round(eig1,4),
#     #             "Eigenvalue 2": np.round(eig2,4),
#     #         })
 

#     df = pd.DataFrame(eigenvals)
#     df.to_csv("Kornreich_results/table/eigenvalues.csv", index=False)
def make_table(x0, nu, DMD_alpha, IRAM_alpha, iteration_k):

    filepath = "Kornreich_results/table/eigenvalues.csv"

    # Default analytic values
    analytic_k = {
        4.5: [0.4243163, 0.9900716],
        4.6: [0.4556758, 1.063244],
    }

    analytic_alpha = {
        4.5: [-0.3196537, -0.006156369],
        4.6: [-0.2932468, 0.03759991],
    }

    # Determine index based on nu
    if nu == 1.5:
        index = 0
    elif nu == 3.5:
        index = 1
    else:
        raise ValueError("Unsupported nu value")

    # Create new row
    new_row = {
        "x0": x0,
        "nu": nu,
        "analytic k": analytic_k[x0][index],
        "Iteration k": iteration_k,
        "analytic alpha": analytic_alpha[x0][index],
        "VDMD alpha": np.real(DMD_alpha),
        "IRAM alpha": np.real(IRAM_alpha),
    }

    # Load existing file if it exists
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

        # Remove existing row for same (x0, nu) to avoid duplicates
        df = df[~((df["x0"] == x0) & (df["nu"] == nu))]

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    # Sort nicely
    df = df.sort_values(by=["x0", "nu"])

    # Round for prettier output
    df = df.round(6)

    df.to_csv(filepath, index=False)

    print("Table updated successfully.")



def plot_k_convergence(k_list, k_bench, N_ang, x0, N_spaces, nu, time_list, coarse_solve, run_ob, normalization_list, M):
    
    
    plt.figure('keff')
    plt.clf()



    nits = len(k_list)
    plt.plot(np.linspace(0, nits, nits), k_list, '-o', mfc = 'none')
    plt.xlabel('iteration', fontsize = 16)
    plt.plot(np.linspace(0, nits, nits), np.ones(nits) * k_bench, 'k-', label = 'benchmark')
    plt.ylabel(r'$k_\mathrm{eff}$', fontsize = 16)
    # plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'Kornreich_results/convergence_plots/k_iterations_Kornreich_{N_ang}_angles_x0={x0}_nu={nu}_{N_spaces}_spatial_cells_{M+1}_bases.pdf', bbox_inches = 'tight')
    plt.show()

    plt.figure('calc time')
    plt.clf()
    plt.semilogy(np.linspace(0, nits, nits)[1:], time_list, '-o', mfc = 'none')
    plt.xlabel('iteration', fontsize = 16)
    plt.ylabel('time [s]', fontsize = 16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'Kornreich_results/time_plots/calc_time_Kornreich_{N_ang}_angles_x0={x0}_nu={nu}_{N_spaces}_spatial_cells_coarse_solve={coarse_solve}_{M+1}_bases.pdf', bbox_inches = 'tight')
    plt.show()

    plt.figure('flux shape')
    plt.clf()
    nits = len(k_list)
    plt.plot(run_ob.xs, run_ob.phi, '-', mfc = 'none')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    plt.legend()
    # plt.savefig('Kornreich_results/scalar_flux_Kornreich.pdf')
    plt.show()



    plt.figure('normalize')
    plt.clf()
    nits = len(k_list)
    plt.plot(np.linspace(0, nits, nits)[1:], normalization_list[1:], '-o', mfc = 'none')
    plt.xlabel('iterations', fontsize = 16)
    plt.ylabel('normalization', fontsize = 16)
    plt.legend()
    plt.savefig('Kornreich_results/convergence_plots/norm_iterations_Kornreich.pdf', bbox_inches = 'tight')
    plt.show()
    plt.show()


    plt.figure('keff_log')
    plt.clf()
    nits = len(k_list)
    plt.loglog(np.linspace(0, nits, nits)[1:], np.abs(np.array(k_list[1:]) - np.array(k_list[:-1])), '-o', mfc = 'none')
    plt.xlabel('iterations', fontsize = 16)
    # plt.loglog(np.linspace(0, nits, nits), np.ones(nits) * 0.4243163, 'k-', label = 'benchmark')
    plt.ylabel(r'$k_\mathrm{eff}$ difference', fontsize = 16)
    plt.legend()
    plt.savefig('Kornreich_results/convergence_plots/k_iterations_Kornreic_log.pdf', bbox_inches = 'tight')
    plt.show()

    plt.figure('solution plot')
    plt.clf()
    plt.xlabel('x [cm]', fontsize = 16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel(r'$\phi$', fontsize = 16)
    plt.plot(run_ob.xs, run_ob.phi, 'k-', mfc = 'none')
    plt.savefig('Kornreich_results/solution_plots/scalar_flux_Kornreich.pdf', bbox_inches = 'tight')
    plt.show()
# from diffeqpy import de
def basis(i, x, a, b):
     return normTn(i, x, a, b)
def RMS(l1, l2):
    return np.sqrt(np.mean((l1-l2)**2))

def coeffs_to_phi(u, xs, N_ang, N_groups, edges, ws, M):
        psi = np.zeros((N_ang, xs.size, N_groups))
        for g in range(N_groups):
            for ang in range(N_ang):
                for count in range(xs.size):
                    idx = np.searchsorted(edges[:], xs[count])
                    if (idx == 0):
                        idx = 1
                    if (idx >= edges.size):
                        idx = edges.size - 1
                    if edges[0] <= xs[count] <= edges[-1]:
                        for i in range(M+1):

                            # radiation = u[g * N_ang:(ig+1) * N_ang,:,:]
                            # psi[ang, count] += u[ang,idx-1,i] * basis(i,xs[count:count+1],float(edges[idx-1]),float(edges[idx]))[0]
                            psi[ang, count, g] += u[g*N_ang +ang,idx-1,i] * basis(i,xs[count:count+1],float(edges[idx-1]),float(edges[idx]))[0]
        output_phi = np.zeros((xs.size, N_groups))

        for g in range(N_groups):
            output_phi[:,g] = np.sum(np.multiply(psi[:, :, g].transpose(), ws), axis = 1)
        psi_out = psi
        phi_out = output_phi

        return output_phi

# prime solver
run = run()
# run.load('transport', 'mesh_parameters_modak_gupta')
# run.plane_IC(0,0)

loader = load()
def Kornreich_benchmark(prime = True, guess_k = 1, sparse_time_points = 7, skip =4, ktol = 5e-5, use_we = False, max_its_kloop = 100, maxits_power = 15, coarse_angles = 4, alpha_tol = 1e-4, nalphas = 2, tf = 5e3, coarse_solve =True):
    # test_normTnintcell()
    # check_norm_flux()
    # assert 0

    with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:
                data = yaml.safe_load(file)
                data['all']['kold'] = float(guess_k)
    with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
                yaml.dump(data, file, sort_keys=False)

    run.load('Kornreich', 'mesh_parameters_Kornreich')

    if prime == True:
        at = float(run.parameters['all']['at']) 
        rt = float(run.parameters['all']['rt'])
        run.parameters['all']['N_spaces'] = [10]
        run.parameters['all']['tfinal'] = 0.001
        run.parameters['all']['Ms'] = [0]
        run.parameters['random_IC']['N_angles'] = [2]

        # run.parameters['fixed_source']['N_angles'] = [2]
        # run.parameters['all']['sigma_f'] = 1.0
        run.custom_source(randomstart=True, uncollided = 0, moving = 0 )

    # First, find k_eff
    # if run.parameters['fixed_source']['shift'] == 0.0:
    if run.parameters['fixed_source']['x0'][0] ==4.5:
        if run.parameters['all']['nu'] == 1.5:
            k_bench = 0.4241317
            alpha_bench = -0.3229855
        elif run.parameters['all']['nu'] == 3.5:
            k_bench = 0.9896407
            alpha_bench = -0.006440766
    elif run.parameters['fixed_source']['x0'][0] ==4.6:
        if run.parameters['all']['nu'] == 1.5:
            # k_bench = 0.4242237
            k_bench = 0.4556758
            # alpha_bench = -0.3213939
            alpha_bench = -0.2932468
        elif run.parameters['all']['nu'] == 3.5:
            # k_bench = 0.9898554
            k_bench = 1.063244
            alpha_bench = 0.03759991
    else: #not ready for other cases. Probably not necessary
        # k_bench = 0.4243163
        raise ValueError('Do not have this case')

    # coarse solve
    run.load('Kornreich', 'mesh_parameters_Kornreich')
    N_ang = run.parameters['fixed_source']['N_angles'][0] 

    N_spaces = run.parameters['all']['N_spaces'][0]
    N_space = N_spaces
    N_groups = run.parameters['all']['N_groups']
    M = run.parameters['all']['Ms'][0]
    run.load('Kornreich', 'mesh_parameters_Kornreich')
    N_ang = run.parameters['fixed_source']['N_angles'][0]
    N_spaces = run.parameters['all']['N_spaces'][0]
    N_groups = run.parameters['all']['N_groups']
    M = run.parameters['all']['Ms'][0]
    x0 = run.parameters['fixed_source']['x0'][0]
    nu =run.parameters['all']['nu']
    if coarse_solve ==True:
        im =0
        k_list, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(guess_k, 'Kornreich', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, max_its = max_its_kloop, coarse_angles=N_ang, coarse_solve=True, coarse_M=im)
        precondition_sol_coeffs = run_ob.sol_ob.y[:,-1].reshape(((N_ang+1)*N_groups, N_spaces, im+1))
        plot_k_convergence(k_list, k_bench, N_ang, x0, N_spaces, nu, time_list, coarse_solve, run_ob, normalization_list, im)
        for im in range(1, M+1):
            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:
                data = yaml.safe_load(file)
                data['all']['Ms'][0] = int(im)
            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
                yaml.dump(data, file, sort_keys=False)
            k_list, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(k_list[-1], 'Kornreich', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, max_its = max_its_kloop, input_phi=precondition_sol_coeffs, coarse_angles=N_ang)
            plot_k_convergence(k_list, k_bench, N_ang, x0, N_spaces, nu, time_list, coarse_solve, run_ob, normalization_list, im)
            precondition_sol_coeffs = run_ob.sol_ob.y[:,-1].reshape(((N_ang+1)*N_groups, N_spaces, im+1))
    else:
         k_list, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(guess_k, 'Kornreich', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, max_its = max_its_kloop, coarse_solve=False, input_phi=None)
         plot_k_convergence(k_list, k_bench, N_ang, x0, N_spaces, nu, time_list, coarse_solve, run_ob, normalization_list, M)
    print(k_list, 'k_list')
    print(k_list[-1], 'k effective')
    print(k_bench, 'benchmark k effective')
    print(time_list, 'computation time required per iterate')
    
    f = h5py.File(f'Kornreich_results/data/Kornreich_keff_S{N_ang}_{N_spaces}_cells_x0={x0}_nu={nu}.h5', 'w')
    f.create_dataset('scalar_flux', data = run_ob.phi)
    f.create_dataset('xs', data = run_ob.xs)
    f.create_dataset('psi', data = run_ob.psi)
    Yminus = run_ob.sol_ob.Y_minus_psi
    f.create_dataset('Y_minus', data = Yminus)
    f.create_dataset('N_angles', data = np.array([run.parameters['fixed_source']['N_angles'][0]]))
    f.create_dataset('t', data = run_ob.sol_ob.t)
    f.create_dataset('k_list', data = k_list)
    f.create_dataset('fission_source', data = sigma_f_vec * nu_vec * phi  )
    f.close()


    # Estimate alpha modes with VDMD

    with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

    # Use yaml.safe_load() for security when dealing with untrusted input
    # For a trusted config file, you might use yaml.FullLoader
            data = yaml.safe_load(file)
            data['all']['integrator'] = 'Euler'
            data['all']['fixed_source'] = False
            data['all']['tfinal'] = 500
            data['all']['guess_steady_state'] = False
            data['all']['Euler_dt_num'] = sparse_time_points
            with open('moving_mesh_transport/input_scripts/Kornreich_DMD.yaml', 'w') as file:
    # Use sort_keys=False to maintain a sensible order (optional)
                yaml.dump(data, file, sort_keys=False)
    with open('moving_mesh_transport/input_scripts/mesh_parameters_Kornreich.yaml', 'r') as file:

    # Use yaml.safe_load() for security when dealing with untrusted input
    # For a trusted config file, you might use yaml.FullLoader
            data = yaml.safe_load(file)
            # data['all']['integrator'] = 'Euler'
            data['dense'] = True
            data['eval_times'] =False

   

            with open('moving_mesh_transport/input_scripts/mesh_parameters_Kornreich_DMD.yaml', 'w') as file:
    # Use sort_keys=False to maintain a sensible order (optional)
                yaml.dump(data, file, sort_keys=False)
    run.load('Kornreich_DMD', 'mesh_parameters_Kornreich_DMD')
    # f = h5py.File('Kornreich_results/Kornreich_results/Kornreich_keff.h5', 'r+')
    # ts = f['t']
    # fission_source = f['fission_source'][:]
    # Y_minus = f['Y_minus'][:,:]
    # N_ang = f['N_angles'][:][0]
    # xs = f['xs']
    # Y_minus_shifted = np.array(Y_minus).copy().reshape(N_ang, xs.size, ts.size)
    # adjust Y- to remove source influence
    
    integrator = run.parameters['all']['integrator']
    sigma_t = run.parameters['all']['sigma_t']
    N_ang = run.parameters['fixed_source']['N_angles'][0] +1
    run.kold = 1
    # skip = 4
    theta = 0
    run.custom_source(randomstart = True, uncollided = 0, moving=0)
    Yminus = run.sol_ob.Y_minus_psi
    ts =   run.sol_ob.t
    xs = run.xs
    phi = run.phi
    fission_source =  phi * 0 
    res_coeffs_VDMD = run.sol_ob.y[:,-1]

    eigen_vals_DMD, eigen_vectors = DMD_func3(Yminus, ts,  'Euler', sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points, source = True, sourcevec =  fission_source* 0, N_ang = N_ang, xs = xs)
    eigen_vals_DMD_coeffs, eigen_vectors_coeffs = DMD_func3(run.sol_ob.y, ts,  'Euler', sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points, source = True, sourcevec =  fission_source* 0, N_ang = N_ang*(M+1), xs = np.zeros(N_spaces))
    eigen_vals_DMD = np.real(np.flip(np.sort(eigen_vals_DMD[eigen_vals_DMD!=0])))
    max_DMD_alpha_coeffs_ind = np.argmin(np.max(eigen_vals_DMD_coeffs)-eigen_vals_DMD_coeffs)
    v0 = eigen_vectors_coeffs[:, max_DMD_alpha_coeffs_ind]
    print(eigen_vals_DMD, 'alpha eigen values VDMD')
    print(alpha_bench, 'benchmark alpha eigen value' )
    print(eigen_vectors.shape, 'eigen vec shape')
    
# Y_minus_residual = Y_minus.copy() 
# for it in range(1, time_list.size):
#     dt = time_list[it] - time_list[it-1]
#     Y_minus_residual -= dt * source
    # eigen_vals = DMD_func3(Y_minus_residual, time_list, 'Euler', sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points)


    # IRAM to get alpha modes
    if k_list[-1] < 1:
        run.load('Kornreich', 'mesh_parameters_Kornreich')
        run.custom_source(randomstart = True, uncollided = 0, moving=0)
        edges = run.edges
        N_space = run.parameters['all']['N_spaces'][0] 
        N_ang = run.parameters['fixed_source']['N_angles'][0] + 1
        M =  run.parameters['all']['Ms'][0]
        N_groups = 1 
        N_groups = run.parameters['all']['N_groups']
        n = N_space * (N_ang) * (M+1) *N_groups
        sigma_f = run.parameters['all']['sigma_f']
        nu = run.parameters['all']['nu'] 
        chi = run.parameters['all']['chi'] 
        euler_dt_num = run.parameters['all']['Euler_dt_num']
        sigma_a = run.parameters['all']['sigma_t'] - run.parameters['all']['sigma_s']
        shift = run.parameters['fixed_source']['shift']
        sigma_f_array = np.ones(run.xs.size) * sigma_f
        nu_array = np.ones(run.xs.size) * nu
        chi_array = np.ones(run.xs.size) * chi
        sigma_a_vec = np.zeros(N_space) 
        sigma_f_vec = np.ones(N_space) * sigma_f
        nu_vec = np.ones(N_space) * nu
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
       

    
        def matvec(x):
            run.load('Kornreich', 'mesh_parameters_Kornreich')

            run.kold = 1
            psi = x.reshape(((N_ang)*N_groups, N_space, M+1))
            fission_source = make_fission_scalar_flux(psi, run.edges, run.ws, N_ang, M, N_space, N_groups, sigma_f_vec * nu_vec)
            run.custom_source(randomstart = False, uncollided = 0, moving = 0, phi_coeffs=psi, sol_coeffs = fission_source)
            res_coefficients = np.copy(run.sol_ob.y[:,-1])
            with open('moving_mesh_transport/input_scripts/mesh_parameters_Kornreich.yaml', 'r') as file:
                data = yaml.safe_load(file)
                # data['all']['integrator'] = 'Euler'
                # data['dense'] = True
                # data['eval_times'] =False
                ts = run.sol_ob.t
                first_step = float(ts[1] - ts[0])
                data['first_step'] = first_step
                data['dense'] = True
                data['eval_times'] =False

                print(run.sol_ob.t[1] - run.sol_ob.t[0], 'first step')
                # assert 0
                with open('moving_mesh_transport/input_scripts/mesh_parameters_Kornreich.yaml', 'w') as file:
        # Use sort_keys=False to maintain a sensible order (optional)
                    yaml.dump(data, file, sort_keys=False)
            # print(res_coefficients.shape)

            return res_coefficients


        A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

# Compute k eigenvalues (largest magnitude by default)
        sigma = None
        try:
            sigma = -1/np.max(eigen_vals_DMD)
        except:
             sigma = None
             print('DMD did not give a nonzero eigenvalue')
        
        v0 = eigen_vectors_coeffs[:, max_DMD_alpha_coeffs_ind]
        #v0 = eigen_vectors[:,0] # will onlt be able to use this guess if VDMD is fed the coefficients, not psi
        vals, vecs = eigs(A, k=nalphas, sigma = sigma, which = 'LM', tol = alpha_tol, maxiter = maxits_power, v0 = v0)
        ws = run.ws
        print(vals, 'vals')

        x0 = run.parameters['fixed_source']['x0'][0]
        nu =run.parameters['all']['nu']
        alphas_IRAM = np.sort(-1/vals)
        phi0 = coeffs_to_phi(vecs[:,0].reshape((N_ang*N_groups, N_space, M+1)), xs, N_ang, N_groups, edges, ws, M)
        phi1 = coeffs_to_phi(vecs[:,1].reshape((N_ang*N_groups, N_space, M+1)), xs, N_ang, N_groups, edges, ws, M)
        print(np.max(alphas_IRAM), 'max alpha IRAM')

        print(alphas_IRAM, 'eigenvalues IRAM')

        f = h5py.File(f'Kornreich_results/data/Kornreich_alpha_S{N_ang}_{N_space}_cells_x0={x0}_nu={nu}.h5', 'w')
        f.create_dataset('alpha_list_IRAM_iteration', data = alphas_IRAM)
        f.create_dataset('eigenvectors', data = [phi0, phi1])
        f.close()
        
     

        plt.figure('eigenvectors')
        plt.xlabel('r [cm]', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.plot(run.xs, phi0, 'k-')
        plt.plot(run.xs, phi1, 'k--')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig('Kornreich_results/solution_plots/IRAM_eigenvectors.pdf')


    # power iteration
    elif k_list[-1]>=1:
        alpha_old = np.max(eigen_vals_DMD)
        alpha_old_old = 0
        # if get_k == True:
        # k_old = k_list[-1]
        coarse_solve = False
        # else:
        k_old = 1e-3
        phi = v0
        coarse_solve = True
        g_old = 0
        sigma_t_base = run.parameters['all']['sigma_t'] 
        N_spaces = run.parameters['all']['N_spaces'][0] 
        x0 = run.parameters['fixed_source']['x0'][0]
        nu = run.parameters['all']['nu']
        alpha_list = []
        alpha_list.append(alpha_old_old)
        alpha_list.append(alpha_old)
        iterations = 2
        phi = v0
        def residual(alpha_new, phi):
            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:
                    data = yaml.safe_load(file)
                    data['all']['sigma_t'] = sigma_t_base + alpha_new
                    with open('moving_mesh_transport/input_scripts/Kornreich_new.yaml', 'w') as file:
         
                        yaml.dump(data, file, sort_keys=False)
            k_list_new, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(k_old, 'Kornreich_new', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, coarse_solve=coarse_solve, max_its = max_its_kloop, input_phi=phi)
            return abs(k_list_new[-1]) -1
             
        
        alpha_final = newton(residual, np.max(eigen_vals_DMD), fprime=None, args=(phi), tol=alpha_tol, maxiter=maxits_power, fprime2=None, x1=None, rtol=alpha_tol, full_output=False, disp=True)


        # while abs(abs(k_old)-1) > alpha_tol and iterations < maxits_power:
        #     g = k_old -1
        #     alpha_new = alpha_old - g * (alpha_old - alpha_old_old) /(g - g_old + 1e-18)
        #     g_old = g
        #     # alpha_old_old = alpha_old
        #     print('## ## ## ## ## ## ## ## ## ## ## ## ## ##')
        #     print(alpha_new, 'alpha')
        #     print('## ## ## ## ## ## ## ## ## ## ## ## ## ##')
        #     print(k_old, 'k')

        #     with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

        # # Use yaml.safe_load() for security when dealing with untrusted input
        # # For a trusted config file, you might use yaml.FullLoader
        #         data = yaml.safe_load(file)
        #         data['all']['sigma_t'] = sigma_t_base + alpha_new
        #         with open('moving_mesh_transport/input_scripts/Kornreich_new.yaml', 'w') as file:
        # # Use sort_keys=False to maintain a sensible order (optional)
        #             yaml.dump(data, file, sort_keys=False)
        #     k_list_new, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(k_old, 'Kornreich_new', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, coarse_solve=coarse_solve, max_its = max_its_kloop, input_phi=phi)
        #     k_old = k_list_new[-1]
        #     alpha_old_old = alpha_old
        #     alpha_old = alpha_new
            
        #     iterations += 1
        #     alpha_list.append(alpha_old)
        #     plt.figure('alpha power method')
        #     plt.clf()
        #     nits = len(alpha_list)
        #     plt.plot(np.linspace(0, nits, nits)[1:], alpha_list[1:], '-o', mfc = 'none')
        #     plt.plot(np.linspace(0, nits, nits)[1:], np.ones(nits-1) * alpha_bench, 'k-', mfc = 'none')

        #     plt.xlabel('iterations', fontsize = 16)
        #     plt.ylabel(r'$\alpha$', fontsize = 16)
        #     plt.legend()
        #     plt.savefig(f'Kornreich_results/convergence_plots/power_method_alpha_Kornreich_{N_ang}_{N_space}_cells_x0={x0}_nu={nu}.pdf')
        #     plt.show()
        #     plt.close()
        #     f = h5py.File(f'Kornreich_results/data/Kornreich_alpha_S{N_ang}_{N_spaces}_cells_x0={x0}_nu={nu}.h5', 'w')
        #     f.create_dataset('alpha_list_power_iteration', data = alpha_list)
        #     f.close()
        # #     iterations += 1
        # print(alpha_list, 'alpha iterations')
        # print('alpha power iteration converged')
        # assert 0
        
        with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:
                    data = yaml.safe_load(file)
                    data['all']['sigma_t'] = sigma_t_base
                    with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
                        yaml.dump(data, file, sort_keys=False)

    # if VDMD_estimate == True and IRAM == True:
    if k_list[-1] < 1:
        nits = len(k_list)
        # plt.figure('alpha_vals')
        # plt.plot(np.linspace(0, nits, nits)[1:], np.ones(nits-1) * alpha_bench, 'k-', mfc = 'none')
        # plt.plot(nits, np.max(eigen_vals_DMD), 'o', label = 'DMD')          
        # plt.plot(nits, np.max(alphas_IRAM[-1]), 'o', label = 'IRAM')
        # plt.legend()
        alpha_final = np.sort(alphas_IRAM)[-1]
        # plt.ylim(-1, 1)
    # else:
        # plt.figure('alpha_vals')
        # plt.plot(np.linspace(0, nits, nits)[1:], np.ones(nits-1) * alpha_bench, 'k-', mfc = 'none')
        # plt.plot(nits, eigen_vals_DMD[-1], 'o', label = 'DMD')          
        # plt.legend()
        # plt.plot(np.linspace(0, nits, nits)[1:], alpha_list[1:], '-o', mfc = 'none', label = 'iterations')
        # alpha_final = np.sort(alpha_list)[-1]
    # plt.savefig('Kornreich_results/convergence_plots/alphas_Kornreich.pdf')
    if k_list[-1] <1:
        print('subcritical')
        print(alpha_bench, 'benchmark alpha')
        print(np.sort(alphas_IRAM)[-1], 'dominant alpha IRAM')
        print(np.sort(eigen_vals_DMD)[-1], 'DMD guess')
    
    f = h5py.File(f'Kornreich_results/data/kalpha_x0={x0}_nu={nu}.h5', 'r+')
    res_str = f'N_spaces={N_space}_N_angles={N_ang}_M={M}'
    if f.__contains__(res_str):
        del f[res_str]
    f.create_dataset(res_str, data = [k_list[-1], alpha_final])
    f.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    make_table(x0, nu, np.max(eigen_vals_DMD), alpha_final, k_list[-1])
    return k_list[-1], alpha_final, alpha_bench, k_bench, np.max(eigen_vals_DMD)


# Kornreich_benchmark(use_we = False, guess_k=  0.2)




def mesh_converge_Kornreich(cells_start = 20, N_angles = 96, max_cells = 200, tf = 5e3, euler_dt =6):
    converged = False
    k_guess = 0.8
    alpha_old = 1e-6
    tol = 1e-3
    k_list = []
    alpha_list = []
    cells_list = []
    DMD_alpha_list = []
    while converged == False:
          with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

    # Use yaml.safe_load() for security when dealing with untrusted input
    # For a trusted config file, you might use yaml.FullLoader
            data = yaml.safe_load(file)
            data['all']['N_spaces'][0] = cells_start
            data['all']['tfinal'] = float(tf)
            data['all']['Euler_dt_num'] = euler_dt
            data['fixed_source']['N_angles'][0] = N_angles
            nu = data['all']['nu']
            x0 = data['fixed_source']['x0'][0]
            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
    # Use sort_keys=False to maintain a sensible order (optional)
                yaml.dump(data, file, sort_keys=False)
          
          k_new, alpha_new, alpha_bench, k_bench, DMD_alpha = Kornreich_benchmark(guess_k=k_guess)
          DMD_alpha_list.append(DMD_alpha)
          cells_list.append(cells_start)
          converged = True
          if (np.abs(k_guess - k_new) <= tol and np.abs(alpha_new-alpha_old) <= tol):
               converged = True
          elif cells_start >= max_cells:
               converged = True
          else:
               k_guess = k_new
               alpha_old = alpha_new
               cells_start = int(cells_start * 1.5)
               k_list.append(k_guess)
               alpha_list.append(alpha_old)
               
               plt.figure('k converge')
               plt.loglog(cells_list, np.abs(np.array(k_list)-k_bench), '-o')
               plt.xlabel('spatial cells', fontsize = 16)
               plt.ylabel(r'$k_\mathrm{eff}$ error', fontsize= 16)
               ax = plt.gca()
               ax.spines['top'].set_visible(False)
               ax.spines['right'].set_visible(False)
               plt.savefig(f'Kornreich_results/convergence_plots/mesh_converge_k_N_angles={N_angles}_nu={nu}_x0={x0}.pdf', bbox_inches = 'tight')
               plt.figure('alpha converge')
               plt.loglog(cells_list, np.abs(np.array(alpha_list)-alpha_bench), '-o')
               plt.xlabel('spatial cells', fontsize = 16)
               plt.ylabel(r'$\alpha$ error', fontsize= 16)
               ax = plt.gca()
               ax.spines['top'].set_visible(False)
               ax.spines['right'].set_visible(False)
               plt.savefig(f'Kornreich_results/convergence_plots/mesh_converge_alpha_N_angles={N_angles}_nu={nu}_x0={x0}.pdf', bbox_inches = 'tight')
               plt.figure('alpha vals iterations')
               plt.semilogx(cells_list, alpha_list, '-o', mfc = 'none')
               plt.semilogx(cells_list, DMD_alpha_list, '-^', mfc = 'none')
               plt.semilogx(cells_list, np.ones(len(cells_list)) * alpha_bench, 'k-')
               ax = plt.gca()
               ax.spines['top'].set_visible(False)
               ax.spines['right'].set_visible(False)
               plt.xlabel('spatial cells', fontsize = 16)
               plt.ylabel(r'$\alpha$', fontsize= 16)
               plt.savefig(f'Kornreich_results/convergence_plots/mesh_converge_alphaval_N_angles={N_angles}_nu={nu}_x0={x0}.pdf', bbox_inches = 'tight')

               


def fill_Kornreich_table():
     x0_list = [4.6, 4.5]
     nu_list = [3.5, 1.5]
    #  x0_list = [4.6]
    #  nu_list = [3.5]
     for x0 in x0_list:
          for nu in nu_list:
            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:
                data = yaml.safe_load(file)
                # data['all']['integrator'] = 'Euler'
                # data['dense'] = True
                # data['eval_times'] =False
                data['all']['nu'] = float(nu)
                data['fixed_source']['x0'][0] = float(x0)
                with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
                    yaml.dump(data, file, sort_keys=False)
            mesh_converge_Kornreich(cells_start = 20, max_cells = 21, N_angles = 64, tf = 5e3, euler_dt =5)

     
fill_Kornreich_table()


# mesh_converge_Kornreich(N_angles = 2)


# mesh_converge_Kornreich(N_angles = 8)


# mesh_converge_Kornreich(cells_start=30, N_angles = 32)


# mesh_converge_Kornreich(N_angles = 32)


# mesh_converge_Kornreich(N_angles = 64)

# mesh_converge_Kornreich(N_angles = 96)



# mesh_converge_Kornreich(N_angles = 128)



     
     