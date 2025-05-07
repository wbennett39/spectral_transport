# imports functions to run package from terminal 

import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# from julia.api import Julia
# jl = Julia(compiled_modules=False)
                      
# from benchmarks import integrate_greens as intg
from moving_mesh_transport.plots import plotting_script as plotter
from moving_mesh_transport import solver
import matplotlib.pyplot as plt
import h5py 

from moving_mesh_transport.solver_classes.functions import *
from moving_mesh_transport.solver_classes.make_phi import make_output

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
from scipy.interpolate import interp1d
import scipy.integrate as integrate



        # output_ob  = make_output(tt, N_ang, ws, xs, (run.sol_ob.y[:,-1] / normalization).reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
        # phi = output_ob.make_phi(uncollided_sol)
        # phi_interp_test = interp1d(xs, phi)
        # test_normalization_integrand = lambda x: phi_interp_test(x)  * run.parameters['all']['nu'] * run.parameters['all']['sigma_t'] * run.parameters['all']['chi']
        # normalization_test = integrate.quad(test_normalization_integrand, run.xs[0], run.xs[-1])[0]/(xs[-1]-xs[0])

        # print(normalization_test, 'should be 1')
def check_normalization(output_ob, uncollided_ob, xs):
    phi = output_ob.make_phi(uncollided_ob)
    phi_interp_test = interp1d(xs, phi)
    test_normalization_integrand = lambda x: phi_interp_test(x)
    normalization_test = integrate.quad(test_normalization_integrand, run.xs[0], run.xs[-1])[0]
    print('--- --- --- --- --- --- ---')
    print(normalization_test, 'should be 1')
    print('--- --- --- --- --- --- ---')


    
def integrate_phi_cell(cs, ws, a, b, M, N_ang):
    cell_volume = 4 * math.pi * (b**3 - a**3)
    res = 0.0
    for l in range(N_ang):
        for j in range(M+1):
            res += ws[l] * cs[l, j] * normTn_intcell(j, a, b)
    return 4 * math.pi * res #* cell_volume


def normalize_phi(VV, edges, ws, N_ang, M, N_space, N_groups, sigma_f, nu, chi):
    norm_phi = np.zeros(N_space)
    for ig in range(N_groups):
        for ix in range(N_space):
            norm_phi[ix] += integrate_phi_cell(VV[ig * N_ang: (ig+1) * N_ang, ix, :], ws, edges[ix], edges[ix+1], M, N_ang)
    return np.sum(norm_phi) # * sigma_f * nu * chi


run = run()
# run.load('transport', 'mesh_parameters_modak_gupta')
# run.plane_IC(0,0)
run.load('k_eff', 'mesh_parameters_keff')
loader = load()

run.parameters['all']['N_spaces'] = [5]
run.parameters['all']['Ms'] = [0]
run.parameters['random_IC']['N_angles'] = [2]
run.custom_source(randomstart=True, uncollided = 0, moving = 0 )

def power_iterate(kguess = 1.0, tol = 1e-4):
    run.load('k_eff', 'mesh_parameters_keff')
    klist = []
    k_old = kguess
    converged = False
    run.custom_source(randomstart = True, uncollided = 0, moving = 0)
    sigma_f = run.parameters['all']['sigma_f']
    nu = run.parameters['all']['nu']
    chi = run.parameters['all']['chi']
    coeffs = run.sol_ob.y[:,-1]
    N_ang = run.parameters['fixed_source']['N_angles'][0]
    N_ang += 2
    ws = run.ws
    N_groups = run.parameters['all']['N_groups']
    M = run.parameters['all']['Ms'][0]
    N_space = run.parameters['all']['N_spaces'][0]
    tt = run.parameters['all']['tfinal']
    uncollided = False
    # geometry = run.parameters['all']['geometry']
    geometry = run.geometry
    uncollided_ob = run.uncollided_ob
    edges = run.edges

    phi_interpolated = interp1d(run.xs, run.phi[:,0])
    # print(run.phi, 'run.phi')
    # print(phi_interpolated(run.xs))
    # print('phi')
    # normalized_integrand = lambda x: phi_interpolated(x) * run.parameters['all']['nu'] * run.parameters['all']['sigma_t'] * run.parameters['all']['chi']
    xs = run.xs
    # normalization = integrate.quad(normalized_integrand, run.xs[0], run.xs[-1])[0]*(xs[-1]-xs[0]) #do I need to normalize in each cell?
    normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups,N_space,M+1)), edges, ws, N_ang, M, N_space, N_groups, sigma_f, nu, chi)
    n_iters = 0
    normalization_list = []
    normalization_list.append(normalization)
    normalization_old = normalization
    while converged == False and n_iters < 50: 
        run.load('k_eff', 'mesh_parameters_keff')
        run.parameters['all']['sigma_f'] = sigma_f / k_old
        # normalize fission source
        normalized_source = np.copy(run.sol_ob.y[:,-1].reshape((N_ang * N_groups,N_space,M+1)))
        normalized_source *= 1 / normalization
        # testing normalization
        output_ob  = make_output(tt, N_ang, ws, xs, normalized_source, M, edges, uncollided, geometry, N_groups)
        check_normalization(output_ob, uncollided_ob, xs)
        # run solver
        run.custom_source(randomstart = False, sol_coeffs = normalized_source, uncollided = 0, moving = 0)
        phi_interpolated_new = interp1d(run.xs, run.phi[:,0])
        integrand = lambda x: k_old * phi_interpolated_new(x) / (phi_interpolated(x) + 1e-12) # because nu and sigma_t are constant right now, I don't need them in the integrand
        xs = run.xs
        plt.figure(201)
        plt.plot(xs, phi_interpolated(xs), label = f'iteration {n_iters}')
        plt.legend()
        plt.show()
        # update k
        # k_new = integrate.quad(integrand, xs[0], xs[-1])[0]/(xs[-1]-xs[0])
        k_new = k_old * normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups,N_space,M+1)), edges, ws, N_ang, M, N_space, N_groups, sigma_f, nu, chi) / normalization
        print(k_new, 'k')
        # converged = True
        if abs(k_new - k_old ) <=tol:
            print('power iteration complete')
            print(k_new, 'k effective')
            converged = True
        else:
            k_old = k_new
            klist.append(k_new)
            n_iters +=1
            phi_interpolated = phi_interpolated_new
            # normalized_integrand = lambda x: phi_interpolated(x) * run.parameters['all']['nu'] * run.parameters['all']['sigma_t'] * run.parameters['all']['chi']
            # normalization_old = normalization
            # normalization = integrate.quad(normalized_integrand, run.xs[0], run.xs[-1])[0]*(xs[-1]-xs[0])
            normalization = normalize_phi(run.sol_ob.y[:, -1].reshape((N_ang * N_groups,N_space,M+1)), edges, ws, N_ang, M, N_space, N_groups, sigma_f, nu, chi)
            # normalization_list.append(normalization)
        plt.figure(74)
        plt.plot(np.linspace(0, n_iters, n_iters+1)[1:], klist, '-o')
        plt.savefig('k_eff_converge.pdf')
        plt.show()
        plt.close()

        # plt.figure(74)
        # plt.plot(np.linspace(0, n_iters, n_iters+1), normalization_list, '-o')
        # plt.savefig('normalize_converge.pdf')
        # plt.show()
        # plt.close()

power_iterate()

