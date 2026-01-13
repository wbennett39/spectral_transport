# imports functions to run package from terminal 

import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
                      
# from benchmarks import integrate_greens as intg
from moving_mesh_transport.plots import plotting_script as plotter
from moving_mesh_transport import solver
import matplotlib.pyplot as plt
import h5py 

from moving_mesh_transport.solver_classes.functions import *
from k_iterate import power_iterate, test_normTnintcell, check_norm_flux
import yaml

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

# from diffeqpy import de

def RMS(l1, l2):
    return np.sqrt(np.mean((l1-l2)**2))



# prime solver
run = run()
# run.load('transport', 'mesh_parameters_modak_gupta')
# run.plane_IC(0,0)

loader = load()
def Kornreich_benchmark(prime = False, get_k = True, VDMD_estimate = False, IRAM = False, power_method = True, guess_k = 1, sparse_time_points = 12, skip =4, ktol = 5e-3, use_we = False, max_its_kloop = 100, coarse_angles = 4, alpha_tol = 1e-5):
    # test_normTnintcell()
    # check_norm_flux()
    # assert 0
    run.load('Kornreich', 'mesh_parameters_Kornreich')
    if prime == True:
        run.parameters['all']['N_spaces'] = [10]
        run.parameters['all']['tfinal'] = 0.00000000001
        run.parameters['all']['Ms'] = [0]
        run.parameters['random_IC']['N_angles'] = [2]
        # run.parameters['fixed_source']['N_angles'] = [2]
        # run.parameters['all']['sigma_f'] = 1.0
        run.custom_source(randomstart=True, uncollided = 0, moving = 0 )

    # First, find k_eff
    if get_k == True:
        # coarse solve
        k_list, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(guess_k, 'Kornreich', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, max_its = max_its_kloop, coarse_angles=coarse_angles, coarse_solve=True)
        # fine solve
        k_list, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(k_list[-1], 'Kornreich', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, max_its = max_its_kloop, input_phi=phi)
        print(k_list, 'k_list')
        print(k_list[-1], 'k effective')
        print(0.4243163, 'benchmark k effective')
        print(time_list, 'computation time required per iterate')
        N_ang = run.parameters['fixed_source']['N_angles'][0]
        N_spaces = run.parameters['all']['N_spaces'][0]
        x0 = run.parameters['fixed_source']['x0'][0]
        nu =run.parameters['all']['nu']
        f = h5py.File(f'Kornreich_results/Kornreich_keff_S{N_ang}_{N_spaces}_cells_x0={x0}_nu={nu}.h5', 'w')
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

        plt.figure('keff')
        if run.parameters['fixed_source']['shift'] == 0.0:
            if run.parameters['fixed_source']['x0'][0] ==4.5:
                if run.parameters['all']['nu'] == 1.5:
                    k_bench = 0.4241317
                elif run.parameters['all']['nu'] == 3.5:
                    k_bench = 0.9896407
            elif run.parameters['fixed_source']['x0'][0] ==4.6:
                if run.parameters['all']['nu'] == 1.5:
                    k_bench = 0.4242237
                elif run.parameters['all']['nu'] == 3.5:
                    k_bench = 0.9898554


        else: #not ready for other cases. Probably not necessary
            k_bench = 0.4243163
        nits = len(k_list)
        plt.plot(np.linspace(0, nits, nits), k_list, '-o', mfc = 'none')
        plt.xlabel('iteration', fontsize = 16)
        plt.plot(np.linspace(0, nits, nits), np.ones(nits) * k_bench, 'k-', label = 'benchmark')
        plt.ylabel(r'$k_\mathrm{eff}$', fontsize = 16)
        plt.legend()
        plt.savefig('Kornreich_results/k_iterations_Kornreich.pdf')
        plt.show()

        plt.figure('calc time')
        plt.loglog(np.linspace(0, nits, nits)[1:], time_list, '-o', mfc = 'none')
        plt.xlabel('iteration', fontsize = 16)
        plt.ylabel('time [s]', fontsize = 16)
        plt.savefig('Kornreich_results/calc_time_Kornreich.pdf')
        plt.show()

        plt.figure('flux shape')
        nits = len(k_list)
        plt.plot(run_ob.xs, run_ob.phi, '-', mfc = 'none')
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.legend()
        # plt.savefig('Kornreich_results/scalar_flux_Kornreich.pdf')
        plt.show()



        plt.figure('normalize')
        nits = len(k_list)
        plt.plot(np.linspace(0, nits, nits)[1:], normalization_list[1:], '-o', mfc = 'none')
        plt.xlabel('iterations', fontsize = 16)
        plt.ylabel('normalization', fontsize = 16)
        plt.legend()
        plt.savefig('Kornreich_results/norm_iterations_Kornreich.pdf')
        plt.show()
        plt.show()


        plt.figure('keff_log')
        nits = len(k_list)
        plt.loglog(np.linspace(0, nits, nits)[1:], np.abs(np.array(k_list[1:]) - np.array(k_list[:-1])), '-o', mfc = 'none')
        plt.xlabel('iterations', fontsize = 16)
        # plt.loglog(np.linspace(0, nits, nits), np.ones(nits) * 0.4243163, 'k-', label = 'benchmark')
        plt.ylabel(r'$k_\mathrm{eff}$ difference', fontsize = 16)
        plt.legend()
        plt.savefig('Kornreich_results/k_iterations_Kornreic_log.pdf')
        plt.show()

        plt.figure('solution plot')
        plt.xlabel('x [cm]', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.plot(run_ob.xs, run_ob.phi, 'k-', mfc = 'none')
        plt.savefig('Kornreich_results/scalar_flux_Kornreich.pdf')
        plt.show()



    # Estimate alpha modes with VDMD
    if VDMD_estimate == True:
        f = h5py.File('Kornreich_results/Kornreich_results/Kornreich_keff.h5', 'r+')
        ts = f['t']
        fission_source = f['fission_source'][:]
        Y_minus = f['Y_minus'][:,:]
        N_ang = f['N_angles'][:][0]
        xs = f['xs']
        # Y_minus_shifted = np.array(Y_minus).copy().reshape(N_ang, xs.size, ts.size)
        # adjust Y- to remove source influence
      
        integrator = run.parameters['all']['integrator']
        sigma_t = run.parameters['all']['sigma_t']
        # skip = 4
        theta = 0
        eigen_vals = DMD_func3(Y_minus, ts,  integrator, sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points, source = True, sourcevec =  fission_source, N_ang = N_ang, xs = xs)
        print(np.flip(eigen_vals), 'alpha eigen values VDMD')
        print(-0.3196537,-0.3229855, 'benchmark first two alpha eigen values' )
    # Y_minus_residual = Y_minus.copy() 
    # for it in range(1, time_list.size):
    #     dt = time_list[it] - time_list[it-1]
    #     Y_minus_residual -= dt * source
    # eigen_vals = DMD_func3(Y_minus_residual, time_list, 'Euler', sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points)


    # IRAM to get alpha modes

    # power iteration
    if power_method == True:
        alpha_old = 1e-5
        alpha_old_old = 0
        k_old = k_list[-1]
        g_old = 0
        sigma_t_base = run.parameters['all']['sigma_t'] 
        alpha_list = []
        alpha_list.append[alpha_old_old]
        alpha_list.append(alpha_old)
        iterations = 2
        while abs(k_old-1) > alpha_tol:
            g = k_old -1
            alpha_new = alpha_old - g * (alpha_old - alpha_old_old) /(g - g_old)
            g_old = g
            alpha_old_old = alpha_old
            print(alpha_new, 'alpha')
            print(k_old, 'k')

            with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:

        # Use yaml.safe_load() for security when dealing with untrusted input
        # For a trusted config file, you might use yaml.FullLoader
                data = yaml.safe_load(file)
                data['all']['sigma_t'] = sigma_t_base + alpha_new
                with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
        # Use sort_keys=False to maintain a sensible order (optional)
                    yaml.dump(data, file, sort_keys=False)
            k_list, time_list, normalization_list, run_ob, sigma_f_vec, nu_vec, phi = power_iterate(k_list[-1], 'Kornreich', 'mesh_parameters_Kornreich', run, tol = ktol, use_we_accel= use_we, max_its = max_its_kloop, input_phi=phi)
            k_old = k_list[-1]
            alpha_old = alpha_new
            iterations += 1
            alpha_list.append(alpha_old)
        plt.figure('alpha power method')
        nits = len(alpha_list)
        plt.plot(np.linspace(0, nits, nits)[1:], alpha_list[1:], '-o', mfc = 'none')
        plt.xlabel('iterations', fontsize = 16)
        plt.ylabel(r'$\alpha$', fontsize = 16)
        plt.legend()
        plt.savefig('Kornreich_results/power_method_alpha_Kornreich.pdf')
        plt.show()
        plt.show()
            
    with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'r') as file:
                data = yaml.safe_load(file)
                data['all']['sigma_t'] = sigma_t_base
                with open('moving_mesh_transport/input_scripts/Kornreich.yaml', 'w') as file:
                    yaml.dump(data, file, sort_keys=False)

Kornreich_benchmark(use_we = False, guess_k=  0.2)