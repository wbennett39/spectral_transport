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
# from diffeqpy import de

def RMS(l1, l2):
    return np.sqrt(np.mean((l1-l2)**2))


def sparsify_estimator(Y):
    U0, s0, VT0 = np.linalg.svd(Y, full_matrices=False)
    it = 0
    stop =False
    tol = 1e-10
    while stop == False and it < s0.size:
        if abs(s0[it]) < tol:
            stop = True
        else:
            it += 1
    print(s0[:it], 's')
    return it



def theta_optimizer(Y_minus, ts,  integrator, sigma_t, skip, theta, benchmark):
    itt = 0
    converged = False
    eigen = np.flip(DMD_func3(Y_minus, ts,  integrator, sigma_t, skip = skip, theta = theta))[0:4]
    RMS_old = RMS(eigen, benchmark)
    direction = -1
    speed = 0.1
    converge_count = 0
    while itt < 200:
        # if converge_count <=50:
            # theta_new = theta + speed * np.random.rand() * (np.random.rand()*2 - 1)
        # else:
        theta_new = np.random.rand()
        converge_count = 0

        if theta_new < 0.0:
            theta_new = 0.0
            direction *= -1
        elif theta_new > 1:
            theta_new = 1
            direction *= -1
        eigen = np.flip(DMD_func3(Y_minus, ts,  integrator, sigma_t, skip = skip, theta = theta_new))[0:4]
        RMS_NEW =  RMS(eigen[0], benchmark[0])
        
        if RMS_NEW < RMS_old:
            theta = theta_new
            # print(theta)
            # print(eigen)
        else:
            converge_count += 1
            direction *= -1 
        
        itt += 1
    
    return theta



run = run()
# run.load('transport', 'mesh_parameters_modak_gupta')
# run.plane_IC(0,0)
run.load('modak_gupta', 'mesh_parameters_modak_gupta')
loader = load()

benchmark_vals = {'0.0': np.array([-.763507, -1.57201, -2.98348, -5.10866]), '0.05': np.array([-.758893, -1.56062, -2.97899, -5.21764]),
                   '0.1': np.array([-.749672, -1.56062, -2.96323, -5.18772]), '0.25': np.array([-.703578, -1.45315, -3.07282, -5.26925]),
                   '0.5': np.array([-.551429, -1.71149, -2.94399, -5.28234])}
grain_sizes = ['0.0', '0.05', '0.1', '0.25', '0.5']



problem_list = ['modak_gupta0', 'modak_gupta05', 'modak_gupta1', 'modak_gupta25', 'modak_gupta5']


def results(theta = 0.55, run_results = False, skip = 3, iterate_theta = False, sparse_time_points = 10):
    if run_results == True:
        # ping save file
        integrator = run.parameters['all']['integrator']
        f = h5py.File(f'modak_gupta_results_{integrator}.h5', 'r+')
        f.close()
        # prime solver
        run.parameters['all']['integrator'] = 'BDF_VODE'
        run.parameters['all']['N_spaces'] = [5]
        run.parameters['all']['Ms'] = [0]
        run.parameters['random_IC']['N_angles'] = [2]
        run.random_IC(0,0)
        
        for sigma_name in problem_list:

            print(sigma_name, 'sigma function')
            run.load('modak_gupta', 'mesh_parameters_modak_gupta')
            run.mesh_parameters['modak_gupta0'] = False
            sigma_name == 'modak_gupta0'
            # run.parameters['all']['integrator'] = 'BDF_VODE'
            if sigma_name == 'modak_gupta0':
                run.parameters['all']['sigma_s'] = 9.5
            run.random_IC(0,0)
            Yminus = run.sol_ob.Y_minus_psi
            # plt.ion()
            # for itt in range(run.sol_ob.y[0, :].size):
            #     plt.plot(run.xs, run.sol_ob.y[:,itt])
            # plt.show()
            print('saving results')
            integrator = run.parameters['all']['integrator']
            # run.parameters['all']['integrator'] = 'BDF'
            f = h5py.File(f'modak_gupta_results_{integrator}.h5', 'r+')
            if f.__contains__(sigma_name):
                del f[sigma_name]
            f.create_group(sigma_name)
            f[sigma_name].create_dataset('Y_minus', data = Yminus)
            f[sigma_name].create_dataset('t', data = run.sol_ob.t)
            f.close()


    print('### ### ### ### ### ### ### ### ### ')
    print('### ### ### ### ### ### ### ### ### ')
    print('### ### ### ### ### ### ### ### ### ')
    print('### ### ### ### ### ### ### ### ### ')
    print('### ### ### ### ### ### ### ### ### ')
    print('### ### ### ### ### ### ### ### ### ')

    for iterator in range(5):
            sigma_name = problem_list[iterator]
            benchmark_eigen = benchmark_vals[grain_sizes[iterator]] 
            integrator = run.parameters['all']['integrator']
            print(integrator, 'integrator')
            # integrator = 'BDF_VODE'
            f = h5py.File(f'modak_gupta_results_{integrator}.h5', 'r+')
            # f = h5py.File(f'modak_gupta_results.h5', 'r+')
            Y_minus = f[sigma_name]['Y_minus'][:,:]
            
            ts = f[sigma_name]['t'][:]
            # sparse_time_points = int(sparsify_estimator(Y_minus) * 1)
            print('number of snapshots', sparse_time_points)
            

            f.close()
            sigma_t = run.parameters['all']['sigma_t']
            if iterate_theta == True and (integrator =='BDF' or integrator == 'BDF_VODE'):
                theta = theta_optimizer(Y_minus, ts,  integrator, sigma_t, skip = skip, theta = theta, benchmark = benchmark_eigen)    
                


            eigen_vals = DMD_func3(Y_minus, ts,  integrator, sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points)
            

            if eigen_vals.size < 4:
                eigen_vals = np.append(np.zeros(4), eigen_vals)
            first_four_eigen_vals = np.flip(eigen_vals)[0:4] 
            print('----------------------------------------------')
            print('theta = ', theta)
            print('grain size: ', grain_sizes[iterator])
            print('solver eigen values ')
            print(first_four_eigen_vals)
            print('benchmark eigen values ')
            print(benchmark_eigen)
            print('error ')
            print(first_four_eigen_vals - benchmark_eigen)




    

# results()