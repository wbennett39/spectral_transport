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
from k_iterate import power_iterate 

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
run.load('Kornreich', 'mesh_parameters_Kornreich')
loader = load()
def Kornreich_benchmark(prime = True, get_k = True, VDMD_estimate = True, IRAM = True, guess_k = np.random.rand()):
    if prime == True:
        run.parameters['all']['N_spaces'] = [10]
        run.parameters['all']['Ms'] = [0]
        run.parameters['random_IC']['N_angles'] = [2]
        run.parameters['all']['sigma_f'] = 1.0
        run.custom_source(randomstart=True, uncollided = 0, moving = 0 )

    # First, find k_eff
    if get_k == True:
        k_list, time_list, normalization_list, run_ob = power_iterate(guess_k, 'Kornreich', 'mesh_parameters_Kornreich', run, tol = 1e-7)
        print(k_list, 'k_list')
        print(k_list[-1], 'k effective')
        print(time_list, 'computation time required per iterate')
        f = h5py.File('Kornreich_keff.h5', 'w')
        f.create_dataset('scalar_flux', data = run_ob.phi)
        f.create_dataset('xs', data = run_ob.xs)
        f.create_dataset('psi', data = run_ob.psi)
        f.create_dataset('Y_minus', data = run_ob.sol_ob.Y_minus_psi)
        f.create_dataset('t', data = run_ob.sol_ob.t)
        f.close()
        # run.parameters['all']['N_spaces'] = [150]
        # run.parameters['all']['Ms'] = [2]
        # run.parameters['random_IC']['N_angles'] = [512]
        # run.parameters['fixed_source']['N_angles'] = [512]
        # k_list, time_list = power_iterate(k_list[-1], 'Kornreich', 'mesh_parameters_Kornreich', run, tol = 1e-12)

    # Estimate alpha modes with VDMD
    # Y_minus_residual = Y_minus.copy() 
    # for it in range(1, time_list.size):
    #     dt = time_list[it] - time_list[it-1]
    #     Y_minus_residual -= dt * source
    # eigen_vals = DMD_func3(Y_minus_residual, time_list, 'Euler', sigma_t, skip = skip, theta = theta, sparse_time_points=sparse_time_points)


    # IRAM to get alpha modes


Kornreich_benchmark()