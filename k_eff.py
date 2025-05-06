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
    k_old = kguess
    converged = False
    run.custom_source(randomstart = True, uncollided = 0, moving = 0)

    coeffs = run.sol_ob.y[:,-1]

    phi_interpolated = interp1d(run.xs, run.phi[:,0])
    print(run.phi, 'run.phi')
    print(phi_interpolated(run.xs))
    print('phi')
    normalized_integrand = lambda x: phi_interpolated(x) * run.parameters['all']['nu'] * run.parameters['all']['sigma_t'] * run.parameters['all']['chi']
    normalization = integrate.quad(normalized_integrand, run.xs[0], run.xs[-1])[0]
    while converged == False: 
        run.load('k_eff', 'mesh_parameters_keff')
        sigma_f = run.parameters['all']['sigma_f']
        run.parameters['all']['sigma_f'] = sigma_f / k_old
        normalized_source = run.sol_ob.y[:,-1] / normalization
        # output_ob = 
        run.custom_source(randomstart = False, sol_coeffs = normalized_source, uncollided = 0, moving = 0)
        phi_interpolated_new = interp1d(run.xs, run.phi[:,0])
        integrand = lambda x: k_old * phi_interpolated_new(x) / (phi_interpolated(x) + 1e-12) # because nu and sigma_t are constant right now, I don't need them in the integrand
        xs = run.xs
        plt.figure(201)
        plt.plot(xs, integrand(xs))
        plt.show()
        # converged = True
        k_new = integrate.quad(integrand, xs[0], xs[-1])[0]/(xs[-1]-xs[0])
        print(k_new, 'k')
        # converged = True
        if abs(k_new - k_old ) <=tol:
            print('power iteration complete')
            print(k_new, 'k effective')
            converged = True
        else:
            k_old = k_new
            phi_interpolated = phi_interpolated_new
            normalized_integrand = lambda x: phi_interpolated(x) * run.parameters['all']['nu'] * run.parameters['all']['sigma_t'] * run.parameters['all']['chi']
            normalization = integrate.quad(normalized_integrand, run.xs[0], run.xs[-1])[0]


power_iterate()

