# imports functions to run package from terminal 

import sys
import matplotlib.pyplot as plt
# sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
                      
# from benchmarks import integrate_greens as intg
from moving_mesh_transport.plots import plotting_script as plotter
from moving_mesh_transport import solver
import matplotlib.pyplot as plt

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

import h5py 

import numpy as np
from moving_mesh_transport.solver_functions.run_functions import run

N_spaces_list = [125, 200, 250, 300, 350, 400]
# N_spaces_list = [215]
# N_spaces_list =. [300]
MM = 1
N_ang = 12

run = run()
run.load()

loader = load()
run.parameters['all']['Ms'] = [0]
run.parameters['all']['N_spaces'] = [6]
run.parameters['all']['tfinal'] = 0.000000000001
run.parameters['integrator'] = 'BDF'
run.mesh_parameters['eval_times'] = False
run.mesh_parameters['Msigma'] = 0
run.parameters['boundary_source']['N_angles'] = [2]

run.boundary_source(0,0)
plt.close()
plt.close()
plt.close()
plt.close()

run.load('marshak')
for it, N_space in enumerate(N_spaces_list):
    
    run.parameters['boundary_source']['x0'] = np.array([1e-3])
    run.parameters['all']['Ms'] = [MM]
    run.mesh_parameters['Msigma'] = MM
    run.parameters['all']['rt'] = 5e-3
    run.parameters['all']['at'] = 5e-4
    run.parameters['boundary_source']['N_angles'] = [N_ang]


    # menis_times = np.array([-22.122309, -9.4484244, -1])
    menis_times = np.array([-22.122309, -9.4484244, -1])

    # menis_times = np.array([-24, -22.122309, -9.4484244])

    # menis_times =  np.array([-25, -23, -22.122309])
    # menis_times = np.array([-29.625, -29.6, -29.5])
    # menis_times = np.array([-29.0, -28.5, -28.0])
    dimensional_times =  29.625647 + menis_times 

    run.mesh_parameters['eval_array'] = dimensional_times * 29.98
    print(run.mesh_parameters['eval_array'], 'evaluation times')
    run.parameters['all']['tfinal'] = (dimensional_times * 29.98)[-1]
    run.parameters['all']['N_spaces'] = [N_space]
    run.mesh_parameters['sigma_func'] = {'constant': False, 'linear': False, 'siewert1': False, 'siewert2': False, 'gaussian': False, 'f_sedov': False, 'converging': False, 'test1': True, 'test2': False, 'test3': False, 'test4': False}
    # run.parameters['all']['tfinal'] = 10.0
    # run.mesh_parameters['eval_times'] = False

    run.boundary_source(0,0)
    f = h5py.File('converging_heat/results_test1_1031.h5','r+')
    M = run.parameters['all']['Ms'] 
    spaces = run.parameters['all']['N_spaces']
    if f.__contains__(f'M={M}_{spaces}_cells'):
        del f[f'M={M}_{spaces}_cells']

    f.create_group(f'M={M}_{spaces}_cells')

    if f[f'M={M}_{spaces}_cells'].__contains__('scalar_flux'):
        del f['scalar_flux']
        del f['energy_density']
        del f['xs']
    if f[f'M={M}_{spaces}_cells'].__contains__('edges'):
        del f['edges']


    f[f'M={M}_{spaces}_cells'].create_dataset('scalar_flux', data = run.phi)
    f[f'M={M}_{spaces}_cells'].create_dataset('energy_density', data = run.e)
    f[f'M={M}_{spaces}_cells'].create_dataset('xs', data = run.xs)
    f[f'M={M}_{spaces}_cells'].create_dataset('edges', data = run.edges)
    # print('###')
    # print(run.phi,'scalar flux')
    # print('###')
    # print(f['scalar_flux'][:],'loaded scalar flux')
    f.close()