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
from moving_mesh_transport.solver_classes.functions import T_bath, converging_time_function

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

# menis_times = np.array([-145, -140, -135])
# menis_times = np.array([-140, -135, -130])
# menis_times = np.array([-160, -27.126998, -1])
# dimensional_times =  145.4733877 + menis_times 
# tlist = np.linspace(dimensional_times[0] * 29.98, dimensional_times[-1]*29.98, 100)
# menis_t = converging_time_function(tlist, {'test1': False, 'test2': False, 'test3': False, 'test4': True})
# print(menis_t)
# run = run()
# run.load()
# bath = tlist * 0
# for it, tt in enumerate(tlist):
#     bath[it] = T_bath(menis_t[it], {'test1': False, 'test2': False, 'test3': False, 'test4': True})
# plt.figure(18)
# plt.plot(menis_t, bath)
# print(bath)
# plt.show()









# N_spaces_list = [45]
MM = 1
N_ang = 8
# N_spaces_list = [10, 15, 20, 25, 50, 75, 100, 150, 200, 500, 1000]
# N_spaces_list = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_spaces_list = [500]

run = run()
run.load()

loader = load()
run.parameters['all']['Ms'] = [0]
run.parameters['all']['N_spaces'] = [6]
run.parameters['all']['tfinal'] = 0.00000000000001
run.parameters['integrator'] = 'BDF'
run.mesh_parameters['eval_times'] = False
run.mesh_parameters['Msigma'] = 0
run.parameters['boundary_source']['N_angles'] = [2]
run.boundary_source(0,0)

run.load('marshak')
for it, N_space in enumerate(N_spaces_list):
    run.parameters['boundary_source']['x0'] = np.array([10.0])
    run.parameters['all']['N_spaces'] = [N_space]
    run.parameters['all']['rt'] = 1e-4
    run.parameters['all']['at'] = 1e-5
    # run.parameters['all']['at'] = 5e-1
    run.parameters['all']['e_initial'] = 0.00001
    run.parameters['boundary_source']['N_angles'] = [N_ang]
    run.parameters['all']['Ms'] = [MM]
    run.mesh_parameters['Msigma'] = MM
    run.parameters['integrator'] = 'BDF'
    print(run.parameters['integrator'], 'integrator methods')

    # menis_times = np.array([-94.706889, -27.126998, -1])
    # menis_times = np.array([-1])
    # menis_times = np.array([-140, -94.706889, -27.126998])
    # menis_times = np.array([-140, -100, -94.706889])
    # menis_times = np.array([-140, -100, -97.706889])
    # menis_times = np.array([-145, -125, -115])
    menis_times = np.array([-145.47, -145.4, -145.3])

    dimensional_times =  145.4733877 + menis_times 

    run.mesh_parameters['eval_array'] = dimensional_times * 29.98
    print(run.mesh_parameters['eval_array'], 'evaluation times')
    run.parameters['all']['tfinal'] = (dimensional_times * 29.98)[-1]
    run.mesh_parameters['sigma_func'] = {'constant': False, 'linear': False, 'siewert1': False, 'siewert2': False, 'gaussian': False, 'f_sedov': False, 'converging': False, 'test1': False, 'test2': False, 'test3': False, 'test4': True}


    # run.parameters['all']['tfinal'] = 10.0
    # run.mesh_parameters['eval_times'] = False

    run.boundary_source(0,0)
    f = h5py.File('converging_heat/results_test4_1030.h5','r+')
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