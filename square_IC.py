# imports functions to run package from terminal 

import sys
import matplotlib.pyplot as plt
# sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# from julia.api import Julia
# jl = Julia(compiled_modules=False)
import sys
sys.path.append('/Users/wbennett/Documents/Github/transport_benchmarks/')
print(sys.path)
from benchmarks import integrate_greens as intg
from moving_mesh_transport.plots import plotting_script as plotter
from moving_mesh_transport import solver
import matplotlib.pyplot as plt
import h5py 
import time

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



# prime solver
run = run()
# run.load('transport', 'mesh_parameters_modak_gupta')
# run.plane_IC(0,0)
run.load('transport', 'mesh_parameters')
loader = load()


def RMSE(l1, l2):
    return np.sqrt(np.mean((l1-l2)**2))
time_list = [0.1, 0.5, 1.0, 5.0, 10.0]
# time_list = [1.0, 5.0]

N_space_list = [5, 10, 20, 40, 80]
def get_bench(xs, t):
    ob = intg.shell_source(t, 100, choose_xs = True, xpnts = xs)
    return ob[1] + ob[2]

def square_IC_converge(time_list = time_list, N_space_list = N_space_list, run_results = True, uncollided = True, moving_mesh = True, M = 3, N_ang = 256):
   
    if run_results == True: #re-run calculations
        f = h5py.File('shell_source.h5', 'r+')
        f.close()
        run.parameters['all']['N_spaces'] = [5]
        run.parameters['all']['Ms'] = [0]
        run.parameters['square_IC']['N_angles'] = [2]
        run.square_IC(0,0)
        run.load('transport', 'mesh_parameters')
        for it, tt in enumerate(time_list):
            for space in N_space_list:    
                run.parameters['all']['Ms'] =  [M] 
                run.parameters['square_IC']['N_angles'] =  [N_ang]
                run.parameters['all']['N_spaces'] = [space]
                run.parameters['all']['tfinal'] = tt
                run.mesh_parameters['Msigma'] = M
                run.square_IC(uncollided, moving_mesh)
                f = h5py.File('shell_source.h5', 'r+')
                save_string = f't={tt}_uncollided={uncollided}_moving_mesh={moving_mesh}_N_space={space}_N_ang={N_ang}_M={M}'
                if f.__contains__(save_string):
                    del f[save_string]
                if f.__contains__(save_string + 'angular_flux'):
                    del f[save_string + 'angular_flux']
                dat = np.zeros((2, run.xs.size))
                dat[0] = run.xs
                dat[1] = run.phi.transpose()
                f.create_dataset(save_string, data = dat)
                f.create_dataset(save_string + 'angular_flux', data = run.psi)
                f.close()
    # plot benchmark results

    for it, tt in enumerate(time_list):
        err_list = []
        for k, space in enumerate(N_space_list):
            f = h5py.File('shell_source.h5', 'r+')
            save_string = f't={tt}_uncollided={uncollided}_moving_mesh={moving_mesh}_N_space={space}_N_ang={N_ang}_M={M}'
            res = f[save_string][:,:]
            f.close()
            xs = res[0]
            phi = res[1]
            # bench = get_bench(xs, tt)
            f2 = h5py.File('shell_IC_benchmarks.h5', 'r+')
            res2 = f2[f't={tt}']
            xsb = res2[0]
            phib = res2[1]
            f2.close
            bench_interp = interp1d(xsb, phib)
            bench = bench_interp(xs)
            # bench = xs*0
            err_list.append(RMSE(phi, bench))
            print(err_list, 'RMSE LIST')
            print(RMSE(phi, bench), 'RMSE')
            plt.plot(xs, phi, '-o', mfc = 'none')
            plt.plot(xs, bench, 'k-')
            plt.savefig(f'shell_source_solution_t={tt}.pdf')
            plt.close()
            

        print(err_list, 'err list')
        plt.loglog(N_space_list, err_list / bench[0], '-o', mfc = 'none')
        plt.xlabel('spatial cells', fontsize = 16)
        plt.ylabel('scaled RMSE')
        plt.title(f'{N_ang} angles, M={M}') 
        plt.savefig(f'shell_source_RMSE_t={tt}_uncollided={uncollided}_moving_mesh={moving_mesh}.pdf')
        plt.close()

       
def calculate_benchmarks():
    for it, tt in enumerate(time_list):
        xs = np.linspace(0.00000000001, 0.5 + tt, 500)
        bench = get_bench(xs, tt)
        f = h5py.File('shell_IC_benchmarks.h5', 'r+')
        if f.__contains__(f't={tt}'):
                    del f[f't={tt}']
        f.create_dataset(f't={tt}', data = np.array([xs, bench]))
        f.close()




# square_IC_converge(moving_mesh=False, uncollided=False, M=0, N_space_list=[50], N_ang = 96, run_results = False)
# square_IC_converge(moving_mesh=False, uncollided=False, M=2, N_space_list=[12, 25, 50, 75, 100], N_ang = 64, run_results = False)
# square_IC_converge(moving_mesh=False, uncollided=False, M=3, N_space_list=[50], N_ang = 64, run_results = True)
square_IC_converge(moving_mesh=False, uncollided=False, M=2, N_space_list=[30], N_ang = 16, run_results = True)
# square_IC_converge(moving_mesh=False, uncollided=False, M=2, N_space_list=[20], N_ang = 8, run_results = True)
# square_IC_converge(moving_mesh=False, uncollided=False, M=2, N_space_list=[20], N_ang = 16, run_results = True)
    


