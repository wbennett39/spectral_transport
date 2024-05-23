import matplotlib.pyplot as plt
from moving_mesh_transport.plots.plot_functions.show import show
from moving_mesh_transport.solver_classes.functions import get_sedov_funcs
from moving_mesh_transport.solver_classes.sedov_funcs import sedov_class
from moving_mesh_transport.solver_classes.cubic_spline import cubic_spline_ob as cubic_spline

from moving_mesh_transport.solver_classes.sedov_uncollided import sedov_uncollided_solutions
from moving_mesh_transport.loading_and_saving.load_solution import load_sol
from moving_mesh_transport.solver_classes.functions import quadrature
import numpy as np
import math
from scipy.special import erf
import scipy.integrate as integrate
from tqdm import tqdm
import sys
import quadpy
sys.path.append('/Users/bennett/Documents/Github/exactpack/')


def opts0(*args, **kwargs): 
       return {'limit':50, 'epsabs':1.5e-12, 'epsrel':1.5e-12}

def toy_blast_psi(mu, tfinal, x, v0, t0source, x0):
        c1 = 1.0
        x0 = -5 
        if mu!= 0:
            t0 =  (x0-x)/mu + tfinal # time the particle is emitted
        else:
             t0 = np.inf
        x02 = 0.0
        sqrt_pi = math.sqrt(math.pi)
        kappa = 2
        rho0 = 0.1
        # beta = c1 * (v0-1) - v0 * (x0/mu + t0)
        
        # b2 =  v0 * (-x0/mu - t0 + c1) / (1+v0/mu)
        b2 = ((v0*x0) - t0*v0*mu)/(v0 + mu)
        b1 = max(x, b2)
        # b2 = 0

        b4 = x0
        # b3 = min(x,0)

        b3 =  min(x, b2)

        # print(b1, b2, b3, b4, 'bs', x, 'x', t0, 't0')

        # t1 = lambda s: -0.5*(mu*sqrt_pi*kappa*erf((beta - (s*(mu + v0))/mu)/kappa))/(mu + v0)

        t1 = lambda s: (sqrt_pi*kappa*mu*erf((v0*(s - x0) + (c1 + s + t0*v0)*mu)/(kappa*mu + 1e-12)))/(1e-12 + 2.*(v0 + mu))
        t2 = lambda s: rho0 * s

        mfp = t1(b1) - t1(b2) + t2(b3) - t2(b4)
        if mu == 0:
             return 0.0
        else:
            if mfp/mu >40:
                mfp = 40 * mu
                # print(np.exp(-mfp/mu))
                return 0.0
                # mfp = rho0 * x - rho0 * (-x0)
                # print(mfp, x, 'mfp')
            if np.isnan(np.exp(-mfp / mu) * np.heaviside(mu - abs(x - x0)/ (tfinal), 0) * np.heaviside(abs(x0-x) - (tfinal-t0source)*mu,0)):
                    print(np.exp(-mfp / mu))
                    print(mu)
                    assert(0)

            if mu > 0:
                return np.exp(-mfp / mu) * np.heaviside(mu - abs(x - x0)/ (tfinal), 0) * np.heaviside(abs(x0-x) - (tfinal-t0source)*mu,0)
            else:
                    return 0.0
        
def toy_blast_phi(tfinal, x, v0, t0, x0):
    aa = 0.0
    bb = 1.0
    if tfinal > t0:
        bb = min(1.0, abs(x-x0)/ (tfinal - t0))
    aa = abs(x-x0) / tfinal
    if aa <= 1.0:     
        res = integrate.nquad(toy_blast_psi, [[aa, bb]], args = (tfinal, x, v0, t0, x0), opts = [opts0])
        return res[0]
    else:
         return 0.0

def toy_blast_phi_vector(tfinal, xs, v0, t0, x0):
     res = xs*0
     for ix, x in enumerate(xs):
        aa = 0.0
        bb = 1.0
        if tfinal > t0:
            bb = min(1.0, abs(x-x0)/ (tfinal - t0))
        aa = abs(x-x0) / tfinal
        if aa <= 1.0:     
            res[ix] = integrate.nquad(toy_blast_psi, [[aa, bb]], args = (tfinal, x, v0, t0, x0), opts = [opts0])[0]
     return res
     
def RMSE(xs, phi, v0, tfinal, t0, x0):
    benchmark = xs*0
    for ix, xx in enumerate(xs):
        benchmark[ix] = toy_blast_phi(tfinal, xx, v0, t0, x0)
    
    res = math.sqrt(np.mean((phi-benchmark)**2))
    return res, benchmark

def RMSETS(benchmark, phi):
    
    res = math.sqrt(np.mean((phi-benchmark)**2))
    return res


def error_toy_blast_wave_absorbing(N_space=32, M=6, v0 = 0.0035, x0 = -5.0, t0 = 15.0):
    plt.ion()
    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = 0.0)
    tlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])
    RMSE_list = tlist*0
    counter1 = 0
    for it, t in enumerate(tqdm(tlist)):
        # plt.figure(1)
        loader.call_sol(t, M, int(-x0), N_space, 'rad', True, True)
        x = loader.xs
        phi = loader.phi
        psi = loader.psi
        res = RMSE(x, phi, v0, t, t0, x0)
        RMSE_list[it] = res[0]
  
        counter1 += 1
        if counter1 == 1:
            plt.figure(23)
            plt.plot(x, phi, 'k--')
            plt.plot(x, res[1], 's', mfc = 'none', label = f't={t}')
            counter1 = 0

            plt.figure(24)
            # plt.plot(x,  'k--')
            plt.plot(x, np.abs(phi-res[1]), '-', mfc = 'none',  label = f't={t}')
            counter1 = 0
        plt.legend()
        plt.show()



    plt.figure(5)
    plt.loglog(tlist, RMSE_list, '-o')
    plt.xlabel('evaluation time', fontsize = 16)
    plt.ylabel("RMSE", fontsize = 16)
    show('blast_wave_absorbing_error')
    plt.show()


def exit_distributions(v0 = 0.000035, x0 = -5.0, t0 = 15.0, tf = 50):
    tlist = np.linspace(0.001, tf, 100)
    left = tlist * 0
    right = tlist * 0
    for it, tt in enumerate(tlist):
          left[it] = toy_blast_phi(tt, x0, v0, t0, x0)
          right[it] = toy_blast_phi(tt, -x0, v0, t0, x0)

    plt.figure(1)
    plt.title('left exit dist')
    plt.plot(tlist, left, '-o')
    plt.show()

    plt.figure(2)
    plt.title('right exit dist')
    plt.plot(tlist, right, '-o')

    plt.show()

    xs = np.linspace(-x0, x0)
    res = xs * 0

    for ix, xx in enumerate(xs):
         res[ix] = toy_blast_psi(0.5, 1.0, xx, v0, t0, x0)
    plt.figure(3)
    plt.plot(xs, res)
    plt.show()



def plot_analytic_solutions(x0 = -5, v0 =0.0035, t0 = 15, tf = 50):
    tlist = np.array([1, 2, 3, 5, 7, 15.0])
    plt.ion()
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    for it, tt in enumerate(tlist):
         xs = np.linspace(-x0, min(x0, tt-x0), 500)
         sol = toy_blast_phi_vector(tt, xs+1e-8, v0, t0, x0)
         plt.plot(xs, sol, 'k-')
    # plt.text(-4.96,0.2,r'$t=0$')
    plt.text(-4.21,0.16,r'$t=1$')
    plt.text(-3.25,0.09,r'$2$')
    plt.text(-2.34,0.09,r'$3$')
    plt.text(-.70,0.092,r'$5$')
    plt.text(0.93,0.047,r'$7$')
    plt.text(0.4,0.2,r'$15$')
    # plt.text(-3.56,0.226,r'$t=17$')
    # plt.text(-2.97,0.157,r'$t=18$')
    # plt.text(-1.46,0.137,r'$t=19$')
    # plt.text(-1.18,0.067,r'$t=20$')
    # plt.text(1.8,0.014,r'$t=22$')
    plt.show()
    show('uncollided_solutions_before')
    tlist = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 30.0])
    plt.ion()
    ax = plt.figure(11)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    for it, tt in enumerate(tlist):
        xs = np.linspace(-x0 , min(x0, tt-x0), 500)
        sol = toy_blast_phi_vector(tt, xs+1e-8 , v0, t0, x0)
        plt.plot(xs, sol, 'k-')
    plt.text(-4.96,1.004,r'$t=15$')
    plt.text(-4.25,0.532,r'$16$')
    plt.text(-3.4,0.424,r'$17$')
    plt.text(-2.85,0.31,r'$18$')
    plt.text(-1.5,0.277,r'$19$')
    plt.text(-1.18,0.161,r'$20$')
    plt.text(-.6,0.04,r'$22$')


    ax.show()
    show('uncollided_solutions_after')



    # tlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])
    tlist = np.linspace(0.001, tf, 500)
    left = tlist * 0
    right = tlist * 0
    for it, tt in enumerate(tlist):
          left[it] = toy_blast_phi(tt, x0, v0, t0, x0)
          right[it] = toy_blast_phi(tt, -x0, v0, t0, x0)
    plt.figure(2)
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    plt.plot(tlist, left, 'k-', )
    plt.show()
    show('left_exit_uncollided')
    plt.figure(3)
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    plt.plot(tlist, right, 'k-')
    # plt.xlabel('t', fontsize = 16)
    # plt.ylabel(r'$\phi$', fontsize = 16)
    plt.show()
    show('right_exit_uncollided')
    

def toy_blast_scattering_profiles(N_space = 16, cc = 0.125, uncollided = True):
    M = 6
    x0 = -5
    fulltlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])

    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    tlist1 = np.array([1, 2, 3, 5, 7, 15.0])
    tlist2 = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 30.0])
    tlist = np.concatenate((tlist1, tlist2[1:]))
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    plt.ion()
    for it, tt in enumerate(tlist1):
        loader.call_sol(tt, M, int(-x0), N_space, 'rad', uncollided, True)
        xs = loader.xs
        phi = loader.phi
        output_phi = loader.exit_phi
        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #     right_exit[it] = output_phi[1]
 
        plt.figure(3)
        string1 = 'before'
    #   plt.text(-4.96,0.2,r'$t=0$')
        plt.text(-4.21,0.16,r'$t=1$')
        plt.text(-3.25,0.09,r'$2$')
        plt.text(-2.34,0.09,r'$3$')
        plt.text(-.70,0.092,r'$5$')
        plt.text(0.93,0.047,r'$7$')
        plt.text(0.4,0.2,r'$15$')
        plt.plot(xs, phi, 'k-')     
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.xlabel('x', fontsize = 16)
         
        show(f'c={cc}_solutions_' + string1)
    
    for it, tt in enumerate(tlist2):
            loader.call_sol(tt, M, int(-x0), N_space, 'rad', uncollided, True)
            xs = loader.xs
            phi = loader.phi
            output_phi = loader.exit_phi
            plt.figure(4)
            string1 = 'after'
            plt.text(-4.96,1.023,r'$t=15$')
            plt.text(-4.25,0.532,r'$16$')
            plt.text(-3.4,0.424,r'$17$')
            plt.text(-2.85,0.31,r'$18$')
            plt.text(-1.86,0.277,r'$19$')
            plt.text(-2.12,0.161,r'$20$')
            plt.text(-.6,.156,r'$22$')
            plt.text(-.6,.04,r'$30$')
            plt.plot(xs, phi, 'k-')
                
            plt.ylabel(r'$\phi$', fontsize = 16)
            plt.xlabel('x', fontsize = 16)
            
            show(f'c={cc}_solutions_' + string1)



    plt.figure(1)
    plt.plot(fulltlist, output_phi[:,0], 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'left_exit_dist_c={cc}')

    plt.figure(2)       
    plt.plot(fulltlist, output_phi[:,1], 'k-')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'right_exit_dist_c={cc}')

    
    plt.show()

# exit_distributions()
# error_toy_blast_wave_absorbing(N_space = 16)
# plot_analytic_solutions()


def error_TS_blast_wave_absorbing(N_space=16, M=6, x0 = 0.5, t0 = 15.0, sigma_t = 0.000005):
    plt.ion()
    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = 0.0)
    tlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])
    RMSE_list = tlist*0
    counter1 = 0
    for it, t in enumerate(tqdm(tlist)):
        # plt.figure(1)
        loader.call_sol(t, M, x0, N_space, 'rad', False, True)
        x = loader.xs
        phi = loader.phi
        psi = loader.psi
        # res = RMSE(x, phi, v0, t, t0, x0)
        # RMSE_list[it] = res[0]
  
        counter1 += 1
        if counter1 == 1:
            plt.figure(23)
            plt.plot(x, phi, 'k--')
            # plt.plot(x, res[1], 's', mfc = 'none', label = f't={t}')
            counter1 = 0

            # plt.figure(24)
            # plt.plot(x,  'k--')
            # plt.plot(x, np.abs(phi-res[1]), '-', mfc = 'none',  label = f't={t}')
            counter1 = 0
        plt.legend()
        plt.show()



    plt.figure(5)
    plt.loglog(tlist, RMSE_list, '-o')
    plt.xlabel('evaluation time', fontsize = 16)
    plt.ylabel("RMSE", fontsize = 16)
    show('blast_wave_absorbing_TSe_error')
    plt.show()


# def TS_blast_scattering_profiles(N_space = 16, cc = 0.0, uncollided = False):
#     M = 6
#     x0 = 1.0
#     fulltlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])

#     loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
#     tlist1 = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 15.0])
#     tlist2 = np.array([15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
#     tlist = np.concatenate((tlist1, tlist2[1:]))
#     tlist = np.array([1.0])
#     left_exit = np.zeros(tlist.size)
#     right_exit = np.zeros(tlist.size)
#     counter = 0

#     g_interp, v_interp, sedov = TS_bench_prime()

#     plt.ion()
#     for it, tt in enumerate(tqdm(tlist)):
#         #  loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, False)
#         #  xs = loader.xs
#         #  phi = loader.phi
#         #  output_phi = loader.exit_phi
#          xs = np.linspace(-x0, x0, 50)
#          phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov)

#         #  eval_times = loader.eval_array
#         #  if eval_times[counter] = tlist[counter]:         
#         #     left_exit[it] = output_phi[counter, 0]
#         #     right_exit[it] = output_phi[1]
#          if tt <= tlist1[-1]:
#               plt.figure(3)
#               string1 = 'before'
#             #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
#             #   plt.text(-4.96,0.2,r'$t=0$')
#             #   plt.text(-4.21,0.16,r'$t=1$')
#             #   plt.text(-3.25,0.09,r'$t=2$')
#             #   plt.text(-2.34,0.09,r'$t=3$')
#             #   plt.text(-.70,0.092,r'$t=5$')
#             #   plt.text(0.93,0.047,r'$t=7$')
#             #   plt.text(0.4,0.2,r'$t=15$')
#          else:
#             plt.figure(4)
#             string1 = 'after'
#             # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
#             # plt.text(-4.96,0.704,r'$t=15$')
#             # plt.text(-4.25,0.532,r'$t=16$')
#             # plt.text(-3.4,0.424,r'$t=17$')
#             # plt.text(-2.85,0.31,r'$t=18$')
#             # plt.text(-1.5,0.277,r'$t=19$')
#             # plt.text(-1.18,0.161,r'$t=20$')
#             # plt.text(-.6,0.04,r'$t=22$')
#         #  plt.plot(xs, phi, 'k-')
#          plt.plot(xs, phi_bench, 'o', mfc = 'none')
#          plt.ylabel(r'$\phi$', fontsize = 16)
#          plt.xlabel('x', fontsize = 16)
         
#          show(f'c={cc}_solutions_TS' + string1)



#     plt.figure(1)
#     # plt.plot(fulltlist, output_phi[:,0], 'k-')
#     plt.xlabel('t', fontsize = 16)
#     plt.ylabel(r'$\phi$', fontsize = 16)
#     show(f'left_exit_dist_c={cc}')

#     plt.figure(2)       
#     # plt.plot(fulltlist, output_phi[:,1], 'k-')  
#     plt.xlabel('t', fontsize = 16)
#     plt.ylabel(r'$\phi$', fontsize = 16)
#     show(f'right_exit_dist_c={cc}')

    
    plt.show()
def TS_blast_absorbing_profiles(N_space = 16, cc = 0.0, uncollided = False, transform = True):
    M = 6
    x0 = 0.5
    fulltlist = np.array([0.05,0.5, 0.6, 0.7,  1.0, 1.5, 2.0, 2.5, 2.5, 2.6,  3.0, 3.5,  4.0, 4.5,  5.0])

    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    # tlist1 = np.array([0.05, 0.4, 0.7, 1.0, 2.5])
    tlist1 = np.array([0.05,0.5, 0.6, 0.7,  1.0, 1.5, 2.0, 2.5])

    # tlist1 = np.array([1.0])
    # tlist2 = np.array([2.5, 2.6, 3.0, 3.5,  4.0,  5.0])
    tlist2 = np.array([2.5, 2.5, 2.6,  3.0, 3.5,  4.0, 4.5,  5.0])
    tlist = np.concatenate((tlist1, tlist2))
    # tlist = np.array([1.0])
    # tlist = tlist1
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    RMSE_list = tlist*0

    g_interp, v_interp, sedov = TS_bench_prime()
    f, (a1, a2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    plt.ion()
    for it, tt in enumerate(tqdm(tlist1)):
         loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)
         xs = loader.xs
         phi = loader.phi
         output_phi = loader.exit_phi
        #  xs = np.linspace(-x0, x0, 200)
         phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov, transform=transform)
         RMSE_list[it] = RMSETS(phi_bench, phi)

        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #
         string1 = 'before'
        
            #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
            #   plt.text(-4.96,0.2,r'$t=0$')
            #   plt.text(-4.21,0.16,r'$t=1$')
            #   plt.text(-3.25,0.09,r'$t=2$')
            #   plt.text(-2.34,0.09,r'$t=3$')
            #   plt.text(-.70,0.092,r'$t=5$')
            #   plt.text(0.93,0.047,r'$t=7$')
            #   plt.text(0.4,0.2,r'$t=15$')
  

            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            # plt.text(-4.96,0.704,r'$t=15$')
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
         xrho = np.linspace(-x0, x0, 500)
         density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
         density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
         plt.text(-.489,0.404,r'$t=0.05$')
         plt.text(-.299,0.154,r'$0.4$')
         plt.text(-.107,0.161,r'$0.7$')
         plt.text(0.335,0.065,r'$1.0$')
         plt.text(.156,0.191,r'$2.5$')
         plt.ion()
         plt.show()

         a2.plot(xs, phi, 'b-o', mfc = 'none')
         a2.plot(xs, phi_bench, 'k-', mfc = 'none')
         a2.set_ylabel(r'$\phi$', fontsize = 16)
         a1.set_ylabel(r'$\rho$', fontsize = 16)
         a1.plot(xrho, density_initial, 'k--')
         a1.plot(xrho, density_final, 'k-')

         plt.xlabel('x', fontsize = 16)
         
         show(f'c={cc}_solutions_TS' + string1)

    f, (a3, a4) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})

    for it2, tt in enumerate(tqdm(tlist2)):
            loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)
            xs = loader.xs
            phi = loader.phi
            output_phi = loader.exit_phi
            #  xs = np.linspace(-x0, x0, 200)
            phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov, transform = transform)
            RMSE_list[it + it2] = RMSETS(phi_bench, phi)

            #  eval_times = loader.eval_array
            #  if eval_times[counter] = tlist[counter]:         
            #     left_exit[it] = output_phi[counter, 0]
            #     right_exit[it] = output_phi[1]
            
                #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
                #   plt.text(-4.96,0.2,r'$t=0$')
                #   plt.text(-4.21,0.16,r'$t=1$')
                #   plt.text(-3.25,0.09,r'$t=2$')
                #   plt.text(-2.34,0.09,r'$t=3$')
                #   plt.text(-.70,0.092,r'$t=5$')
                #   plt.text(0.93,0.047,r'$t=7$')
                #   plt.text(0.4,0.2,r'$t=15$')
            plt.text(-.439,0.9,r'$t=2.5$')
            plt.text(-.402,0.38,r'$2.6$')
            plt.text(-.16,0.16,r'$3.0$')
            plt.text(0.122,0.108,r'$3.5$')
            plt.text(0.335,0.048,r'$4.0$')
           
            string1 = 'after'
            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
            xrho = np.linspace(-x0, x0, 500)
            density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
            density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
            
            plt.ion()
            plt.show()

            a4.plot(xs, phi, 'b-o', mfc = 'none')
            a4.plot(xs, phi_bench, 'k-', mfc = 'none')
            a4.set_ylabel(r'$\phi$', fontsize = 16)
            a3.set_ylabel(r'$\rho$', fontsize = 16)
            a3.plot(xrho, density_initial, 'k--')
            a3.plot(xrho, density_final, 'k-')

            plt.xlabel('x', fontsize = 16)

            show(f'c={cc}_solutions_TS' + string1)





    plt.figure(1)
    # plt.plot(fulltlist, output_phi[:,0], 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'left_exit_dist_c={cc}')
    tlistdense = np.linspace(fulltlist[0], fulltlist[-1], 60)
    t_bench_right = tlistdense * 0
    for it, tt in enumerate(tlistdense):
         t_bench_right[it] = TS_bench(tt, np.array([x0]), g_interp, v_interp, sedov)
    plt.figure(2)       
    plt.plot(tlistdense, t_bench_right, 'k-')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'right_exit_dist_c={cc}')
    
    plt.figure(5)
    plt.loglog(tlist, RMSE_list, '-o', label = f'{N_space} spatial cells')
    plt.xlabel('evaluation time', fontsize = 16)
    plt.ylabel("RMSE", fontsize = 16)
    plt.legend()
    show('blast_wave_absorbing_TSe_error')
    plt.show()
    


def TS_bench(t, xs, interp_g_fun, interp_v_fun, sedov, sigma_t = 0.000005, t0 = 2.5, x0 = 0.5, transform = True):

    xs_quad, ws_quad = quadrature(30, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(512)
    mu_quad = res1.points
    mu_ws = res1.weights
    
    
    sedov_uncol = sedov_uncollided_solutions(xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_t, t0, transform)

    phi = sedov_uncol.uncollided_scalar_flux(xs, t, sedov, interp_g_fun, interp_v_fun)
    
    return phi


def TS_bench_prime(sigma_t = 0.000005):
    f_fun, g_fun, l_fun = get_sedov_funcs()
    sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t)
    foundzero = False
    iterator = 0
    while foundzero == False:
        if g_fun[iterator] == 0.0:
            iterator += 1
        else:
            foundzero = True
        if iterator >= g_fun.size:
                assert(0)
    f_fun = f_fun[iterator:]
    g_fun = g_fun[iterator:]
    l_fun = l_fun[iterator:]
    
    interp_g_fun, interp_v_fun = interp_sedov_selfsim(g_fun, l_fun, f_fun, sedov)

    plt.figure(24)
    plt.ion()
    plt.plot(l_fun, interp_g_fun.eval_spline(l_fun))
    plt.show()
    plt.figure(25)

    plt.figure(24)
    plt.ion()
    plt.plot(l_fun, interp_v_fun.eval_spline(l_fun))
    plt.show()
    plt.figure(25)

    rs = np.linspace(-0.5, 0.5,100)
    plt.ion()
    plt.plot(rs, sedov.interpolate_self_similar(1.0, rs, interp_g_fun))
    plt.show()
    



    return interp_g_fun, interp_v_fun, sedov
    

def interp_sedov_selfsim(g_fun, l_fun, f_fun, sedov_class):

             l_fun = np.flip(l_fun)
             l_fun[0] = 0.0
             g_fun = np.flip(g_fun)
             f_fun[-1] = 0.0
             l_fun[-1] = 1.0
            #  g_fun[-1] = sedov_class.gpogm
            #  l_fun[-1] = 1.0

             interpolated_g = cubic_spline(l_fun, g_fun)
             interpolated_v = cubic_spline(l_fun, np.flip(f_fun))
             print('g interpolated')
             return interpolated_g, interpolated_v


def TS_blast_scattering_profiles(N_space = 8, cc = 0.19999999999999998, uncollided = False):
    M = 6
    x0 = 0.5
    fulltlist = np.array([0.05,0.1, 0.25, 0.4,0.45, 0.5, 0.55, 0.6, 0.7, 1.0, 1.5, 2.0, 2.5, 2.6,  3.0, 3.5,  4.0, 4.5,  5.0])

    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    tlist1 = np.array([0.05, 0.4, 0.7, 1.0, 2.5])
    # tlist1 = np.array([1.0])
    tlist2 = np.array([2.5, 2.6, 3.0, 3.5,  4.0,  5.0])
    tlist = np.concatenate((tlist1, tlist2[1:]))
    # tlist = np.array([1.0])
    # tlist = tlist1
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    RMSE_list = tlist*0

    g_interp, v_interp, sedov = TS_bench_prime()
    f, (a1, a2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    plt.ion()
    for it, tt in enumerate(tqdm(tlist1)):
         loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)
         xs = loader.xs
         phi = loader.phi
         output_phi = loader.exit_phi
        #  xs = np.linspace(-x0, x0, 200)
        #  phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov)
        #  RMSE_list[it] = RMSETS(phi_bench, phi)

        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #
         string1 = 'before'
        
            #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
            #   plt.text(-4.96,0.2,r'$t=0$')
            #   plt.text(-4.21,0.16,r'$t=1$')
            #   plt.text(-3.25,0.09,r'$t=2$')
            #   plt.text(-2.34,0.09,r'$t=3$')
            #   plt.text(-.70,0.092,r'$t=5$')
            #   plt.text(0.93,0.047,r'$t=7$')
            #   plt.text(0.4,0.2,r'$t=15$')
  

            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            # plt.text(-4.96,0.704,r'$t=15$')
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
         xrho = np.linspace(-x0, x0, 500)
         density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
         density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
         plt.text(-.489,0.404,r'$t=0.05$')
         plt.text(-.299,0.154,r'$0.4$')
         plt.text(-.107,0.161,r'$0.7$')
         plt.text(0.335,0.065,r'$1.0$')
         plt.text(.156,0.291,r'$2.5$')
         plt.ion()
         plt.show()

         a2.plot(xs, phi, 'k-', mfc = 'none')
        #  a2.plot(xs, phi_bench, 'k-', mfc = 'none')
         a2.set_ylabel(r'$\phi$', fontsize = 16)
         a1.set_ylabel(r'$\rho$', fontsize = 16)
         a1.plot(xrho, density_initial, 'k--')
         a1.plot(xrho, density_final, 'k-')

         plt.xlabel('x', fontsize = 16)
         
         show(f'c={cc}_solutions_TS' + string1)

    f, (a3, a4) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})

    for it2, tt in enumerate(tqdm(tlist2)):
            loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)
            xs = loader.xs
            phi = loader.phi
            print(phi)
            output_phi = loader.exit_phi
            #  xs = np.linspace(-x0, x0, 200)
            # phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov)
            # RMSE_list[it + it2] = RMSETS(phi_bench, phi)

            #  eval_times = loader.eval_array
            #  if eval_times[counter] = tlist[counter]:         
            #     left_exit[it] = output_phi[counter, 0]
            #     right_exit[it] = output_phi[1]
            
                #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
                #   plt.text(-4.96,0.2,r'$t=0$')
                #   plt.text(-4.21,0.16,r'$t=1$')
                #   plt.text(-3.25,0.09,r'$t=2$')
                #   plt.text(-2.34,0.09,r'$t=3$')
                #   plt.text(-.70,0.092,r'$t=5$')
                #   plt.text(0.93,0.047,r'$t=7$')
                #   plt.text(0.4,0.2,r'$t=15$')
            plt.text(-.439,0.9,r'$t=2.5$')
            plt.text(-.402,0.38,r'$2.6$')
            plt.text(-.16,0.16,r'$3.0$')
            plt.text(0.122,0.108,r'$3.5$')
            plt.text(0.335,0.048,r'$4.0$')
           
            string1 = 'after'
            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
            xrho = np.linspace(-x0, x0, 500)
            density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
            density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
            
            plt.ion()
            # plt.show()

            a4.plot(xs, phi, 'k-', mfc = 'none')
            # a4.plot(xs, phi_bench, 'k-', mfc = 'none')
            a4.set_ylabel(r'$\phi$', fontsize = 16)
            a3.set_ylabel(r'$\rho$', fontsize = 16)
            a3.plot(xrho, density_initial, 'k--')
            a3.plot(xrho, density_final, 'k-')

            plt.xlabel('x', fontsize = 16)

            show(f'c={cc}_solutions_TS' + string1)





    plt.figure(1)
    plt.plot(fulltlist, output_phi[:,0], 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'left_exit_dist_c={cc}_TS')

    plt.figure(2)       
    plt.plot(fulltlist, output_phi[:,1], 'k-')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'right_exit_dist_c={cc}_TS')