from moving_mesh_transport.loading_and_saving.load_solution import load_sol
import matplotlib.pyplot as plt
import numpy as np
from marshak_point_source import exact_sol
import math
from scipy.interpolate import interp1d
from moving_mesh_transport.plots.plot_functions.show import show
import h5py 
from scipy.optimize import minimize



loader_transport = load_sol('transport', 'plane_IC', 'transport', s2 = True, file_name = 'run_data.hdf5')
loader = load_sol('su_olson_thick_s2', 'plane_IC', 'transfer', s2 = True, file_name = 'run_data.hdf5')

def RMSE(l1, l2):
    diff = (l1-l2)**2
    return np.sqrt(np.mean(diff))

def benchmark(xi):
    return 0.5 * (1/math.sqrt(math.pi)) * np.exp(-xi**2/4)
def benchmark2(x, tau, A):
    xi = x/math.sqrt(A*tau)
    return (1/math.sqrt(A*tau))*0.5 * (1/math.sqrt(math.pi)) * np.exp(-xi**2/4)
def dipole_benchmark(xi):
    return 0.5 * (1/math.sqrt(math.pi)) * np.exp(-xi**2/4) * xi

def gaussian_IC_benchmark(xi):
    return 0.5  * np.exp(-xi**2/4)

def gaussian_IC_bench_full(z, tau, omega, A):
    exponential_term = -z**2/(omega**2 + 4* A*tau)
    return  np.exp(exponential_term) / np.sqrt(A * tau) / np.sqrt(4/omega**2 + 1/A/tau)

# def noniso_part_bench(z, tau, omega, epsilon, mu):
#     u = np.exp(-z**2/omega**2)*np.greater(mu,0)
#     ubar = np.exp(-z**2/omega**2)
#     return np.exp(-tau/epsilon**2) * (u-ubar/2)
def noniso_part_bench(x, tau, omega, epsilon, mu):

    # z = find_z_func(x, mu, 8, epsilon)
    # z = x
    # x = z + epsilon * (z**2 - omega**2*np.log(z+0.00001))/((-1 + 3*(-1 + mu)*mu)*omega**2 + 0.000001)

    # u = np.exp(-x**2/omega**2) * (6/7)*(mu**2-mu+0.25)
    u = np.exp(-x**2/omega**2) * (15/16)*mu*(mu**3 + mu) 
    ubar = np.exp(-x**2/omega**2) 

    # dudx = np.exp(-z**2/omega**2)*(-12*(0.25 - mu + mu**2)*z)/(7.*omega**2)
    # dubardx = (-2*z)*np.exp(-z**2/omega**2)/(omega**2)

    # c1 = -mu *(dudx - 0.5*dubardx) / (u -0.5 * ubar) * 0 
    # print(c1 * epsilon)
    tp = tau *  (1) / epsilon**2

    return np.exp(-tp) * (u-ubar/2)
def f(z, mu, omega, x, epsilon):
    if z >0:
        return abs(z + epsilon * (z**2 - omega**2*math.log(z+0.00001))/((-1 + 3*(-1 + mu)*mu)*omega**2 + 0.000001) - x)
    else:
        return 10**7

def find_z_func(x, mu, omega, epsilon):
    res = x*0
    for ix, xx in enumerate(x):
        x0 = xx
        bnds = ((0, None))
        resmin = minimize(f, x0, args = (mu, omega, xx, epsilon))
        res[ix] =  resmin.x
    return res

def noniso_part_benchabsmu(z, tau, omega, epsilon, mu):
    c1 = -1/8 * 0
    tp = tau * (1 + c1 * epsilon) / epsilon**2
    u = np.exp(-z**2/omega**2)*np.abs(mu)
    ubar = np.exp(-z**2/omega**2)
    return np.exp(-tau/epsilon**2) * (u-ubar/2)
    
def noniso_bench_second_order2(x,tau, omega, epsilon, mu):
    tp = tau/epsilon**2
    t1 =  (-2*x/omega**2) * np.exp(-x**2/omega**2)
    return  epsilon * (tp * np.exp(-tp) * (0.25 * t1 - mu * (t1 * np.greater(mu, 0) - 0.5 * t1)) + t1 *(np.exp(-tp)-1) * 0.5) 

def noniso_bench_second_order(x,tau, omega, epsilon, mu):
    c1 = 0
    # z = find_z_func(x, mu, 8, epsilon)
    z = x
    tp = tau * (1) / epsilon**2
    # u = np.exp(-z**2/omega**2)*(6/7)*(mu**2-mu+0.25)
    # ubar = np.exp(-z**2/omega**2)
    # dudx = np.exp(-z**2/omega**2)*(-12*(0.25 - mu + mu**2)*z)/(7.*omega**2)
    # dubardx = (-2*z)*np.exp(-z**2/omega**2)/(omega**2)
    # intdudx = 0.5*(8*z*np.exp(-z**2/omega**2))/(7.*omega**2)
    # CC1 = -(16*(-z))*np.exp(-z**2/omega**2)/(7.*omega**2)
    u = np.exp(-x**2/omega**2) * (15/16)*mu*(mu**3 + mu) 
    ubar = np.exp(-x**2/omega**2) 
    dubardx = (-2*z)*np.exp(-z**2/omega**2)/(omega**2)
    dudx = np.exp(-z**2/omega**2)*(-2*15*(mu*(mu**3 + mu)*z))/(16.*omega**2)
    intdudx = 0
    # res = np.exp(-tp) * (c1 * tp*(2*u-ubar) + mu * tp * (dubardx - 2 * dudx  ) + (np.exp(tp) + tp) * intdudx) + CC1 * np.exp(-tp)
    res = -tp * np.exp(-tp) * mu * (dudx - 0.5 * dubardx) + (np.exp(-tp) - 1 + tp * np.exp(-tp)) * intdudx 

    return res * epsilon


    # return epsilon * ( -0.5*(np.exp(-t - x**2/omega**2)*x)/omega**2 + (np.exp(-t - x**2/omega**2)*x*(np.exp(t) - t - 2*t*mu + 4*t*mu*np.greater(mu,0)))/(2.*omega**2)) * 

# def noniso_bench_second_orderabsmu(x,tau, omega, epsilon, mu):
#     t = tau/epsilon**2

#     third_order = (np.exp(-t - x**2/omega**2)*(omega**2 - 2*x**2))/(6.*omega**4) - (np.exp(-t - x**2/omega**2)*(omega**2 - 2*x**2)*(np.exp(t) - t - t**2/2. - 3*mu**2*t**2 + 6*mu**2*t**2*np.abs(mu)))/(6.*omega**4)


#     return epsilon * (-((t*x*mu)/(np.exp(x**2/omega**2)*omega**2)) + (2*t*x*mu*np.abs(mu))/(np.exp(x**2/omega**2)*omega**2))/np.exp(t) + epsilon**2 * third_order





def ss_gaussian_noniso(N_spaces, epsilon_list = [1.0], tfinal_list_2  = [1.*10**-6, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, \
0.5, 1., 5., 10., 50., 100., 250., 500., 1000., 5000., 10000.], saving = False, moving_mesh = False):
    


    loader_transport = load_sol('transport', 'gaussian_IC', 'transport', s2 = False, file_name = 'run_data.hdf5')
    RMS_list_1 = []
    RMS_list_2 = []
    sigma = 1
    c=29.998
    A = c/3/sigma
    epsilon = epsilon_list[0]
    
    omega = 8.0
    for it, t in enumerate(tfinal_list_2):
        loader_transport.call_sol(t, 8, omega, N_spaces, 'rad', False, moving_mesh, epsilon)
        x = loader_transport.xs / sigma
        tau = t /c /sigma
        transport_phi = loader_transport.phi
        transport_psi = loader_transport.psi
        N_ang = len(transport_psi[:,0])
        dimensionalize = math.sqrt(1/omega**2) * math.sqrt(A*tau)
        xi_sol = x / math.sqrt(A*tau + omega**2/4)


        plt.ion()
        plt.figure(1)
        error1 = 0
        error2 = 0

        plt.figure(11)
        xp = 0.0
        xmid = np.argmin(np.abs(x-xp))
        mus = loader_transport.mus
        # plt.plot(loader_transport.mus, loader_transport.psi[:,xmid], '-')
        # benchmark = 0.5 * gaussian_IC_bench_full(xp, tau, omega, A) + noniso_part_bench(xp, t, omega, epsilon, mus) 
        # plt.plot(mus, loader_transport.psi[:,xmid]-benchmark, '-', mfc = 'none')
        # plt.plot(mus,  noniso_bench_second_order(xp, t, omega, epsilon, mus), '^', mfc = 'none')
        # plt.show()

        for iang in range(0, N_ang):
            mu = loader_transport.mus[iang]
            
            # plt.plot(x, transport_psi[iang,:], label = f"mu = {round(mu,2)}")
            benchmark = 0.5 * gaussian_IC_bench_full(0, tau, omega, A) + noniso_part_bench(0, t, omega, epsilon, mu)
            benchmark2 = 0.5 * gaussian_IC_bench_full(x, tau, omega, A) + noniso_part_bench(x, t, omega, epsilon, mu) 
            # plt.figure(1)
            # plt.plot(x, transport_psi[iang,:] - benchmark, 'o', mfc = 'none', label = f'mu = {round(mu,2)}' )
            # plt.figure(2)
            if iang == 32:
                plt.figure(234)
                plt.plot(x, benchmark2)
                plt.plot(x, transport_psi[iang,:], 'o', mfc = 'none')
            # plt.plot(x, transport_psi[iang,:] - (benchmark + noniso_bench_second_order(x, t, omega, epsilon, mu)), '^', mfc = 'none', label = f'mu = {round(mu,2)}'  ) 
            # plt.figure(iang + 10)
            # plt.ylim(-0.005, 0.005)
            # plt.plot(x, transport_psi[iang,:] - benchmark, '-', mfc = 'none', label = f'mu = {round(mu,2)}' )
            # plt.plot(x,  noniso_bench_second_order(x, t, omega, epsilon, mu), 'o', mfc = 'none' )
            # # np.testing.assert_allclose(noniso_bench_second_order(x, t, omega, epsilon, mu), noniso_bench_second_order2(x, t, omega, epsilon, mu))
            # # plt.xlim(-0.004, 0.004)
            # plt.legend(fontsize=6)

            # plt.show()
            
            error1 += RMSE(transport_psi[iang,xmid], benchmark) / N_ang
            error2 += RMSE(transport_psi[iang,:], benchmark2 + noniso_bench_second_order(x, t, omega, epsilon, mu)) / N_ang
            # print(error1, error2)
            # print(mu)
        RMS_list_1.append(error1)

        RMS_list_2.append(error2)

    

    plt.figure(3)

    plt.loglog(tfinal_list_2, RMS_list_1, '-o', mfc = 'none', label = 'first order')
    # plt.loglog(tfinal_list_2, RMS_list_2, '-^', mfc = 'none', label = 'second order')

    # plt.loglog(tfinal_list_2, np.max(RMS_list_2[:]) * np.array(tfinal_list_2)**2, 'k--', label = r'$O(t^2)$')
    plt.loglog(tfinal_list_2, np.max(RMS_list_1[:]) * np.array(tfinal_list_2), 'k-.', label = r'$O(t)$')
    # plt.loglog(tfinal_list_2,   np.array(tfinal_list_2)**(-1.5), 'k-', label = r'$O\left(t^{-3/2}\right)$')
    plt.ylim(1e-7,.1)
    
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel('RMSE')
    plt.show()
    if saving == True:
        data_array = np.array([tfinal_list_2, RMS_list_1])
        f = h5py.File('aniso_results/' + 'time_RMSE.hdf5', 'r+')
        group_name = 'time_RMSE' + f'_{N_spaces}_spaces' + f'_moving_mesh={moving_mesh}'
        if f.__contains__(group_name):
            del f[group_name]
        dset = f.create_dataset(group_name, data = data_array)
        f.close()
    
    
    RMS_list_1 = []
    RMS_list_2 = []
    for iep, epsilon in enumerate(epsilon_list):
        
        t = tfinal_list_2[3]
        tau = t /c /sigma
        error1 = 0
        error2 = 0


        for iang in range(0, N_ang):
            mu = loader_transport.mus[iang]
            
            # plt.plot(x, transport_psi[iang,:], label = f"mu = {round(mu,2)}")
            benchmark = 0.5 * gaussian_IC_bench_full(x, tau, omega, A) + noniso_part_bench(x, t, omega, epsilon, mu)
            # plt.figure(iang + 10)
            # plt.plot(x, transport_psi[iang,:], 'o', mfc = 'none', label = f'mu = {round(mu,2)}' )
            # plt.plot(x, benchmark, '-')
            # plt.legend()
            # plt.show()
            # plt.figure(7)
            # plt.plot(x, transport_psi[iang,:]- (benchmark + noniso_bench_second_order(x, t, omega, epsilon, mu)), '^', mfc = 'none', label = f'mu = {round(mu,2)}'  ) 
            # plt.figure(iang)
            # plt.plot(x, transport_psi[iang,:] - benchmark, '-', mfc = 'none', label = f'mu = {round(mu,2)}' )
            # plt.plot(x, noniso_bench_second_order(x, t, omega, epsilon, mu), 'o', mfc = 'none' )
            # np.testing.assert_allclose(noniso_bench_second_order(x, t, omega, epsilon, mu), noniso_bench_second_order2(x, t, omega, epsilon, mu))
            plt.figure(232)
            plt.plot(x, benchmark)
            plt.plot(x, transport_psi[iang,:], 'o', mfc = 'none')
            # plt.legend(fontsize=6)
            plt.show()
            
            error1 += RMSE(transport_psi[iang,:], benchmark) / N_ang
            error2 += RMSE(transport_psi[iang,:], benchmark + noniso_bench_second_order(x, t, omega, epsilon, mu)) / N_ang
            
        RMS_list_1.append(error1)

        RMS_list_2.append(error2)

    plt.figure(30)
    plt.loglog(epsilon_list, RMS_list_1, '-o', mfc = 'none', label = 'first order')
    plt.loglog(epsilon_list, RMS_list_2, '-^', mfc = 'none', label = 'second order')
    plt.loglog(epsilon_list, 5e-5 * np.array(epsilon_list)**(-2), 'k--', label = r'$O(\epsilon^{-2})$')
    # plt.loglog(epsilon_list, np.max(RMS_list_1[:]) * np.array(epsilon_list)**(-1), 'k-.', label = r'$O(\epsilon^{-1})$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    

def polar_contours(N_spaces = 16):
    xcontours =  np.array([0, 0.001, 0.025, 0.5, 1, 2, 4, 6, 8, 10, 12])
    tlist = np.array([0.001, 0.01, 0.1, 1])

    loader_transport = load_sol('transport', 'gaussian_IC', 'transport', s2 = False, file_name = 'run_data.hdf5')
    RMS_list_1 = []
    RMS_list_2 = []
    sigma = 1
    c=29.998
    A = c/3/sigma
    epsilon = 1

    
    omega = 8.0
    for it, t in enumerate(tlist):

        loader_transport.call_sol(t, 8, omega, N_spaces, 'rad', False, False, epsilon)
        x = loader_transport.xs / sigma
        tau = t /c /sigma
        transport_phi = loader_transport.phi
        transport_psi = loader_transport.psi
        N_ang = len(transport_psi[:,0])
        dimensionalize = math.sqrt(1/omega**2) * math.sqrt(A*tau)
        xi_sol = x / math.sqrt(A*tau + omega**2/4)

        sol_interp = interp1d(x, transport_psi)
        mulist = np.linspace(-1,1, 100)
        sinthetalist = np.sqrt(1-mulist**2)

        plt.figure(it)
        plt.ion()
        for ix, x in enumerate(xcontours):
            xlist = []
            ylist = []
            for iang in range(N_ang):
                mu = loader_transport.mus[iang]
                sintheta = np.sqrt(1 - mu**2)
                xc = mu * sol_interp(x)[iang]
                yc = sintheta * sol_interp(x)[iang]
                plt.scatter(xc, yc, c='b')
                plt.scatter(xc, -yc, c='b')
                xlist.append(xc)
                ylist.append(yc)

                psibench = 0.5 * gaussian_IC_bench_full(x, tau, omega, A) + noniso_part_bench(x, t, omega, epsilon, mulist)
                plt.plot(mulist *  psibench, sinthetalist * psibench, 'k--')
            
            data_array = np.array([xlist, ylist])

            # data_array = np.array([mulist *  psibench, sinthetalist * psibench])

            f = h5py.File('aniso_results/' + 'contour_results.hdf5', 'r+')
            if f.__contains__(f'contour_t={t}_x={x}'):
                del f[f'contour_t={t}_x={x}']
            dset = f.create_dataset(f'contour_t={t}_x={x}', data = data_array)
            f.close()

          
        plt.show()





      
       
    

                