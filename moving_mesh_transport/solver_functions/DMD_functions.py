from ..solver_classes.make_phi import make_output
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from .theta_DMD import theta_DMD
from .VDMD import VDMD2 as VDMD_func
import random


def DMD_func(rhs, N_ang, N_groups, N_space, M, xs, uncollided_sol, edges, uncollided, geometry, ws, integrator, sigma_t ):
        if (rhs.t_old_list_Y != np.sort(rhs.t_old_list_Y)).any():
            print(rhs.t_old_list_Y)
            raise ValueError('t list nonconsecutive')
            
        # eigen_vals = np.zeros(rhs.t_old_list_Y.size)
        # for it, tt in enumerate(rhs.t_old_list_Y):
        # print(rhs.Y_minus_list)
        # print(rhs.Y_minus_list[:rhs.Y_iterator-1], 'Y-')
        # print(rhs.Y_plus_list[:rhs.Y_iterator-1], 'Y+')
        # eigen_vals = rhs.t_old_list_Y * 0
        Y_minus = rhs.Y_minus_list[:rhs.Y_iterator-1].copy()
        Y_plus = rhs.Y_plus_list[:rhs.Y_iterator-1].copy()
        Y_plus_psi = np.zeros((N_groups * N_ang * xs.size,rhs.Y_iterator-1))
        Y_minus_psi = np.zeros(( N_groups * N_ang * xs.size, rhs.Y_iterator-1))
        Y_m_final = Y_minus_psi.copy()*0
        Y_p_final = Y_plus_psi.copy()*0
        Y_minus_flipped = np.zeros((N_groups * N_ang * N_space * (M+1), rhs.Y_iterator-1))
        # Mdelta = np.zeros((rhs.Y_iterator-1, rhs.Y_iterator-2))
        # Mtheta = np.zeros((rhs.Y_iterator-1, rhs.Y_iterator-2))
        # for it in range(rhs.Y_iterator-1):
        #     delta_t = rhs.t_old_list_Y[it+1] - rhs.t_old_list_Y[it]
        #     if it < rhs.Y_iterator - 2:
        #         Mdelta[it, it] = -1/delta_t 
        #     Mdelta[it + 1, it] = 1/ delta_t
        # print(Mdelta, 'Mdelta')
        # output = make_outpurhs.Y_it(tfinal, N_ang, ws, xs, Y_minus[0,:].reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
        for it in range(2, rhs.Y_iterator-2):
            tt = rhs.t_old_list_Y[it]

            output = make_output(tt, N_ang, ws, xs, Y_minus[it,:].reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
            phi = output.make_phi(uncollided_sol)
            Y_minus_psi[:,it] = output.psi_out.reshape((N_groups * N_ang * xs.size))
            # print(Y_minus_psi[:, it], 'psi')
            Y_minus_flipped[:, it] = Y_minus[it,:]
            # plt.ion()
            # plt.plot(xs, phi)
            # plt.show()
            # if integrator == 'BDF':
            Y_m_final[:, it] = Y_minus_psi[:, it]
            dt = (rhs.t_old_list_Y[it+1] - rhs.t_old_list_Y[it])/sigma_t # because the list is t old, use it+1 and it to calculate dt 
            Y_p_final[:, it] =  3/2/dt * (Y_minus_psi[:, it] - 4 * Y_minus_psi[:, it-1]/3 + Y_minus_psi[:, it-2]/3)

      



            # else:
            #     raise ValueError('This integrator method does not yet support VDMD')


            # output = make_output(tt, N_ang, ws, xs, Y_plus[it,:].reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
            # output.make_phi(uncollided_sol)
            # Y_plus_psi[:,it] = output.psi_out.reshape((N_groups * N_ang * xs.size)) 
            # Y_plus_psi[:, it] *= -np.sign(Y_plus_psi[:, it])

        # skip = int(0.2 * rhs.Y_iterator)skip 
        skip = 4
        # #swap column
        # Y_minus_psi[:,[rhs.Y_iterator-1,0]] = Y_minus_psi[:,[0, rhs.Y_iterator-1]]
        # Y_plus_psi[:,[rhs.Y_iterator-1,0]]= Y_plus_psi[:,[0, rhs.Y_iterator-1]]
        # print(Y_m_final[:, 2:], 'Y-')
        # print(Y_p_final[:, 2:], 'Y+')
        # print(skip, 'skip')
        eigen_vals_DMD = np.sort(np.real(VDMD_func(Y_m_final[:, :-1] + 1e-18, Y_p_final[:, :-1] + 1e-18, skip)))
        print(eigen_vals_DMD[-1],eigen_vals_DMD[-2],eigen_vals_DMD[-3],eigen_vals_DMD[-4], 'First four eigen vals VDMD raw' )
        eigen_vals_DMD = np.sort(np.real(VDMD_func(Y_minus_psi[:, :-1] + 1e-18, Y_plus_psi[:, :-1] + 1e-18, skip)))
        print(eigen_vals_DMD[-1],eigen_vals_DMD[-2],eigen_vals_DMD[-3],eigen_vals_DMD[-4], 'First four eigen vals VDMD psi' )
        # print(-np.max(np.real(eigen_vals_DMD)), 'Largest negative eigenval VDMD')
        # print(np.max(np.real(eigen_vals_DMD)), 'Largest eigenval VDMD')
        positive_vals = True
        close_to_bench = False
        it2 = 1
        # theta = 0.8417871348541741
        theta = 1.0
        theta_all_negative = []
        theta_close_to_bench = []
        theta_old = theta
        eigen_vals_old = theta_DMD(Y_minus_psi[:, skip:]+1e-18, rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/sigma_t, theta = theta)
        err_old = abs(np.max(np.real(eigen_vals_old)) - -0.763507)
        theta_old_list = []
        it_list = []
        stagnancy_count = 0
        err_list = []
        err_2list = []
        print('iterating theta')
        if integrator == 'BDF' or integrator == 'Euler':
            for it2 in tqdm.tqdm(range(250)):
                # print(it2)
                
            # while it2 <= 500:
                # print(rhs.t_old_list_Y[0:rhs.Y_iterator-1].size, 't list size')
                # print(Y_m_final[0, :].size, 'YM size')
                # print(rhs.Y_iterator, 'Y iterator')
                if stagnancy_count < 100:
                    theta_new = theta_old + 0.01 * theta_old * (np.random.rand()*2-1)
                else:
                    theta_new = np.random.rand()
                    # stagnancy_count = 0
                
                if theta_new > 1.0:
                    theta_new = 1.0
                elif theta_new < 0.0:
                    theta_new = 0.0
                # print(theta_new, 'theta')
                eigen_vals2 = theta_DMD(Y_minus_psi[:, skip:]+1e-18, rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/sigma_t, theta = theta_new)
                eigen_vals = theta_DMD(Y_minus_flipped[:, skip:]+1e-18, rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/sigma_t, theta = theta_new)

                # print(abs(np.sort(eigen_vals2)[-1] - np.sort(eigen_vals)[-1]), 'difference in using psi vs coefficients')
                # print(np.max(np.real(eigen_vals)), 'Largest negative eigenval')
                # print(np.max(np.real(eigen_vals)), 'Largest eigenval')
                # print(theta, 'theta')
                # eigen_vals = theta_DMD(Y_minus_flipped[:, skip:], rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/2.998e10/10.0, theta = theta)
                if (eigen_vals < 0).all():
                    # print(theta, 'theta no positive vals')
                    positive_vals = False
                    theta_all_negative.append(theta_new)
                else:
                    positive_vals = True

                # else:
                    
                if abs(np.max(np.real(eigen_vals)) - -.763507) <= 0.1:
                    close_to_bench = True
                    theta_close_to_bench.append(theta_new)
                    # print(abs(np.max(-np.real(eigen_vals)) - 5.10866))
                    # print(np.sort(np.real(eigen_vals))[:4], 'top 4 modes')
                    # print(np.max(np.real(eigen_vals)), 'largest eigenvalue')
        
                it2 += 1
                

                
                # theta = 2 * np.random.rand()
                # print(theta, 'theta')
                if it2 >= 500:
                    print('iterated out')
                # theta_new = np.random.rand() * 2
                err = abs(np.max(np.real(eigen_vals)) - -0.763507)
                err2 = abs(np.sort(np.real(eigen_vals))[-2] - -1.57201)
                # print(err, 'err')
                
                if err < err_old:
                    eigenvals_old = np.sort(np.real(eigen_vals))[0:4]
                    theta = theta_new
                    theta_old = theta_new
                    err_old = err
                    theta_old_list.append(theta_old)
                    it_list.append(it2)
                    stagnancy_count = 0
                    err_list.append(err_old)
                    err_2list.append(err2)
                else:
                    stagnancy_count += 1
                    theta_old_list.append(theta_old)
                    it_list.append(it2)
                    err_list.append(err_old)
                    err_2list.append(err2)

                    # print('updating theta')
                
                if integrator == 'BDF':
                    theta = random.uniform(0.0, 1.0)

            
            theta_old_list = np.array(theta_old_list)
            it_list = np.array(it_list)
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('iterations', fontsize = 16)
            ax1.set_ylabel(r'$\theta$', fontsize = 16)
            ax1.plot(it_list, theta_old_list, '-x', label = r'$\theta$' +f'{N_space} cells, {M+1} basis functions, {N_ang} angles')
            ax1.tick_params(axis='y')
            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('error')  # we already handled the x-label with ax1
            ax2.semilogy(it_list, err_list)
            ax2.semilogy(it_list, err_2list, '--')
            ax2.tick_params(axis='y')

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.show()
            # plt.figure(2)
            # plt.plot(it_list, err_list, '-', label = f'error {N_space} cells, {M+1} basis functions, {N_ang} angles')
            # plt.xlabel('iterations', fontsize = 16)
            # plt.ylabel(r'$\theta$', fontsize = 16)
            # plt.legend()
            ax1.legend()
            ax2.legend()
            plt.savefig(f'theta_iterations_{N_space}_{integrator}.pdf')
            plt.show()
            plt.close()
            # print(eigen_vals, 'theta method')

        print(skip, 'skip')
        print(rhs.t_old_list_Y[skip], 'first time snapshot')
        print(theta_old, 'theta')
        eigen_vals2 = np.sort(np.real(theta_DMD(Y_minus_psi[:, skip:]+1e-18, rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/sigma_t, theta = theta_old)))
        eigen_vals = np.sort(np.real(theta_DMD(Y_minus_flipped[:, skip:]+1e-18, rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/sigma_t, theta = theta_old)))
        return_vals = np.array([eigen_vals[-1]])
        it = 0
        for ix in range(1, eigen_vals.size):
            if abs(eigen_vals[ix] - return_vals[it]) > 1e-12:
                return_vals = np.append(return_vals, eigen_vals[ix])

                it += 1
        sorted_eigs = return_vals
        print(sorted_eigs[-1], sorted_eigs[-2], sorted_eigs[-3], sorted_eigs[-4], 'first four eigen values')
        # print(eigen_vals2[-1], eigen_vals2[-2], eigen_vals2[-3], eigen_vals2[-4], 'first four eigen values')

        # print(-np.max(-np.real(eigen_vals) /2.9E10), 'largest negative eigen value')
        # if len(theta_all_negative) != 0:
        #     print(np.min(np.array(theta_all_negative)),np.max(np.array(theta_all_negative)),'range of thetas for all values negative' )
        # if len(theta_close_to_bench) != 0:
        #     print(np.min(np.array(theta_close_to_bench)),np.max(np.array(theta_close_to_bench)),'range of thetas for eigen close to bench' )
        return return_vals


def DMD_func2(sol, N_ang, N_groups, N_space, M, xs, uncollided_sol, edges, uncollided, geometry, ws, integrator, sigma_t ):

            
        # eigen_vals = np.zeros(rhs.t_old_list_Y.size)
        # for it, tt in enumerate(rhs.t_old_list_Y):
        # print(rhs.Y_minus_list)
        # print(rhs.Y_minus_list[:rhs.Y_iterator-1], 'Y-')
        # print(rhs.Y_plus_list[:rhs.Y_iterator-1], 'Y+')
        # eigen_vals = rhs.t_old_list_Y * 0
        Y_minus = np.zeros((sol.t.size, sol.y[:,0].size))
        for it in range(sol.t.size):
            Y_minus[it, :] = sol.y[:, it]
        # Y_plus = rhs.Y_plus_list[:rhs.Y_iterator-1].copy()
        Y_plus_psi = np.zeros((N_groups * N_ang * xs.size, sol.t.size))
        Y_minus_psi = np.zeros(( N_groups * N_ang * xs.size, sol.t.size))
        Y_m_final = Y_minus_psi.copy()*0
        Y_p_final = Y_plus_psi.copy()*0
        Y_minus_flipped = np.zeros((N_groups * N_ang * N_space * (M+1), sol.t.size))
        # Mdelta = np.zeros((rhs.Y_iterator-1, rhs.Y_iterator-2))
        # Mtheta = np.zeros((rhs.Y_iterator-1, rhs.Y_iterator-2))
        # for it in range(rhs.Y_iterator-1):
        #     delta_t = rhs.t_old_list_Y[it+1] - rhs.t_old_list_Y[it]
        #     if it < rhs.Y_iterator - 2:
        #         Mdelta[it, it] = -1/delta_t 
        #     Mdelta[it + 1, it] = 1/ delta_t
        # print(Mdelta, 'Mdelta')
        # output = make_outpurhs.Y_it(tfinal, N_ang, ws, xs, Y_minus[0,:].reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
        for it in range(1, sol.t.size):
            tt = sol.t[it]
            output = make_output(tt, N_ang, ws, xs, Y_minus[it,:].reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
            phi = output.make_phi(uncollided_sol)
            Y_minus_psi[:,it] = output.psi_out.reshape((N_groups * N_ang * xs.size))
            # print(Y_minus_psi[:, it], 'psi')
            Y_minus_flipped[:, it] = Y_minus[it,:]
            plt.ion()
            plt.plot(xs, phi)
            plt.show()
            # if integrator == 'BDF':
            Y_m_final[:, it] = Y_minus_psi[:, it]
            dt = (sol.t[it] - sol.t[it-1])/sigma_t # because the list is t old, use it+1 and it to calculate dt 
            # Y_p_final[:, it] =  3/2/dt * (Y_minus_psi[:, it] - 4 * Y_minus_psi[:, it-1]/3 + Y_minus_psi[:, it-2]/3)
            if integrator == 'Euler':
                Y_p_final[:, it] = (Y_minus_psi[:, it] - Y_minus_psi[:, it-1])/dt
            elif integrator == 'BDF':
                Y_p_final[:, it] =  3/2/dt * (Y_minus_psi[:, it] - 4 * Y_minus_psi[:, it-1]/3 + Y_minus_psi[:, it-2]/3)

      



            # else:
            #     raise ValueError('This integrator method does not yet support VDMD')


            # output = make_output(tt, N_ang, ws, xs, Y_plus[it,:].reshape((N_ang * N_groups,N_space,M+1)), M, edges, uncollided, geometry, N_groups)
            # output.make_phi(uncollided_sol)
            # Y_plus_psi[:,it] = output.psi_out.reshape((N_groups * N_ang * xs.size)) 
            # Y_plus_psi[:, it] *= -np.sign(Y_plus_psi[:, it])

        # skip = int(0.2 * rhs.Y_iterator)skip 
        skip = 4
        # #swap column
        # Y_minus_psi[:,[rhs.Y_iterator-1,0]] = Y_minus_psi[:,[0, rhs.Y_iterator-1]]
        # Y_plus_psi[:,[rhs.Y_iterator-1,0]]= Y_plus_psi[:,[0, rhs.Y_iterator-1]]
        # print(Y_m_final[:, 2:], 'Y-')
        # print(Y_p_final[:, 2:], 'Y+')
        # print(skip, 'skip')
        eigen_vals_DMD = np.sort(np.real(VDMD_func(Y_m_final[:, :] + 1e-18, Y_p_final[:, :], skip)) )
        # eigen_vals_DMD2 = np.sort(np.real(VDMD_func(Y_minus_psi[:, :-1] + 1e-18, Y_plus_psi[:, :-1], skip)) )
        print(eigen_vals_DMD, 'VDMD function 2 raw')
        # print(eigen_vals_DMD, 'VDMD function 2 psi')
        # print(-np.max(np.real(eigen_vals_DMD)), 'Largest negative eigenval VDMD')
        # print(np.max(np.real(eigen_vals_DMD)), 'Largest eigenval VDMD')
        positive_vals = True
        close_to_bench = False
        it2 = 1
        # theta = 0.8417871348541741
        theta = 1.0
        theta_all_negative = []
        theta_close_to_bench = []
        theta_old = theta
        eigen_vals_old = theta_DMD(Y_minus_psi[:, skip:]+1e-18, sol.t[skip:sol.t.size]/sigma_t, theta = theta)
        err_old = abs(np.max(np.real(eigen_vals_old)) - -0.763507)
        theta_old_list = []
        it_list = []
        stagnancy_count = 0
        err_list = []
        err_2list = []
        tf_it = sol.t.size
        print('iterating theta')
        if integrator == 'BDF' or integrator == 'Euler':
            for it2 in tqdm.tqdm(range(250)):
                # print(it2)
                
            # while it2 <= 500:
                # print(rhs.t_old_list_Y[0:rhs.Y_iterator-1].size, 't list size')
                # print(Y_m_final[0, :].size, 'YM size')
                # print(rhs.Y_iterator, 'Y iterator')
                if stagnancy_count < 100:
                    theta_new = theta_old + 0.01 * theta_old * (np.random.rand()*2-1)
                else:
                    theta_new = np.random.rand()
                    # stagnancy_count = 0
                
                if theta_new > 1.0:
                    theta_new = 1.0
                elif theta_new < 0.0:
                    theta_new = 0.0
                # print(theta_new, 'theta')
                eigen_vals2 = theta_DMD(Y_minus_psi[:, skip:]+1e-18, sol.t[skip:tf_it]/sigma_t, theta = theta_new)
                eigen_vals = theta_DMD(Y_minus_flipped[:, skip:]+1e-18, sol.t[skip:tf_it]/sigma_t, theta = theta_new)

                # print(abs(np.sort(eigen_vals2)[-1] - np.sort(eigen_vals)[-1]), 'difference in using psi vs coefficients')
                # print(np.max(np.real(eigen_vals)), 'Largest negative eigenval')
                # print(np.max(np.real(eigen_vals)), 'Largest eigenval')
                # print(theta, 'theta')
                # eigen_vals = theta_DMD(Y_minus_flipped[:, skip:], rhs.t_old_list_Y[skip:rhs.Y_iterator -1]/2.998e10/10.0, theta = theta)
                if (eigen_vals < 0).all():
                    # print(theta, 'theta no positive vals')
                    positive_vals = False
                    theta_all_negative.append(theta_new)
                else:
                    positive_vals = True

                # else:
                    
                if abs(np.max(np.real(eigen_vals)) - -.763507) <= 0.1:
                    close_to_bench = True
                    theta_close_to_bench.append(theta_new)
                    # print(abs(np.max(-np.real(eigen_vals)) - 5.10866))
                    # print(np.sort(np.real(eigen_vals))[:4], 'top 4 modes')
                    # print(np.max(np.real(eigen_vals)), 'largest eigenvalue')
        
                it2 += 1
                

                
                # theta = 2 * np.random.rand()
                # print(theta, 'theta')
                if it2 >= 500:
                    print('iterated out')
                # theta_new = np.random.rand() * 2
                err = abs(np.max(np.real(eigen_vals)) - -0.763507)
                err2 = abs(np.sort(np.real(eigen_vals))[-2] - -1.57201)
                # print(err, 'err')
                
                if err < err_old:
                    eigenvals_old = np.sort(np.real(eigen_vals))[0:4]
                    theta = theta_new
                    theta_old = theta_new
                    err_old = err
                    theta_old_list.append(theta_old)
                    it_list.append(it2)
                    stagnancy_count = 0
                    err_list.append(err_old)
                    err_2list.append(err2)
                else:
                    stagnancy_count += 1
                    theta_old_list.append(theta_old)
                    it_list.append(it2)
                    err_list.append(err_old)
                    err_2list.append(err2)

                    # print('updating theta')
                
                # if integrator == 'BDF':
                #     theta = random.uniform(0.0, 1.0)
            
          
        print(skip, 'skip')

        print(theta_old, 'theta')
        eigen_vals2 = np.sort(np.real(theta_DMD(Y_minus_psi[:, skip:]+1e-18, sol.t[skip:sol.t.size]/sigma_t, theta = theta_old)))
        eigen_vals = np.sort(np.real(theta_DMD(Y_minus_flipped[:, skip:]+1e-18, sol.t[skip:sol.t.size]/sigma_t, theta = theta_old)))
        return_vals = np.array([eigen_vals[-1]])
        it = 0
        print(eigen_vals, 'theta-DMD function 2 raw')
        print(eigen_vals2, 'theta-DMD function 2 psi')
        for ix in range(1, eigen_vals.size):
            if abs(eigen_vals[ix] - return_vals[it]) > 1e-12:
                return_vals = np.append(return_vals, eigen_vals[ix])

                it += 1
        sorted_eigs = return_vals
        print(sorted_eigs[-1], sorted_eigs[-2], sorted_eigs[-3], sorted_eigs[-4], 'first four eigen values')
        # print(eigen_vals2[-1], eigen_vals2[-2], eigen_vals2[-3], eigen_vals2[-4], 'first four eigen values')

        # print(-np.max(-np.real(eigen_vals) /2.9E10), 'largest negative eigen value')
        # if len(theta_all_negative) != 0:
        #     print(np.min(np.array(theta_all_negative)),np.max(np.array(theta_all_negative)),'range of thetas for all values negative' )
        # if len(theta_close_to_bench) != 0:
        #     print(np.min(np.array(theta_close_to_bench)),np.max(np.array(theta_close_to_bench)),'range of thetas for eigen close to bench' )
        return return_vals





def DMD_func3(Y_minus, t,  integrator, sigma_t, skip = 4, theta = 1):

        Y_plus = np.zeros((Y_minus[:,0].size, t.size))
        # populate Y+ assuming Backward Euler 
        for it in range(1, t.size):
            tt = t[it]
            dt = (t[it] - t[it-1])/sigma_t 
            if integrator == 'Euler' or integrator == 'BDF_VODE':
                Y_plus[:, it] = (Y_minus[:, it] - Y_minus[:, it-1])/dt
            # elif integrator == 'BDF_VODE':
            #     if it > 1:
            #         Y_plus[:, it] =  1.5 * (Y_minus[:, it] - 4 * Y_minus[:, it-1]/3 + Y_minus[:, it-2]/3) / dt 
        if integrator == 'BDF':
            # eigen_vals_DMD = np.sort(np.real(VDMD_func(Y_minus[:, :] + 1e-16, Y_plus[:, :]+ 1e-16, skip)) )
            eigen_vals_DMD = np.sort(np.real(theta_DMD(Y_minus[:, skip:], t[skip:]/sigma_t, theta = theta)))
        elif integrator == 'Euler' or integrator == 'BDF_VODE':
            eigen_vals_DMD = np.sort(np.real(VDMD_func(Y_minus[:, :] , Y_plus[:, :], skip)) )
            # print(Y_plus, 'yp')
        else:
            raise ValueError('Integration method not implemented')

        # eigen_vals_DMD = np.sort(np.real(theta_DMD(Y_minus[:, skip:]+1e-18, t[skip:], theta = 1)))
        
       
        return eigen_vals_DMD