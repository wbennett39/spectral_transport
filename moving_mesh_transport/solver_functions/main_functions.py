#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 07:02:38 2022

@author: bennett
"""
import numpy as np
import scipy.integrate as integrate
from numba import njit
import h5py
# import quadpy 
import matplotlib.pyplot as plt
import math
from pathlib import Path
from ..solver_classes.functions import find_nodes
from ..solver_classes.functions import Pn, normTn
from .Chebyshev_matrix_reader import file_reader

from ..solver_classes.build_problem import build
from ..solver_classes.matrices import G_L
from ..solver_classes.numerical_flux import LU_surf
from ..solver_classes.sources import source_class
from ..solver_classes.uncollided_solutions import uncollided_solution
from ..solver_classes.phi_class import scalar_flux
from ..solver_classes.mesh import mesh_class
from ..solver_classes.rhs_class_1 import rhs_class
from ..solver_classes.functions import quadrature
from ..solver_classes.functions import converging_time_function, converging_r
# from ..solver_classes.rhs_class import rhs_class

from ..solver_classes.make_phi import make_output
from ..solver_classes.radiative_transfer import T_function
from ..solver_classes.opacity import sigma_integrator

from timeit import default_timer as timer
from .wavespeed_estimator import wavespeed_estimator
from .wave_loc_estimator import find_wave
import chaospy
import scipy
# from diffeqpy import ode
# from .jl_integrator import integrator as jl_integrator_func
# from diffeqpy import de



"""
This file contains functions used by solver
"""



def parameter_function(major, N_spaces, Ms, count):
    if major == 'cells':
        M = Ms[count]
        N_space = N_spaces[count]
    elif major == 'Ms':
        N_space = N_spaces[1]
        M = Ms[count]
    return N_space, M


def s2_source_type_selector(sigma, x0, thermal_couple, source_type, weights):
    """ 
    changes the name of the source type in order to select the correct 
    benchmark. For S2 benchmarks 
    """
    # thick source s8 
    if source_type[5] == 1:
        if sigma == 300:
            source_array_rad = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
            source_array_mat = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif sigma == 0.5:
            source_array_rad = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
            source_array_mat = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    elif source_type[2] == 1:
        if x0 == 400:
            source_array_rad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
            source_array_mat = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif x0 == 0.5:
            source_array_rad = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
            source_array_mat = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    return source_array_rad, source_array_mat
    

def time_step_function(t_array):
    N = len(t_array)
    res = np.zeros(N-1)
    for i in range(N-1):
        res[i] = t_array[i+1]-t_array[i]
    return res

def plot_p1_su_olson_mathematica():
    data_folder = Path("moving_mesh_transport/benchmarks")
    benchmark_mat_file_path = data_folder / "S2SuOlMat_t_1..txt"
    benchmark_rad_file_path = data_folder / "S2SuOlRadt_1..txt"
    
    su_olson_rad = np.loadtxt(benchmark_rad_file_path)
    su_olson_mat = np.loadtxt(benchmark_mat_file_path)
    plt.plot(su_olson_rad[:,0],su_olson_rad[:,1], "xk" )
    plt.plot(su_olson_mat[:,0],su_olson_mat[:,1], "xk" )
    
    return [su_olson_rad, su_olson_mat]

def solve(tfinal, N_space, N_ang, M, x0, t0, sigma_t, sigma_s, t_nodes, source_type, 
          uncollided, moving, move_type, thermal_couple, temp_function, rt, at, e_initial, choose_xs, specified_xs, 
          weights, sigma, particle_v, edge_v, cv0, estimate_wavespeed, find_wave_loc, thick, mxstp, wave_loc_array, 
          find_edges_tol, source_strength, move_factor, integrator, l, save_wave_loc, pad, leader_pad, xs_quad_order, 
          eval_times, eval_array, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
          finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, geometry, lumping):

    # if weights == "gauss_lobatto":
    #     mus = quadpy.c1.gauss_lobatto(N_ang).points
    #     ws = quadpy.c1.gauss_lobatto(N_ang).weights
    # elif weights == "gauss_legendre":
    #     mus = quadpy.c1.gauss_legendre(N_ang).points
    #     ws = quadpy.c1.gauss_legendre(N_ang).weights
    # if N_ang == 2:


    speed_of_light = 29.98 # cm/ns
    mus, ws = quadrature(N_ang, weights, testing = True)

    mus_new = np.zeros(N_ang+2)
    ws_new = np.zeros(N_ang+2)
    mus_new[0] = -1
    mus_new[-1] = 1
    mus_new[1:-1] = mus
    ws_new[1:-1] = ws
    mus = mus_new
    ws = ws_new
    
    N_ang += 2
    #     print("mus =", mus)

    # xs_quad = quadpy.c1.gauss_legendre(2*M+1).points
    # ws_quad = quadpy.c1.gauss_legendre(2*M+1).weights

    if geometry['slab'] == True:
        xs_quad, ws_quad = quadrature(2*M+1, 'gauss_legendre')
    elif geometry['sphere'] == True:
        xs_quad, ws_quad = quadrature(max(3*M+1, 3*Msigma+1), 'chebyshev')

    # t_quad = quadpy.c1.gauss_legendre(t_nodes).points
    t_quad, t_ws = quadrature(t_nodes, 'gauss_legendre')
    print(t_quad, 't quad')

    # t_ws = quadpy.c1.gauss_legendre(t_nodes).weights
    half = int((N_space + 1)/2)
    rest = N_space +1 - half
    Mk = int(2*half-1)
    Mkr = int(2*rest-1)
    # quad_thick_source= chaospy.quadrature.clenshaw_curtis(half - 1)[0][0]
    quad_thick_source = quadrature(Mk, 'gauss_legendre')[0][int((Mk-1)/2):] 
    # quad_thick_edge, blank = quadrature(int(N_space/4+1), 'gauss_lobatto')
    # quad_thick_edge = chaospy.quadrature.clenshaw_curtis(rest)[0][0][:-1]
    quad_thick_edge = np.flip(quadrature(Mkr, 'gauss_legendre')[0][int((Mkr-1)/2):])
    # quad_thick_source = quadpy.c1.gauss_lobatto(int(N_space/2+1)).points
    # quad_thick_edge = quadpy.c1.gauss_lobatto(int(N_space/4+1)).points
    # quad_thick_source = ([quad_thick_source_inside, quad_thick_source_outside])

    reader = file_reader()
    give = reader()
    # ob = matrix_builder()



    initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
                       ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
                       thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
                       wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
                        quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
                        finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, geometry, lumping)



    if sigma_func['converging'] == 1:
        f = h5py.File('heat_wavepos.h5', 'r+')
        boundary_temp = f['temperature'][:] / 10 # convert from HeV to keV
        boundary_temp[0] = boundary_temp[1]
        
        boundary_time = (f['times'][:] - f['times'][0]) * speed_of_light * sigma_t
        # print(f['times'][:], 'times')
        f.close()
        # print(boundary_temp**4 * 0.0137225 * 29.98/4/math.pi)
        # print(boundary_time, 'boundary time array')
        initialize.grab_converging_boundary_data(boundary_temp, boundary_time)
        


    

    if thermal_couple['none'] == 1:
        deg_freedom = N_ang*N_space*(M+1)
    else:
        deg_freedom = (N_ang+1)*N_space*(M+1)

    mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                      wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                      domain_width, fake_sedov_v0, boundary_on, t0, geometry, sigma_func) 


    matrices = G_L(initialize, give[0], give[1], give[2], give[3], give[4], give[5], give[6], give[7], give[8], give[9], give[10], give[11], give[12])
    num_flux = LU_surf(initialize)
    source = source_class(initialize)
    uncollided_sol = uncollided_solution(initialize)
    flux = scalar_flux(initialize)
    rhs = rhs_class(initialize)
    transfer = T_function(initialize)
    sigma_class = sigma_integrator(initialize)
    flux.load_AAA(sigma_class.AAA)
    if thermal_couple['none'] != 1:
        mesh.move(0)
        initialize.make_T4_IC(transfer, mesh.edges)
    initialize.make_IC()
    IC = initialize.IC
    xs = find_nodes(mesh.edges, M, geometry)
    phi_IC = make_output(0.0, N_ang, ws, xs, IC, M, mesh.edges, uncollided, geometry)
    phi = phi_IC.make_phi(uncollided_sol)
    
    # print(phi, 'phi IC')
    
    # @njit
    def RHS(t, V):
        return rhs.call(t, V, mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class)
    
    start = timer()
    reshaped_IC = IC.reshape(deg_freedom)

    if estimate_wavespeed == False:
        tpnts = [tfinal]
    elif estimate_wavespeed == True:
        tpnts = np.linspace(0, tfinal, 10000)
    if eval_times == True:
        tpnts = eval_array
        print(tpnts, 'time points')
        tpnts_dense = np.linspace(tpnts[0], tpnts[1], 100)
        for it, tt in enumerate(tpnts_dense):
            mesh.move(tt)
            
            # dimensional_t = tt/29.98
            # menis_t = -29.6255 + dimensional_t
            menis_t = converging_time_function(tt, sigma_func)
            # rfront = 0.01 * (-menis_t) ** 0.679502 
            rfront = converging_r(menis_t, sigma_func)

            plot_edges_converging(tt, mesh.edges, rfront, 23)
        plt.draw()
        # plt.show()


   
    mesh_dry_run(mesh, tfinal)
    

    # sol_JL = jl_integrator_func(RHS, IC, (0, tfinal), tpnts)

    # sol = integrate.solve_ivp(RHS, [0.0,tfinal], reshaped_IC, method=integrator, t_eval = tpnts , rtol = rt, atol = at, max_step = mxstp, min_step = 1e-7)
    if integrator == 'BDF_VODE':
        ode15s = scipy.integrate.ode(RHS)
        ode15s.set_integrator('lsoda', method='bdf', atol = at, rtol = rt)
        ode15s.set_initial_value(reshaped_IC, 0.0)
        sol = sol_class_ode_solver(ode15s.y, ode15s.t, np.array(tpnts))

        for it in range(len(tpnts)):
            
            tf = tpnts[it]
            # print(tf, 'next integration target time')
            # with stdout_redirected():
            ode15s.integrate(tf)
            sol.y[:,it] = ode15s.y
            ode15s.set_initial_value(ode15s.y, tf)

  
    else:
        print(rt, 'rt')
        print(at, 'at')
        print('starting solve')
        sol = integrate.solve_ivp(RHS, [0.0,tfinal], reshaped_IC, method=integrator, t_eval = tpnts , rtol = rt, atol = at, max_step = mxstp)

    # sol = ode15s.y


    print(sol.status, 'solution status')
    # print(sol)
    if sol.status != 0:
        print(sol.message)
    # print(sol)
    # print(sol.y.shape,'sol y shape')
    # print(eval_times, 'eval times')
    end = timer()
    print('solver finished')
    
    if save_wave_loc == True:
        print(save_wave_loc, 'save wave')
        wave_tpnts = rhs.times_list
        wave_xpnts = rhs.wave_loc_list
    else:
        wave_tpnts = np.array([0.0])
        wave_xpnts = np.array([0.0])

    # if estimate_wavespeed == True:
    #     wavespeed_array = wavespeed_estimator(sol, N_ang, N_space, ws, M, uncollided, mesh, 
    #                       uncollided_sol, thermal_couple, tfinal, x0)
    # elif estimate_wavespeed == False:
    wavespeed_array = np.zeros((1,1,1))

    if find_wave_loc == True:
        wave_loc_finder = find_wave(N_ang, N_space, ws, M, uncollided, mesh, uncollided_sol, 
        thermal_couple, tfinal, x0, sol.t, find_edges_tol, source_type, sigma_t)
        left_edges, right_edges, T_front_location = wave_loc_finder.find_wave(sol)
    elif find_wave_loc == False:
        left_edges =  np.zeros(1)
        right_edges = np.zeros(1)
        T_front_location = np.zeros(1)

    
    if thermal_couple['none'] == 1:
        extra_deg_freedom = 0
       
        sol_last = sol.y[:,-1].reshape((N_ang,N_space,M+1))
        if eval_times ==True and sol.status != -1:
            sol_array = sol.y.reshape((eval_array.size, N_ang,N_space,M+1)) 
    elif thermal_couple['none'] != 1:
        extra_deg_freedom = 1
        sol_last = sol.y[:,-1].reshape((N_ang+1,N_space,M+1))
        # print(sol_last[-1,:,:])
        if eval_times == True:
            sol_array = sol.y.reshape((eval_array.size, N_ang+1,N_space,M+1)) 


    
    if sol.t.size > 1:
        timesteps = time_step_function(sol.t)
        print(np.max(timesteps), "max time step")
    
    mesh.move(tfinal)
    edges = mesh.edges
    
    if choose_xs == False:
        xs = find_nodes(edges, M, geometry)
        
    elif choose_xs == True:
        xs = specified_xs
    # print(xs, 'xs')
    if eval_times == False or sol.status == -1:
        output = make_output(tfinal, N_ang, ws, xs, sol_last, M, edges, uncollided, geometry)
        phi = output.make_phi(uncollided_sol)
        psi = output.psi_out # this is the collided psi
        exit_dist, exit_phi = output.get_exit_dist(uncollided_sol)
        xs_ret = xs
        if thermal_couple['none'] == False:
            # print('reconstructing energy density solution')
            e = output.make_e()
            # print(e,'energy density')
        else:
            e = phi*0
    else:
        phi = np.zeros((eval_array.size, xs.size))
        e = np.zeros((eval_array.size, xs.size))
        psi = np.zeros((eval_array.size, N_ang, xs.size))
        exit_dist = np.zeros((eval_array.size, N_ang, 2))
        exit_phi = np.zeros((eval_array.size, 2))
        xs_ret = np.zeros((eval_array.size, xs.size))
        # initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
        #                ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
        #                thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
        #                wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
        #                 quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
        #                 finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, geometry)
        
        fake_mesh  = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                      wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                      domain_width, fake_sedov_v0, boundary_on, t0, geometry, sigma_func) 
        for it, tt in enumerate(eval_array):
            fake_mesh.move(tt)
            edges = fake_mesh.edges
            # print(edges, 'edges', tt, 't')
            if choose_xs == False:
                xs = find_nodes(edges, M, geometry)
            elif choose_xs == True:
                xs = specified_xs
            output = make_output(tt, N_ang, ws, xs, sol.y[:,it].reshape((N_ang+extra_deg_freedom,N_space,M+1)), M, edges, uncollided, geometry)
            phi[it,:] = output.make_phi(uncollided_sol)
            psi[it, :, :] = output.psi_out # this is the collided psi
            exit_dist[it], exit_phi[it] = output.get_exit_dist(uncollided_sol)
            xs_ret[it] = xs
            if thermal_couple['none'] == False:
                e[it,:] = output.make_e()
                
            else:
                e = phi*0
    computation_time = end-start
    
    return xs_ret, phi, psi, exit_dist, exit_phi,  e, computation_time, sol_last, mus, ws, edges, wavespeed_array, tpnts, left_edges, right_edges, wave_tpnts, wave_xpnts, T_front_location, mus



def problem_identifier():
    name_array = []

def plot_edges_converging(t, edges, rf, fign):
    plt.figure(fign)
    for ed in range(edges.size):
        plt.scatter(edges[ed], t, s = 128, c = 'k', marker = "|")
    plt.scatter(rf, t, c='r', marker='x')


def plot_edges(edges,fign):
    plt.figure(fign)
    for ed in range(edges.size):
        plt.scatter(edges[ed], 0.0, s = 128, c = 'k', marker = "|")



def x0_function(x0, source_type, count):
        if source_type[3] or source_type[4] == 1:
            x0_new = x0[count]
        else:
            x0_new = x0[0]
        return x0_new










# def matmul(a,b):

#     res = [[0 for x in range(3)] for y in range(3)] 
    
#     # explicit for loops
#     for i in range(len(matrix1)):
#         for j in range(len(matrix2[0])):
#             for k in range(len(matrix2)):
    
#                 # resulted matrix
#     return res


 
def mesh_dry_run(mesh, tfinal):
    tlist = np.linspace(0.0, tfinal, 500)
    for it, tt in enumerate(tlist):
        mesh.move(tt)
    print('mesh dry run complete')
    mesh.move(0.0)


class sol_class_ode_solver():
    def __init__(self, y, t, tpnts):
        self.y = np.zeros(( y.size, tpnts.size,))
        self.y[:,0] = y
        self.t = tpnts
        self.status = 1
        self.message = 'sup'