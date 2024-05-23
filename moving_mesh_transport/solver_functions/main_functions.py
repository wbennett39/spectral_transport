#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 07:02:38 2022

@author: bennett
"""
import numpy as np
import scipy.integrate as integrate
from numba import njit
# import quadpy 
import matplotlib.pyplot as plt
import math
from pathlib import Path
from ..solver_classes.functions import find_nodes, get_sedov_funcs
from ..solver_classes.functions import Pn, normTn
from .Chebyshev_matrix_reader import file_reader
from ..plots.plot_functions.show import show
import quadpy


from ..solver_classes.build_problem import build
from ..solver_classes.matrices import G_L
from ..solver_classes.numerical_flux import LU_surf
from ..solver_classes.sources import source_class
from ..solver_classes.uncollided_solutions import uncollided_solution
from ..solver_classes.phi_class import scalar_flux
from ..solver_classes.mesh import mesh_class
from ..solver_classes.sedov_funcs import sedov_class
from ..solver_classes.sedov_uncollided import sedov_uncollided_solutions
# from ..solver_classes.rhs_class_1 import rhs_class
from ..solver_classes.functions import quadrature
from ..solver_classes.rhs_class import rhs_class

from ..solver_classes.make_phi import make_output
from ..solver_classes.radiative_transfer import T_function
from ..solver_classes.opacity import sigma_integrator

from timeit import default_timer as timer
from .wavespeed_estimator import wavespeed_estimator
from .wave_loc_estimator import find_wave

from scipy.special import roots_legendre
import numpy.polynomial as poly
import scipy.special as sps
from functools import partial
# from exactpack..solvers.sedov.sedov_similarity_variables import sedov
from exactpack.solvers.sedov.doebling import Sedov as SedovDoebling
# from scipy.interpolate import interp1d as interp


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
          finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, remesh, geometry):


    # if weights == "gauss_lobatto":
    #     mus = quadpy.c1.gauss_lobatto(N_ang).points
    #     ws = quadpy.c1.gauss_lobatto(N_ang).weights
    # elif weights == "gauss_legendre":
    #     mus = quadpy.c1.gauss_legendre(N_ang).points
    #     ws = quadpy.c1.gauss_legendre(N_ang).weights
    # if N_ang == 2:
    mus, ws = quadrature(N_ang, weights, testing = True)
    #     print("mus =", mus)

    # xs_quad = quadpy.c1.gauss_legendre(2*M+1).points
    # ws_quad = quadpy.c1.gauss_legendre(2*M+1).weights
    if geometry['slab'] == True:
        xs_quad, ws_quad = quadrature(2*M+1, 'gauss_legendre')
    elif geometry['sphere'] == True:
        xs_quad, ws_quad = quadrature(6*M+1, 'chebyshev')

    # t_quad = quadpy.c1.gauss_legendre(t_nodes).points
    t_quad, t_ws = quadrature(t_nodes, 'gauss_legendre')

    # t_ws = quadpy.c1.gauss_legendre(t_nodes).weights
    quad_thick_source, blank = quadrature(int(N_space/2+1), 'gauss_lobatto')
    quad_thick_edge, blank = quadrature(int(N_space/4+1), 'gauss_lobatto')
    # quad_thick_source = quadpy.c1.gauss_lobatto(int(N_space/2+1)).points
    # quad_thick_edge = quadpy.c1.gauss_lobatto(int(N_space/4+1)).points
    # quad_thick_source = ([quad_thick_source_inside, quad_thick_source_outside])

    reader = file_reader()
    give = reader()
    # ob = matrix_builder()

    f_fun, g_fun, l_fun = get_sedov_funcs()
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

    
    
    initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
                       ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
                       thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
                       wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
                        quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
                        finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, eval_array, geometry, f_fun, g_fun, l_fun)
                       
    mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                      wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                      domain_width, fake_sedov_v0, boundary_on, t0, eval_array, geometry, sigma_func) 
    
    initialize.make_IC(mesh.edges)
    IC = initialize.IC

    if thermal_couple['none'] == 1:
        deg_freedom = N_ang*N_space*(M+1)
    else:
        deg_freedom = (N_ang+1)*N_space*(M+1)


    
    matrices = G_L(initialize)
    num_flux = LU_surf(initialize)
    source = source_class(initialize)
    uncollided_sol = uncollided_solution(initialize)
    flux = scalar_flux(initialize)
    rhs = rhs_class(initialize)
    transfer = T_function(initialize)
    sigma_class = sigma_integrator(initialize)
    sedov = sedov_class(initialize.g_fun, initialize.f_fun, initialize.l_fun, initialize.sigma_t)
    flux.load_AAA(sigma_class.AAA)
    g_interp = rhs.interp_sedov_selfsim(sedov)
    xs_quadTS, ws_quadTS = quadrature(30, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(2048)
    mu_quadTS = res1.points
    mu_wsTS = res1.weights
    sedov_uncol = sedov_uncollided_solutions(xs_quadTS, ws_quadTS, mu_quadTS, mu_wsTS, x0, sigma_t, t0)
    # phi = sedov_uncol.uncollided_scalar_flux(xs, t, sedov, g_interp, g_interp)
    # plot_sedov_blast_profiles([-200,50.0, 250],  0.4, sedov)
    # plt.figure(72)
    # plt.ion()
    # rs = np.linspace(-0.5, 0.5,100)
    # plt.ion()
    # plt.plot(rs, sedov.interpolate_self_similar(1.0, rs, g_interp))
    # plt.show()
    # assert(0)

    
    def RHS(t, V):
        return rhs.call(t, V, mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class, sedov, g_interp, sedov_uncol)
    
    start = timer()
    reshaped_IC = IC.reshape(deg_freedom)

    if estimate_wavespeed == False:
        tpnts = [tfinal]
    elif estimate_wavespeed == True:
        tpnts = np.linspace(0, tfinal, 10000)
    if eval_times == True:
        tpnts = eval_array
        print(tpnts, 'time points')


    if remesh == False:
        v_hit = sedov.find_contact_time(x0)
     
        
        sol = integrate.solve_ivp(RHS, [0.0,tfinal], reshaped_IC, method=integrator, t_eval = tpnts , rtol = rt, atol = at, max_step = mxstp)

        fake_mesh =  mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                    wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                    domain_width, fake_sedov_v0, boundary_on, t0, eval_array, geometry, sigma_func) 
    
        mesh_edges = get_edges(fake_mesh, tpnts, N_space)
    else:
        if sigma_func['TaylorSedov'] == 1:
            r2_list = []
            mesh_edges = np.zeros((tpnts.size, N_space + 1))
            v_hit = sedov.find_contact_time(x0)
            sedov.physical(tpnts[0])
            r2_list.append(sedov.r2)
            sedov.physical(tpnts[-1])
            r2_list.append(sedov.r2)

            print(v_hit, '**')
            t_pnts1, split_point = split_up_tpnts(tpnts, tpnts[0], v_hit[0])
            sol = integrate.solve_ivp(RHS, [0.0, t_pnts1[-1]], reshaped_IC, method=integrator, t_eval = t_pnts1 , rtol = rt, atol = at, max_step = mxstp)
            mesh.move(t_pnts1[-1])
            edges = mesh.edges
            fake_mesh =  mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                        wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                        domain_width, fake_sedov_v0, boundary_on, t0, eval_array, geometry, sigma_func) 
            mesh_edges[:split_point, :] = get_edges(fake_mesh, t_pnts1[:-1], N_space)
            sol_last = sol.y[:,-1].reshape((N_ang, N_space, M+1))
            # first re-meshing
            move_type = np.array([0,1,0]) 
            t_pnts2, split_point2 = split_up_tpnts(tpnts, v_hit[0], v_hit[1])
            xs_quad, ws_quad = quadrature(60, 'gauss_legendre')
            initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
                        ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
                        thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
                        wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
                            quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
                            finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, eval_array, geometry, f_fun, g_fun, l_fun)
            
            mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                    wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                    domain_width, fake_sedov_v0, boundary_on, t0, tpnts, geometry, sigma_func, sedov.r2 ) 
            
            sedov.physical(v_hit[0])
            mesh.save_thits(v_hit)
            mesh.get_shock_location_TS(sedov.r2, sedov.vr2)

            new_edges = mesh.edges
            print(edges, 'old edges')
            new_quadrature_points = np.zeros((N_space, xs_quad.size))
            old_psi_evaluated = np.zeros((N_ang, N_space, xs_quad.size))
            for space2 in range(N_space):
                b = new_edges[space2+1]
                a = new_edges[space2]
                new_quadrature_points[space2, :] = (b-a)/2*xs_quad + (a+b)/2
            output = make_output(sol.t[-1], N_ang, ws, new_quadrature_points.flatten(), sol_last, M, edges, uncollided, geometry)
            counter = xs_quad.size
            output.make_phi(uncollided_sol)
            for iang in range(N_ang):
                for space2 in range(N_space):
                    old_psi_evaluated[iang, space2, :] = output.psi_out[iang, space2 * counter:space2*counter + counter]
            initialize.make_IC_from_solution(old_psi_evaluated, new_edges)
            # initial_condition = make_output(sol.t[-1], N_ang, ws, np.linspace(new_edges[0], new_edges[-1],50), initialize.IC, M, new_edges, uncollided, geometry)
            # plt.figure(30)
            # plt.ion()
            # plt.plot(np.linspace(new_edges[0], new_edges[-1], 50), initial_condition.make_phi(uncollided_sol))
            # plt.show()
            # assert(0)

            # plt.ion()
            # plt.figure(77)
            # for ii in range(3):
            #     plt.plot(new_quadrature_points.flatten(), old_psi_evaluated[ang_index[ii], :,:].flatten(), 'bo', label = f'mu = {mus[ang_index[ii]]}')
            #     # plt.plot(new_quadrature_points.flatten(), old_psi_2[ang_index[ii], :,:].flatten(), 'k-', label = f'mu = {mus[ang_index[ii]]}')
            # plt.legend()
            # plt.show()
            # assert(0)

            IC2 = initialize.IC
            reshaped_IC2 = IC2.reshape(deg_freedom)

            # fake_mesh =  mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
            #             wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
            #             domain_width, fake_sedov_v0, boundary_on, t0, eval_array, geometry, sigma_func) 
            # edges = mesh.edges
            # mesh_edges[:split_point, :] = get_edges(mesh, t_pnts2[:-1], N_space)
            # mxstp = 0.0005
            sol2 = integrate.solve_ivp(RHS, [t_pnts1[-1], t_pnts2[-1]], reshaped_IC2, method=integrator, t_eval = t_pnts2 , rtol = rt, atol = at, max_step = mxstp)
            
            
            # # second re-meshing
            # edges = mesh.edges
            # sol_last  = sol2.y[:,-1].reshape((N_ang, N_space, M+1))
            # mesh_edges[:split_point, :] = get_edges(mesh, t_pnts2[:-1], N_space)
            # move_type = np.array([0,0,1]) 
            # t_pnts3, split_point3 = split_up_tpnts(tpnts, v_hit[1], tfinal)
            # xs_quad, ws_quad = quadrature(60, 'gauss_legendre')

            # initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
            #             ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
            #             thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
            #             wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
            #                 quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
            #                 finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, eval_array, geometry, f_fun, g_fun, l_fun)
            
            # mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
            #         wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
            #         domain_width, fake_sedov_v0, boundary_on, t0, tpnts, geometry, sigma_func, sedov.r2 ) 
            
            # sedov.physical(v_hit[1])
            # mesh.save_thits(v_hit)
            # mesh.get_shock_location_TS(sedov.r2, sedov.vr2)

            # new_edges = mesh.edges
            # print(edges, 'old edges')
            # new_quadrature_points = np.zeros((N_space, xs_quad.size))
            # old_psi_evaluated = np.zeros((N_ang, N_space, xs_quad.size))
            # for space2 in range(N_space):
            #     b = new_edges[space2+1]
            #     a = new_edges[space2]
            #     new_quadrature_points[space2, :] = (b-a)/2*xs_quad + (a+b)/2
            # output = make_output(sol2.t[-1], N_ang, ws, new_quadrature_points.flatten(), sol_last, M, edges, uncollided, geometry)
            # counter = xs_quad.size
            # phiout = output.make_phi(uncollided_sol)
            # for iang in range(N_ang):
            #     for space2 in range(N_space):
            #         old_psi_evaluated[iang, space2, :] = output.psi_out[iang, space2 * counter:space2*counter + counter]
            # initialize.make_IC_from_solution(old_psi_evaluated, new_edges)
            # # initial_condition = make_output(sol2.t[-1], N_ang, ws, np.linspace(new_edges[0], new_edges[-1],50), initialize.IC, M, new_edges, uncollided, geometry)
            # # plt.figure(30)
            # # plt.ion()
            # # plt.plot(np.linspace(new_edges[0], new_edges[-1], 50), initial_condition.make_phi(uncollided_sol))
            # # plt.show()
            # # assert(0)

            # # plt.ion()
            # # plt.figure(77)
            # # for ii in range(3):
            # #     plt.plot(new_quadrature_points.flatten(), old_psi_evaluated[ang_index[ii], :,:].flatten(), 'bo', label = f'mu = {mus[ang_index[ii]]}')
            # #     # plt.plot(new_quadrature_points.flatten(), old_psi_2[ang_index[ii], :,:].flatten(), 'k-', label = f'mu = {mus[ang_index[ii]]}')
            # # plt.legend()
            # # plt.show()
            # # assert(0)

            # IC3 = initialize.IC
            # reshaped_IC3 = IC3.reshape(deg_freedom)

            # ic_test = make_output(t_pnts2[-1], N_ang, ws, new_quadrature_points.flatten(), IC3, M, new_edges, uncollided, geometry)
            # phi_ic = ic_test.make_phi(uncollided_sol)

            # plt.figure(29)
            # plt.plot(new_quadrature_points.flatten(), phi_ic)
            # plt.plot(new_quadrature_points.flatten(), phiout, 'o', mfc = 'none')
            # plt.plot(new_quadrature_points.flatten(), sedov_uncol.uncollided_scalar_flux(new_quadrature_points.flatten(), v_hit[1], sedov, g_interp, g_interp), 's', mfc = 'none' )

            # plt.show()
            # print(new_quadrature_points.flatten())


            # sol3 = integrate.solve_ivp(RHS, [t_pnts2[-1], tfinal], reshaped_IC3, method=integrator, t_eval = t_pnts3 , rtol = rt, atol = at, max_step = mxstp)
            
            
            sol_last = sol2.y[:,-1].reshape((N_ang, N_space, M+1))

            print(sol2.t[-1], 'sol last evaluation time')
            old_edges = mesh.edges

            print(old_edges, 'old edges, second remesh')
            mesh_edges[split_point:split_point2, :] = get_edges(mesh, t_pnts2[:-1], N_space)
            mesh.move(v_hit[1])
            print(old_edges, 'old edges, second remesh')
            xs_quad, ws_quad = quadrature(80, 'gauss_legendre')
            new_quadrature_points = np.zeros((N_space, xs_quad.size))
            old_psi_evaluated = np.zeros((N_ang, N_space, xs_quad.size))
            move_type = np.array([0,0,1]) 
            mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                    wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                    domain_width, fake_sedov_v0, boundary_on, t0, tpnts, geometry, sigma_func, sedov.r2)
            
            
            sedov.physical(v_hit[1])  
            mesh.save_thits(v_hit)
            mesh.get_shock_location_TS(sedov.r2, sedov.vr2)
            new_edges = mesh.edges

            for space2 in range(N_space):
                b = new_edges[space2+1]
                a = new_edges[space2]
                new_quadrature_points[space2, :] = (b-a)/2*xs_quad + (a+b)/2
            counter = xs_quad.size
            print(old_edges, 'old edges, second remesh')
            output2 = make_output(v_hit[1], N_ang, ws, new_quadrature_points.flatten(), sol_last, M, old_edges, uncollided, geometry)
            phiout = output2.make_phi(uncollided_sol)
            for iang in range(N_ang):
                for space2 in range(N_space):
                    old_psi_evaluated[iang, space2, :] = output2.psi_out[iang, space2 * counter:space2*counter + counter]

            # fake_mesh2 =  mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
            #             wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
            #             domain_width, fake_sedov_v0, boundary_on, t0, eval_array, geometry, sigma_func) 

            
            
            
            
            

            t_pnts3, split_point3 = split_up_tpnts(tpnts, v_hit[1], tfinal)
            

            initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
                        ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
                        thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
                        wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
                            quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
                            finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, eval_array, geometry, f_fun, g_fun, l_fun)
            
            
        
              
            
            print(new_edges, 'new edges, second remesh')
            print(old_edges, 'old edges, second remesh')

            print(sedov.r2, 'shock positon')

            

            # initial_condition = make_output(sol2.t[-1], N_ang, ws, np.linspace(new_edges[0], new_edges[-1], 50), initialize.IC, M, new_edges, uncollided, geometry)
            # plt.figure(30)
            # plt.ion()
            # plt.plot(np.linspace(new_edges[0], new_edges[-1], 50), initial_condition.make_phi(uncollided_sol))
            # plt.show()
            # assert(0)

            initialize.make_IC_from_solution(old_psi_evaluated, new_edges)

            IC3 = initialize.IC
            reshaped_IC3 = IC3.reshape(deg_freedom)

            ic_test = make_output(t_pnts2[-1], N_ang, ws, new_quadrature_points.flatten(), IC3, M, new_edges, uncollided, geometry)
            phi_ic = ic_test.make_phi(uncollided_sol)

            plt.figure(29)
            plt.plot(new_quadrature_points.flatten(), phi_ic, '--')
            plt.plot(new_quadrature_points.flatten(), phiout, ':', mfc = 'none')
            plt.plot(new_quadrature_points.flatten(), sedov_uncol.uncollided_scalar_flux(new_quadrature_points.flatten(), v_hit[1], sedov, g_interp, g_interp), '-')

            plt.show()
            print(new_quadrature_points.flatten())


            sol3 = integrate.solve_ivp(RHS, [v_hit[1], tfinal], reshaped_IC3, method=integrator, t_eval = t_pnts3, rtol = rt, atol = at, max_step = mxstp)
            
            mesh_edges[split_point2:, :] = get_edges(mesh, t_pnts3[:], N_space)

            sol_placeholder = np.zeros(((N_ang*N_space*(M+1)), tpnts.size))
            # print(t_pnts1)
            # print(t_pnts2)
            # print(t_pnts3)
            # print(tpnts)
            # print(split_point)
            # print(split_point2)
            # print(mesh_edges)
            sol_placeholder[:, :split_point] = sol.y[:, :-1]
            sol_placeholder[:, split_point:split_point2] = sol2.y[:,:-1]
            sol_placeholder[:, split_point2:] = sol3.y[:,:]
            if sum(np.concatenate((t_pnts1[:-1], t_pnts2[:-1], t_pnts3))-tpnts)!= 0.0:
                assert(0)
            sol.y = sol_placeholder
            sol.t = tpnts
            print(r2_list, 'r2 list')
            print(t_pnts3, 't3')
            print(t_pnts2, 't2')
            print(t_pnts1, 't1')


            

        elif sigma_func['fake_sedov'] == 1:
            t_stop = mesh.t_hit[0]
            print('t_stop', t_stop)
            split_point = np.argmin(np.abs(tpnts-t_stop)) 
            mesh_edges = np.zeros((tpnts.size, N_space + 1))
            t_pnts1 = tpnts[0:split_point]
            t_pnts2 = tpnts[split_point:]
            if t_pnts1[-1] > t_stop or t_pnts2[0] < t_stop:
                assert(0)
            t_pnts1 = np.append(t_pnts1, t_stop)
            


            # mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
            #               wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
            #               domain_width, fake_sedov_v0, boundary_on, t0, tpnts) 

            sol = integrate.solve_ivp(RHS, [0.0, t_pnts1[-1]], reshaped_IC, method=integrator, t_eval = t_pnts1 , rtol = rt, atol = at, max_step = mxstp)
            if sol.t.size < t_pnts1.size:
                print(sol.t)
                print(t_pnts1)
                print('skipped over an evaluation time')
                assert(0)
            mesh.move(t_stop)
            edges = mesh.edges

            fake_mesh =  mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                        wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                        domain_width, fake_sedov_v0, boundary_on, t0, eval_array, geometry) 
            
            mesh_edges[:split_point, :] = get_edges(fake_mesh, t_pnts1[:-1], N_space)
            xs = np.linspace(-x0, x0, 300)
            sol_last = sol.y[:,-1].reshape((N_ang, N_space, M+1))
            # sol_last = np.zeros(deg_freedom)
            
            # output = make_output(t_stop, N_ang, ws, xs, sol_last, M, edges, uncollided)

            # phi = output.make_phi(uncollided_sol)
            # psi = output.psi_out 

            # mesh_edges[:split_point, :] = mesh.saved_edges[:split_point, :]

            # plt.ion
            # plt.figure(24)
            # plt.plot(xs, psi[0, :])
            # plt.plot(xs, psi[-1, :])
            # plt.show()

            move_type = np.array([0,1,0]) 
            xs_quad, ws_quad = quadrature(60, 'gauss_legendre')
            
            
            initialize = build(N_ang, N_space, M, tfinal, x0, t0, mus, ws, xs_quad,
                        ws_quad, sigma_t, sigma_s, source_type, uncollided, moving, move_type, t_quad, t_ws,
                        thermal_couple, temp_function, e_initial, sigma, particle_v, edge_v, cv0, thick, 
                        wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
                            quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma,
                            finite_domain, domain_width, fake_sedov_v0, test_dimensional_rhs, epsilon, eval_array, geometry, f_fun, g_fun, l_fun)
            
            mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
                    wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
                    domain_width, fake_sedov_v0, boundary_on, t0, tpnts, geometry) 
            
            new_edges = mesh.edges
            print(edges, 'old edges')
            print(new_edges, 'new edges')
            new_quadrature_points = np.zeros((N_space, xs_quad.size))
            old_psi_evaluated = np.zeros((N_ang, N_space, xs_quad.size))
            
            for space2 in range(N_space):
                b = new_edges[space2+1]
                a = new_edges[space2]
                new_quadrature_points[space2, :] = (b-a)/2*xs_quad + (a+b)/2
            # print('##', edges, 'old edges ##')
            # print(edges[:-1]- edges[1:], 'edge spacings' )
            # print('##', new_quadrature_points.flatten(), 'new eval points ##')
            output = make_output(t_stop, N_ang, ws, new_quadrature_points.flatten(), sol_last, M, edges, uncollided, geometry)
            counter = xs_quad.size
            output.make_phi(uncollided_sol)
            for iang in range(N_ang):
                for space2 in range(N_space):
                    old_psi_evaluated[iang, space2, :] = output.psi_out[iang, space2 * counter:space2*counter + counter]
            # plt.figure(97)
            # plt.plot(new_quadrature_points.flatten(), old_psi_evaluated[int(N_ang/2 + 10), :, :].flatten())
            # plt.show()
            # old_psi_2 = old_psi_evaluated *0
            # for iang in range(N_ang): 
            #     for space2 in range(N_space):
            #         for ixx, xx in enumerate(new_quadrature_points.flatten()[space2 * counter:space2*counter + counter]):
            #             old_psi_2[iang, space2, ixx] =  uncollided_sol.fake_sedov_integrand(mus[iang], t_stop, xx )
            
        

            # ang_index = [np.random.randint(int(N_ang/2), N_ang-1) for a in np.zeros(4)]
            # plt.ion()
            # plt.figure(77)
            # for ii in range(3):
            #     plt.plot(new_quadrature_points.flatten(), old_psi_evaluated[ang_index[ii], :,:].flatten(), 'bo', label = f'mu = {mus[ang_index[ii]]}')
            #     # plt.plot(new_quadrature_points.flatten(), old_psi_2[ang_index[ii], :,:].flatten(), 'k-', label = f'mu = {mus[ang_index[ii]]}')
            # plt.legend()
            # plt.show()
            # assert(0)


            initialize.make_IC_from_solution(old_psi_evaluated, new_edges)
            IC = initialize.IC
            reshaped_IC = IC.reshape(deg_freedom)
        
            sol2 = integrate.solve_ivp(RHS, [t_stop, tfinal], reshaped_IC, method=integrator, t_eval = t_pnts2 , rtol = rt, atol = at, max_step = mxstp)
            # mesh_edges[split_point:, :] = mesh.saved_edges[split_point:, :]
            # mesh_edges[-1,:] = mesh.edges
            mesh_edges[split_point:, :] = get_edges(mesh, t_pnts2, N_space)

            plot_edges_xvst(tpnts, mesh_edges, fake_sedov_v0, 'realsedov')

            sol_placeholder = np.zeros(((N_ang*N_space*(M+1)), tpnts.size))

            sol_placeholder[:, :split_point] = sol.y[:, :-1]
            sol_placeholder[:, split_point:] = sol2.y[:,:]

            sol.y = sol_placeholder
            sol.t = tpnts


    


    print(sol.y.shape,'sol y shape')
    print(eval_times, 'eval times')
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
        thermal_couple, tfinal, x0, sol.t, find_edges_tol, source_type, sigma_t, eval_array)
        left_edges, right_edges, T_front_location = wave_loc_finder.find_wave(sol)
    elif find_wave_loc == False:
        left_edges =  np.zeros(1)
        right_edges = np.zeros(1)
        T_front_location = np.zeros(1)

    
    if thermal_couple['none'] == 1:
        extra_deg_freedom = 0
       
        sol_last = sol.y[:,-1].reshape((N_ang,N_space,M+1))
        if eval_times ==True:
            sol_array = sol.y.reshape((eval_array.size, N_ang,N_space,M+1)) 
    elif thermal_couple['none'] != 1:
        extra_deg_freedom = 1
        sol_last = sol.y[:,-1].reshape((N_ang+1,N_space,M+1))
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
    if eval_times == False:
        output = make_output(tfinal, N_ang, ws, xs, sol_last, M, edges, uncollided, geometry)
        phi = output.make_phi(uncollided_sol)
        psi = output.psi_out # this is the collided psi
        exit_dist, exit_phi = output.get_exit_dist(uncollided_sol)
        xs_ret = xs
        if thermal_couple == 1:
            e = output.make_e()
        else:
            e = phi*0
    else:
        ## This is broken for a moving mesh
        phi = np.zeros((eval_array.size, xs.size))
        psi = np.zeros((eval_array.size, N_ang, xs.size))
        exit_dist = np.zeros((eval_array.size, N_ang, 2))
        exit_phi = np.zeros((eval_array.size, 2))
        xs_ret = np.zeros((eval_array.size, xs.size))
        # mesh = mesh_class(N_space, x0, tfinal, moving, move_type, source_type, edge_v, thick, move_factor,
        #               wave_loc_array, pad, leader_pad, quad_thick_source, quad_thick_edge, finite_domain,
        #               domain_width, fake_sedov_v0, boundary_on, t0, eval_array) 
        for it, tt in enumerate(eval_array):
            # mesh.move(tt)
            # edges = mesh.edges
            edges = mesh_edges[it]
            print(tt)
            print(edges)
            if choose_xs == False:
                xs = find_nodes(edges, M, geometry)
            elif choose_xs == True:
                xs = specified_xs
            output = make_output(tt, N_ang, ws, xs, sol.y[:,it].reshape((N_ang+extra_deg_freedom,N_space,M+1)), M, edges, uncollided, geometry)
            phi[it,:] = output.make_phi(uncollided_sol)
            psi[it, :, :] = output.psi_out # this is the collided psi
            exit_dist[it], exit_phi[it] = output.get_exit_dist(uncollided_sol)
            xs_ret[it] = xs
            if thermal_couple == 1:
                e = output.make_e()
            else:
                e = phi*0
    computation_time = end-start

    import h5py 
    f = h5py.File("mesh_edges_blast.h5", 'r+')
    if f.__contains__('edges'):
        del f['edges']
    f.create_dataset('edges', data = mesh_edges)
    f.close()
    
    return xs_ret, phi, psi, exit_dist, exit_phi,  e, computation_time, sol_last, mus, ws, edges, wavespeed_array, tpnts, left_edges, right_edges, wave_tpnts, wave_xpnts, T_front_location, mus



def problem_identifier():
    name_array = []

def plot_edges(edges, fign):
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


def newtons(x0, f, fprime, tol = 1e-14):
    def iterate(x0, f, fprime):
        return x0 - f(x0) / fprime(x0)
    tol_met = False
    while tol_met == False:
        new_x0 = iterate(x0, f, fprime)
        if abs(new_x0-x0) <= tol:
            tol_met = True
        x0 = new_x0
    return x0


def get_edges(mesh_ob, t_pnts, N_space):
    edges = np.zeros((t_pnts.size, N_space+1))
    for it, tt in enumerate(t_pnts):
        mesh_ob.move(tt)
        edges[it] = mesh_ob.edges
    return edges



def plot_edges_xvst(tpnts, edges, v0, type):
    plt.ion()
    plt.figure(77)
    if type == 'sedov':
        for ix in range(edges[0,:].size):
            plt.plot(edges[:, ix], tpnts, 'b-', mfc = 'none')
        plt.plot(tpnts - 5, tpnts ,'k-')
        plt.plot(tpnts * (-v0), tpnts, 'k--')
        plt.plot(np.zeros(tpnts.size), tpnts, 'b--')
        plt.xlim(-5.01, 5.01)
        plt.ylim(0, 20)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('t', fontsize = 16)
        # plt.plot()
        show('edges_vs_t_fakesedov')
    if type == 'realsedov':
        for ix in range(edges[0,:].size):
            plt.plot(edges[:, ix], tpnts, 'b-', mfc = 'none')
        plt.plot(tpnts - 5, tpnts ,'k-')
        plt.plot(tpnts * (-v0), tpnts, 'k--')
        plt.plot(np.zeros(tpnts.size), tpnts, 'b--')
        plt.xlim(-2, 2.01)
        plt.ylim(0, 20)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('t', fontsize = 16)
        # plt.plot()
        show('edges_vs_t_sedov')
    plt.show()





def plot_sedov_blast_profiles(tlist, x0, sedov):
    xlist = np.linspace(-x0, x0, 500)
    for it, tt in enumerate(tlist):
        print('## ## ## ## ## ##')
        print('t=', tt)
        rho, v = sedov.interior_interpolate(tt, xlist)
        plt.ion()
        plt.figure(265)
        plt.text(-0.045, 3.0, f't={tlist[0]}')
        # plt.text(-.1, 3.0, f't={tlist[1]}')
        plt.text(-.16, 3.0, f't={tlist[2]}')
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(r'$\rho$', fontsize = 16)
        plt.plot(xlist, rho, 'k-')
    show('sedov_density_profiles')
    plt.show()
    assert(0)


def split_up_tpnts(tpnts, t_start,  t_stop):
    start_point = np.argmin(np.abs(tpnts-t_start))
    split_point = np.argmin(np.abs(tpnts-t_stop)) 
    t_pnts1 = tpnts[start_point:split_point]
    t_pnts2 = tpnts[split_point:]
    if t_pnts1[-1] > t_stop:
        print(t_pnts1)
        print(t_stop)
        assert(0)
    if t_pnts1[0] < t_start:
        start_point += 1
        t_pnts1 = tpnts[start_point:split_point]
    if t_pnts2[0] < t_stop:
        split_point += 1
        t_pnts1 = tpnts[start_point:split_point]
        t_pnts2 = tpnts[split_point:]
        print(t_pnts2)
        print(t_stop)
    # if t_pnts1[0] > t_start:
    #     print(t_pnts1)
    #     print(t_start)
    #     start_point -= 1
    #     t_pnts1 = tpnts[start_point:split_point]
    t_pnts1 = np.append(t_pnts1, t_stop)
    
    return t_pnts1, split_point
