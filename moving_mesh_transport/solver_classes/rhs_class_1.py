#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:25:35 2022

@author: bennett
"""
import numpy as np
import math
from .cubic_spline import cubic_spline
from .build_problem import build
from .matrices import G_L
from .sources import source_class
from .phi_class import scalar_flux
from .uncollided_solutions import uncollided_solution
from .numerical_flux import LU_surf
from .radiative_transfer import T_function
from .opacity import sigma_integrator
from .functions import shaper
from .functions import finite_diff_uneven_diamond 
import numba as nb
from numba import prange
from numba.experimental import jitclass
from numba import int64, float64, deferred_type, prange
from numba import types, typed

build_type = deferred_type()
build_type.define(build.class_type.instance_type)
matrices_type = deferred_type()
matrices_type.define(G_L.class_type.instance_type)
num_flux_type = deferred_type()
num_flux_type.define(LU_surf.class_type.instance_type)
source_type = deferred_type()
source_type.define(source_class.class_type.instance_type)
flux_type = deferred_type()
flux_type.define(scalar_flux.class_type.instance_type)
uncollided_solution_type = deferred_type()
uncollided_solution_type.define(uncollided_solution.class_type.instance_type)
transfer_class_type = deferred_type()
transfer_class_type.define(T_function.class_type.instance_type)
sigma_class_type = deferred_type()
sigma_class_type.define(sigma_integrator.class_type.instance_type)
cubic_class_type = deferred_type()
cubic_class_type.define(cubic_spline.class_type.instance_type)

# kv_ty = (types.int64, types.unicode_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))


data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('source_type', int64[:]),
        ('t', float64),
        ('sigma_t', float64),
        ('sigma_s', float64),
        ('IC', float64[:,:,:]),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('x0', float64),
        ("xL", float64),
        ("xR", float64),
        ("dxL", float64),
        ("dxR", float64),
        ("L", float64[:,:]),
        ("G", float64[:,:]),
        ("P", float64[:]),
        ("PV", float64[:]),
        ("S", float64[:]),
        ("LU", float64[:]),
        ("U", float64[:]),
        ("H", float64[:]),
        ("V_new", float64[:,:,:]),
        ("V", float64[:,:,:]),
        ("V_old", float64[:,:,:]),
        ('c', float64),
        ('uncollided', int64),
        ('thermal_couple', nb.typeof(params_default)),
        ('test_dimensional_rhs', int64),
        ('told', float64),
        ('division', float64),
        ('c_a', float64),
        ('sigma_a', float64),
        ('mean_free_time', float64),
        ('counter', int64),
        ('delta_tavg', float64),
        ('l', float64),
        ('times_list', float64[:]),
        ('save_derivative', int64),
        ('e_list', float64[:]),
        ('e_xs_list', float64[:]),
        ('wave_loc_list', float64[:]),
        ('sigma_func', nb.typeof(params_default)),
        ('particle_v', float64),
        ('epsilon', float64),
        ('deg_freedom', int64[:]),
        ('geometry', nb.typeof(params_default)),
        ('alphams', float64[:]),
        ('radiative_transfer', nb.typeof(params_default)),
        ('test', float64),
        ('xs_quad', float64[:]),
        ('T_old', float64[:,:]),
        ('T_eval_points', float64[:,:]),
        ('time_points', float64[:]),
        ('t_quad', float64[:]),
        ('t_ws', float64[:])

        ]
##############################################################################
#thermal couple, l scaling


@jitclass(data)
class rhs_class():
    def __init__(self, build):
        self.N_ang = build.N_ang 
        self.N_space = build.N_space
        self.M = build.M
        self.mus = build.mus
        self.geometry = build.geometry
        # if self.geometry['slab'] == True:
        self.ws = build.ws
        self.xs_quad = build.xs_quad
        self.t_quad = build.t_quad
        self.t_ws = build.t_ws
        # elif self.geometry['sphere'] == True:
        #     self.ws = build.t_ws
        #     self.xs_quad = build.t_quad
        self.source_type = np.array(list(build.source_type), dtype = np.int64) 

        self.thermal_couple = build.thermal_couple
        self.uncollided = build.uncollided
        self.test_dimensional_rhs = build.test_dimensional_rhs
        self.told = 0.0
        self.sigma_s = build.sigma_s
        self.sigma_a = build.sigma_a
        self.sigma_t = build.sigma_t
        self.c = build.sigma_s 
        self.particle_v = build.particle_v
        
        self.radiative_transfer = build.thermal_couple
       
        self.c_a = build.sigma_a / build.sigma_t
        
        self.mean_free_time = 1/build.sigma_t
        self.division = 1000
        self.counter = 0
        self.delta_tavg = 0.0
        self.l = build.l
        self.times_list = np.array([0.0])
        self.e_list = np.array([0.0])
        self.e_xs_list = np.array([0.0])
        self.wave_loc_list = np.array([0.0])
        self.save_derivative = build.save_wave_loc
        self.sigma_func = build.sigma_func
        self.deg_freedom = shaper(self.N_ang, self.N_space, self.M + 1, self.thermal_couple)
        self.alphams = np.zeros(self.N_ang + 1)
        self.x0 = build.x0
        timepoints = 2048
        self.time_points = np.linspace(0.0, build.tfinal, timepoints)
        self.T_old = np.zeros((self.N_space, self.xs_quad.size))
        
        for angle2 in range(1, self.N_ang + 1):
            self.alphams[angle2] = self.alphams[angle2-1] - self.ws[angle2-1] * self.mus[angle2-1]

    
    def time_step_counter(self, t, mesh):
        delta_t = abs(self.told - t)
        self.delta_tavg += delta_t / self.division
        if self.counter == self.division:
            print('t = ', t, '|', 'delta_t average= ', self.delta_tavg)
            if self.N_space <= 32:
                if self.geometry['sphere'] == True:
                    print(mesh.edges)
                else:
                    print(mesh.edges)
            print('--- --- --- --- --- --- --- --- --- --- --- --- --- ---')
            self.delta_tavg = 0.0
            self.counter = 0
        else:
            self.counter += 1
        self.told = t

        
    def derivative_saver(self, t,  space, transfer_class):
        if self.save_derivative == True:
            self.e_list = np.append(self.e_list, transfer_class.e_points)
            self.e_xs_list = np.append(self.e_xs_list, transfer_class.xs_points)

        if space == self.N_space - 1:
            deriv = np.copy(self.e_list)*0
            for ix in range(1,self.e_list.size-1):
                dx = self.e_xs_list[ix+1] - self.e_xs_list[ix]
                deriv[ix] = (self.e_list[ix+1] - self.e_list[ix])/dx

            max_deriv = max(np.abs(deriv))
            max_deriv_loc = np.argmin(np.abs(np.abs(self.e_list) - max_deriv))
            heat_wave_loc = self.e_xs_list[max_deriv_loc]
            self.wave_loc_list = np.append(self.wave_loc_list, abs(heat_wave_loc)) 
            self.times_list = np.append(self.times_list,t)
            # print(heat_wave_loc, 'wave x')
        
    def call(self, t, V, mesh, matrices, num_flux, source, uncollided_sol, flux, transfer_class, sigma_class):
        # print out timesteps
        self.time_step_counter(t, mesh) 
        # allocate arrays

        # My (Stephen's) attempt at adding radiative transfer to this code

        # Not sure if this is correct
        if self.radiative_transfer['none'] == False :  
            V_new = V.copy().reshape((self.N_ang + 1, self.N_space, self.M+1))
            V_old = V_new.copy()
        else:
            # V_new = V.copy().reshape((self.deg_freedom[0], self.deg_freedom[1], self.deg_freedom[2]))
            V_new = V.copy().reshape((self.N_ang, self.N_space, self.M+1))
            V_old = V_new.copy()
        # move mesh to time t 
        mesh.move(t)
        # represent opacity as a polynomial expansion
        # self.T_old[:,0] = 1.0
        time_loc = np.argmin(np.abs(self.time_points - t))
        # print(self.T_old[time_loc])
        # if self.T_old[time_loc, 0,0] == np.zeros(self.xs_quad.size):
        #     time_loc -= 1
        self.T_old, self.T_eval_points = self.make_temp(V_old[-1,:,:], mesh, transfer_class)
        
        sigma_class.sigma_moments(mesh.edges, t, self.T_old, self.T_eval_points)
        flux.get_coeffs(sigma_class)
        # sigma_class.check_sigma_coeffs(self.T_eval_points, mesh.edges, self.T_old)
    

        # iterate over all cells
        for space in range(self.N_space):  
            # get mesh edges and derivatives          
            xR = mesh.edges[space+1]
            xL = mesh.edges[space]
            dxR = mesh.Dedges[space+1]
            dxL = mesh.Dedges[space]
            # matrices.matrix_test(True)
            matrices.make_all_matrices(xL, xR, dxL, dxR)

            L = matrices.L
            G = matrices.G
            MPRIME = matrices.MPRIME
             # special matrices for spherical geometries
            if self.geometry['sphere'] == True:
                Mass = matrices.Mass
                Minv = np.linalg.inv(Mass)
                J = matrices.J
                # VVs = matrices.VV
            # make P 
            if self.radiative_transfer['none'] == False:
                flux.make_P(V_old[:-1,space,:], space, xL, xR)
            else:
                flux.make_P(V_old[:,space,:], space, xL, xR)
            # I need to make different functions to call the scattering and absorbing scalar flux
            PV = flux.scalar_flux_term
            # integrate the source
            source.make_source(t, xL, xR, uncollided_sol)
            #print(source.make_source(t, xL, xR, uncollided_sol))
            S = source.S
            H = transfer_class.H

            
            #if ( (S[0]!=0.0) or (S[1]!=0.0) ):
                #print("Nonzero source.S in rhs_class: ", S)

            # radiative transfer term
            
            # def testsoln(a, b):
            #     """ Calculates the value of the analytic solution for H when a simple function is used,
            #     to test whether the temperature function is being integrated properly."""
            #     test = np.sqrt(1/(np.pi*(b-a)) ) * ((a**2 - 2)*np.cos(a) - 2*a*np.sin(a) - (b**2 - 2)*np.cos(b) + 2*b*np.sin(b))
            #     return test

            if self.radiative_transfer['none'] == False:

                # print(V_old[self.N_ang, space, :], 'v old')
                transfer_class.make_H(xL, xR, V_old[-1, space, :], sigma_class, space)

                H = transfer_class.H
                # if (H <0).any():
                #     print(H)
                #     assert 0 
                
               
                #T_old saves the temperature at the zeros of the interpolating polynomial
                # print(H)
                # time_loc = np.argmin(np.abs(self.time_points - t))
                # self.T_old[time_loc, space] = transfer_class.make_T(argument, a, b) 
                # print(self.T_old[space], 'T old')
    
                ######### solve thermal couple ############
                U = V_old[-1,space,:]
                num_flux.make_LU(t, mesh, V_old[-1,:,:], space, 0.0, V_old[-1, 0, :]*0, True)
                RU = num_flux.LU 
                RHS_transfer = U*0
                if self.uncollided == True:
                    RHS_transfer += self.c_a *source.S * 2 
                    #RHS_transfer += source.S * 2
                #print("source.S = ", source.S)
                RHS_transfer -= RU
                # if space == self.N_space -1:
                #     print(RU)
                RHS_transfer += -np.dot(MPRIME, U) + np.dot(G,U) - self.c_a *H /self.sigma_t
                RHS_transfer += self.c_a * PV*2 /self.sigma_t 
                RHS_transfer = np.dot(RHS_transfer, Minv)
                if self.l != 1.0:
                    RHS_transfer = RHS_transfer / self.l
                V_new[-1,space,:] = RHS_transfer 
                if np.isnan(V_new[-1, space, :]).any():
                    print('rhstransfer is nan')
                    assert(0)
                # print(RHS_transfer, 'rhs transfer')

            ########## Loop over angle ############
            for angle in range(self.N_ang):
                
                mul = self.mus[angle]
                # calculate numerical flux
                refl_index = 0
                if space == 0:
                    if angle >= self.N_ang/2:
                        assert(self.mus[angle] > 0)
                        refl_index = self.N_ang-angle-1
                        assert(abs(self.mus[refl_index] - -self.mus[angle])<=1e-10)
                    # print(self.mus[])
                    
                num_flux.make_LU(t, mesh, V_old[angle,:,:], space, mul, V_old[refl_index, 0, :])
                
                    # print(self.mus[self.N_ang-angle-1], -self.mus[angle])
                    
                LU = num_flux.LU
                # Get absorption term
                # sigma_class.sigma_moments(mesh.edges, t, self.T_old, V_old[-1, :, :])
                sigma_class.make_vectors(mesh.edges, V_old[angle,space,:], space)
                VV = sigma_class.VV
                # Initialize solution vector, RHS
                U = np.zeros(self.M+1).transpose()
                # assert(abs(G[0,0]  + 0.16666666666666666*((xL**2 + xL*xR + xR**2)*(dxL - dxR))/((xL - xR)*math.pi))<=1e-10)
                U[:] = V_old[angle,space,:]
                # RHS = np.zeros_like(V_new[angle,space,:])

             

                    # if self.M == 0:
                    #     a = xL
                    #     b = xR
                    #     # print(Minv,  3 * math.pi/ (a*b + b**2 + a**2))
                    #     assert(np.abs(Minv[0,0] - 3 * math.pi/ (a*b + b**2 + a**2)) <=1e-8)
                dterm = U*0
                for j in prange(self.M+1):
                    # vec = (1-self.mus**2) * V_old[:, space, j]
                    # if angle != 0 and angle != self.N_ang-1:
                        
                    # dterm[j] = finite_diff_uneven_diamond_2(self.mus, angle, V_old[:, space, j], self.alphams, self.ws, left = (angle==0), right = (angle == self.N_ang-1))
                    dterm[j] = finite_diff_uneven_diamond(self.mus, angle, V_old[:-1, space, j], left = (angle==0), right = (angle == self.N_ang-1), origin = False)
                     


                if self.geometry['sphere'] == True:  
                    a = xL
                    b = xR
                    RHS = V_old[angle, space, :]*0
                    RHS -=  LU
                    RHS +=  mul*np.dot(L,U)
                    mu_derivative =  np.dot(J, dterm)
                    RHS -= mu_derivative
                    RHS += np.dot(G, U)
                    # RHS += 0.5 * S * self.c #(commented this out because c is included)
                    RHS += 0.5 * S /self.sigma_t / self.l
                    RHS +=  self.c_a * H * 0.5 / self.sigma_t / self.l
                    RHS += PV * self.c /self.sigma_t / self.l
                    # print(VV, 'VV')
                    # print(V_old[angle, space,:], 'vold')
                    
                    # print(VV,'vv')
                    # print(np.dot(Mass, U), 'MU')
                    # VV[0] = U[0]*(math.sqrt(1/(-xL + xR))*(xL**2 + xL*xR + xR**2))/3/(math.pi**1.5)*math.sqrt(math.pi)*math.sqrt(xR-xL)
                    # assert(abs(VV[0] - U[0]*(math.sqrt(1/(-xL + xR))*(xL**2 + xL*xR + xR**2))/3/(math.pi**1.5)*math.sqrt(math.pi)*math.sqrt(xR-xL) ) <=1e-3 )
                    # assert(abs(Mass[0,0] -  (math.sqrt(1/(-xL + xR))*(xL**2 + xL*xR + xR**2))/3/(math.pi**1.5)*math.sqrt(math.pi)*math.sqrt(xR-xL) ) <=1e-3)
                    # assert(abs((VV-np.dot(Mass, U))[0])<=1e-3)
                    RHS -= VV / self.sigma_t / self.l
                    RHS -= np.dot(MPRIME, U)
                    RHS = np.dot(Minv, RHS)

               
                    
                    V_new[angle,space,:] = RHS  

        # print(V_new.shape)

        if self.radiative_transfer['none'] == False:
            # V_new = self.V_new_floor_func(V_new)
            return V_new.reshape((self.N_ang + 1) * self.N_space * (self.M+1))

        else:

            return V_new.reshape((self.N_ang) * self.N_space * (self.M+1))
        
    
    def V_new_floor_func(self, V_new):
        floor = 1e-20
        for ang in range(self.N_ang + 1):
            for space in range(self.N_space):
                for j in range(self.M+1):
                    if abs(V_new[ang, space, j])<=floor:
                        V_new[ang, space, j] = 0.0
        return V_new
    
    def make_temp(self, e_vec, mesh, rad_transfer):
        T_vec = np.zeros((self.N_space, self.xs_quad.size))
        T_eval_points = np.zeros((self.N_space, self.xs_quad.size))
        for space in range(self.N_space):
            xR = mesh.edges[space+1]
            xL = mesh.edges[space]
            rad_transfer.e_vec = e_vec[space,:]
            a = xL
            b = xR
            argument = (b-a)/2*self.xs_quad + (a+b)/2
            argument2 = (b-a)/2*self.t_quad + (a+b)/2
            T_vec[space] = rad_transfer.make_T(argument, a, b)
            # T_test = rad_transfer.make_T(argument2, a, b)
            # spline_ob = cubic_spline(argument, T_vec[space])
            # if np.max(np.abs(spline_ob.eval_spline(argument2)- T_test)/np.abs(T_test+1e-4))>=5e-2:
            #     print(argument, 'arg')
            #     print(argument2, 'arg2')
            #     print(a,b,'a,b')
            #     # print(spline_ob.eval_spline(argument2), T_test)
            #     print((spline_ob.eval_spline(argument2)- T_test)/np.abs(T_test+1e-4))
            #     assert(0)
            T_eval_points[space] = argument
            # print(T_vec[space], 'T vec')
            # print(argument, 'xs')
            if np.isnan(T_eval_points.any()):
                assert(0)
            # if (T_vec[space]).any() <0:
            T_vec[space] = np.mean(T_vec[space]) + T_vec[space] * 0
        # print(T_vec, 'T')
        # print('## ## ## ## ## ## ')
        return T_vec, T_eval_points


