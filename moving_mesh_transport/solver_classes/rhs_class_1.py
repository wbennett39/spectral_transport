#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:25:35 2022

@author: bennett
"""
import numpy as np
import math
from .cubic_spline import cubic_spline
from .make_phi import make_output
from .build_problem import build
from .matrices import G_L
from .sources import source_class
from .phi_class import scalar_flux
from .uncollided_solutions import uncollided_solution
from .numerical_flux import LU_surf
from .radiative_transfer import T_function
from .opacity import sigma_integrator
from .functions import shaper
from .functions import finite_diff_uneven_diamond, alpha_difference, finite_diff_uneven
from .functions import converging_time_function, converging_r, make_u_old, legendre_difference, check_current_legendre
import numba as nb
from numba import prange
from numba.experimental import jitclass
from numba import int64, float64, deferred_type, prange
from numba import types, typed
from .functions import mass_lumper


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
make_phi_class_type = deferred_type()
make_phi_class_type.define(make_output.class_type.instance_type)




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
        ('t_ws', float64[:]),
        ('lumping', int64),
        ('tfinal', float64),
        ('alphas', float64[:]),
        ('ws', float64[:]),
        ('old_percent_complete', float64),
        ('stymie_count', int64),
        ('psi_onehalf_old', float64[:,:]),
        ('index', int64),
        ('edges_old', float64[:]),
        ('ws_quad', float64[:]),
        ('t_old_list', float64[:]),
        ('time_save_points', int64),
        ('slope_limiter', int64),
        ('wavefront_estimator', float64),
        ('Y_plus', float64[:]),
        ('Y_minus', float64[:,:]),
        ('save_Ys', int64),
        ('g', int64),
        ('t_old_list_Y', float64[:]), 
        ('Y_minus_list', float64[:,:]),
        ('Y_plus_list', float64[:,:]),
        ('Y_iterator', int64),
        ('N_groups', int64),
        ('VDMD', int64),
        ('chi', float64),
        ('sigma_f', float64),
        ('nu', float64)

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
        self.ws = build.ws
        self.ws_quad = build.ws_quad
        self.tfinal = build.tfinal
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
        self.c = build.sigma_s/ build.sigma_t 
        print(self.c, 'c')
        self.particle_v = build.particle_v
        self.lumping = build.lumping
        
        self.radiative_transfer = build.thermal_couple
       
        self.c_a = build.sigma_a / build.sigma_t
        
        self.mean_free_time = 1/build.sigma_t
        self.division = 1000
        self.counter = 1000
        self.delta_tavg = 0.0
        self.l = build.l
        self.times_list = np.array([0.0])
        self.e_list = np.array([0.0])
        self.e_xs_list = np.array([0.0])
        self.wave_loc_list = np.array([0.0])
        self.save_derivative = build.save_wave_loc
        self.sigma_func = build.sigma_func
        self.deg_freedom = shaper(self.N_ang, self.N_space, self.M + 1, self.thermal_couple)
        self.alphams = np.zeros(self.N_ang -1)
        self.x0 = build.x0
        timepoints = 2048
        self.time_points = np.linspace(0.0, build.tfinal, timepoints)
        self.T_old = np.zeros((self.N_space, self.xs_quad.size))
        # self.alphas = np.zeros(self.N_ang-1)
        self.alphas = np.zeros(self.N_ang-1)
        # print(self.mus, 'mus')
        # print(self.ws, 'ws')
        self.make_alphas()
        self.old_percent_complete = 0.0
        self.stymie_count = 0
        self.N_groups = build.N_groups
        if build.thermal_couple['none'] == 1:
            self.index = -1
            self.Y_minus = np.zeros((self.N_groups, (self.N_ang) * self.N_space * (self.M+1)))
            self.Y_plus = self.Y_minus[0, :].copy()*0
        else:
            self.Y_minus = np.zeros((self.N_groups, (self.N_ang) * self.N_space * (self.M+1)))
            self.Y_plus = self.Y_minus[0, :].copy()*0
            self.index = -2
        self.edges_old = build.edges_init
        self.time_save_points = 100
        self.t_old_list = np.zeros(1)
        self.slope_limiter = False 
        print('### ### ### ### ### ###')
        print(self.slope_limiter, 'slope limiter')
        self.wavefront_estimator = 0.0
        self.g = 0
        self.t_old_list_Y = np.array([0.0])
        self.Y_iterator = 0
        self.Y_minus_list = np.zeros((1,(self.N_groups * self.N_ang) * self.N_space * (self.M+1)))
        self.Y_plus_list = np.zeros((1,(self.N_groups * self.N_ang) * self.N_space * (self.M+1)))
        self.VDMD = build.VDMD
        # self.VDMD = False
        self.chi = build.chi
        self.nu = build.nu
        self.sigma_f = build.sigma_f
        print(np.sum(self.ws), 'sum ws')
        
        
        
    def V_new_refl_enforce(self, V_new):
        res = V_new.copy()
        # for j in range(self.M+1):
        #     if j%2 != 0:
        #         res[:,0, j] = 0.0
        res[0,:,:] = res[1,:,:]
        return res

    def VDMD_func(self, t, V_old):
        """
        returns the vectors Y+ and Y-, which are the update and solution vector respectively. These are required to perform VDMD in post-processing. 
        Also, this function handles the integrator jumping around and taking negative timesteps by deleting data appropriately. 

        Args:
            t : evaluation time
            V_old : solution vector (before update)
        
        """
        # check for nonsequential evaluation
        if self.t_old_list_Y[-1] > t:
                    last_t = np.argmin(np.abs(self.t_old_list_Y-t))
                    if self.t_old_list_Y[self.Y_iterator] > t:
                        last_t -= 1
                    self.Y_iterator = last_t
                    self.Y_minus_list = self.Y_minus_list[:self.Y_iterator, :].copy()
                    self.Y_plus_list = self.Y_plus_list[:self.Y_iterator, :].copy()

                    # self.Y_minus_list[self.Y_iterator:, :] = 0.0
                    # self.Y_plus_list[self.Y_iterator:, :] = 0.0
                    # t_old_temp = np.zeros(last_t)
                    self.t_old_list_Y = self.t_old_list_Y[:last_t].copy()
                    # self.t_old_list_Y = t_old_temp.copy()
        # reshape solution matrix into a vector
        if self.radiative_transfer['none'] == False:
            res2 = V_old[:-1,:,:].copy().reshape((self.N_ang ) * self.N_space * (self.M+1))
        else:
            res2 = V_old[:,:,:].copy().reshape((self.N_ang ) * self.N_space * (self.M+1))

        # calculate Y+, Y-
        # It may be necessary to calculate Y+ outside of the loop or use the previous two time steps
        if self.t_old_list_Y.size >= 2: 
            dt = (self.t_old_list_Y[-1]-self.t_old_list_Y[-2])
            if dt < 0:
                raise ValueError('negative timestep')
            Y_minus_old = self.Y_minus_list[self.Y_iterator,:].copy().reshape((self.N_ang * self.N_groups, self.N_space, self.M+1))
            Y_minus_old_g = Y_minus_old[self.g * self.N_ang : (self.g+1) * self.N_ang, :, :].copy().reshape((self.N_ang * self.N_space * (self.M+1)))
            # self.Y_plus = ( res2- self.Y_minus[self.g,:])/dt
            self.Y_plus = ( res2- Y_minus_old_g)/dt

        else:
            self.Y_plus = self.Y_minus[self.g, :].copy()*0
        self.Y_minus[self.g,:] = res2.copy()

        list_length = self.Y_minus_list[:,0].size + 1

        # make new lists of vectors to hold the expanded Y-, Y+
        Y_minus_new = np.zeros((list_length,(self.N_groups * self.N_ang) * self.N_space * (self.M+1)))
        Y_plus_new = np.zeros((list_length,(self.N_groups * self.N_ang) * self.N_space * (self.M+1)))
        Y_minus_new[:-1] = self.Y_minus_list[:].copy()
        Y_plus_new[:-1] = self.Y_plus_list[:].copy()  
        self.Y_minus_list = np.copy(Y_minus_new)
        self.Y_plus_list = np.copy(Y_plus_new)

        # Append new Y_, Y- to the new lists of vectors
        Y_minus_temp = self.Y_minus_list.copy().reshape((list_length, self.N_groups * self.N_ang, self.N_space, self.M+1))
        Y_minus_temp[self.Y_iterator, self.g * self.N_ang:(self.g+1)*self.N_ang, :, :] = self.Y_minus[self.g, :].copy().reshape(((self.N_ang), self.N_space, self.M+1))
        Y_plus_temp = np.copy(self.Y_plus_list).reshape((list_length, self.N_groups * self.N_ang, self.N_space, self.M+1))
        Y_plus_temp[self.Y_iterator, self.g * self.N_ang:(self.g+1)*self.N_ang, :, :] = self.Y_plus.copy().reshape(((self.N_ang), self.N_space, self.M+1))

        # Reshape lists of vectors
        self.Y_minus_list = Y_minus_temp.copy().reshape((list_length, self.N_ang * self.N_groups * self.N_space * (self.M+1)))
        self.Y_plus_list = Y_plus_temp.copy().reshape((list_length, self.N_ang * self.N_groups * self.N_space * (self.M+1)))

        # if time is increasing, that means that we are out of the energy group loop and the iterator can advance
        if t > self.t_old_list_Y[-1]:
            self.Y_iterator += 1
            self.t_old_list_Y = np.append(self.t_old_list_Y, t)


    def make_alphas(self):
        """
        This function uses a recursion relation to calculate the alpha coefficients for the diamond difference angular derivative. 
        """
        # middle = int((self.N_ang)/2)
        self.alphas[0] = 0
        # for ia in range(1,self.N_ang):
        for ia in range(1, self.alphas.size):
            self.alphas[ia] = self.alphas[ia-1] - self.mus[ia] * self.ws[ia] * 2
        # print(self.alphas[:middle])
        # print(self.alphas[middle:])
        # self.alphas[:middle] = np.flip(self.alphas[middle:])
        print(self.alphas, 'alphas')



    def time_step_counter(self, t, mesh, V_old):
        delta_t = t - self.told
        # print(delta_t, 'dt in rhs')
        self.told = t
        
        self.delta_tavg += delta_t / self.division
        if self.counter == self.division:
            print('t = ', t, '|', 'delta_t average= ', self.delta_tavg)
            print(np.round((t/self.tfinal) * 100, 3), ' percent complete')
            if np.round((t/self.tfinal) * 100, 3)-self.old_percent_complete <= 0.001:
                self.stymie_count += 1
            # if self.stymie_count >= 15:
            #     raise ValueError('Solver stuck')
            self.old_percent_complete = np.round((t/self.tfinal) * 100, 3)
             
            
            print(self.N_space, 'spatial cells, ', self.M+1, ' basis functions ', self.N_ang, ' angles' )
            print(np.min(mesh.edges[1:]-mesh.edges[:-1]), 'min edge spacing')
            print(np.mean(mesh.edges[1:]-mesh.edges[:-1]), 'mean edge spacing')
            print(mesh.edges, 'edges')
            print(np.max(V_old), 'max u')
            print(np.min(V_old), 'min u')
            # if np.min(V_old) <= -1:
            #     raise ValueError('The solution is becoming too negative')

            third = int((self.N_space+1)/3)
            rest = int(self.N_space+1 - 2*third)
            
            # print(np.argmin(np.abs(V_old - np.min(V_old))), 'location of min')
            
            # print(np.min(V_old.copy().reshape((self.N_ang+1, self.N_space, self.M+1))[-1, :, :]), 'min e vec')
            dimensional_t = t/29.98
            # menis_t = -29.6255 + dimensional_t
            menis_t = converging_time_function(t, self.sigma_func)
            # rfront = 0.01 * (-menis_t) ** 0.679502 
            rfront = converging_r(menis_t, self.sigma_func)
 
            # print(np.min(np.abs(rfront-mesh.edges)), 'closest edge to rf')
            # tracker_edge = int(-third) 
            # print(tracker_edge)
            # print(np.abs(mesh.edges[tracker_edge]-rfront), ' abs diff of wavefront and tracker edge')
            # print(rfront, 'marshak wavefront location')
            # print(self.wavefront_estimator, 'wave loc estimate')
            if mesh.moving == True:
                tracker_edges = mesh.edges[third:third+rest]
                rf_in_tracker_region = tracker_edges[0] <rfront < tracker_edges[-1]           # if self.N_space <= 100:
            # print('is the wavefront in the tracking region?', rf_in_tracker_region)
            #     if self.geometry['sphere'] == True:
            #         print(mesh.edges/self.x0)
                # else:
                #     print(mesh.edges)
            # print(mesh.edges)
            print('--- --- --- --- --- --- --- --- --- --- --- --- --- ---')
            self.delta_tavg = 0.0
            self.counter = 0
        else:
            self.counter += 1
        self.told = t

    def slope_scale(self, V, edges, stop = False):
        floor = -1e-8#floor 1e-4 
        posfloor = floor
        theta = 0.0
        V_new = V.copy() 
        for k in range(self.N_space):
            h = math.sqrt(edges[k+1] - edges[k])
            edgeval = 1 / h / math.sqrt(math.pi)

            B_left0 = edgeval
            B_right0 = edgeval

            B_right1 = math.sqrt(2) * edgeval

            B_left1 = -B_right1

            for angle in range(self.N_ang+1):
                c0 = V[angle, k, 0]
                c1 = V[angle, k, 1]
                
                # if c0 ==0:
                #     V_new[angle,k,1] = 0.0
                if c0 > 0:
                    left_edge = (c0 * B_left0 + c1*B_left1)
                    right_edge = (c0 * B_right0 + c1 * B_right1)
                    if  left_edge < floor:
                        self.wavefront_estimator = (edges[k+1] + edges[k])/2
                        if stop == True and (c0 * B_left0 + c1*B_left1) < floor:
                            print(c0 * B_left0 + c1*B_left1, 'left')
                            print(c0 * B_right0 + c1 * B_right1, 'right')
                            print(c0, 'c0')
                            print(c1, 'c1')
                            print(V[angle, k,:])
                            assert 0
                        # print('left is negative')
                        # V_new[angle, k, 1] = -c0 * B_left0 / B_left1   
                        # posfloor = left_edge * theta 
                        V_new[angle, k, 1] = posfloor/B_left1 -c0 * (-1/math.sqrt(2))   
                        # print(c0 * B_left0 + V_new[angle, k, 1]*B_left1, 'new left solution')
                        # print(c0 * B_right0 + V_new[angle, k, 1]*B_right1, 'new right solution')

                    elif right_edge < floor:
                        # print('right is negative')
                        # V_new[angle, k, 1] = -c0 * B_right0 / B_right1
                        # posfloor = right_edge * theta
                        V_new[angle, k, 1] = posfloor/ B_right1 -c0 * (1/math.sqrt(2))  

                        # if (c0 * B_left0 + V_new[angle, k, 1]*B_left1) < 0:
                            # assert 0
                # elif c0 < 0:
                #     V_new[angle, k, 0] = 0
                #     V_new[angle, k, 1] = 0 
                    # print('negative c0')
        return V_new 


        
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
        if self.radiative_transfer['none'] == False :  
            V_new = V.copy().reshape((self.N_ang + 1, self.N_space, self.M+1))
            V_old = V_new.copy()
        else:
            V_new = V.copy().reshape((self.N_ang, self.N_space, self.M+1))
            V_old = V_new.copy()

        # enforce zero flux in origin cell
        for tangle in range(int(self.N_ang/2), self.N_ang):
                refl_index = self.N_ang-tangle-1
                            # print(self.mus[angle], self.mus[refl_index])
                assert(abs(self.mus[refl_index] - -self.mus[tangle])<=1e-10) 
                V_old[tangle, 0, :] = V_old[refl_index, 0, :]
        # check Pn moments
        for k in range(self.N_space):
            check_current_legendre(2 * self.ws, self.mus, V_old[:, 0, k], self.N_ang, int(2*self.N_ang-1))
            
        # V_old[0, :, 0] = V_old[1,:,0]
        # for j in range(1, self.M+1):
            # V_old[0, :, j] = (-1)**j * V_old[1,:,j]
        # for ang in range(self.N_ang+1): #attempt at positivizing T
        #     new_energy_vec = transfer_class.positivize_temperature_vector(V_old[ang,:,:], mesh.edges)
        #     V_old[ang,:,:] = new_energy_vec
        self.time_step_counter(t, mesh, V_old)
        # self.time_marching_func(t, mesh.told)
        if mesh.told <= t:
            self.edges_old = mesh.edges
        else:
            mesh.move(np.max((self.t_old_list < t)))
            self.edges_old = mesh.edges
        mesh.move(t)

        # if self.slope_limiter == True and self.M>0: # positivity fix
        #     V_old_new = self.slope_scale(V_old, mesh.edges)
        #     V_old = V_old_new
        if self.radiative_transfer['none'] == 0: # make temperature solution for T dependent cross sections
            self.T_old, self.T_eval_points = self.make_temp(V_old[-1,:,:], mesh, transfer_class)
        sigma_class.sigma_moments(mesh.edges, t, self.T_old, self.T_eval_points)
        flux.get_coeffs(sigma_class) # pass scattering cross section expansion coefficients to scalar flux class
        # sigma_class.check_sigma_coeffs(self.T_eval_points, mesh.edges, self.T_old)
        update = True
        # iterate over all cells
        for space in range(self.N_space): 
            # get mesh edges and edge derivatives          
            xR = mesh.edges[space+1]
            xL = mesh.edges[space]
            dxR = mesh.Dedges[space+1]
            dxL = mesh.Dedges[space]
            update = True
            if self.sigma_func['test4']== True: # special converging Marshak case
                menis_t = converging_time_function(t, self.sigma_func)
                rfront = converging_r(menis_t, self.sigma_func)
                if (xR < rfront - self.x0/4) and (rfront - self.x0/4 >0)   :
                    update = False
                else:
                    update = True
            # matrices.matrix_test(True) # tests matrices against analytic functions
            if update == False:
                V_new[:, space, :] = V_old[:, space, :] * 0
                assert 0 
            
            elif update == True:
                # u_old = make_u_old(V_old[0, :,:], self.edges_old, xL, xR, self.xs_quad, self.ws_quad, self.M) # projects psi back to the basis
                u_old = V_old[0, space, :]
                matrices.make_all_matrices(xL, xR, dxL, dxR)
                L = matrices.L # gradient matrix
                G = matrices.G # time derivative correction for moving mesh
                MPRIME = matrices.MPRIME # time derivative of mass matrix. Necessary because Mass is not orthonormal 
                if self.radiative_transfer['none'] == False:
                    flux.make_P(V_old[:-1,space,:], space, xL, xR)
                else:
                    flux.make_P(V_old[:,space,:], space, xL, xR)
                PV = flux.scalar_flux_term
                fixed_source = flux.P_fixed[space, self.g, :]
                source.make_source(t, xL, xR, uncollided_sol)
                S = source.S
                H = transfer_class.H
                if self.geometry['sphere'] == True:
                    Mass = matrices.Mass
                    J = matrices.J
                    if (self.lumping == True) and (self.M >0):
                        assert 0
                        # Mass, Minv = self.mass_lumper(Mass, True) 
                        Mass, Minv = mass_lumper(Mass, xL, xR)
                        # L, Linv = mass_lumper(L, xL, xR, invert = False)
                        # L = self.mass_lumper(L)
                        # G = self.mass_lumper(G)
                        # J = self.mass_lumper(J)
                        # MPRIME = self.mass_lumper(MPRIME)
                    else:
                        Minv = np.linalg.inv(Mass)

                    # VVs = matrices.VV
                # make P 
                
                # integrate the source
                # source.make_source(t, xL, xR, uncollided_sol)
                # radiative transfer term
                if self.radiative_transfer['none'] == False:
                    if self.g ==0:
                        transfer_class.make_H(xL, xR, V_old[-1, space, :], sigma_class, space)
                    H = transfer_class.H
                    # if self.lumping == True: #attempts at lumping transfer matrix
                    #     H = mass_lumper(H, xL, xR)[0]
                    #T_old saves the temperature at the zeros of the interpolating polynomial
                    # time_loc = np.argmin(np.abs(self.time_points - t))
                    # self.T_old[time_loc, space] = transfer_class.make_T(argument, a, b) 
                    ######### solve thermal couple ############
                    U = V_old[-1,space,:]
                    num_flux.make_LU(t, mesh, V_old[-1,:,:], space, 0.0, V_old[-1, 0, :], True)
                    RU = num_flux.LU 
                    RHS_transfer = np.copy(V_old[-1, space, :]*0)
                    if self.uncollided == True:
                        RHS_transfer += self.c_a *source.S * 2 
                    RHS_transfer -= RU
                    RHS_transfer += -np.dot(MPRIME, U) + np.dot(G,U) - self.c_a *H /self.sigma_t
                    RHS_transfer += self.c_a * PV*2 /self.sigma_t 
                    RHS_transfer = np.dot(RHS_transfer, Minv)
                    if self.l != 1.0:
                        RHS_transfer = RHS_transfer / self.l
                    V_new[-1,space,:] = RHS_transfer 
                    # not changing cell if in equilibrium
                    # print(RHS_transfer, 'rhs transfer')
                    # if (np.abs(self.c_a * PV*2 /self.sigma_t - self.c_a *H /self.sigma_t)<=1e-8).all():
                    #     V_new[-1,space,:] = RHS_transfer *0 
                    if np.isnan(V_new[-1, space, :]).any():
                        print('rhstransfer is nan')
                        assert(0)
                ########## Starting direction #########
                # psionehalf = V_old[0, space, :] # should this be make_u_old_func?
                psionehalf = u_old
                ########## Loop over angle ############
                for angle in range(self.N_ang):
                    # psin = make_u_old(V_old[angle, :,:], self.edges_old, xL, xR, self.xs_quad, self.ws_quad, self.M) # projects psi back to the basis
                    psin = V_old[angle, space, :]
                    mul = self.mus[angle]
                    # calculate numerical flux
                    refl_index = 0
                    if space == 0:
                        # if abs(xL) <= 1e-8:
                            if self.mus[angle] > 0:
                                refl_index = self.N_ang-angle-1
                                # print(self.mus[angle], self.mus[refl_index])
                                assert(abs(self.mus[refl_index] - -self.mus[angle])<=1e-10) 
                    num_flux.make_LU(t, mesh, V_old[angle,:,:], space, mul, V_old[refl_index, 0, :])
                    # new r=0 BC # not sure what the reasoning was behind this
                    # num_flux.make_LU(t, mesh, V_old[angle,:,:], space, mul, psionehalf)
                    LU = num_flux.LU 
                    # Get absorption term
                    # sigma_class.sigma_moments(mesh.edges, t, self.T_old, V_old[-1, :, :])
                    sigma_class.make_vectors(mesh.edges, V_old[angle,space,:], space)
                    VV = sigma_class.VV
                    # Initialize solution vector, RHS
                    U = np.zeros(self.M+1).transpose()
                    U[:] = V_old[angle,space,:]
                    # assert((np.abs(U-VV/self.sigma_t) < 1e-6).all())
                    dterm = U.copy()*0
                    mu_derivative = U*0
                    # if angle > 0 and angle != self.N_ang-1:
                    mu_derivative = legendre_difference(self.ws, self.N_ang, int(2*self.N_ang-1), V_old[:, space, :], J, self.M, self.mus, self.mus[angle])
                    #     for j in range(self.M+1):
                    # #         # dterm[j] = finite_diff_uneven_diamond_2(self.mus, angle, V_old[:, space, j], self.alphams, self.ws, left = (angle==0), right = (angle == self.N_ang-1))
                    # #         # dterm[j] = finite_diff_uneven_diamond(self.mus, angle, V_old[:-1, space, j], left = (angle==0), right = (angle == self.N_ang-1), origin = False)
                    #         dterm[j] = alpha_difference(self.alphas[angle], self.alphas[angle-1], self.ws[angle],  psionehalf[j], psin[j])
                    # for j in range(self.M+1):
                    #     vec = (1-self.mus**2) * V_old[:, space, j]
                    #     dterm[j] = finite_diff_uneven(self.mus, angle, vec, left = (angle==0), right = (angle == self.N_ang - 1))
                    if self.geometry['sphere'] == True:  
                    
                        RHS = V_old[angle, space, :]*0
                        RHS -=  LU # numerical flux 
                        RHS +=  mul*np.dot(L,U) #gradient
                        # mu_derivative =  np.dot(J, dterm) 
                        RHS -= mu_derivative # angular derivative
                        RHS += np.dot(G, U) # moving mesh time derivative correction
                        RHS += 0.5 * S /self.sigma_t / self.l # source
                        RHS +=  self.c_a * H * 0.5 / self.sigma_t / self.l # radiative transfer coupling
                        RHS += PV * self.c /self.sigma_t / self.l # scattering
                        RHS += fixed_source * self.sigma_f * self.nu * self.chi / self.sigma_t # fixed fission source
                        RHS -= VV / self.sigma_t / self.l # absorption
                        RHS -= np.dot(MPRIME, U) # time derivative of mass matrix
                        RHS = np.dot(Minv, RHS) # mass matrix 
                        V_new[angle,space,:] = RHS

                        if angle == 0:
                            psionehalf = u_old 
                        else:  
                            # psionehalf_new = 2 * V_old[angle, space,:] - psionehalf
                            psionehalf_new = 2 * psin - psionehalf
                            psionehalf = psionehalf_new
        # V_new = self.V_new_refl_enforce(V_new)
        if self.radiative_transfer['none'] == False:
            # V_new = self.V_new_floor_func(V_new) # This was an attempt at enforcing positivity
            res = V_new.reshape((self.N_ang + 1) * self.N_space * (self.M+1))
            return res
        

        else:
            return V_new.reshape((self.N_ang) * self.N_space * (self.M+1))
        
    

    def V_new_floor_func(self, V_new):
        floor = 1e-16
        for ang in range(self.N_ang + 1):
            for space in range(self.N_space):
                for j in range(self.M+1):
                    if abs(V_new[ang, space, j])<=floor:
                        V_new[ang, space, j] = floor * np.sign(V_new[ang, space, j])
        return V_new
    
    def make_temp(self, e_vec, mesh, rad_transfer):
        # if self.lumping == True and 0 ==1:
        #     T_vec = np.zeros((self.N_space, 2))
        #     T_eval_points = np.zeros((self.N_space, 2))
        #     for space in range(self.N_space):
        #         xR = mesh.edges[space+1]
        #         xL = mesh.edges[space]
        #         rad_transfer.e_vec = e_vec[space,:]
        #         T_vec[space] = rad_transfer.make_T(np.array([xL,xR]), xL, xR)
        #         T_eval_points[space] = np.array([xL,xR])
        # else:
            T_vec = np.zeros((self.N_space, self.xs_quad.size))
            T_eval_points = np.zeros((self.N_space, self.xs_quad.size))
            for space in range(self.N_space):
                xR = mesh.edges[space+1]
                xL = mesh.edges[space]
                rad_transfer.e_vec = e_vec[space,:]
                a = xL
                b = xR
                argument = (b-a)/2*self.xs_quad + (a+b)/2
                # print(argument, 'arg in make temp')
                # argument2 = (b-a)/2*self.t_quad + (a+b)/2
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
                # if np.isnan(T_eval_points.any()):
                #     assert(0)
                # elif T_vec.any() <0:
                #     # raise ValueError('negative temperature')
                #     assert(0)
                # if (T_vec[space]).any() <0:
                # T_vec[space] = np.mean(T_vec[space]) + T_vec[space] * 0

            # for space in range(self.N_space):
            #     for j in range(self.xs_quad.size):
            #         if T_vec[space, j] <0:
            #             print(T_vec[space, j])
            #             assert(0)
            # print('## ## ## ## ## ## ')
            return T_vec, T_eval_points
    def time_marching_func(self, t, told):
        if t > told:
            if self.t_old_list.size <  self.time_save_points:
                self.t_old_list = np.append(self.t_old_list, t)
            else:
                temp = np.zeros(self.time_save_points)

                temp[0:self.time_save_points-1] = self.t_old_list[1:]
                temp[-1] = t
                self.t_old_list = temp


