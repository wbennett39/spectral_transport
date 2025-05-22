#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 07:24:05 2022

@author: William Bennett
"""
import numpy as np
from numba import int64, float64, jit, njit, deferred_type
from numba.experimental import jitclass
from numba import types, typed
# from main import IC_func 
#from mesh import mesh_class
from .mesh import mesh_class
#from functions import normPn, normTn
from .functions import normPn, normTn
from .mutables import IC_func
from .functions import weight_func_Tn
# from mutables import IC_func
# from functions import weight_func_Tn

import yaml
from pathlib import Path
import numba as nb

###############################################################################
mesh_class_type = deferred_type()
mesh_class_type.define(mesh_class.class_type.instance_type)
IC_func_type = deferred_type()
IC_func_type.define(IC_func.class_type.instance_type)
kv_ty = (types.int64, types.unicode_type)
# Explicitly define the types of the key and value:
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))

data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('tfinal', float64),
        ('sigma_t', float64),
        ('sigma_s', float64),
        ('IC', float64[:,:,:]),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('xs_quad', float64[:]),
        ('ws_quad', float64[:]),
        ('x0', float64),
        ('t0', float64),
        ("source_type", int64[:]),
        ("uncollided", int64),
        ("moving", int64),
        ("move_type", int64[:]),
        ("argument", float64[:]),
        ("temp", float64),
        ("t_quad", float64[:]),
        ("t_ws", float64[:]),
        ('scattering_ratio', float64),
        ('thermal_couple', nb.typeof(params_default)),
        ('temp_function', int64[:]),
        ('e_init', float64),
        ('sigma', float64),
        ('particle_v', float64),
        ('edge_v', float64),
        ('cv0', float64),
        ('thick', int64),
        ('wave_loc_array', float64[:,:,:]),
        ('source_strength', float64),
        ('move_factor', float64),
        ('sigma_a', float64),
        ('l', float64),
        ('save_wave_loc', int64),
        ('pad', float64),
        ('leader_pad', float64),
        ('quad_thick_source', float64[:]),
        ('quad_thick_edge', float64[:]),
        ('boundary_on', int64[:]), 
        ('boundary_source_strength', float64),
        ('boundary_source', int64),
        # ('sigma_func', int64[:]),
        ('sigma_func', nb.typeof(params_default)),
        ('Msigma', int64),
        ('domain_width', float64),
        ('finite_domain', int64),
        ('fake_sedov_v0', float64),
        ('x01', float64),
        ('test_dimensional_rhs', int64),
        ('epsilon', float64), 
        ('geometry', nb.typeof(params_default)),
        ('boundary_temp', float64[:]),
        ('boundary_time', float64[:]),
        ('lumping', int64),
        ('T4', float64[:]),
        ('edges_init', float64[:]),
        ('N_groups', int64),
        ('VDMD', int64),
        ('shift', float64),
        ('fixed_source_coeffs', float64[:,:,:]),
        ('chi', float64),
        ('nu', float64[:]),
        ('sigma_f', float64[:]),
        ('legendre_moments', int64),
        ('angular_derivative', nb.typeof(params_default) )
        ]
###############################################################################

@jitclass(data)
class build(object):
    def __init__(self, N_ang, N_space, M, N_groups, tfinal, x0, t0, mus, ws, xs_quad, ws_quad, sigma_t, sigma_s, 
    source_type, uncollided, moving, move_type, t_quad, t_ws, thermal_couple, temp_function, e_initial, sigma, particle_v, 
    edge_v, cv0, thick, wave_loc_array, source_strength, move_factor, l, save_wave_loc, pad, leader_pad, quad_thick_source,
    quad_thick_edge, boundary_on, boundary_source_strength, boundary_source, sigma_func, Msigma, finite_domain, domain_width, 
    fake_sedov_v0, test_dimensional_rhs, epsilon, geometry, lumping, VDMD, fixed_source_coeffs, chi, nu, sigma_f, legendre_moments,
    angular_derivative):
        self.N_ang = N_ang
        print(self.N_ang, 'angles')
        self.N_space = N_space
        self.M = M
        self.lumping = lumping
        self.tfinal = tfinal
        self.VDMD = VDMD
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.N_groups = N_groups
        self.sigma_a = sigma_t-sigma_s
        self.scattering_ratio = self.sigma_s / self.sigma_t
        self.mus = mus
        self.ws = ws/np.sum(ws)
        self.xs_quad = xs_quad
        self.ws_quad = ws_quad
        self.x0 = x0
        self.legendre_moments = legendre_moments
        self.source_type = np.array(list(source_type), dtype = np.int64)
        self.uncollided = uncollided 
        self.moving = moving
        self.move_type = np.array(list(move_type), dtype = np.int64)
        self.t_quad = t_quad
        self.t_ws = t_ws
        self.t0 = t0
        self.sigma_func = sigma_func
        self.angular_derivative = angular_derivative
        self.test_dimensional_rhs = test_dimensional_rhs
        self.thermal_couple = thermal_couple
        self.temp_function = np.array(list(temp_function), dtype = np.int64)
        self.sigma = sigma
        self.particle_v = particle_v
        self.edge_v = edge_v
        self.cv0 = cv0
        self.thick = thick
        self.wave_loc_array = wave_loc_array
        self.source_strength = source_strength
        self.move_factor = move_factor
        self.l = l
        self.save_wave_loc = save_wave_loc
        self.pad = pad
        self.leader_pad = leader_pad
        self.quad_thick_source = quad_thick_source
        self.quad_thick_edge = quad_thick_edge
        self.boundary_on = np.array(list(boundary_on), dtype = np.int64)
        self.boundary_source = boundary_source
        self.boundary_source_strength = boundary_source_strength
        self.Msigma = Msigma
        self.finite_domain = finite_domain
        self.domain_width = domain_width
        self.fake_sedov_v0 = fake_sedov_v0
    

        
        if self.thermal_couple['none'] == 1:
            self.IC = np.zeros((N_ang * self.N_groups, N_space, M+1))
        elif self.thermal_couple['none'] != 1:
            self.IC = np.zeros((N_ang * self.N_groups + 1, N_space, M+1))
        self.epsilon = epsilon
        self.geometry = geometry
       
        self.e_init = e_initial
        self.T4 = np.zeros(self.xs_quad.size * self.N_space)
        self.shift = 0.0
        if self.source_type[16] == 1:
            self.fixed_source_coeffs = fixed_source_coeffs
        else:
            self.fixed_source_coeffs = np.zeros((self.N_ang, self.N_space, self.M+1))
        self.chi = chi
        

        self.sigma_f =np.ones(self.N_space) * sigma_f
        self.nu = np.ones(self.N_space) * sigma_f
        # print(self.randomstart)
        # assert 0

        # self.e_initial = 1e-4
        
        
    def integrate_quad(self, a, b, ang, space, j, ic):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        mu = self.mus[ang]
        self.IC[ang,space,j] = 0.5 * (b-a) * np.sum(self.ws_quad * ic.function(argument, mu) * normPn(j, argument, a, b))
    
    def grab_converging_boundary_data(self, boundary_temp, boundary_time):
        self.boundary_temp = boundary_temp
        self.boundary_time = boundary_time

    def integrate_quad_sphere(self, a, b, ang, space, j, g,  ic):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        mu = self.mus[ang]
        self.IC[ang + g * self.N_ang, space,j] = 0.5 * (b-a) * np.sum(self.ws_quad * ic.function(argument, mu, iarg = space * self.xs_quad.size, earg = (space+1)*self.xs_quad.size) * 2.0 * normTn(j, argument, a, b))
        
    def make_T4_IC(self, RT_class, edges):
        for space in range(self.N_space):
            for j in range(self.M+1):

                a = edges[space]
                b = edges[space+1]

                self.integrate_e_sphere(a, b, space, j)
                RT_class.e_vec[j] = self.IC[self.N_ang * self.N_groups ,space,j]

            argument = (b-a)/2*self.xs_quad + (a+b)/2
            T = RT_class.make_T(argument, a, b)
            self.T4[space * self.xs_quad.size:(space+1) * self.xs_quad.size] = T**4
        # print(self.T4, 'T4')




    def integrate_e(self, a, b, space, j):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        self.IC[self.N_ang * self.N_groups, space,j] = (b-a)/2 * np.sum(self.ws_quad * self.IC_e_func(argument) * normPn(j, argument, a, b))
    
    def integrate_e_sphere(self, a, b, space, j):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        self.IC[self.N_ang * self.N_groups,space,j] = (b-a)/2 * np.sum(self.ws_quad * self.IC_e_func(argument) *2.0* normTn(j, argument, a, b))
    
    def IC_e_func(self,x):
        return np.ones(x.size) * self.e_init
                
    def make_IC(self, edges, randomstart):
        if self.sigma_func['Kornreich'] == True:
            self.sigma_f = np.zeros(self.N_space)
            self.nu = np.zeros(self.N_space)
            for space in range(1, self.N_space):
                if edges[space] <  -3.5 and edges[space-1] > -4.5 or edges[space-1] > 3.5:
                    self.sigma_f[space] = 0.3
                    self.nu[space] = 1


        # edges = mesh_class(self.N_space, self.x0, self.tfinal, self.moving, self.move_type, self.source_type, 
        # self.edge_v, self.thick, self.move_factor, self.wave_loc_array, self.pad,  self.leader_pad, self.quad_thick_source, 
        # self.quad_thick_edge, self.finite_domain, self.domain_width, self.fake_sedov_v0, self.boundary_on, self.t0, self.geometry, self.sigma_func)
        edges_init = edges
        self.edges_init = edges_init
        # as of now, only constant IC's are posible with Radiative transfer 
        if self.thermal_couple['none'] != 1:
            for space in range(self.N_space):
                for j in range(self.M + 1):
                    if self.geometry['slab'] == True:
                        self.integrate_e(edges_init[space], edges_init[space+1], space, j)
                    elif self.geometry['sphere'] == True:
                        self.integrate_e_sphere(edges_init[space], edges_init[space+1], space, j)
            
            ic = IC_func(self.source_type, self.uncollided, self.x0, self.source_strength, self.sigma, 0.0, self.geometry, True, self.T4, randomstart)

            for g in range(self.N_groups):
                for ang in range(self.N_ang):
                    for space in range(self.N_space):
                        for j in range(self.M+1):
                            self.integrate_quad_sphere(edges_init[space], edges_init[space+1], ang, space, j, g, ic)
            # print(self.T4, 'T4')
            # print(self.IC, 'IC')
         



            
        else:

            # The current method for handling delta functions
            if self.moving == False and self.source_type[0] == 1 and self.uncollided == False and self.N_space%2 == 0:
                if self.geometry['slab'] == True:
                    right_edge_index = int(self.N_space/2 + 1)
                    left_edge_index = int(self.N_space/2 - 1)
                    self.x0 = edges_init[right_edge_index] - edges_init[left_edge_index]
                # temp = (edges_init[int(self.N_space/2 + 1)] - edges_init[self.N_space/2 - 1]) 
                elif self.geometry['sphere'] == True:
                    self.x0 = edges_init[1] - edges_init[0]
                    print(self.x0, 'x0')
                



            if self.moving == False and self.source_type[0] == 2 and self.uncollided == False and self.N_space%2 == 0:
                i = 0
                it = 0
                while i==0:
                    if edges_init[it] <= -self.x0 <= edges_init[it+1]:
                        i = 1
                    else:
                        it += 1
                    if it > edges_init.size:
                        assert(0)

                x1 = edges_init[int(self.N_space/2)] - edges_init[int(self.N_space/2)+1]
                ic = IC_func(self.source_type, self.uncollided, self.x0, self.source_strength, self.sigma, x1, self.geometry, False, self.T4, False)
            
            else:
                ic = IC_func(self.source_type, self.uncollided, self.x0, self.source_strength, self.sigma, 0.0, self.geometry, False, self.T4, randomstart)

            for g in range(self.N_groups):
                for ang in range(self.N_ang):
                    for space in range(self.N_space):
                        for j in range(self.M + 1):
                            if self.geometry['slab'] == True:
                                print("Inside slab if statement.")
                                self.integrate_quad(edges_init[space], edges_init[space+1], ang, space, j, ic)
                            elif self.geometry['sphere'] == True:
                                self.integrate_quad_sphere(edges_init[space], edges_init[space+1], ang, space, j,g, ic)

    
        
        

        
        
