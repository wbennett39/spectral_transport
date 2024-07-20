
                
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:42:11 2022

@author: bennett
"""

from .build_problem import build
from .functions import normPn, normTn

#from build_problem import build
#from functions import normPn, normTn

from numba.experimental import jitclass
from numba import int64, float64, deferred_type, prange
import numpy as np
import math
from numba import types, typed
import numba as nb
from .opacity import sigma_integrator
 
build_type = deferred_type()
build_type.define(build.class_type.instance_type)

sigma_class_type = deferred_type()
sigma_class_type.define(sigma_integrator.class_type.instance_type)
kv_ty = (types.int64, types.unicode_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))




data = [('temp_function', int64[:]),
        ('e_vec', float64[:]),
        ('e', float64),
        ('H', float64[:]),
        ('alpha', float64),
        ('a', float64),
        ('M', int64),
        ("xs_quad", float64[:]),
        ("ws_quad", float64[:]),
        ("T", float64[:]),
        ('cv0', float64),
        ('fudge_factor', float64[:]),
        ('clight', float64),
        ('test_dimensional_rhs', int64),
        ('save_derivative', int64),
        ('xs_points', float64[:]),
        ('e_points', float64[:]),
        ('thermal_couple', nb.typeof(params_default)),
        ('geometry', nb.typeof(params_default)),
        ('temperature', float64[:,:]),
        ('space', int64),
        ('sigma_a_vec', float64[:]),
        ('a2', float64)



        ]
###############################################################################

@jitclass(data)
class T_function(object):
    def __init__(self, build):
        self.temp_function = np.array(list(build.temp_function), dtype = np.int64) 
        self.H = np.zeros(build.M+1).transpose()
        self.M = build.M
        
        self.a = 0.0137225 # GJ/cm$^3$/keV$^4
        self.alpha = 4 * self.a
        self.clight = 29.98 # cm/ns
        self.a2 = 5.67e-5 # in ergs

        self.geometry = build.geometry
        
        if self.geometry['slab'] == True:
            self.xs_quad = build.xs_quad
            self.ws_quad = build.ws_quad
        
        if self.geometry['sphere'] == True:
            self.xs_quad = build.t_quad
            self.ws_quad = build.t_ws

        self.cv0 = build.cv0 / self.a 
        if (self.cv0) != 0.0:
            print('cv0 is ', self.cv0)
        self.test_dimensional_rhs = False
        self.save_derivative = build.save_wave_loc
        self.thermal_couple = build.thermal_couple
        self.temperature = np.zeros((build.N_space, self.xs_quad.size))
        self.e_vec = np.zeros(self.M+1)
        
    def make_e(self, xs, a, b):
        temp = xs*0
        for ix in range(xs.size):
            for j in range(self.M+1):
                if self.geometry['slab'] == True:
                    temp[ix] += normPn(j, xs[ix:ix+1], a, b)[0] * self.e_vec[j] 
                elif self.geometry['sphere'] == True:
                    temp[ix] += normTn(j, xs[ix:ix+1], a, b)[0] * self.e_vec[j]
        return temp 
    
    def integrate_quad(self, a, b, j, sigma_class):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        self.H[j] = (b-a)/2 * np.sum(self.ws_quad * self.T_func(argument, a, b, sigma_class, self.space) * normPn(j, argument, a, b))


    def integrate_quad_sphere(self, a, b, j, sigma_class):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        # self.H[j] = 0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b) * 1 * normTn(j, argument, a, b))
        self.H[j] =  0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b, sigma_class, self.space) * 1 * normTn(j, argument, a, b))
        #assert(0)
    
    def get_sigma_a_vec(self, x, sigma_class, temperature):
        self.sigma_a_vec = sigma_class.sigma_function(x, 0.0, temperature)

    def make_T(self, argument, a, b):
        e = self.make_e(argument, a, b)
        if self.temp_function[0] == 1:
            T = self.su_olson_source(e, argument, a, b)
        elif self.temp_function[1] == 1:
             T =  e / self.cv0
        elif self.temp_function[2] == 1:
            #  if np.max(e) <= 1e-20:
            #     T = np.zeros(e.size) + 1e-12
            #  else:
                T = self.meni_eos(e)
                # T = self.su_olson_source(e, argument, a, b)
                
                # T = e / 0.1
                if np.isnan(T).any() or np.isinf(T).any() :
                    print('###                                ###')
                    print('nonreal temperature')
                    print(a,b, 'edges')
                    print(e, 'e solution')
                    print(self.e_vec, 'e vector')
                    print(a, b, 'edges')
                    print('###                                ###')
                    assert(0)
                # print(T)
        return T
             

    def T_func(self, argument, a, b, sigma_class, space):
        T = self.make_T(argument, a, b)
        self.get_sigma_a_vec(argument, sigma_class, T)
        # self.xs_points = argument
        # self.e_points = e

        if self.temp_function[0] == 1:
            self.temperature[space,:] = T
            return  np.power(T,4) * self.fudge_factor  * self.sigma_a_vec
            #return np.sin(argument) 

        elif self.temp_function[1] == 1:
        
                self.temperature[space,:] = T
                return np.power(T,4)  * self.sigma_a_vec 
        
        # elif self.temp_function[2] == 1:
        #     return f * T ** beta * rho ** (1-mu)- self.a * T**4

        elif self.temp_function[2] == 1:
            # print(self.sigma_a_vec)
            self.temperature[space,:] = T
            return  np.power(T,4) * np.abs(self.sigma_a_vec) * self.fudge_factor
        
            

        else:
            assert(0)


        
    def su_olson_source(self, e, x, a, b):
        
        self.fudge_factor = np.ones(e.size)
    
        for count in range(e.size):
            if math.isnan(e[count]) == True:
                            print("nan")
                            print(e)
                            assert 0     
            elif (e[count]) < 0.:
                self.fudge_factor[count] = -1.


        t1 = np.abs(4*e/self.alpha)
        return np.power(self.a*t1,0.25)
    
    def meni_eos(self, e):
        
        self.fudge_factor = np.ones(e.size)
    
        for count in range(e.size):
            # if math.isnan(e[count]) == True:
            #                 print("nan")
            #                 print(e)
            #                 assert 0     
            if (e[count]) < 0.:
                self.fudge_factor[count] = -1.0

        # dimensional e in GJ/cm^3
        ee = e * self.a  / 10**-3 * 0.1**1.6
        T1 = (np.abs(ee))
        # self.alpha = 10**-3
        t1 = np.abs(4*e*self.a/self.alpha)
        # return np.power(t1,0.25) 
        return np.power(T1, 0.625) 
        
        
    def make_H(self, xL, xR, e_vec, sigma_class, space):
        self.e_vec = e_vec
        self.space = space
            
        # Lines commented out are the original lines of code

        #for j in range(self.M+1):
            #self.integrate_quad(xL, xR, j)

        if self.thermal_couple['none'] != True:

            if self.geometry['slab'] == True:

                for j in range(self.M+1):
                    self.integrate_quad(xL, xR, j, sigma_class)

            elif self.geometry['sphere'] == True:

                for j in range(self.M+1):
                    self.integrate_quad_sphere(xL, xR, j, sigma_class)
        
