#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:38:34 2022

@author: bennett
"""
import numpy as np
import math
from numba import float64, int64, deferred_type
from numba.experimental import jitclass

from .build_problem import build
from .functions import normPn, normTn
from .functions import numba_expi as expi
from .uncollided_solutions import uncollided_solution
from scipy.special import expi as expi2
from numba import types, typed
import numba as nb
###############################################################################
build_type = deferred_type()
build_type.define(build.class_type.instance_type)
uncollided_solution_type = deferred_type()
uncollided_solution_type.define(uncollided_solution.class_type.instance_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))

data = [("S", float64[:]),
        ("source_type", int64[:]),
        ("uncollided", int64),
        ("moving", int64),
        ("M", int64),
        ("x0", float64),
        ("t", float64),
        ("xL", float64),
        ("xR", float64),
        ("argument", float64[:]),
        ("source_vector", float64[:]),
        ("temp", float64[:]),
        ("abxx", float64),
        ("xx", float64),
        ("ix", int64),
        ("xs_quad", float64[:]),
        ("ws_quad", float64[:]),
        ("mag", float64),
        ("term1", float64),
        ("term2", float64),
        ("tfinal", float64),
        ("t0", float64),
        ("t1", float64),
        ("t2", float64), 
        ("t3", float64),
        ("tau", float64),
        ("sigma", float64),
        ('source_strength', float64),
        ('sigma_s', float64),
        ('geometry', nb.typeof(params_default)),
        ('g', int64),
        ('shift', float64)
        
        ]
###############################################################################
@jitclass(data)
class source_class(object):
    def __init__(self, build):
        self.S = np.zeros(build.M+1).transpose()
        self.source_type = np.array(list(build.source_type), dtype = np.int64) 
        self.uncollided = build.uncollided
        self.x0 = build.x0
        self.M = build.M
        self.xs_quad = build.xs_quad
        self.ws_quad = build.ws_quad
        self.moving = build.moving
        self.tfinal = build.tfinal
        self.t0 = build.t0
        self.sigma = build.sigma
        self.sigma_s = build.sigma_s
        # self.source_strength = 0.0137225 * 299.98
        self.source_strength = build.source_strength
        self.geometry = build.geometry
        self.shift = 0.0
    
    def integrate_quad(self, t, a, b, j, func):
        a = 1
        # argument = (b-a)/2 * self.xs_quad + (a+b)/2
        # self.S[j] = (b-a)/2 * np.sum(self.ws_quad * func(argument, t) * normPn(j, argument, a, b))

    
    def integrate_quad_sphere(self, t, a, b, j, func):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        #self.S[j] = 0.5 * (b-a) * np.sum((argument**2) * np.sqrt(1- self.xs_quad**2) * self.ws_quad * func(argument, t) * 1 * normTn(j, argument, a, b))
        self.S[j] = 0.5 * (b-a) * np.sum( (argument**2) * self.ws_quad * func(argument, t) * normTn(j, argument, a, b))
        #self.S[j] = 0.5 * j  # Testing simple function
        #print("self.S[j] inside integrate_quad_sphere function: ", self.S[j])
        #if self.S[j] != 0:
            #print("Non zero self.S[j] inside integrate_quad_sphere function: ", self.S[j])


    def integrate_quad_not_isotropic(self, t, a, b, j, mu, func):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        self.S[j] = (b-a)/2 * np.sum(self.ws_quad * func(argument, t, mu) * normPn(j, argument, a, b))
    
    def MMS_source(self, xs, t, mu):
        temp = xs*0
        for ix in range(xs.size):
            if -t - self.x0 <= xs[ix] <= t + self.x0:
                # temp[ix] = - math.exp(-xs[ix]*xs[ix]/2)*(1 + (1+t)*xs[ix]*mu)/((1+t)**2)/2
                temp[ix] = -0.5*(1 + (1 + t)*xs[ix]*mu)/(math.exp(xs[ix]**2/2.)*(1 + t)**2)
        return temp*2.0
    
    def square_source(self, xs, t):
        temp = xs*0

        for ix in range(xs.size):
            if abs(xs[ix]-self.shift) <= self.x0:# and (t <self.t0):
        #     if ((abs(xs[ix]) - 510) < self.x0) and (t < self.t0):
                temp[ix] = 1.0
        if self.geometry['slab'] == True:
            return temp
        elif self.geometry['sphere'] == True:
            return temp/(4*np.pi*self.x0**3) 
            
    def gaussian_source(self, xs, t):
        temp = xs*0
        for ix in range(xs.size):
            x = xs[ix]
            if t <= self.t0:
                temp[ix] = math.exp(-x*x/self.sigma**2)
        return temp
        
        
    def make_source(self, t, xL, xR, uncollided_solution):
        if self.geometry['slab'] == True:
            if self.uncollided == True:
                if (self.source_type[0] == 1) and  (self.moving == True):
                        self.S[0] = uncollided_solution.plane_IC_uncollided_solution_integrated(t, xL, xR)
                else:
                    for j in range(self.M+1):
                        self.integrate_quad(t, xL, xR, j, uncollided_solution.uncollided_solution)
                self.S = self.S * self.sigma_s
            elif self.uncollided == False:
                if self.source_type[2] == 1:
                    for j in range(self.M+1):
                        self.integrate_quad(t, xL, xR, j, self.square_source)
                elif self.source_type[5] == 1:
                    for j in range(self.M+1):
                        self.integrate_quad(t, xL, xR, j, self.gaussian_source)
        
        elif self.geometry['sphere'] == True:
            # print("In spherical if statement.") # This statement is evaluating to true
            if self.uncollided == True:
                if (self.source_type[1] == 1) or (self.source_type[2] == 1):
                 
                    for j in range(self.M+1):
                        self.integrate_quad_sphere(t, xL, xR, j, uncollided_solution.uncollided_solution)
            elif self.uncollided == False:
                #print("In uncollided off if statement.") # This if statement is evaluating to true.
                if self.source_type[2] == 1:
                    #print("In square source if statement.")  
                    for j in range(self.M+1):
                        # assert 0
                        self.integrate_quad_sphere(t, xL, xR, j, self.square_source)
                        #print("Integrate_quad_sphere function is returning ", self.integrate_quad_sphere(t, xL, xR, j, self.square_source))
                elif self.source_type[5] == 1:
                    for j in range(self.M+1):
                        self.integrate_quad_sphere(t, xL, xR, j, self.gaussian_source)

        #print(self.integrate_quad_sphere(t, xL, xR, j, self.square_source).type)
                

                    
                # if self.source_type[0] == 1:
                #     for j in range(self.M+1):
                #         if (xL <= t <= xR):
                #             t = t + 1e-10
                #             self.S[j] = math.exp(-t)/4/math.pi/t * normTn(j, np.array([t]), xL, xR)[0]                

        self.S = self.S * self.source_strength
        #if ( (self.S[0] != 0) or (self.S[1] != 0)):
        #    print(self.S)s

    # def test_square_source(xL, xR):
    #     """Used to test if the solution obtained for the square source without the uncollided treatment is 
    #     correct."""

    #     return (math.sqrt(1/(-xL + xR))*(-0.3333333333333333*xL**3 + xR**3/3.))/math.sqrt(math.pi) 

    def make_source_not_isotropic(self, t, mu, xL, xR):
            if self.source_type[4] ==1:
                for j in range(self.M+1):
                    self.integrate_quad_not_isotropic(t, xL, xR, j, mu, self.MMS_source)
                

        
        

            
