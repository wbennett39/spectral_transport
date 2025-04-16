 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:34:07 2022

@author: bennett
"""
from numba import njit, jit, int64, float64
from numba.experimental import jitclass
import numpy as np
import math
from numba import types, typed
import numba as nb




###############################################################################
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))
data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('tfinal', float64),
        ('IC', float64[:,:,:]),
        ('x0', float64),
        ('source', int64[:]),
        ("source_type", int64[:]),
        ("uncollided", int64),
        ('x', float64[:]),
        ('source_strength', float64),
        ('sigma', float64),
        ('x1', float64),
        ('mu', float64),
        ('geometry', nb.typeof(params_default)),
        ('thermal_couple', int64),
        ('T4', float64[:])
        ]
@jitclass(data)
class IC_func(object):
    def __init__(self, source_type, uncollided, x0, source_strength, sigma, x1, geometry, thermal_couple, T4):
        self.source_type = np.array(list(source_type), dtype = np.int64)
        self.uncollided = uncollided
        self.x0 = x0
        self.source_strength = source_strength
        self.sigma = sigma
        self.x1 = x1
        self.geometry = geometry
        self.thermal_couple = thermal_couple
        self.T4 = T4
        


    def function(self, x, mu, iarg = 0, earg = 0):
        if self.geometry['slab'] == True:
            if self.thermal_couple == True:
                return self.T4
            else:
                if self.uncollided == True:
                    return np.zeros(x.size)
                elif self.uncollided == False and self.source_type[0] == 1:
                    return self.plane_and_square_IC(x)/self.x0/2.0
                    # return self.gaussian_plane(x)/2.0
                elif self.uncollided == False and self.source_type[1] == 1:
                    return self.plane_and_square_IC(x)
                elif self.uncollided == False and self.source_type[2] == 1:
                    return np.zeros(x.size)
                elif self.uncollided == False and self.source_type[3] == 1:
                    if self.source_type[-1] == 1:
                        return self.gaussian_IC_noniso(x,mu)
                    else:
                        return self.gaussian_IC(x)
                elif self.source_type[4] == 1 and self.source_type[3] == 0:
                    return self.MMS_IC(x)
                elif self.source_type[0] == 2:
                    return self.dipole(x)/abs(self.x1)
                elif self.source_type[0] == 3:
                    return self.self_sim_plane(x)
                elif self.source_type[15] == 1:
                    self.random_IC(x)
                else:
                    return np.zeros(x.size)

        elif self.geometry['sphere'] == True:
            if self.thermal_couple == True:

                return self.T4[iarg:earg] * 0.5
            else:
                if self.uncollided == False:
                    if self.source_type[0] == 1:
                        # return self.plane_and_square_IC(x)/self.x0/2.0 
                        return self.point_pulse(x)/(self.x0**3)
                    elif self.source_type[1] == 1:
                        return self.shell_IC(x)
                    elif self.source_type[15] == 1:
                        self.random_IC(x)
                    # This elif below added by Stephen
                    # The problem that was causing the code not to run seems to have been here (19-06-24)
                    #elif ((self.source_type[2] == 1):
                    #    for j in range(x.size):
                    #        if np.less(np.abs(x[j]) - 510, self.x0)):
                    #            return np.zeros(x.size)
                    else:
                        return np.zeros(x.size)
                else:
                    return np.zeros(x.size)

    def point_pulse(self, x):
        temp = (np.greater(x, 0) - np.greater(x, self.x0))*self.source_strength
        return temp


    def random_IC(self, x):
        temp = np.random.random((x.size)) * self.source_strength + 1e-12
        return temp



    def plane_and_square_IC(self, x):
        temp = (np.greater(x, - self.x0) - np.greater(x, self.x0))*self.source_strength
            # temp = x/x
        return temp/2.0
    
    def shell_IC(self, x):
        R = self.x0
        a = 0
        temp = (np.greater(x, a) - np.greater(x, R))*self.source_strength * 3 / 4 / math.pi / R**3
            # temp = x/x
        return temp / 2.0 

    def gaussian_plane(self, x):
        RES = math.sqrt(1/math.pi/2.0)/self.x0 * np.exp(-0.5 * x**2/self.x0**2)
        print(RES)
        return RES
    
    def gaussian_IC(self, x):
        temp = np.exp(-x*x/self.sigma**2)*self.source_strength
        return temp/2.0

    def gaussian_IC_noniso(self, x, mu):
        temp = 2*np.exp(-x*x/self.sigma**2)*self.source_strength*np.greater(mu, 0)
        return temp/2.0
    
    def MMS_IC(self, x):
        # temp = np.greater(x, -self.x0)*1.0 - np.greater(x, self.x0)*1.0 * np.exp(-x*x/2)/(2)
        temp = np.exp(-x*x/2)/(2)
        return temp
    
    def dipole(self, x):
        x1 = abs(self.x1)
        dx = 1e-10
        temp = -(np.greater(x, -x1) - np.greater(x, 0))*self.source_strength +  (np.greater(x, 0) - np.greater(x, x1))*self.source_strength 
        return temp/2
    
    def self_sim_plane(self, x):
        c = 29.998
        kappa = 800
        A = c/3/kappa
        t = 0.01
        arg = -x**2/4/A/t
        temp = 1 / math.sqrt(math.pi*0.5) / math.sqrt(A * t) * np.exp(arg)
        return temp / 2.0 


        
        
