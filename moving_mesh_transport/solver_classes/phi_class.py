#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 09:26:00 2022

@author: bennett
"""
import numpy as np
import math

from .build_problem import build
from .opacity import sigma_integrator
from .GMAT_sphere import VV_matrix, VVmatLUMPED

from numba import float64, int64, deferred_type
from numba.experimental import jitclass
from numba import types, typed
from .functions import mass_lumper
import numba as nb
###############################################################################
build_type = deferred_type()
build_type.define(build.class_type.instance_type)
sigma_class_type = deferred_type()
sigma_class_type.define(sigma_integrator.class_type.instance_type)
kv_ty = (types.int64, types.unicode_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))


data = [("P", float64[:]),
        ("ws", float64[:]),
        ("M", int64),
        ("u", float64[:,:]),
        ('thermal_couple', nb.typeof(params_default)),
        ("vec", float64[:,:]),
        ('AAA', float64[:,:,:]),
        # ('sigma_func', int64[:]),
        ('sigma_s', float64),
        ('N_ang', int64),
        ('Msigma', int64),
        ('cs', float64[:,:]),
        ('edges', float64),
        ('PV', float64[:]),
        ('sigma_func', nb.typeof(params_default)),
        ('scalar_flux_term', float64[:]),
        ('geometry', nb.typeof(params_default)),
        ('lumping', int64),
        ('N_groups', int64),
        ('g', int64)
        ]
###############################################################################
@jitclass(data)
class scalar_flux(object):
    def __init__(self, build):
        self.P = np.zeros(build.M+1).transpose()
        self.PV = np.zeros(build.M+1).transpose()
        self.M = build.M
        self.ws = build.ws
        self.thermal_couple = build.thermal_couple
        self.sigma_func = build.sigma_func
        self.sigma_s = build.sigma_s
        self.N_ang = build.N_ang
        self.Msigma = build.Msigma
        self.scalar_flux_term = np.zeros(self.M+1)
        self.geometry = build.geometry
        self.lumping = build.lumping
        self.N_groups = build.N_groups
        self.g = 0
        

    # def make_P(self, u):
    #     if self.thermal_couple == 1:
    #         vec = u[:-1,:]
    #     elif self.thermal_couple == 0:
    #         vec = u
    #     for i in range(0, self.M+1):
    #         self.P[i]  = np.sum(np.multiply(vec[:,i],self.ws)) 
    #     return self.P

    def make_P(self, u, space, xL, xR):
        # if self.sigma_func['constant'] == True: # if the opacity is constant
        #     for i in range(0, self.M+1):
        #         self.P[i]  = np.sum(np.multiply(u[:,i],self.ws))
        #     if self.geometry['slab'] == True:
        #         self.scalar_flux_term = self.P
        #     elif self.geometry['sphere'] == True:
        #         self.scalar_flux_term = self.P 

        # else:
            if self.g == 0: #resets PV sum at the next step. Otherwise, adds over groups
                self.PV = self.PV*0
            VV = np.zeros((self.M+1, self.M+1))

            # for g in range(self.N_groups):
            for i in range(self.M+1):
                for l in range(self.N_ang):
                    for j in range(self.M+1):
                        for k in range(self.Msigma + 1):
                            if self.geometry['slab'] == True:
                                self.PV[i] +=  self.cs[space, k] * u[l,j] * self.ws[l] * self.AAA[i, j, k] 
                            elif self.geometry['sphere'] == True:
                                if self.lumping == True: # this is a very inefficient implementation 
                                    if k ==0:
                                        for ii in range(self.M+1):
                                            for jj in range(self.M+1):
                                                VV[ii,jj] = VV_matrix(ii, jj,k, xL, xR) / (math.pi**1.5)
                                        VV_lumped = mass_lumper(VV, xL, xR)[0]
                                    # self.PV[i] += self.cs[space, k] * u[l,j] * self.ws[l] * VVmatLUMPED(i, j,k, xL, xR) / (math.pi**1.5)
                                    self.PV[i] += self.cs[space, k] * u[l,j] * self.ws[l] * VV_lumped[i,j]
                                elif self.lumping == False:
                                    self.PV[i] += self.cs[space, k] * u[l,j] * self.ws[l] * VV_matrix(i, j,k, xL, xR) / (math.pi**1.5)
                                # else:


            if self.geometry['slab'] == True:                    
                self.scalar_flux_term = self.PV / math.sqrt(xR-xL)
            elif self.geometry['sphere'] == True:
                self.scalar_flux_term = self.PV
                # a = xL
                # b = xR
                # assert((abs(self.cs[0,space]- math.sqrt(math.pi) * math.sqrt(b-a))<=1e-5))
                
                            # self.PV[i] += self.ws[l] * u[l,i]

        # print(self.PV)
    
    def call_P_noncon(self, xL, xR):
        dx = math.sqrt(xR - xL)
        return self.PV/dx

    def load_AAA(self, AAA):
        """ Gets the tensor of integrals over Bi Bj Bk from opacity_class
        """
        self.AAA = AAA


    def get_coeffs(self, opacity_class):
        self.cs = opacity_class.cs


    

        
