
                
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
        ('a2', float64),
        ('sigma_func',nb.typeof(params_default)),
        ('lumping', float64)



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
        self.lumping = build.lumping

        self.geometry = build.geometry
        self.sigma_func = build.sigma_func
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
    
    def integrate_trap_sphere(self, a, b, j, sigma_class):
        
        # self.H[j] = 0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b) * 1 * normTn(j, argument, a, b))
        left = (a**2 * self.T_func(np.array([a]), a, b, sigma_class, self.space) * 1 * normTn(j, np.array([a]), a, b))[0]
        right = (b**2 * self.T_func(np.array([b]), a, b, sigma_class, self.space) * 1 * normTn(j, np.array([b]), a, b))[0]

        self.H[j] =  0.5 * (b-a) * (left + right)
        #assert(0)
    
    def get_sigma_a_vec(self, x, sigma_class, temperature):
        self.sigma_a_vec = sigma_class.sigma_function(x, 0.0, temperature)

    def make_T(self, argument, a, b):
        e = self.make_e(argument, a, b)
        # e = self.positivize_e(e, argument, a,b)
        # self.find_minimum(a,b)
        # if e.any() <0:
        #      raise ValueError('Negative energy density')

        if self.temp_function[0] == 1:
            T = self.su_olson_source(e, argument, a, b)
        elif self.temp_function[1] == 1:
             T =  e / self.cv0
        elif self.temp_function[2] == 1:
            #  if np.max(e) <= 1e-20:
            #     T = np.zeros(e.size) + 1e-12
            #  else:
                T = self.meni_eos(e, argument)
  
                # for it, TT in enumerate(T):
                #     if TT<0.0:
                #     #  raise ValueError('negative temperature')
                #         print(TT)
                #         assert(0)
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
            return  np.power(T,4) * self.sigma_a_vec * np.sign(T)
        
            

        else:
            assert(0)


    def positivize_e(self, e, argument, a, b):
        #  e = self.make_e(argument, a,b)
         enew = e
         floor = 1e-5 
         ubar = self.cell_average(a,b)
         if (e<0).any() and ubar >0.0 :
            tol = 1000
            
        

            m = self.find_minimum(a,b)
            if abs(m - ubar) <=1e-14:
                theta = 1.0
            else:    
                theta = min(1, abs(-ubar/(m-ubar)))

            
            #  floor = np.max(e)/tol
                    

            #  if ubar <0.0 and abs(ubar) < floor:
            #     #   print(ubar, 'ubar')
            #       ubar = 0.0
            #       theta = min(1, abs(-ubar/(m-ubar+1e-16)))
            #     #   enew = theta * (e-ubar) + ubar 
            #       enew = 0 * e + floor

            #  if ubar <0.0 and abs(ubar) >= floor:
            #       enew = e * 0
            #       print(ubar, 'ubar')
            #       print(e, 'e')
            #       print(a,b, 'cell edges')
            #     #   if self.M !=0:
                    
            #     #     raise ValueError('negative ubar')
            
            #  elif ubar >= 0.0:
            enew = theta * (e-ubar) + ubar 
            
            
            # if (enew<0).any():
            #         m = self.find_minimum(a,b, tol1 = 1e10)

            #         theta = min(1, abs(-ubar/(m-ubar+1e-15)))
            #         enew = theta * (e-ubar) + ubar
                    # if (enew<0).any():
                        

                    #     #  print(self.xs_quad, 'xs')
                    #     #  e2 = self.make_e(np.linspace(a,b,1000), a,b)
                    #     #  print(np.mean(e2), 'mean e2')
                    #     #  if ((m - e2) <0).all():
                    #     #     assert(0)
                    #     #  else:
                    #     #     #   print(np.min(np.abs((m-e2))), 'm-e2')
                    #     #     #   print(np.min(e2),'min e2')
                    #     #     #   print(np.min(enew), 'min enew')

                    #     #     #   assert(0)
                    #      basee = np.mean(e)
                    #      tol = 10**5
                    #      for ix, ee in enumerate(enew):
                    #             if ee < 0:
                    #                if abs(ee) < floor:
                    #                     enew[ix] = 0.0
                    #                else:
                    #                     print(theta, 'theta')
                    #                     print(ubar, 'ubar')
                    #                     print(e,'e')
                    #                     print(m,'m')
                    #                     print(enew, 'enew')
                    #                     assert 0


        #  if enew == []:
        #       assert 0
         return enew
         



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
    
    def meni_eos(self, e, x):
        
        self.fudge_factor = np.ones(e.size)
    
        # for count in range(e.size):
            # if math.isnan(e[count]) == True:
            #                 print("nan")
            #                 print(e)
            #                 assert 0
            #
            #
            # if e[0] > 0.:
            #      e[0] = np.mean(e)     
            # if (e[count]) < 0.:
            #     self.fudge_factor[count] = -1.0
                # e[count] = e[count-1]

        # dimensional e in GJ/cm^3
        if self.sigma_func['test1'] == True:
             rho = 19.3
             ee = e * self.a  * (0.1**1.6) / 10**-3  / 3.4 / (rho **.86)
             T1 = (np.abs(ee))
             return np.power(T1, 0.625) * np.sign(e)
        elif self.sigma_func['test2'] == True:
             rho = (x+1e-5)**.5
             ee = e * self.a * 0.1**2 / 10**-3  / 3.0/ (rho **.4)
             T1 = (np.abs(ee))
             return np.power(T1, 0.5) * np.sign(e)
        elif self.sigma_func['test3'] == True:
             rho = (np.mean(x)+1e-6) **(-.45)

             ee = e * self.a * 0.1**2 / 10**-3  / (rho **.75)
             T1 = (np.abs(ee))
             return np.power(T1, 0.5) * np.sign(e)
             
        elif self.sigma_func['test4'] == True:
             
             ee = e * self.a   / 10**-2  * 4 / 5 / 1.372017 * 5
             T1 = (np.abs(ee))
             return np.power(T1, 0.25) * np.sign(e)
                  
        else:
            ee = e * self.a  / 10**-3 /0.3
            T1 = (np.abs(ee))
            # self.alpha = 10**-3
            # t1 = np.abs(4*e*self.a/self.alpha)
            # return np.power(t1,0.25) 
            return np.power(T1, 1.0) * np.sign(e)
        
        
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
                    if self.lumping == False:
                        self.integrate_quad_sphere(xL, xR, j, sigma_class)
                    else:
                         self.integrate_trap_sphere(xL, xR, j, sigma_class)
        


    def cell_average(self, a, b):
        # dx = 4/3 * math.pi * (b**3-a**3)
        dx = b-a
        if dx <=1e-16:
             print(a,b,'edges')
             assert 0
        
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        e = self.make_e(argument,a,b)
        ubar = 0.5 * (b-a) * np.sum(e * self.ws_quad)

        # if np.abs(ubar/dx - self.e_vec[0] * normTn(0,argument,a,b)).any() >=1e-8:
        #      assert 0
        # if ubar <0:
        #      print('negative ubar')
        #      assert(0)
        return ubar / dx
    
     
     

    def find_minimum(self, a, b, tol1 = 2**6):
        dx = (b-a)/10
        pool_size = 1
        npts = 51
        converged = False
        tol = 1e-15

        initial_guess = np.linspace(a,b,npts)
        ee = self.make_e(initial_guess, a, b)
        emins_initial = np.sort(ee)[0:pool_size]
        xvals = np.zeros(pool_size)
        emins = np.zeros(pool_size)
        emins = emins_initial
        for n in range(pool_size):
            xvals[n] = initial_guess[np.argmin(np.abs(ee-emins_initial[n]))]


        it = 0
        # while converged == False:
        #     emins_old = emins
        for ix in range(pool_size):
                # xs = np.linspace(xvals[ix]-dx, xvals[ix]+dx, npts)

                # ee = self.make_e(xs, a, b)
                # emin = np.sort(ee)[0]

                # xval = xs[np.argmin(np.abs(ee-emin))]
                xvals[ix] = self.gradient_descent(xvals[ix] -dx, xvals[ix]+dx, xvals[ix], a,b, tol1)
                emins[ix] = self.make_e(np.array([xvals[ix]]), a, b)[0]
                

        if (np.sort(emins)[0] > emins_initial).any():
             print(emins, 'min vals')
             print(emins_initial, 'initial min values')
            #  assert 0


        return np.sort(emins)[0]
    
    def gradient_descent(self,x1,x2,x0, a,b, tolf):
        step = (x2-x1)/10
        tol = step/tolf
        loc = x0
        loc_old = loc
        direction = 1.0
        converged = False
        abstol = 10**-15
        # print(it, 'it')
        # print(step, 'step', tol, 'tol', converged)
        # print(tolf,'tol')
        while step > tol and converged == False:
            # print(it)
            loc += step * direction
            f1 = self.make_e(np.array([loc]), a, b)[0]
            f2 = self.make_e(np.array([loc_old]), a, b)[0]
            if abs(f1-f2)<=1e-12:
                converged = True
            elif step <= abstol:
                 converged = True
            elif f1 > f2 or loc <=a or loc >= b:
                step = step/2.0
                direction = direction * -1

            
            # elif loc <= a or loc >=b:
            #      converged = True 

                
            loc_old = loc
        # print('found min')
        return loc
    





        
                             


              
             
             
        


         