
                
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
from .functions import mass_lumper
from .GMAT_sphere import VV_matrix
 
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
        ('lumping', float64),
        ('xs_quad_chebyshev', float64[:]),
        ('ws_quad_chebyshev', float64[:]),
        ('N_space', int64),
        ('cs', float64[:,:]),
        ('Msigma', int64),
        ('cs_T4', float64[:]),
        ('H2', float64[:]),
        ('g', int64)




        ]
###############################################################################

@jitclass(data)
class T_function(object):
    def __init__(self, build):
        self.temp_function = np.array(list(build.temp_function), dtype = np.int64) 
        self.H = np.zeros(build.M+1).transpose()
        self.M = build.M
        self.Msigma = build.Msigma
        
        self.a = 0.0137225 # GJ/cm$^3$/keV$^4
        self.alpha = 4 * self.a
        self.clight = 29.98 # cm/ns
        self.a2 = 5.67e-5 # in ergs
        # self.lumping = build.lumping
        self.lumping = build.lumping

        self.geometry = build.geometry
        self.sigma_func = build.sigma_func
        if self.geometry['slab'] == True:
            self.xs_quad = build.xs_quad
            self.ws_quad = build.ws_quad
        
        if self.geometry['sphere'] == True:
            self.xs_quad = build.t_quad
            self.ws_quad = build.t_ws
            self.xs_quad_chebyshev = build.xs_quad
            self.ws_quad_chebyshev = build.ws_quad

        self.cv0 = build.cv0 / self.a 
        if (self.cv0) != 0.0:
            print('cv0 is ', self.cv0)
        self.test_dimensional_rhs = False
        self.save_derivative = build.save_wave_loc
        self.thermal_couple = build.thermal_couple
        self.temperature = np.zeros((build.N_space, self.xs_quad.size))
        self.e_vec = np.zeros(self.M+1)
        self.N_space = build.N_space
        self.cs = np.zeros((self.N_space, self.M+1))
        self.cs_T4 = np.zeros(self.M+1)
    
        self.g  =0
    def make_e(self, xs, a, b):
        temp = xs*0
        for ix in range(xs.size):
            for j in range(self.M+1):
                if self.geometry['slab'] == True:
                    temp[ix] += normPn(j, xs[ix:ix+1], a, b)[0] * self.e_vec[j] 
                elif self.geometry['sphere'] == True:
                    temp[ix] += normTn(j, xs[ix:ix+1], a, b)[0] * self.e_vec[j]
        return temp 
    def make_sigma_vec(self, xs, a, b, k):
        temp = xs*0
        for ix in range(xs.size):
            for j in range(self.Msigma+1):
                if self.geometry['slab'] == True:
                    temp[ix] += normPn(j, xs[ix:ix+1], a, b)[0] * self.cs[k,j] 
                elif self.geometry['sphere'] == True:
                    temp[ix] += normTn(j, xs[ix:ix+1], a, b)[0] * self.cs[k,j]
        return temp 
    def integrate_quad(self, a, b, j, sigma_class):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        self.H[j] = (b-a)/2 * np.sum(self.ws_quad * self.T_func(argument, a, b, sigma_class, self.space) * normPn(j, argument, a, b))


    def integrate_quad_sphere(self, a, b, j, sigma_class):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        # self.H[j] = 0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b) * 1 * normTn(j, argument, a, b))
        self.H[j] =  0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b, sigma_class, self.space) * 1 * normTn(j, argument, a, b))
    
    
    def integrate_moments_sphere(self, a, b, j, T4):
        # self.ws_quad, self.xs_quad = quadrature(2*self.M+1, 'chebyshev')
        
        argument = 0.5*(b-a)*self.xs_quad_chebyshev + (a+b) * 0.5
        # argument = (-b-a + 2 * self.xs_quad) / (b-a)
        # if np.abs(argument - T_eval_points[k]).any() >=1e-16:
        #     print(argument - T_eval_points[k])
        #     assert(0)

        # opacity = self.sigma_function(self.xs_quad, t, T_old)
        #  
        self.cs_T4[j] =  0.5 * (b-a) * np.sum(self.ws_quad_chebyshev * T4 * 2.0 * normTn(j, argument, a, b)) 
    
    def integrate_trap_sphere(self, a, b, j, sigma_class, k):                      
        self.get_sigma_a_vec(np.array([a]), sigma_class, self.make_T(np.array([a]), a, b))
        # self.H[j] = 0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b) * 1 * normTn(j, argument, a, b))
        # left = (a**2 * self.make_T(np.array([a]), a, b)**4 * 1 * normTn(j, np.array([a]), a, b))[0]
        # right = (b**2 * self.T_func(np.array([b]), a, b)**4 * 1 * normTn(j, np.array([b]), a, b))[0]
        self.H = self.H * 0
        for n in range(self.Msigma + 1):
            Ta = self.make_T(np.array([a]), a, b)
            Tb = self.make_T(np.array([b]), a, b)
            left = (a**2 * Ta**4* np.sign(Ta) * normTn(j, np.array([a]), a, b) * normTn(n, np.array([a]), a, b))[0]
            right = (b**2 * Tb**4 * np.sign(Tb) * normTn(j, np.array([b]), a, b) * normTn(n, np.array([b]), a, b))[0]
            self.H[j] += self.cs[k, n] * 0.5 * (b-a) * (left + right)

    def project_T4_to_basis(self, a, b, sigma_class):
        argument = 0.5*(b-a)*self.xs_quad_chebyshev + (a+b) * 0.5
        T = self.make_T(argument, a, b)
        T4 = T**4 * np.sign(T)
        self.get_sigma_a_vec(argument, sigma_class, T)
        for j in range(0, self.M+1):
            self.integrate_moments_sphere(a, b, j, T4)

        # self.H[j] =  0.5 * (b-a) * (left + right)
        #assert(0)
    
    def get_sigma_a_vec(self, x, sigma_class, temperature):
        self.sigma_a_vec = sigma_class.sigma_function(x, 0.0, temperature)
        self.cs = sigma_class.cs

    def make_T(self, argument, a, b):
        # print(argument, 'argument')
        e = self.make_e(argument, a, b)
        # e = self.positivize_e(e, argument, a,b)
        # self.find_minimum(a,b)

        # if (e <-1e-5).any():
        #      print(e)
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
        self.sigma_a_vec = self.make_sigma_vec(argument, a, b, space)
        # self.xs_points = argument
        # self.e_points = e
        

        if self.temp_function[0] == 1:
            self.temperature[space,:] = T
            return  np.power(T,4) * self.fudge_factor  * self.sigma_a_vec
            #return np.sin(argument) 

        elif self.temp_function[1] == 1:
        
                self.temperature[space,:] = T
                return np.power(T,4)  * self.sigma_a_vec  * np.sign(T)
        
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
         floor = 1e-14
         ubar = self.cell_average(a,b)
        #  if ubar < -floor:
        #       print('negative ubar', ubar)
         if abs(ubar) <=floor:
              enew = 0 * e
         
         if (e<0).any() and ubar >floor :
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
            if (np.abs(enew) < floor).any():
                 for it, tt in enumerate(enew):
                      if abs(tt) < floor:
                           enew[it] = floor
                           
            if (enew<0).any():
                 print(enew, 'enew')
                 print(theta, 'theta')
                 print(ubar, 'ubar')
                 print(e, 'e')
                 print(m, 'm')
                 assert(0)
            # print(enew.size)
            # print(argument)
            
            
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
             rho = (np.mean(x)+1e-8)**.5
             ee = e * self.a * 0.1**2 / 10**-3  / 3.0/ (rho **.4)
             T1 = (np.abs(ee))
             return np.power(T1, 0.5) * np.sign(e)
        elif self.sigma_func['test3'] == True:
             rho = (np.mean(x)) **(-.45)

             ee = e * self.a * 0.1**2 / 10**-3  / (rho **.75)
             T1 = (np.abs(ee))
             return np.power(T1, 0.5) * np.sign(e)
             
        elif self.sigma_func['test4'] == True:
             
             ee = e * self.a   / 10**-2  * 4  / 1.372017 
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
        # if self.g == 0:
        self.H = self.H * 0
            
        # Lines commented out are the original lines of code

        #for j in range(self.M+1):
            #self.integrate_quad(xL, xR, j)

        if self.thermal_couple['none'] != True:

            if self.geometry['slab'] == True:

                for j in range(self.M+1):
                    self.integrate_quad(xL, xR, j, sigma_class)

            elif self.geometry['sphere'] == True:
                if self.lumping == False:
                    for j in range(self.M+1):
                            self.integrate_quad_sphere(xL, xR, j, sigma_class)
                else:
                    self.H2 = self.H.copy()*0
                    self.project_T4_to_basis(xL, xR, sigma_class)
                    
                    VV = np.zeros((self.M+1, self.M+1))
                    for i in range(self.M + 1):
                        for j in range(self.M + 1):
                            for k in range(self.Msigma + 1):
                                if self.geometry['sphere'] == True:
                                        if k ==0:
                                            for ii in range(self.M+1):
                                                for jj in range(self.M+1):
                                                    VV[ii,jj] = VV_matrix(ii, jj,k, xL, xR) / (math.pi**1.5)
                                            VV_lumped = mass_lumper(VV, xL, xR)[0]
                                        self.H2[i] +=   self.cs[space, k] * self.cs_T4[j] * VV_lumped[i,j]
                    self.H = self.H2.copy()
        


    def cell_average(self, a, b):
        # This isn't right because sometimes Gauss-Legendre quadrature is used, sometimes Chebyshev
        # dx = 4/3 * math.pi * (b**3-a**3)

        if self.M < 2:
             
            ubar = self.e_vec[0] * 1/(math.sqrt(1/(-a + b))*math.sqrt(math.pi) )
        else:
             raise ValueError('Not implemented to this order of M yet')



        dx = b-a
        # if dx <=1e-16:
        #      print(a,b,'edges')
        #      assert 0
        
        # argument = (b-a)/2*self.xs_quad + (a+b)/2
        # e = self.make_e(argument,a,b)
        # ubar2 = 0.5 * (b-a) * np.sum(e * self.ws_quad)
        # print(ubar2-ubar)

        # # if np.abs(ubar/dx - self.e_vec[0] * normTn(0,argument,a,b)).any() >=1e-8:
        # #      assert 0
        # if ubar <0:
        #      print('negative ubar')
        #      print(ubar, 'ubar')
        #      assert(0)

        # do I need to divide by dx?
        return ubar /dx
    
     
     

    def find_minimum(self, a, b, tol1 = 2**6):

        if self.M == 1:
            f1 = self.make_e(np.array([a]), a, b)[0]
            f2 = self.make_e(np.array([b]), a, b)[0]
            if f1 > f2:
                 return f2
            else:
                 return f1


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
    def positivize_temperature_vector(self, e_old, edges):
        # self.ws_quad, self.xs_quad = quadrature(2*self.M+1, 'chebyshev')

        e_new = e_old * 0
        for k in range(self.N_space):
            a = edges[k]
            b = edges[k+1]
            self.e_vec = e_old[k]

            argument = 0.5*(b-a)*self.xs_quad_chebyshev + (a+b) * 0.5
            e = self.make_e(argument, a, b)

            if (e<0).any():
            # if 1 ==0:

                ee = self.positivize_e(e, argument, a,b)
                
                for j in range(self.M+1):
                    e_new[k, j] =  0.5 * (b-a) * np.sum(self.ws_quad_chebyshev * ee * 2.0 * normTn(j, argument, a, b))
                # reconstruction test
                # etest = e*0
                # for count in range(argument.size):
                #     for i in range(self.M+1):
                #             etest[count] += e_new[k, i] * normTn(i,argument[count:count+1],float(edges[k]),float(edges[k+1]))[0]
                # if np.sqrt(np.max((etest - e)**2)) > 1e-10:
                    #  assert 0 
            else:

                 e_new[k, :] = e_old[k]
        return e_new
        



        # argument = (-b-a + 2 * self.xs_quad) / (b-a)
        # if np.abs(argument - T_eval_points[k]).any() >=1e-16:
        #     print(argument - T_eval_points[k])
        #     assert(0)

        # opacity = self.sigma_function(self.xs_quad, t, T_old)
        #  
         


         
    





        
                             


              
             
             
        


         