import numpy as np
import math
from .build_problem import build
# from .radiative_transfer import T_function

from numba.experimental import jitclass
from numba import int64, float64, deferred_type, prange
from .functions import Pn, normPn, normTn
from numba import types, typed
import numba as nb
from .GMAT_sphere import VV_matrix, VVmatLUMPED
from .functions import quadrature

build_type = deferred_type()
build_type.define(build.class_type.instance_type)
# RT_type = deferred_type()
# RT_type.define(T_function.class_type.instance_type)
kv_ty = (types.int64, types.unicode_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))

data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('sigma_t', float64),
        ('sigma_s', float64),
        ('sigma_a', float64),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('x0', float64),
        ("xL", float64),
        ("xR", float64),
        ('sigma_func', nb.typeof(params_default)),
        ('Msigma', int64),
        ('AAA', float64[:,:,:]),
        ('xs_quad', float64[:]),
        ('ws_quad', float64[:]),
        ('edges', float64[:]),
        ('std', float64), 
        ('cs', float64[:,:]), 
        ('VV', float64[:]),
        ('VP', float64[:]),
        ('moving', float64),
        ('sigma_v', float64), 
        ('fake_sedov_v0', float64),
        ('geometry', nb.typeof(params_default)),
        ('T', float64[:]),
        ('V_old', float64[:,:]),
        ('current_space', int64),
        ('opacity_vec', float64[:,:]),
        ('a', float64),
        ('lumping', int64)

        ]


@ jitclass(data)
class sigma_integrator():
    def __init__(self, build):
        self.sigma_t = build.sigma_t
        self.sigma_s = build.sigma_s
        print(self.sigma_s,'sigma_s')
        self.sigma_a = self.sigma_t - self.sigma_s
        print(self.sigma_a,'sigma_a')
        self.sigma_func = build.sigma_func
        self.M = build.M
        self.a = 0.0137225
        self.Msigma = build.Msigma
        self.xs_quad = build.xs_quad
        self.ws_quad = build.ws_quad
        self.std = 2
        self.N_space = build.N_space
        self.edges = np.zeros(self.N_space + 1)
        self.cs = np.zeros((self.N_space, self.Msigma+ 1))
        self.VV = np.zeros(self.M+1)
        self.VP = np.zeros(self.M+1)
        self.AAA = np.zeros((self.M+1, self.M + 1, self.Msigma + 1))
        self.moving = False
        self.x0 = build.x0
        # if self.sigma_func['fake_sedov'] == True:
        #     self.moving = True
        # self.sigma_v = 0.005
        self.sigma_v = build.fake_sedov_v0
        self.geometry = build.geometry
        self.lumping = build.lumping
        
        # initialize integrals of Basis Legendre polynomials
        self.create_integral_matrices()

    def integrate_quad(self, a, b, i, j, k):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        fact = np.sqrt(2*i + 1) * np.sqrt(2*j + 1) * np.sqrt(2*k + 1) / 2
        self.AAA[i,j,k] = fact * (b-a)/2 * np.sum(self.ws_quad *  Pn(i, argument, a, b) * Pn(j, argument, a, b) * Pn(k, argument, a, b))
    
    def integrate_moments(self, a, b, j, k, t, T_old):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        opacity = self.sigma_function(argument, t, T_old)
        self.cs[k, j] = (b-a)/2 * np.sum(self.ws_quad * opacity * normPn(j, argument, a, b))
     
    def integrate_moments_sphere(self, a, b, j, k, t, T_old, T_eval_points, checkfunc = False):
        # self.ws_quad, self.xs_quad = quadrature(2*self.M+1, 'chebyshev')
        
        argument = 0.5*(b-a)*self.xs_quad + (a+b) * 0.5
        # argument = (-b-a + 2 * self.xs_quad) / (b-a)
        # if np.abs(argument - T_eval_points[k]).any() >=1e-16:
        #     print(argument - T_eval_points[k])
        #     assert(0)
        opacity = self.sigma_function(argument, t, T_old)
        # opacity = self.sigma_function(self.xs_quad, t, T_old)
        #  
        self.cs[k, j] =  0.5 * (b-a) * np.sum(self.ws_quad * opacity * 2.0 * normTn(j, argument, a, b)) 

    # def integrate_moments_sphere_trap(self, a, b, j, k, t, T_old, T_eval_points, checkfunc = False):
    #     # self.ws_quad, self.xs_quad = quadrature(2*self.M+1, 'chebyshev')
    #     self.cs[k, j] = 0.5 * (b-a) 
    def integrate_trap_sphere(self, a, b, j, k, t, T_old):
        # I probably need rho here
        
        #before using this, I need to check the T eval points

        # self.H[j] = 0.5 * (b-a) * np.sum((argument**2) * self.ws_quad * self.T_func(argument, a, b) * 1 * normTn(j, argument, a, b))
        left = (a**2 * self.sigma_function(np.array([a]), t, T_old) * 1 * normTn(j, np.array([a]), a, b))[0]
        right = (b**2 * self.sigma_function(np.array([b]), t, T_old) * 1 * normTn(j, np.array([b]), a, b))[0]


        self.cs[k, j] =  0.5 * (b-a) * (left + right) 
        
        

        
        # self.cs[k, j] = 0.5 * (b-a) * np.sum(self.ws_quad * opacity * 2.0 * normTn(j, self.xs_quad, a, b))
        


                      

        
        
        # assert(abs(self.cs[j,k]- math.sqrt(math.pi) * math.sqrt(b-a))<=1e-5)
        
    def check_sigma_coeffs(self, T_eval_points, edges, T_old):
        # for space in range(self.N_space):
        #     T_old[space,:] = np.ones(self.xs_quad.size) * np.mean(T_old[space])
        # self.sigma_moments(edges, 0.0,T_old, T_eval_points)
        for space in range(self.N_space):
            a = edges[space]
            b = edges[space+1]
            xs = T_eval_points[space]
            assert(np.mean(np.abs(xs-0.5*(b-a)*self.xs_quad - 0.5*(b+a)))<=1e-14)
            # print(xs, 'xs')
            # print(space,'space]')
            test_func = xs*0
            for ix, xx in enumerate(xs):
                for j in range(0,self.Msigma + 1):
                    test_func[ix] += self.cs[space, j] * normTn(j, xs[ix:ix+1],a,b)[0]
            if abs(np.max(np.abs(test_func - self.sigma_function(xs, 0, T_old[space]))/ self.sigma_function(xs, 0, T_old[space])) ) >=1e0 : 
                    print((test_func - self.sigma_function(xs, 0, T_old[space]))/ self.sigma_function(xs, 0, T_old[space]) , 'difference')
                    print(test_func,'test')
                    print(self.sigma_function(xs, 0, T_old[space]), 'sigma')
                    print(space,'space')
                    print(self.cs[space], 'coeffs')
                    print(T_old[space],'T')
                    # assert(0)
                    # assert(0)

    def both_even_or_odd(self, i, j, k):
        if i % 2 == 0:
            if (j + k) % 2 == 0:
                return True
            else:
                return False
        if i % 2 != 0:
            if (j + k) % 2 != 0:
                return True
            else:
                return False

    def create_integral_matrices(self):
        """
        creates a matrix with every integral over [-1,1] of the three normalized Legendre polynomials of order
        i, j, k. Each entry must be divided by sqrt(xR-xL) 
        """
        for i in range(self.M + 1):
            for j in range(self.M + 1):
                for k in range(self.Msigma + 1):
                    if (j + k >= i) and (self.both_even_or_odd(i, j, k)):
                        self.integrate_quad(-1, 1, i, j, k)
        # print(self.AAA)
    
    def sigma_moments(self, edges, t, T_old, T_eval_points):
        # self.V_old = V_old
        self.edges = edges
        for k in range(self.N_space):
            # self.current_space = int(i)
            # if (edges[i] != self.edges[i]) or (edges[i+1] != self.edges[i+1]) or self.moving == True:
            for j in range(0, self.Msigma + 1):
                if self.geometry['slab'] == True:
                    self.integrate_moments(edges[k], edges[k+1], j, k, t, T_old[k,:])
                elif self.geometry['sphere'] == True:
                    if self.lumping == True:
                        # self.integrate_trap_sphere(edges[k], edges[k+1], j, k, t, T_old[k,:])
                        self.integrate_moments_sphere(edges[k], edges[k+1], j, k, t, T_old[k,:], T_eval_points)
                    else:
                        self.integrate_moments_sphere(edges[k], edges[k+1], j, k, t, T_old[k,:], T_eval_points)

        
    
    def xi2(self, x, t, x0, c1, v0tilde):
        return -x - c1 - v0tilde*(t)

    def heaviside(self,x):
        if x < 0.0:
            return 0.0
        else:
            return 1.0

    def heaviside_vector(self, x):
        return_array = np.ones(x.size)
        for ix, xx in enumerate(x):
            if xx < 0:
                return_array[ix] = 0.0
        return return_array

    def sigma_function(self, x, t, T_old):

        if self.sigma_func['constant'] == 1:
            return x * 0 + 1.0
        
        elif self.sigma_func['converging'] == 1 or self.sigma_func['test1'] == 1 or self.sigma_func['test2'] == 1 or self.sigma_func['test3'] == 1 or self.sigma_func['test4'] == 1:
            # self.get_temp(x, a, b, RT)
            if np.isnan(T_old).any() or np.isinf(T_old).any():
                print(T_old, 'T')
                print(x,'x')
                assert(0)
            # if (T_old<0).any():
            #     T_old = np.mean(T_old) + T_old*0
            # resmax = 134183.7512857635 / self.x0
            
            

            if self.sigma_func['test1'] == 1:
                # resmax = 1e8
                # resmax = 4e6
                resmax = 5e5
                floor = 5e-3
                result = np.where(T_old<0.0, 0.0, T_old)
                # result = np.abs(T_old)
                rho = 19.3
                res = 7200 *  (result+1e-8) ** (-1.5) * (0.1**1.5) * rho **1.2
                if (res > resmax).any():
                    for ix, xx in enumerate(res):
                        if res[ix] > resmax:
                            res[ix] = resmax

                # print(np.max(res))

            elif self.sigma_func['test2'] == 1:
                floor = 5e-2
                resmax = 5e4
                # resmax = 1e3 
                result = np.where(T_old<0.0, 0.0, T_old)
                rho = (np.mean(x)+1e-8)**.5
                res = 1.5e4 * (result+1e-10) ** -3.0 * (0.1**3) * rho ** 1.4
                if (res > resmax).any():
                    for ix, xx in enumerate(res):
                        if res[ix] > resmax:
                            res[ix] = resmax
                # if res.any() > resmax:
                #     res = np.zeros(result.size) + resmax

            elif self.sigma_func['test3'] == 1:
                floor = 5e-2
                resmax = 1e6 * (1)
                result = np.where(T_old<0, 0.0, T_old)
                rho = (np.mean(x)) ** (-.45)
                res = 10**3 * (result +1e-12) ** -3.5 * (0.1**3.5) * (rho) **1.4
                if (res > resmax).any():
                    for ix, xx in enumerate(res):
                        if res[ix] > resmax:
                            res[ix] = resmax
                # if res.any() > resmax:
                #     res = np.zeros(result.size) + resmax
            elif self.sigma_func['test4'] == 1:
                # floor = 5e-3
                # resmax = 6e3
                a = x[0]
                b = x[-1]
                resmax = 15000 * (b-a)
                # resmax = 1e5
                # resmax = 950
                # resmax = 5e3
                # resmax = 500
                # if(T_old<0).any():
                #     assert 0
                result = np.where(T_old<0.0, 0.0, T_old)
                rho = np.mean(x )
                if (x<0).any():
                    assert(0)
                res = (result+1e-10) ** -3.5 * rho ** 2
                if (res<0).any():
                    assert 0
                if (res > resmax).any():
                    for ix, xx in enumerate(res):
                        if res[ix] > resmax:
                            res[ix] = resmax
                        # if res[ix] <floor:
                        #     res[ix] = floor
                # if (res!=15e3).any():
                #     print(res, x)
          
                # if res.any() > resmax:
                #     res = np.zeros(result.size) + resmax
            else:
                floor = 5e-2
                result = np.where(T_old<0.0, 0.0, T_old)
                # res = 5 * 10**(3) * (result + floor) ** -1.5 * (0.1**1.5)
                res = 300 * (result+floor) ** -3
            
            # if (res>resmax).any():
            #     for ix, xx in enumerate(res):
            #         if res[ix] > resmax:
            #             res[ix] = resmax

            # for ie, elem in enumerate(res):
                # if elem >= 1e16:
                #     res[ie] = 1e16
                #     print('ceiling')
                # if elem < 0:
                #     res[ie] = 0.0
                    # print('negative')
            # res = 5* 10**3 + T_old * 0
            # res = T_old *0 + 100
            if np.isnan(res).any() or np.isinf(res).any():
                print(res, 'res')
                print(T_old, 'T old')
                assert(0)

             
            # for ix, xx in enumerate(res):
            #     if abs(res[ix]) >1e16:
            #         res[ix] = 1e16
            
            # res = 1 * 10**(2) + x*0
            return res
        
        elif self.sigma_func['gaussian'] == 1:
            return np.exp(- x**2 /(2* self.std**2))  # probably shouldn't have sigma_a here
            # return x * 0 + 1.0
        elif self.sigma_func['siewert1'] == 1: # siewert with omega_0 = 1, s = 1
            return np.exp(-x - 2.5)
        elif self.sigma_func['siewert2'] == 1:
            return np.exp(-x/100000000000)
        elif self.sigma_func['fake_sedov'] == 1:
            # return np.exp(-(x- self.sigma_v * t)**2/(2*self.std**2))
            c1 = 1
            xi2x = self.xi2(x, t, 0, c1, self.sigma_v)
            rho2 = 0.2
            res = np.exp(-xi2x**2/self.std**2) * self.heaviside_vector(-xi2x - c1) + rho2*self.heaviside_vector(xi2x + c1)
            # vec_test = self.heaviside_vector(-xi2x - c1)
            # found = False
            # index = 0
            # if np.any(vec_test == 0):
            #     while found == False and index < x.size:
            #         if vec_test[index] == 1:
            #             found == True
            #             print(x[index], 'location of shock', t, 't')
            #             print(vec_test)
            #             print(-self.sigma_v*t - x[index])
            #             print("#--- --- --- --- --- --- ---#")
            #         index += 1

            return res

       
            
            # # assert(0)
            # # self.T = x 
            #
            # res = x * 0 + 1.0
            # return res

        else:
            raise Exception('no opacity function selected')

        

    # def get_temp(self, x, a, b, RT):
    #     e = self.V_old[self.current_space,:]
    #     RT.make_H(a, b, e)
    #     self.T = RT.T_func(x, a, b)




    def make_vectors(self, edges, u, space):
        self.VV = u * 0
        # self.sigma_moments(edges) # take moments of the opacity
        xL = edges[space]
        xR = edges[space+1]
        dx = math.sqrt(xR-xL)
        # if self.sigma_func['constant'] == True:
        #     self.VV = u * self.sigma_t
        # else:
        for i in range(self.M + 1):
                for j in range(self.M + 1):
                    for k in range(self.Msigma + 1):
                        if self.geometry['slab'] == True:
                            self.VV[i] +=   self.cs[space, k] * u[j] * self.AAA[i, j, k] / dx
                        elif self.geometry['sphere'] == True:
                            if self.lumping == False:
                                self.VV[i] +=   self.cs[space, k] * u[j] * VV_matrix(i, j, k, xL, xR) / (math.pi**1.5) 
                            else:
                                self.VV[i] +=   self.cs[space, k] * u[j] * VVmatLUMPED(i, j, k, xL, xR) / (math.pi**1.5) 

                            # assert(abs(self.cs[j,k]- math.sqrt(math.pi) * math.sqrt(xR-xL))<=1e-5)
                            # self.VV[i] +=  self.cs[space, k] * u[j]  / (math.pi**1.5) * (math.sqrt(1/(-xL + xR))*(xL**2 + xL*xR + xR**2))/3
            # self.VV = u * 1


                        




    