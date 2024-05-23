import numba as nb
import numpy as np
import math
from numba.experimental import jitclass
from numba import types, typed
from numba import int64, float64, jit, njit, deferred_type
from .build_problem import build
from .cubic_spline import cubic_spline_ob as cubic_spline
from .functions import newtons

build_type = deferred_type()
build_type.define(build.class_type.instance_type)

spline_type = deferred_type()
spline_type.define(cubic_spline.class_type.instance_type)

data = [('rho2', float64),
        ('u2', float64),
        ('gamma', float64),
        ('rho1', float64),
        ('gamp1', float64),
        ('gamm1', float64),
        ('gpogm', float64),
        ('r2', float64),
        ('eblast', float64),
        ('rho0', float64),
        ('omega', float64),
        ('xg2', float64),
        ('f_fun', float64[:]),
        ('g_fun', float64[:]),
        ('l_fun', float64[:]),
        ('us', float64),
        ('alpha', float64),
        ('sigma_t', float64),
        ('vr2', float64),
        ('t_shift', float64)
        ]

@jitclass(data)
class sedov_class(object):
    def __init__(self, g_fun, f_fun, l_fun, sigma_t):
        t = 1
        self.gamma = 7.0/5.0
        self.gamm1 = self.gamma - 1.0
        self.gamp1 = self.gamma + 1.0
        self.gpogm = self.gamp1 / self.gamm1
        self.rho0 = 1.0
        self.omega = 0.0
        geometry = 1
        self.alpha = self.gpogm * 2**(geometry) /\
                (geometry*(self.gamm1*geometry + 2.0)**2)
        print(self.alpha, 2/self.gamm1/self.gamp1)
        self.eblast = 0.851072
        # self.us = (2.0/self.xg2) * self.r2 / t
        # self.u2 = 2.0 * self.us / self.gamp1
        self.xg2 = 1 + 2.0 - self.omega
        self.sigma_t = sigma_t
        self.physical(t)
        print(self.sigma_t, 'sigma_t')
        # self.find_r2(t)


        self.g_fun = g_fun
        self.f_fun = f_fun
        self.l_fun = l_fun

        

        
    def find_r2(self, tt):
        t = tt + self.t_shift
        self.r2 = (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) *\
                t**(2.0/self.xg2)
        if math.isnan(self.r2):
            print(tt)
            print(t)
            print((self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2))
            
            assert(0)

        self.vr2 = (1/ self.sigma_t/29.98) * 1e-8 * 2/self.xg2 * (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) *\
                (tt + self.t_shift)**(2/self.xg2 -1)
        
        self.rho1 = self.rho0 * self.r2**(-self.omega)
        self.rho2 = self.gpogm * self.rho1
        self.us = (2.0/self.xg2) * self.r2 / t
        self.u2 = 2.0 * self.us / self.gamp1


    def physical(self, tt):
        '''Returns physical variables from single values of Sedov functions'''
        self.t_shift = (500/29.98)*1e-8/self.sigma_t
        t_sh = tt / 29.98 / self.sigma_t # convert mean free times to shakes
        t = t_sh * 1e-8 # convert shakes to seconds
        self.find_r2(t)
        density = self.rho2 * self.g_fun
        velocity = self.u2 * self.f_fun
        rs = self.l_fun * self.r2
        return density, velocity, rs
    
    def splice_blast(self, x, interpolated_sol, elseval):
        res = x * 0 
        for ix, xx in enumerate(x):
            if xx < self.r2:
                res[ix] = interpolated_sol.eval_spline(np.array([xx]))[0]
            else:
                res[ix] = elseval
        return res


    def evaluate_sedov(self, x, interpolated_rho, interpolated_v):
        rho = x * 0
        v = x * 0 

        v = self.splice_blast(x, interpolated_v, 0.0)
        rho = self.splice_blast(x, interpolated_rho, self.rho0)

        return rho, v

    def interpolate_solution(self, t, xs, interpolated_rho, interpolated_v):
       
        # density, velocity, rs = self.physical(tt + t_shift)
        # interpolated_density = cubic_spline(rs, density)
        
        # interpolated_velocity = cubic_spline(rs, velocity)
        

        # if (xs > self.r2).all():
        #     print(self.r2)
        #     return self.rho0 * np.ones(xs.size), np.zeros(xs.size)
        # else:
            # interpolated_density = cubic_spline(rs2, np.flip(density))
            # interpolated_velocity = cubic_spline(rs2, np.flip(velocity))
            
            # get mirrored Taylor-Sedov blast solution
            res_rho = xs * 0 
            res_v = xs * 0 
            res = self.evaluate_sedov(np.abs(xs), interpolated_rho, interpolated_v)
            res_rho = res[0]
            res_v = res[1]
            
            
            return res_rho, res_v
    
    def interpolate_self_similar(self, t, xs, interpolated_g):
        self.physical(t)
        
        res = xs * 0
        if self.r2 == 0.0:
            assert(0)
        for ix, xx in enumerate(xs):
            if abs(xx) <= self.r2:
                
                res[ix] = self.rho2 * interpolated_g.eval_spline(np.array([abs(xx) / self.r2]))[0]
                # res[ix] = self.rho0
            else:
                res[ix] = self.rho0
        return res
   
    def interpolate_self_similar_v(self, t, xs, interpolated_v):
        self.physical(t)
        res = xs * 0
        if self.r2 == 0.0:
            assert(0)
        for ix, xx in enumerate(xs):
            if abs(xx) <= self.r2:
                res[ix] = self.u2 * interpolated_v.eval_spline(np.array([abs(xx) / self.r2]))[0]
            else:
                res[ix] = 0.0
        return res
        

    def interior_interpolate(self, t, xs):
        density, velocity, rs = self.physical(t)
        rs2 = np.flip(rs)
        rs2[0] = 0.0
        density2 = np.flip(density)
        density2[0] = 0.0
        density2[-1] = self.gpogm
        rs2[-1] = self.r2

        interpolated_density = cubic_spline(rs2, density2)
        # interpolated_density = cubic_spline(xs, np.cos(xs))
        # interpolated_velocity = cubic_spline(xs, np.cos(xs))
        interpolated_velocity = cubic_spline(rs2, np.flip(velocity))
        res_rho, res_v = self.interpolate_solution(t, xs, interpolated_density, interpolated_velocity)
        return res_rho, res_v
    
    def find_contact_time(self, x0):
       t_hits = np.zeros(2)
       contact_time = self.bisection(self.contact_func, 0.0, x0, x0)
       t_hits[0] = contact_time
       contact_time2 = self.bisection(self.contact_func2, contact_time, 2*x0, x0)
       t_hits[1] = contact_time2
       self.physical(contact_time)
       print(self.r2, 'r2 at t_hit1')
       self.physical(contact_time2)
       print(self.r2, 'r2 at t_hit2')
       return t_hits - 1e-3
    
    def chi_func(self, s, x, mu, t):
        return (s-x) / mu + t
    
    def r2_func(self, t):
        self.physical(t)
        return self.r2
   
    def integral_bounds_func(self, s, x, t, mu):
        r2 = self.r2_func(self.chi_func(s, x, mu, t))
        return  s - r2
    def integral_bounds_func2(self, s, x, t, mu):
        r2 = self.r2_func(self.chi_func(s, x, mu, t))
        return -r2 - s
    
    def find_r2_in_transformed_space(self, x, t, mu, x0):
        a = 0
        b = x0
        shock_point1 = self.bisection2(self.integral_bounds_func, a, b, x, t, mu)
        a = -x0
        b = 0
        shock_point2 = self.bisection2(self.integral_bounds_func2, a, b, x, t, mu)

        return shock_point2, shock_point1
    
    def contact_func(self, t, x0):
        self.physical(t)
        return t - x0 + self.r2
    
    def contact_func2(self, t, x0):
        self.physical(t)
        return t - x0 - self.r2


    def bisection(self, f, a, b, x0, tol=1e-14):
        assert np.sign(f(a, x0)) != np.sign(f(b, x0))
        while b-a > tol:
            m = a + (b-a)/2
            fm = f(m, x0)
            if np.sign(f(a, x0)) != np.sign(fm):
                b = m
            else:
                a = m
        return m
    
    def bisection2(self, f, a, b, x,t, mu, tol=1e-5):
            if np.sign(f(a, x, t, mu)) == np.sign(f(b, x, t, mu)):
                return x
            else:
                while b-a > tol:
                    m = a + (b-a)/2
                    fm = f(m, x, t, mu)
                    if np.sign(f(a, x, t, mu)) != np.sign(fm):
                        b = m
                    else:
                        a = m
                return m