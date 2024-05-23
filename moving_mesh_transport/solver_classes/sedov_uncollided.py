import numba as nb
import numpy as np
from numba.experimental import jitclass
from numba import types, typed
from numba import int64, float64, jit, njit, deferred_type
from .build_problem import build
from .cubic_spline import cubic_spline_ob as cubic_spline
from .sedov_funcs import sedov_class
from tqdm import tqdm



spline_type = deferred_type()
spline_type.define(cubic_spline.class_type.instance_type)
sedov_type = deferred_type()
sedov_type.define(sedov_class.class_type.instance_type)
build_type = deferred_type()
build_type.define(build.class_type.instance_type)

data = [('x0', float64),
        ('xs_quad', float64[:]),
        ('ws_quad', float64[:]),
        ('sigma_a', float64),
        ('lambda1', float64),
        ('t0source', float64),
        ('mu_quad', float64[:]),
        ('mu_ws', float64[:]),
        ('transform', float64)
        ]

@jitclass(data)
class sedov_uncollided_solutions(object):
    def __init__(self, xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_a, t0, transform = True):
        self.xs_quad = xs_quad
        self.ws_quad = ws_quad
        self.x0 = x0
        self.lambda1 = 1.75
        # self.lambda1 = 1.0 
        self.sigma_a = 1.0
        self.t0source = t0
        # print(self.t0source, 't0')
        self.mu_quad = mu_quad
        self.mu_ws = mu_ws
        self.transform = transform

    
    # def get_sedov_density(self, rho_interp, v_interp, xs, sedov_class):
    #     rho, v  = sedov_class.interpolate_solution(0.0, xs, rho_interp, v_interp)

    #     return rho
    # def get_sedov_velocity(self, rho_interp, v_interp, xs, sedov_class):
    #     rho, v  = sedov_class.interpolate_solution(0.0, xs, rho_interp, v_interp)

    #     return v

    def get_upper_integral_bounds(self, x, mu, t, sedov_class, v_interpolated):

        tp = (-self.x0 - x) / mu + t

        if tp < t:
            # tau_int = self.integrate_quad_velocity(tp, t, x, sedov_class, v_interpolated)
            tau_int = 0.0
        else:
            tau_int = 0.0
        

        return x - tau_int*0 


    def integrate_quad_velocity(self, a, b, x, sedov_class, v_interpolated):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        func = self.xs_quad * 0
        for ix, xx in enumerate(self.xs_quad):
            func[ix] = sedov_class.interpolate_self_similar_v(argument[ix], np.array([x]), v_interpolated)[0] * np.sign(-x)
        res = (b-a)/2 * np.sum(self.ws_quad * func )
        return  res
    
    def integrate_quad_sigma(self, a, b, mu, x, t, sedov, g_interpolated):
        func = self.xs_quad * 0 
        # integral_bound = sedov.find_r2_in_transformed_space(x, t, mu, self.x0)
        # print(integral_bound)
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        for ix, xx in enumerate(self.xs_quad):
            func[ix] = self.transformed_sigma(argument[ix], x, t, mu, sedov, g_interpolated)
        res = (b-a)/2 * np.sum(self.ws_quad * func )
        return res
        

    def sigma_func(self, x, t, sedov_class, g_interpolated):
        sigma = self.sigma_a * sedov_class.interpolate_self_similar(t, np.array([x]), g_interpolated)[0] ** self.lambda1

        return sigma
    
    def chi_func(self, s, x, mu, t):
        return (s-x) / mu + t
    
    def integrate_sigma(self, x, mu, t, sedov_class, g_interpolated, v_interpolated):
        lower_bound = -self.x0
        
        integral_bound1, integral_bound2 = sedov_class.find_r2_in_transformed_space(x, t, mu, self.x0)
        # # upper_bound = self.get_upper_integral_bounds(x, mu, t, sedov_class, v_interpolated)
        upper_bound = x
        res = 0.0
        # print(integral_bound1, integral_bound2, x)
        if integral_bound1 > lower_bound and integral_bound1 < upper_bound:

            res += self.integrate_quad_sigma(lower_bound, integral_bound1, mu, x, t, sedov_class, g_interpolated)
            if integral_bound2 < upper_bound:
                res += self.integrate_quad_sigma(integral_bound1, integral_bound2, mu, x, t, sedov_class, g_interpolated)
                res += self.integrate_quad_sigma(integral_bound2, upper_bound, mu, x, t, sedov_class, g_interpolated)
                
            else:
                res += self.integrate_quad_sigma(integral_bound1, upper_bound, mu, x, t, sedov_class, g_interpolated)




        # if (-integral_bound < upper_bound) and (-integral_bound > -self.x0) and integral_bound != -1:
        #     res += self.integrate_quad_sigma(lower_bound, -integral_bound, mu, x, t, sedov_class, g_interpolated)
        #     if integral_bound < upper_bound:
        #         res += self.integrate_quad_sigma(-integral_bound, integral_bound, mu, x, t, sedov_class, g_interpolated)
        #         res += self.integrate_quad_sigma(integral_bound, upper_bound, mu, x, t, sedov_class, g_interpolated)
        #     else:
        #         res += self.integrate_quad_sigma(-integral_bound, upper_bound, mu, x, t, sedov_class, g_interpolated)

        else: 
            res += self.integrate_quad_sigma(lower_bound, upper_bound, mu, x, t, sedov_class, g_interpolated)
        return res
    
    def transformed_sigma(self, s, x, t, mu, sedov_class, g_interpolated):
        tt = self.chi_func(s, x, mu, t)
        if self.transform == True:
            return self.sigma_func(s, tt, sedov_class, g_interpolated)
        else:
            return self.sigma_func(s, t, sedov_class, g_interpolated)
        # return self.sigma_func(s, t, sedov_class, g_interpolated)
    
    def heaviside(self, x):
        res = x * 0.0
        for ix, xx in enumerate(x):
            if xx > 0.0:
                res[ix] = 1.0
        return res
    
    def uncollided_angular_flux(self, xs, tfinal, mu, sedov_class, g_interpolated, v_interpolated):
        heaviside_array = self.heaviside(mu - np.abs(xs + self.x0)/ (tfinal)) * self.heaviside(np.abs(-self.x0-xs) - (tfinal-self.t0source)*mu)
        mfp_array = xs * 0

        if mu > 0.0:
            for ix, xx, in enumerate(xs):
                if heaviside_array[ix] != 0.0:
                    mfp_array[ix] = self.integrate_sigma(xx, mu, tfinal, sedov_class, g_interpolated, v_interpolated)
            return np.exp(-mfp_array / mu) * heaviside_array
        else:
            return xs * 0.0
    
    def integrate_angular_flux(self, a, b, x, tfinal, sedov_class, g_interpolated, v_interpolated):
        func = self.mu_quad * 0
        argument = (b-a)/2*self.mu_quad + (a+b)/2

        for ix, xx in enumerate(self.mu_quad):
            func[ix] = self.uncollided_angular_flux(np.array([x]), tfinal, argument[ix], sedov_class, g_interpolated, v_interpolated)[0]
        res = (b-a)/2 * np.sum(self.mu_ws * func)
        
        return res

    def uncollided_scalar_flux(self, xs, tfinal, sedov_class, g_interpolated, v_interpolated):
        phi = xs * 0
        a = 0.0
        b = 1.0
        if tfinal > 0.0:
            for ix, xx in enumerate(xs):
                bb = 1.0
                if tfinal > self.t0source:
                    bb = min(1.0, abs(xx+self.x0)/ (tfinal - self.t0source))
                aa = abs(xx+self.x0) / tfinal
                if aa <= 1.0:
                    phi[ix] = self.integrate_angular_flux(aa, bb, xx, tfinal, sedov_class, g_interpolated, v_interpolated)
        return phi

        
