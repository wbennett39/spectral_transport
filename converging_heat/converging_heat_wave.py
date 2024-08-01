import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from .heat_wave import AbstractHeatWave, Units
import h5py
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConvergingHeatWave")

EVENT_OCCURED = 1
EPSILON = sys.float_info.epsilon

class ConvergingHeatWave(AbstractHeatWave):
    """An object to compute the solution to the Converging Marshak heat wave problem
    This problem has a similarity solution of the second kind.

    The input are the general power law heat wave pramemeters (see the AbstractHeatWave object).
    The medium has a spatial power law density of the form:

    rho(r) = rho0 * r ** (-omega)

    The total (radiation+matter) energy per unit volume is given by a temperature and density power law:
    u(T, rho) = f * T ** beta * rho ** (1-mu)
    
    Similarly, the Rosseland oapcity is given as:
    1/kappa_Rosseland(T, rho) = g * T ** alpha * rho ** -lambda

    """

    def __init__(self, *,
                 geometry:str,       # symmetry of the problem: can be 'spherical', 'cylindrical', 'planar'
                 rho0:float,         # spatial density coeff
                 omega:float,        # spatial density power
                 g:float,            # opacity coeff
                 alpha:float,        # opacity temmperature power
                 lambdap:float,      # opacity density power
                 f:float,            # energy coeff
                 beta:float,         # energy temmperature power
                 mu:float,           # energy density power
                 
                 # --- length and times scales (in c.g.s.) of the problem
                 r_front:float=1.,   # position of heat wave front at the given time 't_front'
                 t_front:float=-1.,  # the given time

                 #  Numerical parameters
                 Z0:float=-1e-5      # the initial Z coordinate of integration in the Z-V plance
                 ):
        
        super().__init__(rho0=rho0,
                         omega=omega,
                         g=g,
                         alpha=alpha,
                         lambdap=lambdap,
                         f=f,
                         beta=beta,
                         mu=mu)

        self.m = omega*(1.-mu)
        self.k = omega*(1+lambdap)
        
        self.A = 16.*Units.sigma_sb*g/(3.*beta*f**((4.+alpha)/beta)*rho0**(lambdap+1.+(1.-mu)*(alpha+4.)/beta))

        logger.info(f"geometry={geometry!r}")
        if geometry == "spherical":     d=3
        elif geometry == "cylindrical": d=2
        elif geometry == "planar":      d=1
        else:
            logger.fatal(f"invalid given geoemtry")
            sys.exit(1)

        self.dim = d
        
        self.a = -(self.m-d+1.)
        self.b = -(self.k+self.m)
        self.n = (4. + alpha - beta)/beta

        self.Z0 = Z0
        self.ell = -1./(2.+self.b)
        self.fac = self.A**self.ell
        self.r_front = r_front
        self.t_front = t_front

        logger.info(f"a={self.a}")
        logger.info(f"b={self.b}")
        logger.info(f"n={self.n}")
        logger.info(f"Z0={self.Z0}")
        logger.info(f"A={self.A}")
        logger.info(f"r_front={self.r_front} [cm]")
        logger.info(f"t_front={self.t_front} [sec]")

        assert self.a >= 0
        assert self.n >= 1, f"n = {self.n}"
        assert self.b > -0.7 and self.b <= 29., f"b = {self.b}"
        assert self.Z0 < 0.
        assert self.t_front < 0.
        assert self.r_front > 0.

        self.kappa = (2.0 + self.b) + self.n * (self.a + 1.0)

        self.max_delta = 1.
        self.min_delta = 1. / (2. + self.b)

        self.delta = None
        self.Z_negative_time = None
        self.V_negative_time = None
        self.max_xsi_negative_time = None

        self.event_lambda = lambda Z, V_arr, delta: zero_slope_event(Z, V_arr, n=self.n, b=self.b)
        self.event_lambda.terminal = True

        self.calc_delta()
        self.B = r_front*self.fac / (np.abs(t_front)**self.delta)
        logger.info(f"B={self.B}")
        logger.info(f"r_front(t/ns)={self.B/self.fac*(Units.nsec)**self.delta}*(-t/ns)^{self.delta:g}")

        self.create_interpolation_functions()

    def get_density(self, r):
        return self.rho0*r**(-self.omega)
    
    def get_energy_volume(self, *, w, r):
        return w / r**self.m
    
    def get_temperature(self, *, w, r):
        energy_volume = self.get_energy_volume(w=w, r=r)
        return self.energy_volume_to_temperature(energy_volume, self.get_density(r))
    
    def get_flux(self, *, w, v, r):
        return self.A**(self.ell*(self.b+1.)+1.)*r**(self.k + self.b)*w*v

    def calc_delta(self, delta_initial_guess=None):
        """
        Calculate the similarity exponent delta as the root of self.func
        """

        logger.info(f"Begin calculation of delta...")

        if delta_initial_guess is None:
            delta_initial_guess = 0.5*(self.max_delta + self.min_delta)

        assert delta_initial_guess < self.max_delta and delta_initial_guess > self.min_delta, f"delta initial guess = {delta_initial_guess:g}"
        self.delta = root(self.func, x0=[delta_initial_guess], tol=1e-8).x[0]
        err = abs(self.func([self.delta])[0])
        logger.info(f"Found delta={self.delta} err={err:g}")
        assert err < 1e-8
        return self

    def func(self, delta_arr):
        """
        The root of this function has the correct delta
        Calculates the difference between the end points of the integration from A and O
        """
        delta = delta_arr[0]
        if delta >= self.max_delta or delta <= self.min_delta: return [1.]

        sol_A = self.integrate_from_A(delta)
        sol_O = self.integrate_from_O(delta)

        if sol_A.status == EVENT_OCCURED and sol_O.status == EVENT_OCCURED:
            return [sol_A.t_events[0][0] - sol_O.t_events[0][0]]
        
        return [sol_A.t[-1] - sol_O.t[-1]]
    
    def integrate_from_A(self, delta, dense_output=False):
        """
        Integration from A=(0, delta) until a point with an infinite slope
        """
        V0 = delta
        V0 += (self.kappa*delta-1.) * self.Z0 / (self.n*(self.n+1.))

        Zmax = -self.n*delta

        solution = solve_ivp(self.dVdZ, t_span=(self.Z0, Zmax), y0=[V0], args=[delta], events=self.event_lambda, rtol=1e-8, method='LSODA', dense_output=dense_output)
        
        return solution

    def integrate_from_O(self, delta:float, dense_output:bool=False):
        """
        Integration from O=(0, 0) until point with an infinite slope
        """
        V0 = - (2.*delta - 1.) / (self.n*delta) * self.Z0
        
        alpha = -((2.+self.b)*delta - 1.) / (self.n*delta)
        gamma = -alpha*(self.kappa-1.-self.n)/(self.n*delta**2.)
        
        V0 = alpha*self.Z0 + gamma*self.Z0**2
        
        Zmax = -self.n*delta

        solution = solve_ivp(self.dVdZ, t_span=(self.Z0, Zmax), y0=[V0], args=[delta], events=self.event_lambda, rtol=1e-8, method='LSODA', dense_output=dense_output)
        
        return solution
    
    def dVdZ(self, Z, V_arr, delta:float):
        V = V_arr[0]

        numer = Z*((2.+self.b)*delta-1.)
        numer += self.n*(self.a+1.)*V*Z
        numer += self.n*(delta-V)*V

        denom = self.n*Z*((2.+self.b)*Z + self.n*V)

        return [numer / denom]
    
    def dln_xsi_dZ(self, Z, ln_xsi_arr, V_Z):
        """
        dln_xsi/dZ
        V_Z is a function V(Z)
        """
        return [-1.0/((2.+self.b)*Z + self.n*V_Z(Z))]

    def create_interpolation_functions(self):
        """
        Creates the functions V(ln_xsi), Z(ln_xsi)
        """
        assert self.delta is not None
        
        # integrate from A and O
        sol_V_Z_A = self.integrate_from_A(self.delta, dense_output=True)
        sol_V_Z_O = self.integrate_from_O(self.delta, dense_output=True)

        assert sol_V_Z_A.status == EVENT_OCCURED and sol_V_Z_O.status == EVENT_OCCURED

        # create V(Z) from A and from O, since V(Z) is not a function
        # notice integration is from 0.0 since the integration sol_V_Z_A is from Z0
        V_Z_from_A = interp1d([0.0, *sol_V_Z_A.t], [self.delta, *sol_V_Z_A.y[0]], kind='linear', bounds_error=False, fill_value=self.delta)
        V_Z_from_O = interp1d(sol_V_Z_O.t, sol_V_Z_O.y[0], kind='linear', bounds_error=True)
        
        # end Z of integration from A
        Zend_A = sol_V_Z_A.t_events[0][0]
        
        # we split the ln_eta(Z) integration too because we give the functions V_Z_from_A, V_Z_from_O as args
        sol_ln_xsi_Z_A = solve_ivp(self.dln_xsi_dZ, t_span=(0., Zend_A), y0=[0.0], args=[V_Z_from_A], method='LSODA', rtol=1e-12, atol=1e-8)

        # starting from close to the point of infinte slope (else we have infinite slope and an error)
        Zstart_O = sol_V_Z_O.t_events[0][0]*(1.- 1e3*EPSILON)
        ln_xsi_end_A_start_O = sol_ln_xsi_Z_A.y[0][-1]
        
        sol_ln_xsi_Z_O = solve_ivp(self.dln_xsi_dZ, t_span=(Zstart_O, self.Z0), y0=[ln_xsi_end_A_start_O], args=[V_Z_from_O], method='LSODA', rtol=1e-12, atol=1e-8)

        # sol_ln_xsi_Z_A end at the same ln_xsi that sol_ln_xsi_Z_O starts
        # Z(ln_xsi)
        Z_ln_xsi = interp1d(np.append(sol_ln_xsi_Z_A.y[0][:-1], sol_ln_xsi_Z_O.y[0]), np.append(sol_ln_xsi_Z_A.t[:-1], sol_ln_xsi_Z_O.t), kind='linear', bounds_error=True)

        ln_xsi_end = sol_ln_xsi_Z_O.y[0][-1]

        ln_xsi_from_A_grid = np.linspace(0.0, ln_xsi_end_A_start_O, int(1e4), endpoint=False)
        ln_xsi_from_O_grid = np.linspace(ln_xsi_end_A_start_O, ln_xsi_end, int(1e4))

        V_on_grid = np.append(V_Z_from_A(Z_ln_xsi(ln_xsi_from_A_grid)), V_Z_from_O(Z_ln_xsi(ln_xsi_from_O_grid)))

        # V(ln_xsi) = V(Z(ln_xsi))
        V_ln_xsi = interp1d(np.append(ln_xsi_from_A_grid, ln_xsi_from_O_grid), V_on_grid, kind='linear', bounds_error=True)

        self.V_negative_time = V_ln_xsi
        self.Z_negative_time = Z_ln_xsi

        # maximal xsi we can solve
        self.max_xsi_negative_time = np.exp(ln_xsi_end)
    
    def solve(self, r:np.ndarray, t:float):
        """
        returns the solution at time `t` on a given radial grid `r`
        """
        assert self.delta is not None
        assert (self.V_negative_time is not None) and (self.Z_negative_time is not None) and self.max_xsi_negative_time is not None

        assert t < 0.
        x = self.fac*r
        xsi = x / (self.B*np.abs(t)**self.delta)
        
        Z = np.zeros_like(xsi)
        V = np.zeros_like(xsi)

        xsi_before_front = xsi[xsi < 1.]

        if any(xsi > self.max_xsi_negative_time):
            max_r = self.max_xsi_negative_time * self.B*np.abs(t)**self.delta / self.fac
            logger.warning(f"Some points are outside the limits, solving from r=0.0 up to r={max_r:g}")
            logger.warning(f"RETURNING NaN FOR THE UNSOLVED r's")

        xsi_after_front = xsi[np.logical_and(xsi >= 1., xsi <= self.max_xsi_negative_time)]

        ln_xsi_after_front = np.log(xsi_after_front)

        start_index = len(xsi_before_front)
        end_index = start_index+len(xsi_after_front)
        
        Z[start_index:end_index] = self.Z_negative_time(ln_xsi_after_front)
        V[start_index:end_index] = self.V_negative_time(ln_xsi_after_front)

        Z[end_index:] = np.nan
        V[end_index:] = np.nan

        w = (x**(2+self.b)/np.abs(t)*np.abs(Z))**(1./self.n)
        v = x/t*V

        energy_volume = self.get_energy_volume(w=w, r=r)
        temperature = self.get_temperature(w=w, r=r)
        flux = self.get_flux(w=w, v=v, r=r)

        return dict(
            energy_volume=energy_volume,
            temperature=temperature,
            flux=flux,
            w=w,
            v=v,
            Z=Z,
            V=V,
            i=end_index,
        )
    
    def xsi_over_r(self, *, time):
        assert time < 0.
        return self.B / self.fac*np.abs(time)**self.delta

    def heat_wave_position(self, *, t:float):
        assert self.delta is not None
        assert t <= 0.

        return self.B/self.fac*np.abs(t)**self.delta
    
    def heat_wave_time(self, *, r:float):
        assert self.delta is not None
        assert r >= 0.

        return -(r*self.fac/self.B)**(1./self.delta)

    def temperature_bc(self, *, r, t):
        assert r > 0
        assert t < 0.
        assert self.V_negative_time is not None and self.Z_negative_time is not None

        sol = self.solve(np.array([r]), t)
        w_bc = sol['w'][0]
        T_bc = self.get_temperature(w=w_bc, r=r)
        
        return T_bc
    
    def temperature_bath_bc(self, *, r, t):
        assert r > 0.
        assert t < 0.
        assert self.V_negative_time is not None and self.Z_negative_time is not None

        sol = self.solve(np.array([r]), t)

        w_bc, v_bc = sol['w'][0], sol['v'][0]

        T_bc = self.get_temperature(w=w_bc, r=r)
        F_bc = self.get_flux(w=w_bc, v=v_bc, r=r)

        T_bath = (T_bc**4. - F_bc*2./(Units.arad*Units.clight))**0.25

        return T_bath

def zero_slope_event(Z, V_arr, n, b):
    return ((2.+b)*Z + n*V_arr[0])

if __name__ == "__main__":

    geometry = "spherical"
    Tscale = Units.hev_kelvin
    rho0 = 1.
    omega = 0.
    beta, mu = 1.6, 0.
    alpha, lambdap = 1.5, 0.
    f = 1e13*Tscale**(-beta)
    k0_cm = 5e3
    g = 1. / (Tscale**alpha * k0_cm)
    L = 1e-1 #system size
    fanal = f
    
    t_end = -1.*Units.nsec

    solver = ConvergingHeatWave(
        geometry=geometry,
        rho0=rho0,
        omega=omega,
        g=g,
        alpha=alpha,
        lambdap=lambdap,
        f=fanal,
        beta=beta,
        mu=mu,

        # scale the solution such that at time -1ns the heat front will be at r=L/10
        t_front=-1.*Units.nsec,
        r_front=L/10.,
    )

    t_init = solver.heat_wave_time(r=L)
    assert abs(t_end/solver.heat_wave_time(r=L/10.)-1.) < 1e-10

    # plot the boundary condition of the drive:
    # bath and surface temperatures applied at r=L
    times = np.geomspace(t_init, t_end, 10000)
    Tbath = np.array([solver.temperature_bath_bc(t=t, r=L) for t in times])
    Tsurface = np.array([solver.temperature_bc(t=t, r=L) for t in times])
    plt.plot(times/Units.nsec, Tbath/Units.hev_kelvin, c="r", ls="-", label="Bath temperature")
    plt.plot(times/Units.nsec, Tsurface/Units.hev_kelvin, c="b", ls="--", label="Surface temperature")
    plt.legend()
    plt.grid()
    plt.xlabel("time [ns]")
    plt.ylabel("$T(t)$ [HeV]")
    plt.ylim(ymin=0.)
    plt.tight_layout()
    plt.show()
    
    f = h5py.File('heat_wavepos.h5', 'r+')
    del f['temperature']
    del f['times']
    f.create_dataset('temperature', data = Tbath/Units.hev_kelvin )
    tm = times/Units.nsec
    print(tm + 29.6255)
    

    f.create_dataset('times', data = times/Units.nsec)
    f.close()

    # plot the heat front position as a function of time
    rfront = np.array([solver.heat_wave_position(t=t) for t in times])
    plt.plot(times/Units.nsec, rfront, c="b")
    plt.legend()
    plt.grid()
    plt.xlabel("time [ns]")
    plt.ylabel("$r_{{\\mathrm{{front}}}}(t)$ [cm]")
    plt.ylim(ymin=0.)
    plt.tight_layout()
    plt.show()

    # plot temperature, energy density and flux profiles at several times
    r = np.linspace(0., L, 1000)
    times = [0.95*t_init, solver.heat_wave_time(r=L*0.8), solver.heat_wave_time(r=L*0.5), solver.heat_wave_time(r=L*0.3), t_end]

    for t in times:
        
        sol = solver.solve(r, t)

        plt.figure("T")
        plt.plot(r, sol['temperature']/Units.hev_kelvin, label=f"$t={t/Units.nsec:g}$ ns")

        plt.figure("u")
        plt.plot(r, sol['energy_volume']/1e13, label=f"$t={t/Units.nsec:g}$ ns")

        plt.figure("flux")
        plt.plot(r, sol['flux'], label=f"$t={t/Units.nsec:g}$ ns")

    plt.figure("T")
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.grid()
    plt.legend()
    plt.xlabel("r [cm]")
    plt.ylabel("$T$ [HeV]")
    plt.tight_layout()

    plt.figure("u")
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.grid()
    plt.legend()
    plt.xlabel("r [cm]")
    plt.ylabel("$u$ [$10^{{13}}$erg/cm$^3$]")
    plt.tight_layout()

    plt.figure("flux")
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.grid()
    plt.legend()
    plt.xlabel("r [cm]")
    plt.ylabel("$F$ [erg/cm$^2$/sec]")
    plt.tight_layout()



    plt.show()


def run_converging(tt):

    t = tt - 29.6255
    geometry = "spherical"
    Tscale = Units.hev_kelvin
    rho0 = 1.
    omega = 0.
    beta, mu = 1.6, 0.
    alpha, lambdap = 1.5, 0.
    f = 1e13*Tscale**(-beta)
    k0_cm = 5e3
    g = 1. / (Tscale**alpha * k0_cm)
    L = 1e-1 #system size
    fanal = f
    
    t_end = t*Units.nsec
    # print(Units.nsec)
    
    

    solver = ConvergingHeatWave(
        geometry=geometry,
        rho0=rho0,
        omega=omega,
        g=g,
        alpha=alpha,
        lambdap=lambdap,
        f=fanal,
        beta=beta,
        mu=mu,

        # scale the solution such that at time -1ns the heat front will be at r=L/10
        t_front=-1.*Units.nsec,
        r_front=L/10.,
    )
    t_init = solver.heat_wave_time(r=L)
    times = np.geomspace(t_init, t_end, 1000)

    
#    assert abs(t_end/solver.heat_wave_time(r=L/10.)-1.) < 1e-10
    r = np.linspace(0., L, 1000)
    print(times[-1])
    sol = solver.solve(r, t*Units.nsec)
    
    return r, sol['temperature']/Units.hev_kelvin

