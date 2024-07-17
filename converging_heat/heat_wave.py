from abc import ABC, abstractmethod
import numpy as np

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('HeatWave')

class Units:
    sigma_sb = 5.670374419e-5
    clight = 2.99792458e10
    arad = 4. * sigma_sb / clight 
    ev_kelvin = 1.160451812e4
    hev_kelvin = 100. * ev_kelvin
    kev_kelvin = 1000. * ev_kelvin
    nsec = 1e-9
    
class AbstractHeatWave(ABC):
    """An abstract object to computes the solution to the Radiation Diffusion equation.

    The spatial density is given as:
    rho(r) = rho0 * r ** (-omega)

    The sie (energy per units mass) is given by:

    sie(T, rho) = f * (T ** beta) * (rho ** -mu)
    
    The Rosseland oapcity is given as:
    1/kappa_Rosseland(T, rho) = g * (T ** alpha) * (rho ** -lambda)
    """

    def __init__(self, *,
                 rho0,      # spatial density coeff
                 omega,     # spatial density power
                 g,         # opacity coeff
                 alpha,     # opacity temmperature power
                 lambdap,   # opacity density power
                 f,         # energy coeff
                 beta,      # energy temmperature power
                 mu,        # energy density power
                 ):
        
        super().__init__()
        logger.info(f"creating a HeatWave calculator...")

        self.rho0 = rho0
        self.omega = omega
        self.g = g
        self.alpha = alpha
        self.lambdap = lambdap
        self.f = f
        self.beta = beta
        self.mu = mu

        logger.info(f"rho0={self.rho0:g}")
        logger.info(f"omega={self.omega:g}")
        logger.info(f"g={self.g:g}")
        logger.info(f"alpha={self.alpha:g}")
        logger.info(f"lambdap={self.lambdap:g}")
        logger.info(f"f={self.f:g}")
        logger.info(f"beta={self.beta:g}")
        logger.info(f"mu={self.mu:g}")

        assert self.rho0 > 0.
        assert self.f > 0.
        assert self.g > 0.
        assert self.beta != 0.

        self.n = (self.alpha - self.beta + 4.) / self.beta
        logger.info(f"n={self.n:g}")
        assert self.n >= 0.

        self.nonlinear = self.n > 0.
        logger.info(f"nonlinear={self.nonlinear}")

        self.m = self.omega*(1. - self.mu)
        self.k = self.omega*(1. + self.lambdap)

        logger.info(f"k={self.k:g}")
        logger.info(f"m={self.m:g}")

        self.title = f"$    \\omega={self.omega:.4g}" \
                     f", \\ \\alpha={self.alpha:.4g}" \
                     f", \\ \\beta={self.beta:.4g}" \
                     f", \\ \\lambda={self.lambdap:.4g}" \
                     f", \\ \\mu={self.mu:.4g}$"
        
        # self.title = f"$    \\omega={self.omega:.4g}$"

        # ------- auxiliary lambda functions
        self.power_law_density = lambda rcell: self.rho0 * np.abs(rcell)**(-self.omega)

        # energy density <-> temperature conversion functions
        self.energy_volume_to_temperature = lambda energy_volume, density: (energy_volume/(self.f*density**(1.-self.mu)))**(1./self.beta)
        self.temperature_to_energy_volume = lambda temperature, density: self.f*density**(1.-self.mu)*temperature**self.beta

    @abstractmethod
    def xsi_over_r(self, *, time):
        pass
    
    @abstractmethod
    def heat_wave_position(self, *, time):
        """
        returns heat wave position at the given time
        """
        pass

    @abstractmethod
    def heat_wave_time(self, *, r):
        """
        returns time for heat wave to reach position r
        """
        pass
    
    @abstractmethod
    def solve(self, *, rcell, time):
        pass
    
    def solve_energy_volume(self, *, rcell, time):
        return self.solve(rcell=rcell, time=time)["energy_volume"]

    def solve_temperature(self, *, rcell, time):
        return self.solve(rcell=rcell, time=time)["temperature"]

    def solve_flux(self, *, rcell, time):
        return self.solve(rcell=rcell, time=time)["flux"]