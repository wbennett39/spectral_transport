"""Small standalone k-effective driver for moving_mesh_transport.

This file keeps the historical ``power_iterate`` entry point but removes import-
time execution, hard-coded plotting, and unconditional printing. For reusable
benchmark work, prefer the more general implementation in ``k_iterate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate

from moving_mesh_transport.solver_classes.functions import normTn_intcell
from moving_mesh_transport.solver_classes.make_phi import make_output
from moving_mesh_transport.solver_functions.run_functions import run as Run

LOGGER = logging.getLogger(__name__)


@dataclass
class SimpleKResult:
    k_history: list[float]
    normalization_history: list[float]
    iteration_times: list[float]
    run: object
    converged: bool

    @property
    def k_eff(self) -> float:
        return self.k_history[-1]


def configure_logging(verbose: bool = False) -> None:
    LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)
    if verbose and not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.propagate = False


def integrate_phi_cell(cs, ws, a: float, b: float, degree: int, n_angles: int) -> float:
    """Integrate angular expansion coefficients over one spherical cell."""
    psi = np.zeros(n_angles)
    for angle in range(n_angles):
        for mode in range(degree + 1):
            psi[angle] += cs[angle, mode] * normTn_intcell(mode, a, b)
    return float(4.0 * math.pi * np.sum(psi * ws))


def normalize_phi(
    values: np.ndarray,
    edges: np.ndarray,
    ws: np.ndarray,
    n_angles: int,
    degree: int,
    n_space: int,
    n_groups: int,
    sigma_f=None,
    nu=None,
    chi=None,
) -> float:
    """Integrate scalar flux over all cells and energy groups."""
    del sigma_f, nu, chi  # retained for historical call compatibility
    total = 0.0
    for group in range(n_groups):
        for cell in range(n_space):
            total += integrate_phi_cell(
                values[group * n_angles : (group + 1) * n_angles, cell, :],
                ws,
                edges[cell],
                edges[cell + 1],
                degree,
                n_angles,
            )
    return total


def check_normalization(output_ob, uncollided_ob, xs, *, verbose: bool = False, plot: bool = False) -> float:
    """Numerically integrate reconstructed scalar flux for diagnostics."""
    phi = output_ob.make_phi(uncollided_ob)[:, 0]
    interpolant = interp1d(xs, phi)
    integral = integrate.quad(
        lambda x: interpolant(x) * x**2 * 4.0 * math.pi,
        xs[0],
        xs[-1],
    )[0]
    if verbose:
        LOGGER.info("reconstructed flux normalization = %.12g", integral)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(xs, interpolant(xs) * xs**2 * 4.0 * math.pi)
        ax.set_xlabel("x")
        ax.set_ylabel("integrand")
        plt.show()
        plt.close(fig)
    return float(integral)


def power_iterate(
    kguess: float = 0.5,
    tol: float = 1e-5,
    *,
    transport_parameters: str = "k_eff",
    mesh_parameters: str = "mesh_parameters_keff",
    n_spaces: Optional[int] = None,
    degree: Optional[int] = None,
    n_angles: Optional[int] = None,
    max_iterations: int = 100,
    verbose: bool = False,
    plot: bool = False,
    return_result: bool = False,
):
    """Run the historical simple k-effective iteration without import side effects."""
    configure_logging(verbose)
    solver = Run()
    solver.load(transport_parameters, mesh_parameters)

    if n_spaces is not None:
        solver.parameters["all"]["N_spaces"] = [int(n_spaces)]
    if degree is not None:
        solver.parameters["all"]["Ms"] = [int(degree)]
    if n_angles is not None:
        solver.parameters["random_IC"]["N_angles"] = [int(n_angles)]

    solver.custom_source(randomstart=True, uncollided=0, moving=0)

    n_angles_actual = int(solver.parameters["fixed_source"]["N_angles"][0])
    if solver.parameters["all"]["angular_derivative"]["diamond"]:
        n_angles_actual += 1
    n_groups = int(solver.parameters["all"]["N_groups"])
    degree_actual = int(solver.parameters["all"]["Ms"][0])
    n_space_actual = int(solver.parameters["all"]["N_spaces"][0])
    tfinal = float(solver.parameters["all"]["tfinal"])
    ws = solver.ws
    edges = solver.edges
    xs = solver.xs

    coeffs_old = np.copy(
        solver.sol_ob.y[:, -1].reshape(
            (n_angles_actual * n_groups, n_space_actual, degree_actual + 1)
        )
    )
    phi_old = interp1d(xs, solver.phi[:, 0])
    normalization = integrate.quad(
        lambda x: phi_old(x) * x**2 * 4.0 * math.pi,
        xs[0],
        xs[-1],
    )[0]

    k_old = float(kguess)
    k_history = [k_old]
    normalization_history = [float(normalization)]
    iteration_times = []
    converged = False

    for iteration in range(max_iterations):
        solver.load(transport_parameters, mesh_parameters)
        output_ob = make_output(
            tfinal,
            n_angles_actual,
            ws,
            xs,
            coeffs_old,
            degree_actual,
            edges,
            False,
            solver.geometry,
            n_groups,
        )
        if verbose:
            check_normalization(output_ob, solver.uncollided_ob, xs, verbose=True)

        start = time.time()
        solver.custom_source(
            randomstart=False,
            sol_coeffs=coeffs_old,
            uncollided=0,
            moving=0,
        )
        iteration_times.append(time.time() - start)

        coeffs_old = np.copy(
            solver.sol_ob.y[:, -1].reshape(
                (n_angles_actual * n_groups, n_space_actual, degree_actual + 1)
            )
        )
        phi_new = interp1d(solver.xs, solver.phi[:, 0])
        numerator = integrate.quad(
            lambda x: phi_new(x) * x**2 * 4.0 * math.pi,
            xs[0],
            xs[-1],
        )[0]
        denominator = integrate.quad(
            lambda x: phi_old(x) * x**2 * 4.0 * math.pi,
            xs[0],
            xs[-1],
        )[0]
        k_new = k_old * numerator / denominator
        if k_new < 0.0:
            raise ValueError(f"negative k_eff encountered: {k_new}")

        k_history.append(float(k_new))
        normalization_history.append(float(numerator))
        if verbose:
            LOGGER.info("iteration %d: k=%.12g, |dk|=%.3e", iteration + 1, k_new, abs(k_new - k_old))

        if plot:
            fig, ax = plt.subplots()
            ax.plot(xs, phi_old(xs), label="previous")
            ax.plot(xs, phi_new(xs), label="current")
            ax.legend()
            plt.show()
            plt.close(fig)

        if abs(k_new - k_old) <= tol:
            converged = True
            break
        k_old = float(k_new)
        phi_old = phi_new

    result = SimpleKResult(
        k_history=k_history,
        normalization_history=normalization_history,
        iteration_times=iteration_times,
        run=solver,
        converged=converged,
    )
    if verbose:
        LOGGER.info("k-effective %s at %.12g", "converged" if converged else "stopped", result.k_eff)

    if return_result:
        return result
    return result.k_eff


def main() -> None:
    power_iterate(verbose=True, plot=False)


if __name__ == "__main__":
    main()
