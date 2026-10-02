"""Reusable k-effective power iteration utilities for moving_mesh_transport.

This module contains the numerical machinery for a fixed-source k iteration.
Problem-specific material layouts are supplied through a callback instead of
being hard-coded into the iteration routine.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Callable, Optional

import numpy as np
import yaml
from scipy import integrate

from moving_mesh_transport.solver_classes.functions import (
    njit,
    normTn,
    normTn_intcell,
    normalize_fission_source,
)

LOGGER = logging.getLogger(__name__)

MaterialModel = Callable[[np.ndarray, np.ndarray, dict], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class KIterationResult:
    """Result of a k-effective power iteration."""

    k_history: list[float]
    iteration_times: list[float]
    source_history: list[float]
    run: object
    sigma_f_x: np.ndarray
    nu_x: np.ndarray
    scalar_flux: np.ndarray
    converged: bool
    iterations: int

    def as_legacy_tuple(self):
        """Return the historical tuple used by existing benchmark scripts."""
        return (
            self.k_history,
            self.iteration_times,
            self.source_history,
            self.run,
            self.sigma_f_x,
            self.nu_x,
            self.scalar_flux,
        )


def _log(verbose: bool, message: str, *args) -> None:
    if verbose:
        LOGGER.info(message, *args)


def coeff_for_const_one(a: float, b: float, n: int) -> float:
    """Coefficient of mode ``n`` for the constant function one on a cell."""
    if n == 0:
        return math.sqrt(math.pi) * math.sqrt(b - a)
    return 0.0


def make_fission_scalar_flux(
    coeffs: np.ndarray,
    edges: np.ndarray,
    ws: np.ndarray,
    n_angles: int,
    degree: int,
    n_space: int,
    n_groups: int,
    fission_vector: np.ndarray,
) -> np.ndarray:
    """Collapse angular coefficients to scalar-flux fission coefficients."""
    del n_angles, n_space, n_groups  # dimensions are carried by coeffs/edges
    phi = np.zeros((edges.size - 1, degree + 1))
    for cell in range(edges.size - 1):
        for mode in range(degree + 1):
            phi[cell, mode] = (
                np.sum(ws * coeffs[:, cell, mode]) * fission_vector[cell]
            )
    return phi


@njit
def integrate_phi_cell(phi_coeffs, a, b, degree):
    """Integrate a scalar-flux expansion over a spherical cell."""
    acc = 0.0
    for mode in range(degree + 1):
        acc += phi_coeffs[mode] * normTn_intcell(mode, a, b)
    return acc


@njit
def total_flux(values, edges, degree):
    """Integrate scalar flux over a spherical domain."""
    total = 0.0
    for cell in range(values.shape[0]):
        total += integrate_phi_cell(values[cell, :], edges[cell], edges[cell + 1], degree)
    return 4.0 * math.pi * total


@njit
def total_fission_production(values, edges, degree, sigma_f, nu):
    """Integrate total fission production over groups and cells."""
    n_groups, n_cells, _ = values.shape
    total = 0.0
    for group in range(n_groups):
        for cell in range(n_cells):
            cell_integral = 0.0
            for mode in range(degree + 1):
                cell_integral += values[group, cell, mode] * normTn_intcell(
                    mode, edges[cell], edges[cell + 1]
                )
            total += nu[group] * sigma_f[group, cell] * cell_integral
    return 4.0 * math.pi * total


@njit
def renormalize_fission_source(values, edges, degree, sigma_f, nu, target):
    """Scale flux coefficients to a requested total fission production."""
    production = total_fission_production(values, edges, degree, sigma_f, nu)
    scale = target / production
    values *= scale
    return scale, target



def check_norm_flux(verbose: bool = False) -> list[tuple[int, float, float]]:
    """Diagnostic comparison of cell-wise and whole-domain basis integrals."""
    results = []
    radius = 4.5
    for n_cells in (10, 20, 30, 40, 60):
        edges = np.linspace(0.0, radius, n_cells + 1)
        coeffs = np.array([coeff_for_const_one(edges[i], edges[i + 1], 0) for i in range(n_cells)])
        cell_sum = 4.0 * math.pi * sum(
            coeffs[i] * normTn_intcell(0, edges[i], edges[i + 1])
            for i in range(n_cells)
        )
        whole = 4.0 * math.pi * normTn_intcell(0, 0.0, radius) * np.sum(coeffs)
        results.append((n_cells, float(cell_sum), float(whole)))
        _log(verbose, "normalization diagnostic N=%d: cells=%.12g, whole=%.12g", n_cells, cell_sum, whole)
    return results

def test_normTnintcell() -> None:
    """Regression test for the analytic cell-basis integral."""
    from scipy.interpolate import interp1d

    for mode in range(3):
        for n_cells in (10, 20, 30, 40, 50):
            edges = np.linspace(0.0, 4.5, n_cells + 1)
            for cell in range(edges.size - 1):
                xs = np.linspace(edges[cell], edges[cell + 1], 15000)
                interpolant = interp1d(
                    xs, normTn(mode, xs, edges[cell], edges[cell + 1])
                )
                numerical = integrate.quad(
                    lambda x: interpolant(x) * x**2,
                    edges[cell],
                    edges[cell + 1],
                )[0]
                analytic = normTn_intcell(mode, edges[cell], edges[cell + 1])
                np.testing.assert_allclose(analytic, numerical, atol=1e-6, rtol=1e-6)


def build_fission_source(coeffs: np.ndarray, fission_vector: np.ndarray) -> np.ndarray:
    """Multiply coefficient arrays by a cell-wise fission vector."""
    source = coeffs.copy()
    for cell in range(coeffs.shape[1]):
        source[:, cell, :] *= fission_vector[cell]
    return source


def boundary_leakage_from_angular_flux(
    psi: np.ndarray, ws: np.ndarray, mus: np.ndarray, radius: float
) -> float:
    """Compute outward leakage at a spherical outer boundary."""
    mask = mus > 0.0
    return float(2.0 * math.pi * radius**2 * np.sum(ws[mask] * mus[mask] * psi[mask, -1]))


def wynn_epsilon(sequence: np.ndarray) -> np.ndarray:
    """Construct the Wynn epsilon tableau used for optional acceleration."""
    n = sequence.size
    width = n - 1
    tableau = np.zeros((n + 1, width + 2))
    tableau[1:, 1] = sequence
    for col in range(2, width + 2):
        for row in range(col, n + 1):
            delta = tableau[row, col - 1] - tableau[row - 1, col - 1]
            tableau[row, col] = tableau[row - 1, col - 2] + 1.0 / delta
    return tableau


def transfer_coefficients(coeffs_old: np.ndarray, degree: int) -> np.ndarray:
    """Embed coefficients from a lower-order basis in a higher-order basis."""
    n_angles, n_cells, old_modes = coeffs_old.shape
    coeffs_new = np.zeros((n_angles, n_cells, degree + 1), dtype=coeffs_old.dtype)
    copy_modes = min(old_modes, degree + 1)
    coeffs_new[:, :, :copy_modes] = coeffs_old[:, :, :copy_modes]
    return coeffs_new


def uniform_material_model(
    edges: np.ndarray, xs: np.ndarray, parameters: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Default material model: uniform fission parameters everywhere."""
    n_cells = edges.size - 1
    sigma_f_cells = np.full(n_cells, float(parameters["all"]["sigma_f"]))
    nu_cells = np.full(n_cells, float(parameters["all"]["nu"]))
    chi_cells = np.full(n_cells, float(parameters["all"]["chi"]))
    return sigma_f_cells, nu_cells, chi_cells


def material_arrays_on_points(
    xs: np.ndarray,
    parameters: dict,
    material_model: Optional[MaterialModel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a material model on plotting/output points.

    The callback interface is cell based, so point locations are represented as
    degenerate cells here. A problem-specific model may ignore ``edges`` and use
    the supplied centers directly.
    """
    if material_model is None:
        sigma_f = np.full(xs.size, float(parameters["all"]["sigma_f"]))
        nu = np.full(xs.size, float(parameters["all"]["nu"]))
        chi = np.full(xs.size, float(parameters["all"]["chi"]))
        return sigma_f, nu, chi

    # Build tiny intervals whose centers are exactly xs.
    if xs.size == 1:
        edges = np.array([xs[0] - 0.5, xs[0] + 0.5])
    else:
        mids = 0.5 * (xs[:-1] + xs[1:])
        edges = np.concatenate(([xs[0] - (mids[0] - xs[0])], mids, [xs[-1] + (xs[-1] - mids[-1])]))
    return material_model(edges, xs, parameters)


def _prepare_material_vectors(run, material_model: Optional[MaterialModel]):
    if material_model is None:
        material_model = uniform_material_model
    return material_model(run.edges, run.xs, run.parameters)


def _write_first_step(mesh_yaml: str, first_step: float) -> None:
    with open(mesh_yaml, "r") as stream:
        data = yaml.safe_load(stream)
    data["first_step"] = float(first_step)
    data["dense"] = True
    data["eval_times"] = False
    with open(mesh_yaml, "w") as stream:
        yaml.dump(data, stream, sort_keys=False)


def power_iterate_result(
    kguess: float,
    transport_parameters: str,
    mesh_parameters: str,
    run,
    *,
    tol: float = 1e-12,
    use_we_accel: bool = False,
    max_its: int = 100,
    coarse_angles: int = 4,
    coarse_solve: bool = False,
    input_phi: Optional[np.ndarray] = None,
    input_psi=None,
    ss_tol: float = 1e-10,
    coarse_M: int = 0,
    precon_mat=None,
    material_model: Optional[MaterialModel] = None,
    verbose: bool = False,
    plot: bool = False,
    strict_convergence: bool = False,
    source_yaml: str = "moving_mesh_transport/input_scripts/Kornreich.yaml",
    coarse_yaml: str = "moving_mesh_transport/input_scripts/Kornreich_new.yaml",
    mesh_yaml: str = "moving_mesh_transport/input_scripts/mesh_parameters_Kornreich.yaml",
    coarse_transport_parameters: str = "Kornreich_new",
) -> KIterationResult:
    """Run fixed-source power iteration for ``k_eff``.

    ``material_model`` is the principal extension point for other problems. It
    receives ``(edges, xs, parameters)`` and returns cell-wise
    ``(sigma_f, nu, chi)`` arrays.
    """
    del input_psi, ss_tol  # retained for API compatibility

    if coarse_solve:
        with open(source_yaml, "r") as stream:
            data = yaml.safe_load(stream)
        data["all"]["Ms"][0] = int(coarse_M)
        data["fixed_source"]["N_angles"][0] = int(coarse_angles)
        with open(coarse_yaml, "w") as stream:
            yaml.dump(data, stream, sort_keys=False)
        run.load(coarse_transport_parameters, mesh_parameters)
    else:
        run.load(transport_parameters, mesh_parameters)

    n_angles = int(run.parameters["fixed_source"]["N_angles"][0])
    if run.parameters["all"]["angular_derivative"]["diamond"]:
        n_angles += 1
    n_groups = int(run.parameters["all"]["N_groups"])
    degree = int(run.parameters["all"]["Ms"][0])
    n_space = int(run.parameters["all"]["N_spaces"][0])

    sigma_f_cells, nu_cells, _ = _prepare_material_vectors(run, material_model)
    sigma_f_x, nu_x, _ = material_arrays_on_points(run.xs, run.parameters, material_model)

    at = float(run.parameters["all"]["at"])
    rt = float(run.parameters["all"]["rt"])
    at_schedule = np.logspace(-1, np.log10(at), 3)
    rt_schedule = np.logspace(-1, np.log10(rt), 3)

    if coarse_solve:
        run.parameters["all"]["rt"] = 1.0
        run.parameters["all"]["at"] = 1e-3
        run.parameters["all"]["integrator"] = "Euler"
        run.parameters["all"]["kold"] = float(kguess)

    start = time.time()
    if coarse_solve:
        run.custom_source(randomstart=True, uncollided=0, moving=0)
    elif input_phi is not None:
        run.parameters["all"]["kold"] = float(kguess)
        phi_guess = transfer_coefficients(input_phi, degree)
        transfer_source = make_fission_scalar_flux(
            phi_guess,
            run.edges,
            run.ws,
            n_angles,
            degree,
            n_space,
            n_groups,
            sigma_f_cells * nu_cells,
        )
        run.parameters["all"]["rt"] = float(rt_schedule[0])
        run.parameters["all"]["at"] = float(at_schedule[0])
        transfer_source = normalize_fission_source(
            transfer_source, n_space, 0, 1.0 / kguess, run.edges
        )
        run.custom_source(
            randomstart=False,
            uncollided=0,
            moving=0,
            input_phi_coeffs=phi_guess,
            sol_coeffs=transfer_source,
            input_A=precon_mat,
        )
    else:
        run.custom_source(randomstart=True, uncollided=0, moving=0)

    first_time = time.time() - start
    coeffs_old = np.copy(
        run.sol_ob.y[:, -1].reshape((n_angles * n_groups, n_space, degree + 1))
    )
    initial_condition = run.fission_source

    new_source = make_fission_scalar_flux(
        coeffs_old,
        run.edges,
        run.ws,
        n_angles,
        degree,
        n_space,
        n_groups,
        sigma_f_cells * nu_cells,
    )
    initial_source = make_fission_scalar_flux(
        initial_condition,
        run.edges,
        run.ws,
        n_angles,
        degree,
        n_space,
        n_groups,
        sigma_f_cells * nu_cells,
    )

    from moving_mesh_transport.solver_classes.functions import normalize_phi

    source_new = float(normalize_phi(new_source, run.edges, run.ws, n_angles, degree, n_space, n_groups))
    source_old = float(normalize_phi(initial_source, run.edges, run.ws, n_angles, degree, n_space, n_groups))

    k_history = [float(kguess), source_new]
    source_history = [source_old, source_new]
    iteration_times = [first_time]
    k_old = source_new
    old_source = normalize_fission_source(new_source, n_space, degree, 1.0 / k_old, run.edges)
    n_iters = 1
    converged = False

    _log(verbose, "Initial k estimate: %.12g", k_old)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure("k_it scalar flux")
        plt.plot(run.xs, run.phi[:, -1], "--", label="initial")
        plt.legend()

    while not converged and n_iters < max_its:
        run.load(coarse_transport_parameters if coarse_solve else transport_parameters, mesh_parameters)
        if n_iters < 3:
            run.parameters["all"]["at"] = float(at_schedule[n_iters])
            run.parameters["all"]["rt"] = float(rt_schedule[n_iters])
        run.parameters["all"]["kold"] = float(k_old)

        start = time.time()
        run.custom_source(
            randomstart=False,
            sol_coeffs=old_source,
            phi_coeffs=coeffs_old,
            uncollided=0,
            moving=0,
            input_A=precon_mat,
        )
        iteration_times.append(time.time() - start)

        ts = run.sol_ob.t
        if len(ts) > 1:
            _write_first_step(mesh_yaml, float(ts[1] - ts[0]))

        coeffs_new = run.sol_ob.y[:, -1].reshape(
            (n_angles * n_groups, n_space, degree + 1)
        )
        new_source = make_fission_scalar_flux(
            coeffs_new,
            run.edges,
            run.ws,
            n_angles,
            degree,
            n_space,
            n_groups,
            sigma_f_cells * nu_cells,
        )
        k_new = float(
            normalize_phi(new_source, run.edges, run.ws, n_angles, degree, n_space, n_groups)
        )

        delta_k = abs(k_new - k_old)
        k_history.append(k_new)
        source_history.append(k_new)
        _log(verbose, "k iteration %d: k=%.12g, |dk|=%.3e", n_iters, k_new, delta_k)

        coeffs_old = coeffs_new
        old_source = normalize_fission_source(new_source, n_space, degree, 1.0 / k_new, run.edges)

        if delta_k <= tol:
            converged = True
            k_old = k_new
            break

        k_old = k_new
        if use_we_accel and n_iters > 4:
            tableau = wynn_epsilon(np.asarray(k_history))
            index = n_iters - 1 if n_iters % 2 == 0 else n_iters
            accelerated = tableau[index:, index]
            if accelerated.size and np.isfinite(accelerated[-1]):
                k_old = float(accelerated[-1])
                _log(verbose, "Wynn-accelerated k guess: %.12g", k_old)

        n_iters += 1

    if strict_convergence and not converged:
        raise RuntimeError(
            f"k iteration failed to converge after {max_its} iterations; "
            f"last k={k_history[-1]:.12g}"
        )

    return KIterationResult(
        k_history=k_history,
        iteration_times=iteration_times,
        source_history=source_history,
        run=run,
        sigma_f_x=sigma_f_x,
        nu_x=nu_x,
        scalar_flux=run.phi[:, -1],
        converged=converged,
        iterations=n_iters,
    )


def power_iterate(*args, **kwargs):
    """Backward-compatible wrapper returning the historical seven-item tuple."""
    return power_iterate_result(*args, **kwargs).as_legacy_tuple()
