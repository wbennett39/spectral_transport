"""Kornreich benchmark driver built on reusable transport eigenvalue utilities.

The module is intentionally split into small pieces:

* Kornreich-specific material/benchmark definitions.
* k-effective solution orchestration.
* DMD alpha estimation.
* inverse-operator IRAM alpha refinement.
* warm-started secant alpha refinement near criticality.
* result persistence and plotting.

Importing this module does not run a convergence study. Use ``run_converge`` or
call ``Kornreich_benchmark`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

from k_iterate import power_iterate_result
from moving_mesh_transport.loading_and_saving.load_solution import load_sol
from moving_mesh_transport.solver_classes.functions import normTn
from moving_mesh_transport.solver_functions.DMD_functions import DMD_func3
from moving_mesh_transport.solver_functions.run_functions import run as Run

LOGGER = logging.getLogger(__name__)

INPUT_DIR = Path("moving_mesh_transport/input_scripts")
KORNREICH_YAML = INPUT_DIR / "Kornreich.yaml"
KORNREICH_NEW_YAML = INPUT_DIR / "Kornreich_new.yaml"
KORNREICH_IRAM_YAML = INPUT_DIR / "Kornreich_IRAM.yaml"
KORNREICH_DMD_YAML = INPUT_DIR / "Kornreich_DMD.yaml"
MESH_YAML = INPUT_DIR / "mesh_parameters_Kornreich.yaml"
MESH_DMD_YAML = INPUT_DIR / "mesh_parameters_Kornreich_DMD.yaml"
RESULTS_DIR = Path("Kornreich_results")

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False



@dataclass(frozen=True)
class BenchmarkValues:
    k_eff: float
    alpha: float


@dataclass
class DMDResult:
    alphas: np.ndarray
    vectors: np.ndarray
    dominant_index: int

    @property
    def dominant_alpha(self) -> float:
        return float(np.real(self.alphas[self.dominant_index]))

    @property
    def dominant_vector(self) -> np.ndarray:
        return self.vectors[:, self.dominant_index]


@dataclass
class AlphaResult:
    alpha: float
    method: str
    converged: bool
    history: list[tuple[float, float]]
    candidates: Optional[np.ndarray] = None
    vectors: Optional[np.ndarray] = None


@dataclass
class KornreichResult:
    k_eff: float
    alpha: float
    alpha_dmd: float
    benchmark: BenchmarkValues
    k_converged: bool
    alpha_converged: bool
    run: object


BENCHMARKS = {
    (4.5, 1.5): BenchmarkValues(k_eff=0.4241317, alpha=-0.3229855),
    (4.5, 3.5): BenchmarkValues(k_eff=0.9896407, alpha=-0.006440766),
    (4.6, 1.5): BenchmarkValues(k_eff=0.4556758, alpha=-0.2932468),
    (4.6, 3.5): BenchmarkValues(k_eff=1.063244, alpha=0.03759991),
}


def configure_logging(verbose: bool = False) -> None:
    """Configure this module's log level without forcing global verbosity."""
    LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)
    if verbose and not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.propagate = False


def _load_yaml(path: Path) -> dict:
    with path.open("r") as stream:
        return yaml.safe_load(stream)


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w") as stream:
        yaml.dump(data, stream, sort_keys=False)


def update_yaml(path: Path, updater: Callable[[dict], None], output: Optional[Path] = None) -> Path:
    """Load, mutate, and write a YAML file."""
    data = _load_yaml(path)
    updater(data)
    destination = output or path
    _write_yaml(destination, data)
    return destination


def get_benchmark(x0: float, nu: float) -> BenchmarkValues:
    try:
        return BENCHMARKS[(float(x0), float(nu))]
    except KeyError as exc:
        raise ValueError(f"No Kornreich benchmark is registered for x0={x0}, nu={nu}") from exc


def kornreich_material_model(edges: np.ndarray, xs: np.ndarray, parameters: dict):
    """Return cell-wise fission data for the Kornreich benchmark.

    The central region ``[-3.5, 3.5]`` is non-fissioning. Outside it the YAML
    values for sigma_f, nu, and chi are used. The spatial shift is applied in
    the same coordinate convention as the original script.
    """
    del xs
    centers = 0.5 * (edges[:-1] + edges[1:])
    shift = float(parameters["fixed_source"].get("shift", 0.0))
    physical_centers = centers - shift

    sigma_f = np.full(centers.size, float(parameters["all"]["sigma_f"]))
    nu = np.full(centers.size, float(parameters["all"]["nu"]))
    chi = np.full(centers.size, float(parameters["all"]["chi"]))

    non_fissioning = (physical_centers >= -3.5) & (physical_centers < 3.5)
    sigma_f[non_fissioning] = 0.0
    nu[non_fissioning] = 0.0
    chi[non_fissioning] = 0.0
    return sigma_f, nu, chi


def basis(index: int, x, a: float, b: float):
    return normTn(index, x, a, b)


def rms(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((lhs - rhs) ** 2)))


# Historical alias.
RMS = rms


def coeffs_to_phi(
    coeffs: np.ndarray,
    xs: np.ndarray,
    n_angles: int,
    n_groups: int,
    edges: np.ndarray,
    ws: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Reconstruct group scalar flux from cell/basis angular coefficients."""
    psi = np.zeros((n_angles, xs.size, n_groups), dtype=np.result_type(coeffs, float))
    for group in range(n_groups):
        for angle in range(n_angles):
            for point_index, x in enumerate(xs):
                cell = np.searchsorted(edges, x)
                cell = min(max(cell, 1), edges.size - 1) - 1
                if edges[0] <= x <= edges[-1]:
                    for mode in range(degree + 1):
                        psi[angle, point_index, group] += coeffs[
                            group * n_angles + angle, cell, mode
                        ] * basis(mode, np.array([x]), float(edges[cell]), float(edges[cell + 1]))[0]

    phi = np.zeros((xs.size, n_groups), dtype=psi.dtype)
    for group in range(n_groups):
        phi[:, group] = np.sum(psi[:, :, group].T * ws, axis=1)
    return phi


def prepare_case_yaml(*, x0: float, nu: float, degree: int, n_spaces: int, n_angles: int, tf: float, euler_steps: int) -> None:
    """Update the primary Kornreich YAML for a benchmark case."""
    def mutate(data: dict) -> None:
        data["all"]["nu"] = float(nu)
        data["all"]["Ms"][0] = int(degree)
        data["all"]["N_spaces"][0] = int(n_spaces)
        data["all"]["tfinal"] = float(tf)
        data["all"]["Euler_dt_num"] = int(euler_steps)
        data["fixed_source"]["N_angles"][0] = int(n_angles)
        data["fixed_source"]["x0"][0] = float(x0)

    update_yaml(KORNREICH_YAML, mutate)


def solve_keff(
    run,
    *,
    guess_k: float,
    ktol: float,
    max_iterations: int,
    use_wynn: bool,
    coarse_solve: bool,
    verbose: bool,
    plot: bool,
):
    """Solve k-effective, optionally increasing polynomial order from M=0."""
    n_angles = int(run.parameters["fixed_source"]["N_angles"][0]) + 1
    n_spaces = int(run.parameters["all"]["N_spaces"][0])
    n_groups = int(run.parameters["all"]["N_groups"])
    target_degree = int(run.parameters["all"]["Ms"][0])

    if not coarse_solve:
        return power_iterate_result(
            guess_k,
            "Kornreich",
            "mesh_parameters_Kornreich",
            run,
            tol=ktol,
            use_we_accel=use_wynn,
            max_its=max_iterations,
            coarse_solve=False,
            input_phi=None,
            material_model=kornreich_material_model,
            verbose=verbose,
            plot=plot,
        )

    result = power_iterate_result(
        guess_k,
        "Kornreich",
        "mesh_parameters_Kornreich",
        run,
        tol=ktol,
        use_we_accel=use_wynn,
        max_its=max_iterations,
        coarse_angles=n_angles - 1,
        coarse_solve=True,
        coarse_M=0,
        material_model=kornreich_material_model,
        verbose=verbose,
        plot=plot,
    )
    warm_coeffs = result.run.sol_ob.y[:, -1].reshape((n_angles * n_groups, n_spaces, 1))

    for degree in range(1, target_degree + 1):
        def set_degree(data: dict) -> None:
            data["all"]["Ms"][0] = int(degree)

        update_yaml(KORNREICH_YAML, set_degree)
        result = power_iterate_result(
            result.k_history[-1],
            "Kornreich",
            "mesh_parameters_Kornreich",
            run,
            tol=ktol,
            use_we_accel=use_wynn,
            max_its=max_iterations,
            input_phi=warm_coeffs,
            material_model=kornreich_material_model,
            verbose=verbose,
            plot=plot,
        )
        warm_coeffs = result.run.sol_ob.y[:, -1].reshape(
            (n_angles * n_groups, n_spaces, degree + 1)
        )

    return result


def prepare_dmd_yaml(vdmd_timesteps: int) -> None:
    def mutate_transport(data: dict) -> None:
        data["all"]["integrator"] = "Euler"
        data["all"]["fixed_source"] = False
        data["all"]["tfinal"] = 5000.0
        data["all"]["guess_steady_state"] = False
        data["all"]["Euler_dt_num"] = int(vdmd_timesteps)
        data["all"]["fission_operator"] = True

    update_yaml(KORNREICH_YAML, mutate_transport, KORNREICH_DMD_YAML)

    def mutate_mesh(data: dict) -> None:
        data["dense"] = True
        data["eval_times"] = False

    update_yaml(MESH_YAML, mutate_mesh, MESH_DMD_YAML)


def estimate_alpha_dmd(
    run,
    *,
    k_eff: float,
    sparse_time_points: int,
    skip: int,
    verbose: bool,
) -> DMDResult:
    """Estimate alpha modes from the time-dependent coefficient history."""
    sigma_t = float(run.parameters["all"]["sigma_t"])
    n_angles = int(run.parameters["fixed_source"]["N_angles"][0]) + 1
    n_spaces = int(run.parameters["all"]["N_spaces"][0])
    degree = int(run.parameters["all"]["Ms"][0])
    target = "negative" if k_eff < 1.0 else "positive"

    run.custom_source(randomstart=True, uncollided=0, moving=0)
    zeros = np.zeros_like(run.phi)

    # Scalar-flux DMD is retained for diagnostic parity, but coefficient-space
    # DMD supplies the vector used to warm start the iterative alpha methods.
    DMD_func3(
        run.sol_ob.Y_minus_psi,
        run.sol_ob.t,
        "Euler",
        sigma_t,
        skip=skip,
        theta=0,
        sparse_time_points=sparse_time_points,
        source=True,
        sourcevec=zeros,
        N_ang=n_angles,
        xs=run.xs,
        target=target,
    )

    alphas, vectors, _ = DMD_func3(
        run.sol_ob.y,
        run.sol_ob.t,
        "Euler",
        sigma_t,
        skip=skip,
        theta=0,
        sparse_time_points=sparse_time_points,
        source=True,
        sourcevec=zeros,
        N_ang=n_angles * (degree + 1),
        xs=np.zeros(n_spaces),
        target=target,
    )

    nonzero = alphas != 0
    alphas = alphas[nonzero]
    vectors = vectors[:, nonzero]
    if alphas.size == 0:
        raise RuntimeError("DMD returned no nonzero alpha modes")

    dominant_index = int(np.argmax(np.real(alphas)))
    if verbose:
        LOGGER.info("DMD alpha candidates: %s", np.asarray(alphas))
        LOGGER.info("DMD dominant alpha: %.12g", np.real(alphas[dominant_index]))
    return DMDResult(alphas=alphas, vectors=vectors, dominant_index=dominant_index)


def _prepare_iram_yaml() -> None:
    def mutate(data: dict) -> None:
        data["all"]["guess_steady_state"] = True
        data["all"]["fixed_source"] = True
        data["all"]["fission_operator"] = True

    update_yaml(KORNREICH_YAML, mutate, KORNREICH_IRAM_YAML)


def build_inverse_transport_operator(run, run_ob, *, atol: float, verbose: bool) -> LinearOperator:
    """Build the historical inverse steady-state alpha operator.

    This operator has eigenvalues ``lambda = 1/alpha``. Inputs are normalized
    before the inner solve so that absolute tolerances do not make the operator
    scale dependent.
    """
    edges = run.edges
    n_space = int(run.parameters["all"]["N_spaces"][0])
    n_angles = int(run.parameters["fixed_source"]["N_angles"][0]) + 1
    n_groups = int(run.parameters["all"]["N_groups"])
    degree = int(run.parameters["all"]["Ms"][0])
    matrices = run_ob.matrices
    dimension = n_space * n_angles * (degree + 1) * n_groups

    def matvec(x):
        x = np.asarray(x)
        xnorm = np.linalg.norm(x)
        if xnorm == 0.0:
            return np.zeros_like(x)

        run.load("Kornreich_IRAM", "mesh_parameters_Kornreich")
        run.kold = 1
        input_vec = -x.reshape((n_angles * n_groups, n_space, degree + 1)) / xnorm

        for cell in range(n_space):
            matrices.make_all_matrices(edges[cell], edges[cell + 1], 0, 0)
            mass = matrices.Mass
            for angle in range(n_angles * n_groups):
                input_vec[angle, cell, :] = mass @ input_vec[angle, cell, :]

        psi_old = input_vec.copy()
        coeffs_old = psi_old.ravel()
        diff = np.inf
        res_coefficients = coeffs_old.copy()

        for inner_iteration in range(2):
            run.custom_source(
                randomstart=False,
                uncollided=0,
                moving=0,
                phi_coeffs=psi_old,
                input_A=None,
                input_coeffs=input_vec,
            )
            res_coefficients = np.copy(run.sol_ob.y[:, -1])
            diff = np.max(np.abs(res_coefficients - coeffs_old))
            if verbose:
                LOGGER.info("inverse matvec inner %d: diff=%.3e", inner_iteration, diff)
            if diff <= atol:
                break
            coeffs_old = res_coefficients.copy()
            psi_old = res_coefficients.reshape((n_angles * n_groups, n_space, degree + 1))

        return xnorm * res_coefficients

    return LinearOperator((dimension, dimension), matvec=matvec, dtype=np.float64)


def solve_alpha_iram(
    run,
    run_ob,
    dmd: DMDResult,
    *,
    alpha_tol: float,
    n_modes: int,
    max_iterations: int,
    verbose: bool,
) -> AlphaResult:
    """Refine alpha using IRAM on the inverse steady-state operator."""
    _prepare_iram_yaml()
    operator = build_inverse_transport_operator(
        run, run_ob, atol=float(run.parameters["all"]["at"]), verbose=verbose
    )

    v0 = np.real_if_close(dmd.dominant_vector).astype(float)
    try:
        lambdas, vectors = eigs(
            operator,
            k=n_modes,
            which="LM",
            tol=alpha_tol,
            maxiter=max_iterations,
            v0=v0,
        )
        converged = True
    except ArpackNoConvergence as error:
        lambdas = error.eigenvalues
        vectors = error.eigenvectors
        converged = False
        if lambdas is None or len(lambdas) == 0:
            raise RuntimeError("IRAM failed before any eigenpair converged") from error
        LOGGER.warning("IRAM returned only %d converged eigenpairs", len(lambdas))

    valid = np.isfinite(lambdas) & (np.abs(lambdas) > 1e-12)
    if not np.any(valid):
        raise RuntimeError("IRAM returned no usable inverse eigenvalues")

    alpha_candidates = 1.0 / lambdas[valid]
    valid_vectors = vectors[:, valid]
    target = dmd.dominant_alpha
    selected = int(np.argmin(np.abs(np.real(alpha_candidates) - target)))
    alpha = float(np.real(alpha_candidates[selected]))

    if verbose:
        for idx, candidate in enumerate(alpha_candidates):
            LOGGER.info(
                "IRAM candidate %d: lambda=%s, alpha=%s, |alpha-DMD|=%.3e",
                idx,
                lambdas[valid][idx],
                candidate,
                abs(np.real(candidate) - target),
            )
        LOGGER.info("Selected IRAM alpha: %.12g", alpha)

    return AlphaResult(
        alpha=alpha,
        method="IRAM inverse operator",
        converged=converged,
        history=[],
        candidates=alpha_candidates,
        vectors=valid_vectors,
    )


def solve_alpha_secant(
    evaluate: Callable[[float, np.ndarray, float], tuple[float, np.ndarray, float]],
    *,
    alpha_initial: float,
    phi_initial: np.ndarray,
    k_initial: float,
    tol: float,
    max_iterations: int,
    initial_step: float = 0.01,
    verbose: bool = False,
) -> AlphaResult:
    """Warm-started secant solve of ``k(alpha) - 1 = 0``."""
    alpha0 = float(alpha_initial)
    step = max(abs(alpha_initial) * 0.1, initial_step)
    alpha1 = alpha0 + step

    g0, phi0, k0 = evaluate(alpha0, phi_initial, k_initial)
    g1, phi1, k1 = evaluate(alpha1, phi0, k0)
    history = [(alpha0, g0), (alpha1, g1)]

    for iteration in range(max_iterations):
        if abs(g1) <= tol:
            return AlphaResult(alpha=alpha1, method="warm-started secant", converged=True, history=history)

        denominator = g1 - g0
        if abs(denominator) < 1e-14:
            raise RuntimeError(
                "Secant alpha solve stalled because successive residuals are indistinguishable"
            )

        alpha2 = alpha1 - g1 * (alpha1 - alpha0) / denominator
        g2, phi2, k2 = evaluate(alpha2, phi1, k1)
        history.append((float(alpha2), float(g2)))
        if verbose:
            LOGGER.info(
                "alpha secant %d: alpha=%.12g, k-1=%.3e, k=%.12g",
                iteration,
                alpha2,
                g2,
                k2,
            )

        alpha0, g0 = alpha1, g1
        alpha1, g1 = float(alpha2), float(g2)
        phi1, k1 = phi2, float(k2)

    return AlphaResult(alpha=alpha1, method="warm-started secant", converged=False, history=history)


def make_alpha_evaluator(
    run,
    *,
    ktol: float,
    max_k_iterations: int,
    use_wynn: bool,
    verbose: bool,
):
    """Create the expensive ``g(alpha)=k(alpha)-1`` callback for secant search."""
    n_angles = int(run.parameters["fixed_source"]["N_angles"][0]) + 1
    n_groups = int(run.parameters["all"]["N_groups"])
    n_space = int(run.parameters["all"]["N_spaces"][0])
    degree = int(run.parameters["all"]["Ms"][0])

    def evaluate(alpha: float, phi_guess: np.ndarray, k_guess: float):
        def mutate(data: dict) -> None:
            data["all"]["alpha_shift"] = float(alpha)

        update_yaml(KORNREICH_YAML, mutate, KORNREICH_NEW_YAML)
        result = power_iterate_result(
            k_guess,
            "Kornreich_new",
            "mesh_parameters_Kornreich",
            run,
            tol=ktol,
            use_we_accel=use_wynn,
            coarse_solve=False,
            max_its=max_k_iterations,
            input_phi=phi_guess,
            material_model=kornreich_material_model,
            verbose=verbose,
            strict_convergence=False,
        )
        k_new = result.k_history[-1]
        phi_new = result.run.sol_ob.y[:, -1].reshape(
            (n_angles * n_groups, n_space, degree + 1)
        )
        if verbose:
            LOGGER.info(
                "alpha evaluation: alpha=%.12g, k=%.12g, residual=%.3e, k_iters=%d%s",
                alpha,
                k_new,
                k_new - 1.0,
                result.iterations,
                "" if result.converged else " (not converged)",
            )
        return k_new - 1.0, phi_new, k_new

    return evaluate


def save_case_data(run_ob, k_result, *, x0: float, nu: float, n_angles: int, n_spaces: int) -> None:
    data_dir = RESULTS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"Kornreich_keff_S{n_angles}_{n_spaces}_cells_x0={x0}_nu={nu}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("scalar_flux", data=run_ob.phi)
        handle.create_dataset("xs", data=run_ob.xs)
        handle.create_dataset("psi", data=run_ob.psi)
        handle.create_dataset("Y_minus", data=run_ob.sol_ob.Y_minus_psi)
        handle.create_dataset("N_angles", data=np.array([n_angles - 1]))
        handle.create_dataset("t", data=run_ob.sol_ob.t)
        handle.create_dataset("k_list", data=k_result.k_history)
        handle.create_dataset(
            "fission_source",
            data=k_result.sigma_f_x * k_result.nu_x * k_result.scalar_flux,
        )


def save_summary(result: KornreichResult, *, n_space: int, n_angles: int, degree: int) -> None:
    data_dir = RESULTS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    params = result.run.parameters
    x0 = float(params["fixed_source"]["x0"][0])
    nu = float(params["all"]["nu"])
    path = data_dir / f"kalpha_x0={x0}_nu={nu}.h5"
    key = f"N_spaces={n_space}_N_angles={n_angles}_M={degree}"
    with h5py.File(path, "a") as handle:
        if key in handle:
            del handle[key]
        handle.create_dataset(
            key,
            data=[
                result.k_eff,
                result.alpha,
                result.benchmark.alpha,
                result.benchmark.k_eff,
                result.alpha_dmd,
            ],
        )


def make_table(x0: float, nu: float, DMD_alpha: float, IRAM_alpha: float, iteration_k: float, *, verbose: bool = False) -> None:
    """Update the benchmark CSV table without printing unless requested."""
    table_dir = RESULTS_DIR / "table"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "eigenvalues.csv"
    benchmark = get_benchmark(x0, nu)
    row = {
        "x0": x0,
        "nu": nu,
        "analytic k": benchmark.k_eff,
        "Iteration k": iteration_k,
        "analytic alpha": benchmark.alpha,
        "VDMD alpha": np.real(DMD_alpha),
        "IRAM alpha": np.real(IRAM_alpha),
    }
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[~((frame["x0"] == x0) & (frame["nu"] == nu))]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.sort_values(by=["x0", "nu"]).round(6).to_csv(path, index=False)
    if verbose:
        LOGGER.info("Updated %s", path)


def plot_k_convergence(k_result, benchmark: BenchmarkValues, *, x0: float, nu: float, n_angles: int, n_spaces: int, degree: int, show: bool = False) -> None:
    """Create the principal k-convergence plots."""
    out_dir = RESULTS_DIR / "convergence_plots"
    time_dir = RESULTS_DIR / "time_plots"
    solution_dir = RESULTS_DIR / "solution_plots"
    for directory in (out_dir, time_dir, solution_dir):
        directory.mkdir(parents=True, exist_ok=True)

    history = np.asarray(k_result.k_history)
    iterations = np.arange(history.size)

    fig, ax = plt.subplots()
    ax.plot(iterations, history, "-o", mfc="none")
    ax.axhline(benchmark.k_eff)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$k_\mathrm{eff}$")
    fig.savefig(out_dir / f"k_iterations_Kornreich_{n_angles}_angles_x0={x0}_nu={nu}_{n_spaces}_cells_M={degree}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if k_result.iteration_times:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(1, len(k_result.iteration_times) + 1), k_result.iteration_times, "-o", mfc="none")
        ax.set_xlabel("iteration")
        ax.set_ylabel("time [s]")
        fig.savefig(time_dir / f"calc_time_Kornreich_{n_angles}_angles_x0={x0}_nu={nu}_{n_spaces}_cells_M={degree}.pdf", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(k_result.run.xs, k_result.run.phi)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi$")
    fig.savefig(solution_dir / f"scalar_flux_Kornreich_x0={x0}_nu={nu}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def Kornreich_benchmark(
    prime: bool = True,
    guess_k: float = 1.0,
    sparse_time_points: int = 11,
    skip: int = 3,
    ktol: float = 5e-4,
    use_we: bool = False,
    max_its_kloop: int = 25,
    maxits_power: int = 100,
    coarse_angles: int = 4,
    alpha_tol: float = 5e-4,
    nalphas: int = 4,
    tf: float = 5e3,
    coarse_solve: bool = True,
    switch_to_power: float = 0.9,
    VDMD_timesteps: int = 60,
    get_alpha: bool = True,
    *,
    verbose: bool = False,
    plot: bool = False,
    return_result: bool = False,
):
    """Run one Kornreich k/alpha benchmark.

    The historical five-value return tuple is retained unless
    ``return_result=True``.
    """
    del coarse_angles, tf  # case YAML already contains these values
    configure_logging(verbose)

    def set_guess(data: dict) -> None:
        data["all"]["kold"] = float(guess_k)

    update_yaml(KORNREICH_YAML, set_guess)
    solver = Run()
    load_sol()  # preserve initialization side effects used by the original scripts
    solver.load("Kornreich", "mesh_parameters_Kornreich")

    if prime:
        original_m = int(solver.parameters["all"]["Ms"][0])
        solver.parameters["all"]["N_spaces"] = [10]
        solver.parameters["all"]["tfinal"] = 0.001
        solver.parameters["all"]["Ms"] = [0]
        solver.parameters["random_IC"]["N_angles"] = [2]
        solver.custom_source(randomstart=True, uncollided=0, moving=0)
        solver.load("Kornreich", "mesh_parameters_Kornreich")
        solver.parameters["all"]["Ms"][0] = original_m

    x0 = float(solver.parameters["fixed_source"]["x0"][0])
    nu = float(solver.parameters["all"]["nu"])
    benchmark = get_benchmark(x0, nu)

    k_result = solve_keff(
        solver,
        guess_k=guess_k,
        ktol=ktol,
        max_iterations=max_its_kloop,
        use_wynn=use_we,
        coarse_solve=coarse_solve,
        verbose=verbose,
        plot=plot,
    )
    k_eff = float(k_result.k_history[-1])

    if plot:
        plot_k_convergence(
            k_result,
            benchmark,
            x0=x0,
            nu=nu,
            n_angles=int(solver.parameters["fixed_source"]["N_angles"][0]) + 1,
            n_spaces=int(solver.parameters["all"]["N_spaces"][0]),
            degree=int(solver.parameters["all"]["Ms"][0]),
            show=False,
        )

    prepare_dmd_yaml(VDMD_timesteps)
    solver.load("Kornreich_DMD", "mesh_parameters_Kornreich_DMD")
    dmd = estimate_alpha_dmd(
        solver,
        k_eff=k_eff,
        sparse_time_points=sparse_time_points,
        skip=skip,
        verbose=verbose,
    )

    if not get_alpha:
        alpha_result = AlphaResult(alpha=0.0, method="disabled", converged=True, history=[])
    elif k_eff < switch_to_power:
        # The inverse operator is well behaved farther from criticality.
        solver.load("Kornreich", "mesh_parameters_Kornreich")
        solver.custom_source(randomstart=True, uncollided=0, moving=0)
        alpha_result = solve_alpha_iram(
            solver,
            k_result.run,
            dmd,
            alpha_tol=alpha_tol,
            n_modes=nalphas,
            max_iterations=maxits_power,
            verbose=verbose,
        )
    else:
        # Near criticality alpha -> 0 and the inverse operator becomes poorly
        # conditioned. Solve k(alpha)-1=0 directly with warm starts instead.
        solver.load("Kornreich", "mesh_parameters_Kornreich")
        n_angles = int(solver.parameters["fixed_source"]["N_angles"][0]) + 1
        n_groups = int(solver.parameters["all"]["N_groups"])
        n_space = int(solver.parameters["all"]["N_spaces"][0])
        degree = int(solver.parameters["all"]["Ms"][0])
        phi_initial = np.real_if_close(dmd.dominant_vector).reshape(
            (n_angles * n_groups, n_space, degree + 1)
        )
        evaluator = make_alpha_evaluator(
            solver,
            ktol=ktol,
            max_k_iterations=max_its_kloop,
            use_wynn=use_we,
            verbose=verbose,
        )
        alpha_result = solve_alpha_secant(
            evaluator,
            alpha_initial=dmd.dominant_alpha,
            phi_initial=phi_initial,
            k_initial=k_eff,
            tol=alpha_tol,
            max_iterations=maxits_power,
            verbose=verbose,
        )

    result = KornreichResult(
        k_eff=k_eff,
        alpha=float(alpha_result.alpha),
        alpha_dmd=dmd.dominant_alpha,
        benchmark=benchmark,
        k_converged=k_result.converged,
        alpha_converged=alpha_result.converged,
        run=solver,
    )

    n_space = int(solver.parameters["all"]["N_spaces"][0])
    n_angles = int(solver.parameters["fixed_source"]["N_angles"][0]) + 1
    degree = int(solver.parameters["all"]["Ms"][0])
    save_summary(result, n_space=n_space, n_angles=n_angles, degree=degree)
    make_table(x0, nu, result.alpha_dmd, result.alpha, result.k_eff, verbose=verbose)

    if verbose:
        LOGGER.info(
            "Kornreich result: k=%.10g (benchmark %.10g), alpha=%.10g via %s (benchmark %.10g), DMD=%.10g",
            result.k_eff,
            benchmark.k_eff,
            result.alpha,
            alpha_result.method,
            benchmark.alpha,
            result.alpha_dmd,
        )

    if return_result:
        return result
    return result.k_eff, result.alpha, benchmark.alpha, benchmark.k_eff, result.alpha_dmd


def mesh_converge_Kornreich(
    cells_start: int = 20,
    N_angles: int = 96,
    max_cells: int = 200,
    tf: float = 5e3,
    euler_dt: int = 8,
    *,
    verbose: bool = False,
    plot: bool = False,
):
    """Refine the spatial mesh until k and alpha cease changing appreciably."""
    k_guess = 0.8
    alpha_old = 1e-6
    tol = 1e-3
    history = []

    cells = int(cells_start)
    while True:
        data = _load_yaml(KORNREICH_YAML)
        degree = int(data["all"]["Ms"][0])
        x0 = float(data["fixed_source"]["x0"][0])
        nu = float(data["all"]["nu"])
        prepare_case_yaml(
            x0=x0,
            nu=nu,
            degree=degree,
            n_spaces=cells,
            n_angles=N_angles,
            tf=tf,
            euler_steps=euler_dt,
        )
        result = Kornreich_benchmark(
            guess_k=k_guess,
            verbose=verbose,
            plot=plot,
            return_result=True,
        )
        history.append((cells, result.k_eff, result.alpha, result.alpha_dmd))

        if abs(k_guess - result.k_eff) <= tol and abs(alpha_old - result.alpha) <= tol:
            break
        if cells >= max_cells:
            break

        k_guess = result.k_eff
        alpha_old = result.alpha
        cells = min(max_cells, max(cells + 1, int(cells * 1.5)))

    return history


def fill_Kornreich_table(
    N_ang: int = 16,
    M: int = 2,
    NSPACE: int = 10,
    *,
    x0_values=(4.5,),
    nu_values=(1.5, 3.5),
    verbose: bool = False,
    plot: bool = False,
):
    """Run selected Kornreich benchmark cases at one discretization."""
    outputs = []
    for x0 in x0_values:
        for nu in nu_values:
            prepare_case_yaml(
                x0=x0,
                nu=nu,
                degree=M,
                n_spaces=NSPACE,
                n_angles=N_ang,
                tf=5e3,
                euler_steps=8,
            )
            outputs.append(
                mesh_converge_Kornreich(
                    cells_start=NSPACE,
                    max_cells=NSPACE,
                    N_angles=N_ang,
                    tf=5e3,
                    euler_dt=8,
                    verbose=verbose,
                    plot=plot,
                )
            )
    return outputs


def plot_results(N_ang_list=(2, 4, 8, 16, 32, 64), N_space: int = 10, M: int = 2, *, show: bool = False) -> None:
    """Plot stored angular-convergence results.

    Alpha absolute errors for all available ``nu`` values are placed on one
    figure. Line style identifies ``nu`` (dashed for 1.5, solid for 3.5),
    while marker shape identifies the estimator. All data curves are black;
    legend entries are generated with empty plots so individual curves do not
    each require their own label.
    """
    angle_dir = RESULTS_DIR / "angle_converge"
    angle_dir.mkdir(parents=True, exist_ok=True)

    for x0 in (4.5,):
        stored = {}

        for nu in (1.5, 3.5):
            path = RESULTS_DIR / "data" / f"kalpha_x0={x0}_nu={nu}.h5"
            if not path.exists():
                continue

            n_plot, k_values, alpha_values, dmd_values = [], [], [], []
            benchmark = get_benchmark(x0, nu)
            with h5py.File(path, "r") as handle:
                for n_angle in N_ang_list:
                    key = f"N_spaces={N_space}_N_angles={n_angle + 1}_M={M}"
                    if key not in handle:
                        continue
                    values = np.asarray(handle[key])
                    n_plot.append(n_angle)
                    k_values.append(values[0])
                    alpha_values.append(values[1])
                    dmd_values.append(values[4] if values.size > 4 else np.nan)

            if not n_plot:
                continue

            stored[nu] = {
                "angles": np.asarray(n_plot),
                "k": np.asarray(k_values),
                "alpha": np.asarray(alpha_values),
                "dmd": np.asarray(dmd_values),
                "benchmark": benchmark,
            }

            # Keep the existing per-nu plots except alpha-error, which is
            # intentionally combined below.
            figure_specs = [
                ("kerr", np.abs(stored[nu]["k"] - benchmark.k_eff), r"$k_\mathrm{eff}$ absolute error", True),
                ("k", stored[nu]["k"], r"$k_\mathrm{eff}$", False),
                ("alpha", stored[nu]["alpha"], r"$\alpha$", False),
            ]
            for name, values, ylabel, logarithmic_y in figure_specs:
                fig, ax = plt.subplots()
                if logarithmic_y:
                    ax.loglog(n_plot, values, "-o", mfc="none", label="iterative")
                else:
                    ax.semilogx(n_plot, values, "-o", mfc="none", label="iterative")

                if name == "alpha":
                    ax.semilogx(n_plot, stored[nu]["dmd"], "-^", mfc="none", label="DMD")
                    ax.axhline(benchmark.alpha)
                elif name == "k":
                    ax.axhline(benchmark.k_eff)

                ax.set_xlabel("angles")
                ax.set_ylabel(ylabel)
                if len(ax.get_legend_handles_labels()[0]) > 1:
                    ax.legend()
                fig.savefig(angle_dir / f"{name}_x0={x0}_nu={nu}.pdf", bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)

        if stored:
            # Combined k-eigenvalue error plot: linestyle -> nu.
            fig, ax = plt.subplots()
            for nu, result in stored.items():
                linestyle = "--" if np.isclose(nu, 1.5) else "-"
                angles = result["angles"]
                benchmark = result["benchmark"]
                k_error = np.abs(result["k"] - benchmark.k_eff)

                ax.loglog(
                    angles,
                    k_error,
                    color="k",
                    linestyle=linestyle,
                    marker="o",
                    mfc="none",
                )

            # Dummy artists: line style identifies nu without duplicating
            # legend entries for every data curve.
            ax.plot([], [], "k--", label=r"$\nu=1.5$")
            ax.plot([], [], "k-", label=r"$\nu=3.5$")
            ax.plot([], [], "ko", mfc="none", linestyle="None", label=r"$k$ iteration")

            ax.set_xlabel("angles")
            ax.set_ylabel(r"$k_\mathrm{eff}$ absolute error")
            ax.legend()
            fig.savefig(angle_dir / f"kerr_x0={x0}.pdf", bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

            # Combined alpha-error plot: linestyle -> nu, marker -> method.
            # For the current Kornreich workflow, nu=1.5 is refined with IRAM
            # and nu=3.5 is refined with the warm-started secant solve.
            fig, ax = plt.subplots()
            for nu, result in stored.items():
                linestyle = "--" if np.isclose(nu, 1.5) else "-"
                marker = "s" if np.isclose(nu, 1.5) else "o"
                angles = result["angles"]
                benchmark = result["benchmark"]
                iterative_error = np.abs(result["alpha"] - benchmark.alpha)
                dmd_error = np.abs(result["dmd"] - benchmark.alpha)

                ax.loglog(
                    angles,
                    iterative_error,
                    color="k",
                    linestyle=linestyle,
                    marker=marker,
                    mfc="none",
                )
                ax.loglog(
                    angles,
                    dmd_error,
                    color="k",
                    linestyle=linestyle,
                    marker="^",
                    mfc="none",
                )

            # Dummy artists make a compact, semantic legend:
            # line style -> nu; marker -> estimator.
            ax.plot([], [], "k--", label=r"$\nu=1.5$")
            ax.plot([], [], "k-", label=r"$\nu=3.5$")
            ax.plot([], [], "k^", mfc="none", linestyle="None", label="DMD")
            ax.plot([], [], "ks", mfc="none", linestyle="None", label="IRAM")
            ax.plot([], [], "ko", mfc="none", linestyle="None", label="secant")

            ax.set_xlabel("angles")
            ax.set_ylabel(r"$\alpha$ absolute error")
            ax.legend()
            fig.savefig(angle_dir / f"alphaerr_x0={x0}.pdf", bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)



def run_converge(
    *,
    nspace: int = 50,
    degree: int = 3,
    angles=(2, 4, 8, 16),
    verbose: bool = False,
    plot: bool = False,
    solve: bool = True,
) -> None:
    """Run the standard angular-convergence study."""
    completed = []
    for n_angles in angles:
        if solve:
            fill_Kornreich_table(
                n_angles,
                M=degree,
                NSPACE=nspace,
                verbose=verbose,
                plot=plot,
            )
        completed.append(n_angles)
        plot_results(completed, N_space=nspace, M=degree, show=False)


if __name__ == "__main__":
    run_converge(verbose=False)
