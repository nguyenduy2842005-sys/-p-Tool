
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from fem_core import (
    FrameNode, FrameElement, FrameSupport, FramePointLoad,
    PlaneFrameInput, solve_plane_frame,
)


@dataclass
class OptimizationSettings:
    E: float = 200e6              # kN/m²
    A_min: float = 1e-4           # m²
    A_max: float = 0.10           # m²
    I_min: float = 1e-8           # m⁴
    I_max: float = 1e-2           # m⁴
    sigma_allow: float = 160e3    # kN/m²
    disp_allow: float = 0.020     # m
    bending_stress_allow: float = 160e3  # kN/m²
    c: float = 0.10               # m, outer-fibre distance
    optimize_A: bool = True
    optimize_I: bool = False
    volume_weight_A: float = 1.0
    volume_weight_I: float = 0.0


@dataclass
class OptimizationResult:
    success: bool
    message: str
    A: np.ndarray
    I: np.ndarray
    objective: float
    max_displacement: float
    max_axial_stress: float
    max_bending_stress: float
    kkt: dict
    fem_result: object
    history: list[dict]


def make_single_member_model(
    length: float,
    n_segments: int,
    A: np.ndarray,
    I: np.ndarray,
    E: float,
    load_type: str,
    load: float,
):
    nodes = [
        FrameNode(length * i / n_segments, 0.0)
        for i in range(n_segments + 1)
    ]
    elements = [
        FrameElement(
            i_node=i,
            j_node=i + 1,
            E=E,
            A=float(A[i]),
            I=float(I[i]),
            udl_local=0.0,
        )
        for i in range(n_segments)
    ]
    supports = [
        FrameSupport(node=0, ux_fixed=True, uy_fixed=True, rz_fixed=True)
    ]

    if load_type == "axial":
        point_loads = [
            FramePointLoad(node=n_segments, Fx=float(load), Fy=0.0, Mz=0.0)
        ]
    elif load_type == "bending":
        point_loads = [
            FramePointLoad(node=n_segments, Fx=0.0, Fy=float(load), Mz=0.0)
        ]
    else:
        point_loads = [
            FramePointLoad(node=n_segments, Fx=float(load), Fy=float(load), Mz=0.0)
        ]

    return PlaneFrameInput(nodes, elements, supports, point_loads)


def evaluate_design(
    x: np.ndarray,
    length: float,
    n_segments: int,
    settings: OptimizationSettings,
    load_type: str,
    load: float,
):
    n = n_segments
    k = 0
    if settings.optimize_A:
        A = x[k:k+n]
        k += n
    else:
        A = np.full(n, settings.A_min * 10.0)

    if settings.optimize_I:
        I = x[k:k+n]
    else:
        I = np.full(n, settings.I_min * 10.0)

    fem_input = make_single_member_model(
        length, n_segments, A, I, settings.E, load_type, load
    )
    result = solve_plane_frame(fem_input)

    Le = length / n_segments
    objective = float(np.sum(A) * Le)

    if settings.optimize_I:
        # A and I are intentionally treated as independent continuous design fields
        # in this educational variational model.
        objective += settings.volume_weight_I * float(np.sum(I) * Le)

    disp = np.linalg.norm(result.node_displacements[:, :2], axis=1)
    max_disp = float(np.max(disp))

    max_N = 0.0
    max_M = 0.0
    for er in result.element_results:
        max_N = max(max_N, float(np.max(np.abs(er.axial))))
        max_M = max(max_M, float(np.max(np.abs(er.moment))))

    axial_stress = max_N / max(float(np.min(A)), 1e-30)
    bending_stress = max_M * settings.c / max(float(np.min(I)), 1e-30)

    # Elementwise stress measures are needed for optimization constraints.
    axial_by_elem = np.array([
        float(np.max(np.abs(er.axial))) / max(float(A[i]), 1e-30)
        for i, er in enumerate(result.element_results)
    ])
    bending_by_elem = np.array([
        float(np.max(np.abs(er.moment))) * settings.c / max(float(I[i]), 1e-30)
        for i, er in enumerate(result.element_results)
    ])

    return {
        "objective": objective,
        "max_disp": max_disp,
        "max_N": max_N,
        "max_M": max_M,
        "axial_stress": axial_stress,
        "bending_stress": bending_stress,
        "axial_by_elem": axial_by_elem,
        "bending_by_elem": bending_by_elem,
        "A": A,
        "I": I,
        "fem": result,
    }


def _finite_gradient(fun, x, rel_step=2e-5):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        h = rel_step * max(1.0, abs(x[i]))
        xp = x.copy(); xm = x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (fun(xp) - fun(xm)) / (2*h)
    return g


def kkt_verify(x_opt, objective_fun, constraint_funs, active_tol=2e-5):
    """
    Numerical KKT verification for g_i(x) <= 0:
        ∇f + Σ λ_i ∇g_i = 0
        λ_i >= 0
        g_i <= 0
        λ_i g_i = 0

    Multipliers are obtained by least-squares on active constraints.
    """
    x_opt = np.asarray(x_opt, dtype=float)
    grad_f = _finite_gradient(objective_fun, x_opt)

    vals = np.array([float(g(x_opt)) for g in constraint_funs])
    active = [i for i, v in enumerate(vals) if v >= -active_tol]

    if active:
        G = np.column_stack([
            _finite_gradient(constraint_funs[i], x_opt) for i in active
        ])
        lam, *_ = np.linalg.lstsq(G, -grad_f, rcond=None)
        stationarity = grad_f + G @ lam
    else:
        lam = np.zeros(0)
        stationarity = grad_f

    full_lambda = np.zeros(len(constraint_funs))
    for j, i in enumerate(active):
        full_lambda[i] = lam[j]

    primal = float(max([0.0, *vals]))
    dual = float(max([0.0, *(-full_lambda)]))
    complementarity = float(max([
        0.0,
        *(abs(full_lambda[i] * vals[i]) for i in range(len(vals)))
    ])) if len(vals) else 0.0

    return {
        "grad_f": grad_f,
        "constraint_values": vals,
        "active": active,
        "lambda": full_lambda,
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "primal_violation": primal,
        "dual_violation": dual,
        "complementarity": complementarity,
        "stationarity": stationarity,
        "passed": (
            primal < 5e-4
            and dual < 5e-4
            and complementarity < 5e-3
            and np.linalg.norm(stationarity) < max(1e-3, 0.05*np.linalg.norm(grad_f)+1e-6)
        ),
    }


def optimize_single_member(
    length: float,
    n_segments: int,
    settings: OptimizationSettings,
    load_type: str,
    load: float,
    initial_A: float,
    initial_I: float,
):
    n = int(n_segments)
    x0_parts = []
    bounds = []

    if settings.optimize_A:
        x0_parts.append(np.full(n, initial_A))
        bounds += [(settings.A_min, settings.A_max)] * n

    if settings.optimize_I:
        x0_parts.append(np.full(n, initial_I))
        bounds += [(settings.I_min, settings.I_max)] * n

    x0 = np.concatenate(x0_parts)

    history = []

    def ev(x):
        return evaluate_design(x, length, n, settings, load_type, load)

    def objective(x):
        return ev(x)["objective"]

    constraints = []
    constraint_names = []

    # Elementwise axial stress constraints.
    if settings.optimize_A:
        for i in range(n):
            def g(x, i=i):
                e = ev(x)
                return e["axial_by_elem"][i] / settings.sigma_allow - 1.0
            constraints.append(g)
            constraint_names.append(f"σ_axial[{i+1}]")

    # Global displacement constraint.
    constraints.append(lambda x: ev(x)["max_disp"] / settings.disp_allow - 1.0)
    constraint_names.append("u_max")

    # Elementwise bending stress constraints.
    if settings.optimize_I and load_type in ("bending", "combined"):
        for i in range(n):
            def g(x, i=i):
                e = ev(x)
                return e["bending_by_elem"][i] / settings.bending_stress_allow - 1.0
            constraints.append(g)
            constraint_names.append(f"σ_bending[{i+1}]")

    def callback(xk):
        e = ev(xk)
        history.append({
            "iteration": len(history) + 1,
            "objective": e["objective"],
            "max_disp": e["max_disp"],
            "max_axial_stress": e["axial_stress"],
            "max_bending_stress": e["bending_stress"],
        })

    scipy_constraints = [{"type": "ineq", "fun": lambda x, g=g: -g(x)} for g in constraints]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        callback=callback,
        options={"maxiter": 120, "ftol": 1e-8, "disp": False},
    )

    e = ev(res.x)

    # KKT is checked against the original g(x) <= 0 convention.
    kkt = kkt_verify(res.x, objective, constraints)

    return OptimizationResult(
        success=bool(res.success),
        message=str(res.message),
        A=e["A"],
        I=e["I"],
        objective=e["objective"],
        max_displacement=e["max_disp"],
        max_axial_stress=e["axial_stress"],
        max_bending_stress=e["bending_stress"],
        kkt=kkt,
        fem_result=e["fem"],
        history=history,
    )


def variational_axial_solution(
    N_of_x,
    length: float,
    E: float,
    sigma_allow: float,
    disp_allow: float,
    n=400,
):
    """
    Closed-form Euler-Lagrange/KKT construction for:

        min  ∫ A(x) dx
        s.t. ∫ N(x)^2/[E A(x)] dx <= δ_allow
             |N(x)|/A(x) <= σ_allow

    With only the displacement constraint active:
        A*(x) = |N(x)| sqrt(lambda/E)

    With stress constraint:
        A*(x) = max(|N|/sigma_allow, |N| sqrt(lambda/E))

    lambda is found by bisection so the displacement constraint is active
    whenever it can be active.
    """
    x = np.linspace(0.0, length, n)
    N = np.abs(np.asarray(N_of_x(x), dtype=float))
    N = np.maximum(N, 1e-14)

    A_stress = N / sigma_allow

    def area_for_lambda(lam):
        return np.maximum(A_stress, N * np.sqrt(max(lam, 0.0) / E))

    def disp_for_lambda(lam):
        A = area_for_lambda(lam)
        return float(np.trapezoid(N**2 / (E*A), x))

    d0 = disp_for_lambda(0.0)

    if d0 <= disp_allow:
        lam = 0.0
    else:
        lo, hi = 0.0, 1.0
        while disp_for_lambda(hi) > disp_allow:
            hi *= 2.0
            if hi > 1e20:
                break
        for _ in range(100):
            mid = 0.5*(lo+hi)
            if disp_for_lambda(mid) > disp_allow:
                lo = mid
            else:
                hi = mid
        lam = hi

    A = area_for_lambda(lam)
    displacement = float(np.trapezoid(N**2/(E*A), x))
    volume = float(np.trapezoid(A, x))

    return {
        "x": x,
        "N": N,
        "A": A,
        "lambda": lam,
        "displacement": displacement,
        "volume": volume,
        "A_stress": A_stress,
        "A_EL": N * np.sqrt(max(lam, 0.0) / E),
    }
