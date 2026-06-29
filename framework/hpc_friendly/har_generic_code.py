"""Hit and run generic (scitas)

Baud Candice
Tues Apr 07 17:42:00 2026
"""


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import cdd

# ============================================================
# Utilities: feasibility + finding an initial feasible point
# ============================================================

def is_feasible(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float = 1e-10) -> bool:
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.asarray(x, float)
    return np.all(A @ x <= b + tol)

def find_feasible_point(
    A: np.ndarray,
    b: np.ndarray,
    method: str = "scipy_lp",
    seed: int = 0,
    max_tries: int = 4000,
) -> np.ndarray:
    """
    Try to find x0 s.t. A x0 <= b.
    - method="scipy_lp": uses scipy.optimize.linprog (recommended, robust)
    - otherwise: random search (works if feasible region is easy to hit)
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, d = A.shape

    if method == "scipy_lp":
        try:
            from scipy.optimize import linprog
            c = np.zeros(d)
            res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * d, method="highs")
            if not res.success:
                raise RuntimeError(f"linprog failed: {res.message}")
            x0 = res.x
            if not is_feasible(A, b, x0):
                raise RuntimeError("linprog returned non-feasible point (unexpected).")
            return x0
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    scales = np.linspace(0.1, 50.0, 20)
    for s in scales:
        for _ in range(max_tries // len(scales)):
            x0 = rng.normal(size=d) * s
            if is_feasible(A, b, x0):
                return x0

    raise RuntimeError(
        "Could not find a feasible point. "
        "Install SciPy and use method='scipy_lp', or provide x0 manually."
    )

# ============================================================
# Hit-and-Run geometry
# ============================================================

def random_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    u = rng.normal(size=d)
    n = np.linalg.norm(u)
    if n == 0.0:
        u[0] = 1.0
        return u
    return u / n

def chord_interval(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    tol: float = 1e-14,
) -> Tuple[float, float]:
    """
    Compute [t_min, t_max] such that x + t u is feasible:
        A(x + t u) <= b.

    For each row i:
        (A_i u) t <= b_i - A_i x
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.asarray(x, float)
    u = np.asarray(u, float)

    Ax = A @ x
    Au = A @ u
    rhs = b - Ax

    t_min = -np.inf
    t_max = np.inf

    pos = Au > tol
    neg = Au < -tol
    zero = ~(pos | neg)

    # If Au_i == 0 then constraint requires rhs_i >= 0 (already true if x feasible)
    if np.any(zero):
        if np.any(rhs[zero] < -1e-12):
            return (1.0, 0.0)  # empty interval

    if np.any(pos):
        t_max = min(t_max, np.min(rhs[pos] / Au[pos]))
    if np.any(neg):
        t_min = max(t_min, np.max(rhs[neg] / Au[neg]))

    return float(t_min), float(t_max)


from typing import Tuple
def extract_equalities_from_opposites(A: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find equalities of the form a^T x = beta implied by:
        a^T x <= beta  and  (-a)^T x <= -beta.

    Returns:
      A_eq, b_eq, keep_mask  where keep_mask is True for rows NOT used in A_eq
      (i.e., remaining inequalities).
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, d = A.shape

    used = np.zeros(m, dtype=bool)
    Aeq = []
    beq = []

    # normalize rows to make matching more robust
    def norm_row(a, beta):
        na = np.linalg.norm(a)
        if na <= tol:
            return a, beta
        return a / na, beta / na

    An = np.zeros_like(A)
    bn = np.zeros_like(b)
    for i in range(m):
        An[i], bn[i] = norm_row(A[i], b[i])

    for i in range(m):
        if used[i]:
            continue
        ai, bi = An[i], bn[i]
        # search for j such that Aj ~ -Ai and bj ~ -bi
        # brute force is fine for moderate m
        diffs = np.linalg.norm(An + ai[None, :], axis=1)  # Aj + ai ~ 0 => Aj ~ -ai
        cand = np.where((~used) & (diffs <= tol) & (np.abs(bn + bi) <= tol))[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = int(cand[0])
            used[i] = True
            used[j] = True
            # equality is on the ORIGINAL scaling of row i
            Aeq.append(A[i].copy())
            beq.append(b[i].copy())

    A_eq = np.array(Aeq, dtype=float) if Aeq else np.zeros((0, d), dtype=float)
    b_eq = np.array(beq, dtype=float) if beq else np.zeros((0,), dtype=float)

    keep_mask = ~used
    return A_eq, b_eq, keep_mask


def nullspace_basis(Aeq: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Orthonormal basis N for Null(Aeq) using SVD.
    Returns N with shape (d, k) such that Aeq @ N ≈ 0.
    If Aeq has 0 rows, caller should treat as full space.
    """
    Aeq = np.asarray(Aeq, float)
    if Aeq.size == 0:
        return None
    U, S, Vt = np.linalg.svd(Aeq, full_matrices=True)
    rank = int(np.sum(S > tol))
    N = Vt[rank:].T  # (d, d-rank)
    return N


def project_onto_equalities(x, A_eq, b_eq):
    """
    Project x onto the affine space A_eq x = b_eq.

    Solves:
        x_proj = argmin ||x_proj - x|| s.t. A_eq x_proj = b_eq
    """
    if A_eq.shape[0] == 0:
        return x

    residual = A_eq @ x - b_eq

    if np.linalg.norm(residual) < 1e-14:
        return x

    # Solve (A A^T) lambda = residual
    M = A_eq @ A_eq.T
    lam = np.linalg.solve(M, residual)

    correction = A_eq.T @ lam
    return x - correction



def dirichlet_sample_duration_from_vertices(vertices, alpha):
    """
    Samples a feasible duration vector from a set of polytope vertices.

    The duration vector is generated as a convex combination of the
    vertices, using Dirichlet weights to ensure feasibility.

    Parameters
    ----------
    vertices : ndarray
        Array of vertices defining the feasible polytope.

    alpha : ndarray
        Array of values defining the Dirichlet distribution.

    Returns
    -------
    (ndarray, ndarray)
        The Dirichlet weights and the resulting sampled duration vector.
    """
    # dirichlet_sample = np.random.dirichlet(alpha)
    dirichlet_sample = alpha/np.sum(alpha)
    duration_vector = np.dot(dirichlet_sample, vertices)
    return duration_vector


def compute_polytope_vertices(A, b):
    #Scitas version
    """
    Computes the vertices of a polytope defined by linear inequalities.

    The polytope is defined by constraints of the form A x ≤ b. The
    function converts the inequality representation into a generator
    representation and extracts the vertices of the feasible region.
    """
    if A is not None and b is not None:
        A = np.asarray(A)
        b = np.asarray(b).reshape(-1)

        # cdd expects inequalities as:  [b | -A]  representing  b - A x >= 0  <=>  A x <= b
        H = np.hstack([b.reshape(-1, 1), -A])

        # Build cdd matrix (pycddlib 2.1.7 API)
        # number_type='fraction' is the safest (exact rational arithmetic)
        mat = cdd.Matrix(H.tolist(), number_type='fraction')
        mat.rep_type = cdd.RepType.INEQUALITY

        # Build polyhedron and get generators
        poly = cdd.Polyhedron(mat)
        gens = poly.get_generators()

        if gens is None:
            raise ValueError(
                "No vertices found, the polytope is empty: "
                "check that the constraints do not contradict each other."
            )

        # gens is a cdd.Matrix-like object; convert to numpy
        G = np.array(gens, dtype=object)

        if len(G) > 0:
            # First column indicates type: 1 -> vertex, 0 -> ray/line (depending on representation)
            verts = G[G[:, 0] == 1][:, 1:]
            if verts.size == 0:
                raise ValueError(
                    "No vertices found, the polytope is empty: "
                    "check that the constraints do not contradict each other."
                )
            # Convert Fractions to float (keep this if downstream expects numeric ndarray)
            vertices = np.array(verts, dtype=float)
            return vertices
        else:
            raise ValueError(
                "No vertices found, the polytope is empty: "
                "check that the constraints do not contradict each other."
            )
    else:
        return A


# ============================================================
# Diagnostics
# ============================================================

def gelman_rubin(chains):
    """
    Compute R-hat for each dimension.
    chains: array (M, N, d)
    """
    M, N, d = chains.shape
    chain_means = np.mean(chains, axis=1)
    mean_total = np.mean(chain_means, axis=0)
    B = N * np.var(chain_means, axis=0, ddof=1)
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    var_hat = (N - 1) / N * W + B / N
    R_hat = np.sqrt(var_hat / W)
    return R_hat

def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode="full")
    result = result[result.size // 2:]
    return result / result[0]

def effective_sample_size(chains):
    """
    Compute ESS per dimension.
    chains: (M, N, d)
    """
    M, N, d = chains.shape
    ess = np.zeros(d)

    for k in range(d):
        x = chains[:, :, k]
        rho_sum = 0.0
        for m in range(M):
            ac = autocorr(x[m])
            positive = ac[1:]
            positive = positive[positive > 0]
            rho_sum += 2 * np.sum(positive)
        rho_sum /= M
        ess[k] = M * N / (1 + rho_sum)

    return ess






