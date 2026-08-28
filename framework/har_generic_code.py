"""Hit and run generic

Baud Candice
Fri July 10 10:33:00 2026
"""


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

# ============================================================
# Utilities: feasibility + finding an initial feasible point
# ============================================================


import scipy.sparse as sp
from scipy.optimize import linprog

def chebyshev_center(A, b, sparse=False):
    """
    Find a robustly-feasible point for A @ x <= b via Chebyshev centering:
    maximize the radius r of the largest ball {x + u : ||u|| <= r} that fits
    inside the polytope, i.e.

        max_{x, r}  r
        s.t.        a_i^T x + r * ||a_i||_2 <= b_i   for every row i
                    r >= 0

    This is itself a linear program (one extra variable, same rows), solved
    with HiGHS via scipy. Unlike a plain feasibility LP -- which returns a
    vertex sitting exactly on several constraint boundaries -- this returns
    a point strictly inside the polytope whenever one exists, which is what
    you want as a seed for any later sampling (hit-and-run, etc.) since a
    boundary point can make the very next proposal step infeasible.

    Parameters
    ----------
    A : array-like, shape (n_rows, n_cols)
    b : array-like, shape (n_rows,)
    sparse : bool
        Pass True for large systems: converts A to a sparse CSR matrix
        before handing it to HiGHS. Worth it here because every row of
        these matrices has only 2-3 nonzero entries, so sparse formatting
        cuts memory and solve time substantially once n_cols is in the
        thousands.

    Returns
    -------
    x : np.ndarray, shape (n_cols,)
        A feasible point (interior if r > 0).
    r : float
        The inscribed-ball radius. r > 0 => strictly feasible.
        r ~ 0 => the polytope is (numerically) full of degenerate/tight
        constraints -- x is still feasible but sits on the boundary.
        Negative / LP infeasible => no feasible point exists.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n_rows, n_cols = A.shape

    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1e-12  # guard against any theoretically-empty row

    c = np.zeros(n_cols + 1)
    c[-1] = -1.0  # maximize r <=> minimize -r
    bounds = [(None, None)] * n_cols + [(0, None)]

    if sparse:
        A_aug = sp.hstack([sp.csr_matrix(A), sp.csr_matrix(norms.reshape(-1, 1))], format="csr")
    else:
        A_aug = np.hstack([A, norms.reshape(-1, 1)])

    res = linprog(c, A_ub=A_aug, b_ub=b, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"Chebyshev-center LP did not solve: {res.message}")

    x = res.x[:n_cols]
    r = res.x[-1]
    return x, r


def extract_fixed_coordinates_from_equalities(A_eq, b_eq, tol=1e-12):
    """
    Extract explicit one-coordinate equalities x[j] = value.

    Only equality rows with exactly one numerically nonzero coefficient are
    used. Coupled equalities are deliberately ignored: overwriting one of
    their coordinates independently could violate the other coordinates in
    the equality.
    """
    A_eq = np.asarray(A_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float)

    if A_eq.ndim != 2 or b_eq.shape != (A_eq.shape[0],):
        raise ValueError("A_eq and b_eq have incompatible shapes.")

    fixed = {}

    for row, beta in zip(A_eq, b_eq):
        row_scale = float(np.max(np.abs(row))) if row.size else 0.0

        if row_scale == 0.0:
            if abs(beta) > tol:
                raise RuntimeError("The equality system contains 0 = nonzero.")
            continue

        nonzero = np.flatnonzero(np.abs(row) > tol * row_scale)

        if nonzero.size != 1:
            continue

        j = int(nonzero[0])
        value = float(beta / row[j])

        # Preserve an explicitly specified zero exactly, rather than keeping
        # a least-squares/SVD residual such as 1e-15.
        if abs(beta) <= tol * row_scale:
            value = 0.0

        if j in fixed and not np.isclose(
            fixed[j],
            value,
            rtol=tol,
            atol=tol,
        ):
            raise RuntimeError(
                f"Inconsistent fixed values detected for coordinate {j}."
            )

        fixed[j] = value

    fixed_idx = np.array(sorted(fixed), dtype=int)
    fixed_values = np.array([fixed[j] for j in fixed_idx], dtype=float)
    return fixed_idx, fixed_values


def extract_all_fixed_coordinates_from_equalities(A_eq, b_eq, tol=1e-12):
    """
    Determine every coordinate that is uniquely and EXACTLY pinned by the
    equality system A_eq @ x = b_eq -- whether directly (a row with a
    single nonzero coefficient, x[j] = value, the case already handled
    by extract_fixed_coordinates_from_equalities) or indirectly through a
    chain of coupled rows.

    Example: "gap_duration - main_duration = 0" (coupled, two nonzero
    coefficients) combined with "main_duration = 0" (single-coefficient)
    together force gap_duration = 0 too, even though no single row
    mentions gap_duration in isolation. The single-row scan alone leaves
    gap_duration to the general nullspace projector, which only drives it
    to within machine epsilon of 0 (e.g. -7.3e-16) rather than exact 0.

    This resolves coordinates by repeated substitution: whenever a row
    has only one remaining unknown coefficient (after substituting in
    values already found), that unknown is solved for using ordinary
    subtraction/division on the ORIGINAL coefficients -- not an SVD or
    least-squares step -- so the result is as numerically clean as the
    equality system itself. Newly resolved coordinates can turn other
    rows into single-unknown rows in turn, so this repeats to a fixed
    point (handles chains of any length, in any row order).

    A row where two or more coordinates remain genuinely coupled after
    every substitution pass (e.g. two attributes that trade off against
    each other but are neither one individually fixed) is left alone --
    those coordinates really do move within the affine hull and stay the
    responsibility of the general projector (project_onto_equalities /
    enforce_equalities_exactly).

    Returns
    -------
    fixed_idx : np.ndarray[int]
    fixed_values : np.ndarray[float]
    """
    A_eq = np.asarray(A_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float)

    if A_eq.ndim != 2 or b_eq.shape != (A_eq.shape[0],):
        raise ValueError("A_eq and b_eq have incompatible shapes.")

    m = A_eq.shape[0]
    known = {}
    resolved_rows = set()

    changed = True
    while changed:
        changed = False

        for i in range(m):
            if i in resolved_rows:
                continue

            row = A_eq[i]
            beta = b_eq[i]
            row_scale = float(np.max(np.abs(row))) if row.size else 0.0

            if row_scale == 0.0:
                if abs(beta) > tol:
                    raise RuntimeError(
                        "The equality system contains 0 = nonzero."
                    )
                resolved_rows.add(i)
                continue

            nonzero = [int(j) for j in np.flatnonzero(np.abs(row) > tol * row_scale)]
            unknown_cols = [j for j in nonzero if j not in known]

            if len(unknown_cols) > 1:
                continue  # still genuinely coupled -- leave to the projector

            rhs = beta - sum(row[j] * known[j] for j in nonzero if j in known)

            if len(unknown_cols) == 0:
                if abs(rhs) > tol * row_scale:
                    raise RuntimeError(
                        "Inconsistent equality system detected during "
                        f"substitution (row {i})."
                    )
                resolved_rows.add(i)
                continue

            j = unknown_cols[0]
            value = float(rhs / row[j])

            # Preserve an explicitly implied zero exactly, rather than a
            # division residual.
            if abs(rhs) <= tol * row_scale:
                value = 0.0

            if j in known and not np.isclose(known[j], value, rtol=tol, atol=tol):
                raise RuntimeError(
                    f"Inconsistent fixed values detected for coordinate {j}."
                )

            known[j] = value
            resolved_rows.add(i)
            changed = True

    fixed_idx = np.array(sorted(known), dtype=int)
    fixed_values = np.array([known[j] for j in fixed_idx], dtype=float)
    return fixed_idx, fixed_values


def chebyshev_center_affine(
    A_ineq,
    b_ineq,
    A_eq,
    b_eq,
    N,
    tol=1e-10,
):
    """
    Chebyshev center in the affine hull A_eq @ x = b_eq.

    N must be an orthonormal basis of null(A_eq).
    The returned radius is measured only in movable directions.

    Equalities must not be included again in A_ineq. A zero projected
    inequality row is constant on the affine hull: it is checked for
    feasibility and then removed rather than being assigned an artificial
    positive norm.
    """
    A_ineq = np.asarray(A_ineq, dtype=float)
    b_ineq = np.asarray(b_ineq, dtype=float)
    A_eq = np.asarray(A_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float)

    d = A_ineq.shape[1]

    # Particular solution of A_eq @ x = b_eq
    if A_eq.shape[0] == 0:
        x_particular = np.zeros(d)
    else:
        x_particular, *_ = np.linalg.lstsq(
            A_eq,
            b_eq,
            rcond=None,
        )

        residual = A_eq @ x_particular - b_eq

        if np.max(np.abs(residual)) > tol:
            raise RuntimeError(
                "The equality system is inconsistent."
            )

    N = np.asarray(N, dtype=float)

    if N.ndim != 2 or N.shape[0] != d:
        raise ValueError(
            "N must have shape (n_variables, nullspace_dimension)."
        )

    k = N.shape[1]

    # The feasible set is a single point
    if k == 0:
        if np.all(A_ineq @ x_particular <= b_ineq + tol):
            return x_particular, 0.0
        raise RuntimeError(
            "The equality solution violates the inequalities."
        )

    # Reduced inequalities in z-space:
    # x = x_particular + N @ z
    A_reduced = A_ineq @ N
    b_reduced = b_ineq - A_ineq @ x_particular

    norms = np.linalg.norm(A_reduced, axis=1)

    # Rows with zero projected norm are constant on the affine hull.
    constant_rows = norms <= tol

    if np.any(b_reduced[constant_rows] < -tol):
        raise RuntimeError(
            "An inequality is infeasible on the equality affine hull."
        )

    keep = ~constant_rows
    A_reduced = A_reduced[keep]
    b_reduced = b_reduced[keep]
    norms = norms[keep]

    # No inequality restricts the movable dimensions.
    if A_reduced.shape[0] == 0:
        return x_particular, np.inf

    c = np.zeros(k + 1)
    c[-1] = -1.0

    A_aug = np.hstack([
        A_reduced,
        norms[:, None],
    ])

    bounds = [(None, None)] * k + [(0, None)]

    res = linprog(
        c,
        A_ub=A_aug,
        b_ub=b_reduced,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(
            f"Chebyshev-center LP did not solve: {res.message}"
        )

    z_center = res.x[:k]
    radius = res.x[-1]

    x_center = x_particular + N @ z_center

    # SVD and matrix multiplication satisfy the equality system only up to
    # floating-point precision -- and that holds for coupled multi-coordinate
    # rows too, not only explicit single-coordinate ones. Project the whole
    # system back onto the affine hull in one shot, then hard-set the
    # explicit single-coordinate equalities exactly.
    fixed_idx, fixed_values = extract_all_fixed_coordinates_from_equalities(
        A_eq,
        b_eq,
        tol=tol,
    )
    x_center = enforce_equalities_exactly(
        x_center,
        A_eq,
        b_eq,
        exact_fixed_idx=fixed_idx,
        exact_fixed_values=fixed_values,
    )

    # Guard against numerical errors in the reconstructed full vector.
    if A_eq.shape[0] > 0:
        equality_error = np.max(np.abs(A_eq @ x_center - b_eq))
        if equality_error > tol:
            raise RuntimeError(
                "Computed center does not satisfy the equality system "
                f"within tolerance (maximum error {equality_error:.3e})."
            )

    if np.any(A_ineq @ x_center > b_ineq + tol):
        raise RuntimeError(
            "Computed center violates the inequality system."
        )

    return x_center, radius

def feasible_point(A, b, sparse=False):
    """
    Cheaper alternative to chebyshev_center: a plain Phase-1 feasibility
    LP (zero objective). Faster, but HiGHS is free to hand back any vertex
    of the polytope, which typically has many constraints active at once
    (e.g. a birth date pinned exactly to a fertility-window edge). Use this
    only when you just need *a* feasible x and don't care where in the
    polytope it sits.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n_cols = A.shape[1]

    c = np.zeros(n_cols)
    bounds = [(None, None)] * n_cols
    A_ub = sp.csr_matrix(A) if sparse else A

    res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Feasibility LP did not solve: {res.message}")
    return res.x


def find_feasible_point(A, b, method="chebyshev", sparse=False, verbose=True):
    """
    Find a feasible point for one system A @ x <= b.

    Agnostic to where (A, b) came from -- pass the global system's
    (A_global, b_global) or a single already-built cluster's
    (A_cluster, b_cluster) from
    ExistenceClusterSpec.build_cluster_constraint_system(), it makes no
    difference here. Because of that, this function needs nothing beyond
    the matrices themselves: no pop, no warmup/cooldown ids, no spec or
    cluster-spec object. If you're solving several already-built clusters,
    just call this once per (A_c, b_c) in your own loop and keep the
    per-cluster member_ids (returned alongside A_c, b_c by
    build_cluster_constraint_system) on your side to place each x_c back
    into a global vector -- that placement isn't something this function
    needs to know about.

    Unlike chebyshev_center/feasible_point, this does not raise on
    infeasibility -- it reports it and returns (None, None), so a caller
    looping over several systems can keep going and collect a result (or
    a clear miss) for every system instead of the first infeasible one
    stopping the loop.

    If a system does come back infeasible, two structural causes are
    worth checking first -- widening windows alone won't fix either one:
      1) a warmup founder sits only 1-2 pedigree-hops from a cooldown
         leaf, so fertility_age_max*(hops) can't bridge the gap between
         the warmup and cooldown birth windows;
      2) the id-sorted sibling-order rows (t_c_j <= t_c_next) clash with
         fertility/alive-at-birth constraints whenever a lower-id sibling
         is structurally forced later than a higher-id sibling -- since
         individual id has no relationship to generation depth or
         birth-window type, this is common, not an edge case.

    Parameters
    ----------
    A : array-like, shape (n_rows, n_cols)
    b : array-like, shape (n_rows,)
    method : {"chebyshev", "phase1"}
        "chebyshev" (default): robust interior point, see chebyshev_center.
        "phase1": cheaper, may return a boundary vertex, see feasible_point.
    sparse : bool
        Forwarded to the solver; worth turning on once n_cols is large
        (each row here typically has only 2-3 nonzero entries).
    verbose : bool
        Print a one-line result for this system.

    Returns
    -------
    x : np.ndarray or None
        A feasible point, or None if the system is infeasible.
    r : float or None
        Chebyshev radius (None for method="phase1", or when infeasible).
    """
    A_arr = np.asarray(A, dtype=float)
    shape_tag = f"{A_arr.shape[0]} rows, {A_arr.shape[1]} cols"

    try:
        if method == "chebyshev":
            x, r = chebyshev_center(A, b, sparse=sparse)
        else:
            x, r = feasible_point(A, b, sparse=sparse), None
    except RuntimeError as e:
        # if verbose:
            # print(f"infeasible ({shape_tag}): {e}")
        return None, None

    if verbose:
        tag = f"r={r:.4g}" if r is not None else "ok"
        # print(f"feasible ({shape_tag}): {tag}")
    return x, r


def is_feasible(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float = 1e-10) -> bool:
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.asarray(x, float)
    return np.all(A @ x <= b + tol)



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


def build_equality_projector(A_eq: np.ndarray):
    """
    Precompute the projector used by project_onto_equalities.

    A_eq is fixed for the lifetime of one HAR chain (one block, one
    individual, one dimension), so its pseudoinverse should be computed
    ONCE at setup and reused on every proposal, instead of re-deriving a
    correction from a fresh linear solve at every MCMC iteration.

    Uses the Moore-Penrose pseudoinverse rather than solving
    (A_eq @ A_eq.T) @ lambda = residual directly: this stays well-defined
    even if two equality rows end up linearly dependent, which
    extract_equalities_from_opposites does not guarantee against.

    Returns
    -------
    P : np.ndarray, shape (d, m_eq), or None
        None when A_eq has zero rows (nothing to project).
    """
    A_eq = np.asarray(A_eq, dtype=float)
    if A_eq.shape[0] == 0:
        return None
    return np.linalg.pinv(A_eq)


def project_onto_equalities(x, A_eq, b_eq, P=None):
    """
    Project x onto the affine space A_eq x = b_eq.

    Solves:
        x_proj = argmin ||x_proj - x|| s.t. A_eq x_proj = b_eq

    Parameters
    ----------
    P : np.ndarray or None
        Precomputed projector from build_equality_projector(A_eq). Pass
        this whenever calling repeatedly (e.g. once per MCMC proposal)
        to avoid rebuilding the pseudoinverse from scratch every time.
    """
    A_eq = np.asarray(A_eq, dtype=float)
    if A_eq.shape[0] == 0:
        return x

    residual = A_eq @ x - b_eq

    if P is None:
        P = build_equality_projector(A_eq)

    return x - P @ residual


def enforce_equalities_exactly(
    x,
    A_eq,
    b_eq,
    P=None,
    exact_fixed_idx=None,
    exact_fixed_values=None,
):
    """
    Snap x back onto the FULL equality system A_eq @ x = b_eq, then
    hard-set any explicit single-coordinate equalities exactly.

    extract_fixed_coordinates_from_equalities only catches equality rows
    with a single nonzero coefficient (x[j] = value); coupled rows such
    as x[i] + x[j] = c are deliberately left untouched by that mechanism
    (see its docstring), since overwriting one coordinate independently
    could violate the other row. Those coupled rows were instead left to
    N's floating-point orthogonality to A_eq to hold "for free" -- true
    only up to the SVD's numerical precision, and that small residual
    compounds a little more every time an already-slightly-off
    x_current seeds the next proposal.

    Calling this after every proposal (and at initialization) re-solves
    the whole equality system in one minimum-norm correction, so the
    residual is reset to machine precision at every step instead of
    drifting. This correction is a fixed linear map that depends only on
    A_eq (not on the random proposal direction), so it doesn't introduce
    any asymmetry into the Hit-and-Run step -- it only cleans up
    floating-point noise on a point that already, up to that noise, lies
    on the equality hull.
    """
    x = np.asarray(x, dtype=float).copy()
    x = project_onto_equalities(x, A_eq, b_eq, P=P)

    if exact_fixed_idx is not None and exact_fixed_idx.size > 0:
        x[exact_fixed_idx] = exact_fixed_values

    return x


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