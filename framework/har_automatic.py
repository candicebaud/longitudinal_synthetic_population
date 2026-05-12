"""
Automatic blocked Hit-and-Run sampler for individual prior sampling.

This sampler detects blocks of dimensions from inter-event constraints.
It then samples durations block by block, while always evaluating the
full individual joint density.

Baud Candice
"""

import copy
import numpy as np
from typing import Optional

from har_generic_code import (
    compute_polytope_vertices,
    dirichlet_sample_duration_from_vertices,
    is_feasible,
    extract_equalities_from_opposites,
    nullspace_basis,
    chord_interval,
)


def _prepare_block_geometry(
    block_data,
    eq_tol=1e-12,
    fixed_row_tol=1e-14,
):
    """
    Precompute equality/inequality decomposition and nullspace basis
    for one block.

    Parameters
    ----------
    block_data : dict
        Block dictionary containing "A" and "b".

    Returns
    -------
    block_data : dict
        Same dictionary, enriched with:
            - A_eq
            - b_eq
            - A_ineq
            - b_ineq
            - N
            - fixed_idx
            - k
    """

    A = np.asarray(block_data["A"], dtype=float)
    b = np.asarray(block_data["b"], dtype=float)

    _, d = A.shape

    A_eq, b_eq, keep_mask = extract_equalities_from_opposites(
        A,
        b,
        tol=eq_tol,
    )

    A_ineq = A[keep_mask]
    b_ineq = b[keep_mask]

    N = nullspace_basis(A_eq, tol=eq_tol)

    if N is None:
        N = np.eye(d)

    k = N.shape[1]

    row_norms = np.linalg.norm(N, axis=1)

    if row_norms.size == 0:
        fixed_idx = np.arange(d, dtype=int)
    else:
        max_row = float(np.max(row_norms))

        if max_row > 0:
            fixed_idx = np.where(row_norms <= fixed_row_tol * max_row)[0]
        else:
            fixed_idx = np.arange(d, dtype=int)

    block_data["A_eq"] = A_eq
    block_data["b_eq"] = b_eq
    block_data["A_ineq"] = A_ineq
    block_data["b_ineq"] = b_ineq
    block_data["N"] = N
    block_data["fixed_idx"] = fixed_idx
    block_data["k"] = k

    return block_data


def _initialize_block_vector(
    block_data,
    x0_block=None,
    feas_tol=1e-10,
):
    """
    Initialize a feasible duration vector for one block.

    If x0_block is provided, it is checked for feasibility.
    Otherwise, a feasible point is built from the vertices of the block
    polytope, using the same logic as the existing samplers.
    """

    A = np.asarray(block_data["A"], dtype=float)
    b = np.asarray(block_data["b"], dtype=float)

    if x0_block is None:
        vertices = compute_polytope_vertices(A, b)

        x_block = dirichlet_sample_duration_from_vertices(
            vertices,
            np.ones(vertices.shape[0]),
        )

    else:
        x_block = np.asarray(x0_block, dtype=float).copy()

        if not is_feasible(A, b, x_block, tol=feas_tol):
            raise ValueError(
                f"Provided x0 is not feasible for {block_data['name']}."
            )

    return x_block


def _propose_block_duration_vector(
    block_data,
    rng,
):
    """
    Propose a new duration vector for one block using Hit-and-Run.

    The proposal is symmetric, so no proposal-density correction is needed.

    Returns
    -------
    x_prop : np.ndarray or None
        Proposed block vector. None if no valid move was possible.

    moved : bool
        Whether a valid proposal was generated.
    """

    x_current = block_data["x_current"]
    N = block_data["N"]
    k = block_data["k"]

    if k == 0:
        return None, False

    v = rng.normal(size=k)
    norm_v = np.linalg.norm(v)

    if norm_v == 0:
        return None, False

    v /= norm_v
    u = N @ v

    t_min, t_max = chord_interval(
        block_data["A_ineq"],
        block_data["b_ineq"],
        x_current,
        u,
    )

    if (
        not np.isfinite(t_min)
        or not np.isfinite(t_max)
        or t_max <= t_min
    ):
        return None, False

    tau = rng.uniform(t_min, t_max)
    x_prop = x_current + tau * u

    fixed_idx = block_data["fixed_idx"]

    if fixed_idx.size > 0:
        x_prop[fixed_idx] = x_current[fixed_idx]

    return x_prop, True


def automatic_block_hit_and_run(
    model,
    existence_instance,
    n_samples: int,
    n_gibbs: int = 1000,
    burnin: int = 0,
    thin: int = 1,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    feas_tol: float = 1e-10,
    eq_tol: float = 1e-12,
    fixed_row_tol: float = 1e-14,
):
    """
    Automatic blocked Hit-and-Run sampler for one individual.

    Parameters
    ----------
    model : IndividualSpec
        Individual specification.

    existence_instance : dict
        Already sampled existence instance, typically:
            {"Birth": {...}}

    n_samples : int
        Number of retained samples.

    n_gibbs : int
        Number of Gibbs iterations for indicators before duration sampling.

    burnin : int
        Burn-in iterations for duration/attribute MCMC.

    thin : int
        Thinning interval.

    seed : int
        Random seed.

    x0 : np.ndarray or None
        Optional full duration vector, in the same order as the full
        individual vector. If provided, each block extracts its own part
        using block["global_indices"].

    Returns
    -------
    output_instance : dict
        Sampled output container.

    accept_rate_dur : list[int]
        Acceptance indicators for duration block proposals.

    accept_rate_attr : list[int]
        Acceptance indicators for MH attribute proposals.

    block_data : list[dict]
        Blocks enriched with matrices, geometry, and final x_current.
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # 1. Create full current instance
    # ----------------------------------------------------
    instance_current = model.create_empty_instance()
    instance_current["Existence"] = existence_instance

    # ----------------------------------------------------
    # 2. Initialize indicators
    # ----------------------------------------------------
    model.initialize_indicators(instance_current, rng=rng)

    for _ in range(n_gibbs):
        model.exact_gibbs_update_indicators(instance_current, rng=rng)

    # ----------------------------------------------------
    # 3. Initialize attributes
    # ----------------------------------------------------
    model.initialize_instance_attributes(instance_current, rng=rng)

    # ----------------------------------------------------
    # 4. Build block matrices
    # ----------------------------------------------------
    block_data = model.build_all_block_constraint_matrices(instance_current)

    # ----------------------------------------------------
    # 5. Initialize one feasible vector per block
    # ----------------------------------------------------
    for block in block_data:

        if x0 is None:
            x0_block = None
        else:
            x0 = np.asarray(x0, dtype=float)
            x0_block = x0[block["global_indices"]]

        x_block = _initialize_block_vector(
            block,
            x0_block=x0_block,
            feas_tol=feas_tol,
        )

        block["x_current"] = x_block

        # Fill the corresponding dimensions in the full instance
        model.update_instance_from_block_vector(
            x_block,
            instance_current,
            block,
        )

        # Precompute geometry after x_current exists
        _prepare_block_geometry(
            block,
            eq_tol=eq_tol,
            fixed_row_tol=fixed_row_tol,
        )

    # ----------------------------------------------------
    # 6. Output container
    # ----------------------------------------------------
    output_instance = model.create_empty_output_container(
        n_samples,
        instance_current,
    )

    output_instance["Existence"] = existence_instance

    # ----------------------------------------------------
    # 7. Initial full joint density
    # ----------------------------------------------------
    logp_current = float(model.joint_dist_duration(instance_current))

    if not np.isfinite(logp_current):
        logp_current = np.log(1e-300)
        print("Warning: initial full joint density is not finite.")

    # ----------------------------------------------------
    # 8. MCMC loop
    # ----------------------------------------------------
    accept_rate_dur = []
    accept_rate_attr = []

    stored_idx = 0
    total_iters = burnin + n_samples * thin

    for t in range(total_iters):

        # ====================================================
        # A. Attribute update
        # ====================================================

        # Gibbs attributes are accepted directly.
        model.update_instance_attributes_gibbs(instance_current, rng=rng)
        logp_current = float(model.joint_dist_duration(instance_current))

        if not np.isfinite(logp_current):
            logp_current = np.log(1e-300)

        # MH attributes need accept/reject.
        save_current_instance = copy.deepcopy(instance_current)

        log_q_forward, log_q_backward = model.update_instance_attributes_mh(
            instance_current,
            rng=rng,
        )

        logp_prop = float(model.joint_dist_duration(instance_current))

        log_alpha_attr = (
            logp_prop
            + log_q_backward
            - logp_current
            - log_q_forward
        )

        if np.isfinite(logp_prop) and (
            log_alpha_attr >= 0.0
            or np.log(rng.uniform()) < log_alpha_attr
        ):
            logp_current = logp_prop
            accept_rate_attr.append(1)

        else:
            instance_current = save_current_instance
            accept_rate_attr.append(0)

        # ====================================================
        # B. Duration updates, block by block
        # ====================================================

        # Randomize block order to avoid systematic scan effects.
        block_order = rng.permutation(len(block_data))

        for block_pos in block_order:

            block = block_data[int(block_pos)]

            x_prop, moved = _propose_block_duration_vector(
                block,
                rng=rng,
            )

            if not moved:
                continue

            save_current_instance = copy.deepcopy(instance_current)
            x_previous = block["x_current"].copy()

            # Fill only the dimensions of the current block
            model.update_instance_from_block_vector(
                x_prop,
                instance_current,
                block,
            )

            logp_prop = float(model.joint_dist_duration(instance_current))

            log_alpha_dur = logp_prop - logp_current

            if np.isfinite(logp_prop) and (
                log_alpha_dur >= 0.0
                or np.log(rng.uniform()) < log_alpha_dur
            ):
                block["x_current"] = x_prop
                logp_current = logp_prop
                accept_rate_dur.append(1)

            else:
                instance_current = save_current_instance
                block["x_current"] = x_previous
                accept_rate_dur.append(0)

        # ====================================================
        # C. Store
        # ====================================================
        if t >= burnin and ((t - burnin) % thin) == 0:
            model.store_sample(
                output_instance,
                instance_current,
                stored_idx,
            )
            stored_idx += 1

    return output_instance, accept_rate_dur, accept_rate_attr, block_data