"""Hit and run functions for sampling inside a dimension 

Baud Candice
Fri Feb 13 17:42:00 2026
"""

import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
from typing import Optional
import copy

from har_generic_code import *



def all_dim_hit_and_run(
    model,
    existence_instance,
    n_samples: int,
    n_gibbs: int = 1000,
    burnin: int = 0,
    thin: int = 1,
    seed: int = 0,
    x0=None,
    feas_tol: float = 1e-10,
    eq_tol: float = 1e-12,
    fixed_row_tol: float = 1e-14,
):
    """
    MH sampler with fixed existence:

      - Existence is fixed externally.
      - Indicators are warmed up with exact Gibbs.
      - Each MCMC iteration does:
            (i) propose new attributes and accept/reject
            (ii) propose new durations x via Hit-and-Run and MH accept/reject

    The HAR chord proposal is symmetric, so it cancels in the MH ratio.
    The attribute proposal ratio must be included.
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # Current instance
    # ----------------------------------------------------
    instance_current = model.create_empty_instance()
    instance_current["Existence"] = existence_instance

    # Warm up indicators
    model.initialize_indicators(instance_current, rng = rng)
    for _ in range(n_gibbs):
        model.exact_gibbs_update_indicators(instance_current, rng = rng)

    # Initialize attributes
    model.initialize_instance_attributes(instance_current, rng = rng)

    # ----------------------------------------------------
    # Output container
    # ----------------------------------------------------
    output_instance = model.create_empty_output_container(
        n_samples, instance_current
    )
    output_instance["Existence"] = existence_instance

    # ----------------------------------------------------
    # Build constraints
    # ----------------------------------------------------
    A, b = model.build_constraint_matrix(instance_current)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, d = A.shape

    # ----------------------------------------------------
    # Initial feasible x
    # ----------------------------------------------------
    vertices = compute_polytope_vertices(A, b)

    if x0 is None:
        x_current = dirichlet_sample_duration_from_vertices(
            vertices, np.ones(vertices.shape[0])
        )
    else:
        x_current = np.asarray(x0, float).copy()
        if not is_feasible(A, b, x_current, tol=feas_tol):
            raise ValueError("Provided x0 is not feasible.")

    model.update_instance_from_vector(x_current, instance_current)

    # ----------------------------------------------------
    # Extract equalities
    # ----------------------------------------------------
    A_eq, b_eq, keep_mask = extract_equalities_from_opposites(
        A, b, tol=eq_tol
    )
    A_ineq = A[keep_mask]
    b_ineq = b[keep_mask]

    # ----------------------------------------------------
    # Nullspace basis
    # ----------------------------------------------------
    N = nullspace_basis(A_eq, tol=eq_tol)
    if N is None:
        N = np.eye(d)

    k = N.shape[1]

    # Detect structurally fixed coordinates
    row_norms = np.linalg.norm(N, axis=1)
    max_row = float(np.max(row_norms)) if row_norms.size > 0 else 0.0
    if max_row > 0:
        fixed_idx = np.where(row_norms <= fixed_row_tol * max_row)[0]
    else:
        fixed_idx = np.arange(d)

    # ----------------------------------------------------
    # Initial log density
    # ----------------------------------------------------
    logp_current = float(model.joint_dist_duration(instance_current))
    if not np.isfinite(logp_current):
        logp_current = np.log(0.000000000000001) # put a very very small likelihood
        print("Warning Initial log density not finite.")
        # raise ValueError("Initial log density not finite.")

    # ----------------------------------------------------
    # If no movement possible: durations fixed, attributes only
    # ----------------------------------------------------
    if k == 0:
        accept_rate_dur = []
        accept_rate_attr = []
        stored_idx = 0
        total_iters = burnin + n_samples * thin

        for t in range(total_iters):
            accept_rate_dur.append(1)
            # Step 1: propose new attributes
            model.update_instance_attributes_gibbs(instance_current, rng = rng)
            save_current_instance = copy.deepcopy(instance_current)

            log_q_forward, log_q_backward = model.update_instance_attributes_mh(
                instance_current, rng = rng
            )
            logp_prop = float(model.joint_dist_duration(instance_current))

            log_alpha = logp_prop + log_q_backward - logp_current - log_q_forward

            if (log_alpha >= 0.0) or (np.log(rng.uniform()) < log_alpha):
                logp_current = logp_prop
                accept_rate_attr.append(1)
            else:
                accept_rate_attr.append(0)
                instance_current = save_current_instance

            if t >= burnin and ((t - burnin) % thin) == 0:
                model.store_sample(output_instance, instance_current, stored_idx)
                stored_idx += 1

        return output_instance, accept_rate_dur, accept_rate_attr, A, b, fixed_idx

    # ----------------------------------------------------
    # MCMC loop
    # ----------------------------------------------------
    accept_rate_attr = []
    accept_rate_dur = []
    stored_idx = 0
    total_iters = burnin + n_samples * thin

    for t in range(total_iters): 
        # ====================================================
        # 1) Propose attributes
        # ====================================================
        model.update_instance_attributes_gibbs(instance_current, rng = rng)
        save_current_instance = copy.deepcopy(instance_current)
        
        log_q_forward, log_q_backward = model.update_instance_attributes_mh(
            instance_current, rng = rng
        )
        logp_attr = float(model.joint_dist_duration(instance_current))

        log_alpha_attr = (
            logp_attr + log_q_backward - logp_current - log_q_forward
        )

        if (log_alpha_attr >= 0.0) or (np.log(rng.uniform()) < log_alpha_attr):
            logp_current = logp_attr
            accept_rate_attr.append(1)
        else:
            accept_rate_attr.append(0)
            instance_current = save_current_instance

        # Save state after attribute move / before duration move
        save_current_instance = copy.deepcopy(instance_current)

        # ====================================================
        # 2) Propose durations via Hit-and-Run
        # ====================================================
        v = rng.normal(size=k)
        norm_v = np.linalg.norm(v)

        if norm_v != 0:
            v /= norm_v
            u = N @ v

            t_min, t_max = chord_interval(A_ineq, b_ineq, x_current, u)

            if (
                np.isfinite(t_min)
                and np.isfinite(t_max)
                and (t_max > t_min)
            ):
                tau = rng.uniform(t_min, t_max)
                x_prop = x_current + tau * u

                # Strictly enforce equalities
                # x_prop = project_onto_equalities(x_prop, A_eq, b_eq)

                # Enforce structurally fixed coordinates
                if fixed_idx.size > 0:
                    x_prop[fixed_idx] = x_current[fixed_idx]

                # Update instance with proposed durations
                model.update_instance_from_vector(x_prop, instance_current)

                logp_prop = float(model.joint_dist_duration(instance_current))

                log_alpha = logp_prop - logp_current

                if (log_alpha >= 0.0) or (np.log(rng.uniform()) < log_alpha):
                    x_current = x_prop
                    logp_current = logp_prop
                    accept_rate_dur.append(1)
                else:
                    instance_current = save_current_instance
                    accept_rate_dur.append(0)

        # ====================================================
        # 3) Store
        # ====================================================
        if t >= burnin and ((t - burnin) % thin) == 0:
            model.store_sample(output_instance, instance_current, stored_idx)
            stored_idx += 1

    return output_instance, accept_rate_dur, accept_rate_attr, A, b, fixed_idx