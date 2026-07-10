"""Hit and run functions for sampling inside a dimension 

Baud Candice
Fri July 10 10:33:00 2026
"""

import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
from typing import Optional
import copy

from har_generic_code import *

def inside_dimension_hit_and_run(
    model,
    dob: int,
    lifespan: int,
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
    # model is the dimension specification, as we can sample dimension per dimension
    # dob, lifespan given for the bounds 
    # n_samples = 1 for one draw
    # ngibbs : for indicators
    # burnin : burnin for the durations and attributes 

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # Internal instance
    # ----------------------------------------------------
    instance_inside = model.create_empty_instance()

    model.initialize_indicators(instance_inside, lifespan, rng = rng)

    for _ in range(n_gibbs):
        model.exact_gibbs_update_indicators(instance_inside, lifespan, rng = rng)

    output_instance = model.create_empty_output_container(n_samples, instance_inside)

    # ----------------------------------------------------
    # Build constraints
    # ----------------------------------------------------
    A, b = model.build_dimension_constraint_matrix(instance_inside, lifespan)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, d = A.shape

    # initialize attributes
    model.initialize_instance_attributes(instance_inside, rng = rng)

    # ----------------------------------------------------
    # Initial feasible point : changed scipy method to vertices because more efficient
    # ----------------------------------------------------
    vertices = compute_polytope_vertices(A, b)  
    if x0 is None: 
        x_current = dirichlet_sample_duration_from_vertices(vertices, np.ones(vertices.shape[0]))
    else:
        x_current = np.asarray(x0, float).copy()
        if not is_feasible(A, b, x_current, tol=feas_tol):
            raise ValueError("Provided x0 is not feasible.")

    model.update_instance_from_vector(x_current, instance_inside, dob)

    # ----------------------------------------------------
    # Extract equalities
    # ----------------------------------------------------
    A_eq, b_eq, keep_mask = extract_equalities_from_opposites(A, b, tol=eq_tol)
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
    fixed_idx = np.where(row_norms <= fixed_row_tol * np.max(row_norms))[0]

    # ----------------------------------------------------
    # If no movement possible: durations fixed, sample attributes only
    # ----------------------------------------------------
    if k == 0:
        logp_current = float(model.log_density(instance_inside, dob, lifespan))
        if not np.isfinite(logp_current):
            raise ValueError("Initial log_density not finite.")

        accept_rate_dur = []
        accept_rate_attr = []
        stored_idx = 0
        total_iters = burnin + n_samples * thin

        for t in range(total_iters):
            accept_rate_dur.append(1)
        
            # Propose new attributes with gibbs, and save (those shouldn't be rejected anywhere)
            model.update_instance_attributes_gibbs(instance_inside, rng = rng)
            save_current_instance = copy.deepcopy(instance_inside)

            # Propose the new attributes with MH and accept/reject
            log_q_forward, log_q_backward = model.update_instance_attributes_mh(instance_inside, rng = rng)
            logp_prop = float(model.log_density(instance_inside, dob, lifespan))

            log_alpha = logp_prop + log_q_backward - logp_current - log_q_forward

            if (log_alpha >= 0.0) or (np.log(rng.uniform()) < log_alpha):
                logp_current = logp_prop
                accept_rate_attr.append(1)
            else:
                instance_inside = save_current_instance
                accept_rate_attr.append(0)

            if t >= burnin and ((t - burnin) % thin) == 0:
                model.store_sample(output_instance, instance_inside, stored_idx)
                stored_idx += 1

        return output_instance, accept_rate_dur, accept_rate_attr, A, b, fixed_idx
    
    # ----------------------------------------------------
    # Initial log density
    # ----------------------------------------------------
    logp_current = float(model.log_density(instance_inside, dob, lifespan))
    if not np.isfinite(logp_current):
        raise ValueError("log_density(x0) not finite.")
    
    accept_rate_dur = []
    accept_rate_attr = []
    stored_idx = 0
    total_iters = burnin + n_samples * thin

    for t in range(total_iters):     
        # ----------------------------------------------------
        # Step 1: Propose new attributes with a proposal
        # ----------------------------------------------------
        model.update_instance_attributes_gibbs(instance_inside, rng = rng)
        save_current_instance = copy.deepcopy(instance_inside)

        # Propose the new attributes with MH and accept/reject
        log_q_forward, log_q_backward = model.update_instance_attributes_mh(instance_inside, rng = rng)
        logp_current_attributes = float(model.log_density(instance_inside, dob, lifespan))
        log_alpha_attributes = logp_current_attributes + log_q_backward - logp_current - log_q_forward 
        if (log_alpha_attributes >= 0.0) or (np.log(rng.uniform()) < log_alpha_attributes):
            logp_current = logp_current_attributes
            accept_rate_attr.append(1)
        else:
            instance_inside = save_current_instance
            accept_rate_attr.append(0)

        save_current_instance = copy.deepcopy(instance_inside)

        # ----------------------------------------------------
        # Step 2: Propose new durations (after attributes have been updated)
        # ----------------------------------------------------
        v = rng.normal(size=k)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            continue
        v /= norm_v
        u = N @ v

        t_min, t_max = chord_interval(A_ineq, b_ineq, x_current, u)

        if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max <= t_min):
            continue

        t_prop = rng.uniform(t_min, t_max)
        x_prop = x_current + t_prop * u
        # x_prop = project_onto_equalities(x_prop, A_eq, b_eq)

        # Enforce fixed coordinates explicitly for the proposed duration
        if fixed_idx.size > 0:
            x_prop[fixed_idx] = x_current[fixed_idx]

        # Update instance with the new duration values
        model.update_instance_from_vector(x_prop, instance_inside, dob)
        logp_prop_durations = float(model.log_density(instance_inside, dob, lifespan))

        # Accept/reject durations move
        log_alpha_durations = logp_prop_durations - logp_current
        if (log_alpha_durations >= 0.0) or (np.log(rng.uniform()) < log_alpha_durations):
            x_current = x_prop
            logp_current = logp_prop_durations
            accept_rate_dur.append(1)
        else:
            instance_inside = save_current_instance  # Restore previous state (durations)
            accept_rate_dur.append(0)

        # ----------------------------------------------------
        # Store the state if it's past the burn-in phase
        # ----------------------------------------------------
        if t >= burnin and ((t - burnin) % thin) == 0:
            model.store_sample(output_instance, instance_inside, stored_idx)
            stored_idx += 1

    return output_instance, accept_rate_dur, accept_rate_attr, A, b, fixed_idx


def run_multiple_inside_dimension(
    model,
    dob,
    lifespan,
    n_samples,
    burnin,
    thin=1,
    n_chains=4,
    base_seed=12345,
    x0s=None,
    n_gibbs=1000,
):
    """
    Multi-chain runner for the JOINT MH sampler:
        - Hit-and-Run for x
        - MH for attributes

    Diagnostics computed ONLY on:
        - main_duration
        - gap_duration
        - main_start_date
        - gap_start_date

    Attributes are also returned as chains (NO diagnostics).

    Chains where a coordinate is structurally fixed
    are excluded from diagnostics for that variable.

    Indicators are returned per chain (no diagnostics).
    """

    accept_rates_dur = []
    accept_rates_attr = []
    results = {}
    chain_fixed_indices = []
    indicators = {}

    # NEW: store attributes
    attr_results = {}

    # -------------------------------------------------
    # Run chains
    # -------------------------------------------------
    for c in range(n_chains):

        seed = base_seed + c

        x0 = None
        if x0s is not None:
            x0 = np.asarray(x0s[c], float)

        output, acc_dur, acc_attr, _, _, fixed_idx = \
            inside_dimension_hit_and_run(
                model=model,
                dob=dob,
                lifespan=lifespan,
                n_samples=n_samples,
                burnin=burnin,
                thin=thin,
                seed=seed,
                x0=x0,
                n_gibbs=n_gibbs,
            )

        accept_rates_dur.append(acc_dur)
        accept_rates_attr.append(acc_attr)
        chain_fixed_indices.append(set(fixed_idx))

        # -------------------------------------------------
        # Collect per-event variables (durations)
        # -------------------------------------------------
        for event_idx, (event_name, event_dict) in enumerate(output.items()):

            if event_name not in results:
                results[event_name] = {}

            if event_name not in indicators:
                indicators[event_name] = []

            # ---- Store indicator (one per chain) ----
            indicators[event_name].append(event_dict["indicator"])

            for var_name, var_dict in event_dict.items():

                if var_name not in {
                    "main_duration",
                    "gap_duration",
                    "main_start_date",
                    "gap_start_date",
                }:
                    continue

                if "chains" not in var_dict:
                    continue

                if var_name not in results[event_name]:
                    results[event_name][var_name] = []

                results[event_name][var_name].append(
                    np.asarray(var_dict["chains"])
                )

        # -------------------------------------------------
        # Collect attributes USING THE SPEC (NEW)
        # -------------------------------------------------
        for ev in model.list_event_spec:

            if ev.max_count == 1:
                event_names = [ev.name]
            else:
                event_names = [f"{ev.name}_{m+1}" for m in range(ev.max_count)]

            for event_name in event_names:

                if ev.attributes_spec is None:
                    continue

                if event_name not in attr_results:
                    attr_results[event_name] = {}

                for att in ev.attributes_spec:

                    att_name = att.name

                    if att_name not in attr_results[event_name]:
                        attr_results[event_name][att_name] = []

                    attr_results[event_name][att_name].append(
                        np.asarray(output[event_name][att_name]["chains"])
                    )

    # -------------------------------------------------
    # Compute diagnostics (structural filtering)
    # -------------------------------------------------
    final_output = {}

    for event_idx, (event_name, vars_dict) in enumerate(results.items()):

        final_output[event_name] = {}

        # Duration coordinate indices in x
        main_dur_index = 2 * event_idx
        gap_dur_index  = 2 * event_idx + 1

        for var_name, chain_list in vars_dict.items():

            chains_array = np.array(chain_list)  # shape (M, N)

            # Map variable to coordinate index in x
            if var_name in {"main_duration", "main_start_date"}:
                coord_index = main_dur_index
            elif var_name in {"gap_duration", "gap_start_date"}:
                coord_index = gap_dur_index
            else:
                continue

            # Keep only chains where coordinate was NOT structurally fixed
            movable_mask = np.array([
                coord_index not in chain_fixed_indices[c]
                for c in range(n_chains)
            ])

            movable_chains = chains_array[movable_mask]

            if movable_chains.shape[0] >= 2:
                chains_3d = movable_chains[:, :, None]
                R_hat = gelman_rubin(chains_3d)[0]
                ESS = effective_sample_size(chains_3d)[0]
            else:
                R_hat = None
                ESS = None

            final_output[event_name][var_name] = {
                "chains": chains_array,
                "R_hat": R_hat,
                "ESS": ESS,
            }

        # ---- Add indicators ----
        final_output[event_name]["indicator"] = np.array(indicators[event_name])

    # -------------------------------------------------
    # Add attributes (NO diagnostics)
    # -------------------------------------------------
    for event_name, attrs_dict in attr_results.items():

        if event_name not in final_output:
            final_output[event_name] = {}

        for att_name, chain_list in attrs_dict.items():

            final_output[event_name][att_name] = {
                "chains": np.array(chain_list)
            }

    final_output["accept_rate_dur"] = accept_rates_dur
    final_output["accept_rate_attr"] = accept_rates_attr

    return final_output

