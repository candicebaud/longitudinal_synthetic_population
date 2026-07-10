"""Hit and run functions for Existence 

Baud Candice
Fri July 10 10:33:00 2026
"""


import numpy as np 
from typing import Callable, Optional, Tuple, Dict, Any

from har_generic_code import *

def hit_and_run_existence(
    model,
    A: np.ndarray,
    b: np.ndarray,
    n_samples: int,
    burnin: int = 0,
    thin: int = 1,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    feas_tol: float = 1e-10,
):

    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, d = A.shape

    rng = np.random.default_rng(seed)

    instance_inside = model.create_empty_instance()
    output_instance = model.create_empty_output_container(n_samples)

    model.initialize_instance_attributes(instance_inside, rng = rng)

    # ---- Initialize x ----
    vertices = compute_polytope_vertices(A, b)  
    if x0 is None: 
        x_current = dirichlet_sample_duration_from_vertices(vertices, np.ones(vertices.shape[0]))
    else:
        x_current = np.asarray(x0, float).copy()
        if not is_feasible(A, b, x_current, tol=feas_tol):
            raise ValueError("Provided x0 is not feasible.")
        
    model.update_instance_from_vector(x_current, instance_inside)
    logp_current = float(model.log_density(instance_inside))

    accept_rate_dur = []
    accept_rate_attr = []
    stored_idx = 0

    total_iters = burnin + n_samples * thin

    for t in range(total_iters):

        # ====================================================
        # PROPOSE NEW STATE
        # ====================================================

        # --- 1) Propose x using Hit-and-Run ---
        u = random_unit_vector(rng, d)
        t_min, t_max = chord_interval(A, b, x_current, u)

        if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max <= t_min):
            continue

        t_prop = rng.uniform(t_min, t_max)
        x_prop = x_current + t_prop * u

        if not is_feasible(A, b, x_prop, tol=1e-9):
            continue

        # --- Save full current state ---
        old_state = {
            "x": x_current.copy(),
            "attributes": {
                att.name: instance_inside[model.name][att.name]
                for att in (model.attributes_spec or [])
            }
        }

        # --- 2) Set proposed x ---
        model.update_instance_from_vector(x_prop, instance_inside)

        # --- Evaluate target ---
        logp_prop = float(model.log_density(instance_inside))

        # -------- ACCEPTANCE of x---------
        if np.isfinite(logp_prop):

            log_alpha = (
                logp_prop
                - logp_current
            )

            if log_alpha >= 0.0 or np.log(rng.uniform()) < log_alpha:
                # Accept
                x_current = x_prop
                logp_current = logp_prop
                accept_rate_dur.append(1)
            else:
                # Reject → restore full state
                accept_rate_dur.append(0)
                x_current = old_state["x"]
                model.update_instance_from_vector(x_current, instance_inside)

        else:
            accept_rate_dur.append(0)
            # Reject automatically
            x_current = old_state["x"]
            model.update_instance_from_vector(x_current, instance_inside)


        # --- 3) Propose attributes ---
        model.update_instance_attributes_gibbs(instance_inside, rng = rng) #update with gibbs 
        old_state["attributes"] ={
                att.name: instance_inside[model.name][att.name]
                for att in (model.attributes_spec or [])
            } #update the attributes that are saved as those won't be accepted/rejected

        # propose for the ones with mh
        log_q_forward, log_q_backward = \
            model.update_instance_attributes_mh(instance_inside, rng = rng)
        # --- Evaluate target ---
        logp_prop = float(model.log_density(instance_inside))

        if np.isfinite(logp_prop):

            log_alpha = (
                logp_prop
                - logp_current
                + log_q_backward
                - log_q_forward
            )
            if log_alpha >= 0.0 or np.log(rng.uniform()) < log_alpha:
                # Accept
                logp_current = logp_prop
                accept_rate_attr.append(1)

            else:
                # Reject → restore full state
                accept_rate_attr.append(0)
                for name, val in old_state["attributes"].items():
                    instance_inside[model.name][name] = val
        else:
            accept_rate_attr.append(0)
            # Reject automatically
            for name, val in old_state["attributes"].items():
                instance_inside[model.name][name] = val

        # ====================================================
        # STORE
        # ====================================================
        if t >= burnin and ((t - burnin) % thin) == 0:
            model.store_sample(
                output_instance,
                instance_inside,
                x_current,
                stored_idx
            )
            stored_idx += 1

    return output_instance, accept_rate_dur, accept_rate_attr


def run_multiple_chains_hitandrun_existence(
    model,
    A: np.ndarray,
    b: np.ndarray,
    n_samples: int,
    burnin: int = 0,
    thin: int = 1,
    n_chains: int = 4,
    base_seed: int = 12345,
    x0s: Optional[np.ndarray] = None,  # shape (n_chains, d)
):
    """
    Returns an aggregated instance with:
      - chains for main_start_date, main_duration, gap_start_date
      - accept rates per chain
      - R_hat / ESS (your existing functions) if n_chains > 1
    """
    all_main_start = []
    all_main_duration = []
    all_gap_start = []
    accept_rates_dur = []
    accept_rates_attr = []

    # ---- Attributes storage ----
    attr_arrays = {}
    attr_names = []
    if model.attributes_spec is not None:
        attr_names = [att.name for att in model.attributes_spec]
        attr_arrays = {name: [] for name in attr_names}

    for c in range(n_chains):
        seed = base_seed + c
        x0 = None
        if x0s is not None:
            x0 = np.asarray(x0s[c], float)

        instance, acc_dur, acc_attr = hit_and_run_existence(
            model=model,
            A=A, b=b,
            n_samples=n_samples,
            burnin=burnin,
            thin=thin,
            seed=seed,
            x0=x0,
        )

        accept_rates_dur.append(acc_dur)
        accept_rates_attr.append(acc_attr)

        chain_main_start = instance[model.name]["main_start_date"]["chains"]
        chain_duration   = instance[model.name]["main_duration"]["chains"]
        chain_gap_start  = instance[model.name]["gap_start_date"]["chains"]

        all_main_start.append(chain_main_start)
        all_main_duration.append(chain_duration)
        all_gap_start.append(chain_gap_start)

        for name in attr_names:
            attr_arrays[name].append(instance[model.name][name]["chains"])

    all_main_start = np.array(all_main_start)
    all_main_duration = np.array(all_main_duration)
    all_gap_start = np.array(all_gap_start)

    if n_chains > 1:
        chains_matrix = np.stack([all_main_start, all_main_duration, all_gap_start], axis=2)  # (M, N, 3)
        R_hat = gelman_rubin(chains_matrix)
        ESS = effective_sample_size(chains_matrix)
    else:
        R_hat = [None, None, None]
        ESS = [None, None, None]

    result_instance = {
        model.name: {
            "indicator": 1,
            "main_start_date": {"chains": all_main_start, "R_hat": R_hat[0], "ESS": ESS[0]},
            "main_duration": {"chains": all_main_duration, "R_hat": R_hat[1], "ESS": ESS[1]},
            "gap_start_date": {"chains": all_gap_start, "R_hat": R_hat[2], "ESS": ESS[2]},
            "gap_duration": 0,
        },
        "accept_rate_dur": accept_rates_dur,
        "accept_rate_attr": accept_rates_attr,
    }

    if model.attributes_spec is not None:
        for name in attr_names:
            result_instance[model.name][name] = {"chains": np.array(attr_arrays[name])}

    return result_instance