"""Update other dimensions using cross-sectional data

Baud Candice
Fri July 10 10:33:00 2026
"""

import numpy as np
from scipy.linalg import block_diag
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import copy
import os

from concurrent.futures import ProcessPoolExecutor


import pickle
import tempfile

from pre_coded_function import *
from classes_data import *
from constraints_events import *
from spec_existence import *
from constraints import *
from har_generic_code import *
from har_individual_level import *
from spec_other_dim import *


# convert data in existence trajectories
def dfex_to_trajex(ex_dim_spec, df_ex):
    """
    Build existence trajectories from dataframe.

    Returns
    -------
    dict
        {id: trajectory}
    """

    id_to_traj = {}

    # -----------------------------
    # Get correct names
    # -----------------------------
    dim_name = ex_dim_spec.name
    ex_event_spec = ex_dim_spec.list_event_spec[0]
    event_name = ex_event_spec.name

    # -----------------------------
    # Filter dataframe (robust)
    # -----------------------------
    df_ex = df_ex[
        (df_ex["dim_name"] == dim_name) &
        (df_ex["event_name"] == event_name)
    ]

    # -----------------------------
    # Build trajectories
    # -----------------------------
    for row in df_ex.itertuples(index=False):

        i = row.id

        main_start_date = row.main_start_date
        main_duration = row.main_duration
        gap_start_date = row.gap_start_date
        gap_duration = row.gap_duration

        duration = main_duration + gap_duration

        attrs = []
        if ex_event_spec.attributes_spec is not None:
            attrs = [
                Attribute(att.name, att.type, getattr(row, att.name))
                for att in ex_event_spec.attributes_spec
            ]

        birth_event = Event(
            event_name,
            ex_event_spec.type,
            main_start_date,
            gap_start_date,
            duration,
            main_duration,
            gap_duration,
            attrs
        )

        existence_traj = Trajectory(
            dim_name,
            [birth_event],
            [1]
        )

        id_to_traj[i] = existence_traj

    return id_to_traj # dictionary


class DimKernel:
    """
    Likelihood kernel for ONE cross-sectional dataset.

    The mapping(df, t) returns the simulated individuals available at time t.
    A second optional filter, sim_filter, can restrict which simulated
    individuals contribute to each observed individual.

    If sim_filter is None, all mapped simulated individuals contribute,
    so the behavior is the same as before.
    """

    def __init__(
        self,
        df_obs,
        t,
        f,
        mapping,
        weights=None,
        normalize_weights=False,
        sim_filter=None,
    ):

        self.df_obs = df_obs
        self.t = t
        self.f = f
        self.mapping = mapping
        self.sim_filter = sim_filter

        # -------------------------
        # HANDLE WEIGHTS
        # -------------------------
        if weights is None:
            self.weights = np.ones(len(df_obs), dtype=float)

        elif isinstance(weights, str):
            if weights not in df_obs.columns:
                raise ValueError(f"Weight column '{weights}' not found in df_obs")
            self.weights = df_obs[weights].to_numpy(dtype=float)

        else:
            self.weights = np.asarray(weights, dtype=float)

        if normalize_weights:
            self.weights = self.weights * (len(self.weights) / self.weights.sum())

        if len(self.weights) != len(df_obs):
            raise ValueError("Weights must have same length as df_obs")

        # Cache observed arrays once
        self.obs_dict = {
            c: self.df_obs[c].to_numpy()
            for c in self.df_obs.columns
        }

        self.n_obs = len(df_obs)

    def _build_sim_dict(self, time_dep_df):
        return {
            c: time_dep_df[c].to_numpy()
            for c in time_dep_df.columns
        }

    def compute_F_and_mask(self, time_dep_df, it):
        """
        Compute the kernel matrix and the admissibility mask.

        Returns
        -------
        ids : np.ndarray
            Simulated individual ids.

        F : np.ndarray
            Masked contribution matrix, shape (n_obs, n_sim).

        M : np.ndarray
            Boolean admissibility matrix, shape (n_obs, n_sim).
        """
        n_sim = len(time_dep_df)

        if n_sim == 0:
            return (
                np.array([], dtype=int),
                np.zeros((self.n_obs, 0), dtype=float),
                np.zeros((self.n_obs, 0), dtype=bool),
            )

        sim_dict = self._build_sim_dict(time_dep_df)

        F = self.f(self.obs_dict, sim_dict, it)

        if F.shape != (self.n_obs, n_sim):
            raise ValueError(
                f"Kernel returned wrong shape {F.shape}, "
                f"expected {(self.n_obs, n_sim)}"
            )

        if self.sim_filter is None:
            M = np.ones((self.n_obs, n_sim), dtype=bool)
        else:
            M = self.sim_filter(self.obs_dict, sim_dict, it)

            if M.shape != (self.n_obs, n_sim):
                raise ValueError(
                    f"sim_filter returned wrong shape {M.shape}, "
                    f"expected {(self.n_obs, n_sim)}"
                )

            M = M.astype(bool)

        F = F * M

        ids = time_dep_df["id"].to_numpy()

        return ids, F, M

    def loglik(self, time_indep_pop, it):
        """
        Full recomputation fallback.
        """
        time_dep_df = self.mapping(time_indep_pop, self.t)

        ids, F, M = self.compute_F_and_mask(time_dep_df, it)

        if len(ids) == 0:
            return -np.inf

        S = F.sum(axis=1)
        M_sum = M.sum(axis=1).astype(float)

        if np.any(M_sum <= 0):
            return -np.inf

        density_vals = S / M_sum

        if np.any(density_vals <= 0) or np.any(~np.isfinite(density_vals)):
            return -np.inf

        return float(np.sum(self.weights * np.log(density_vals)))


class IndividualState:
    """
    State of ONE individual in the posterior MCMC.

    Contains
    --------
    instance : dict
        Full event structure of the individual.

    log_p : float
        Current full individual prior log density.

    kernel_indices : list[int]
        Indices of kernels where this individual is alive.

    block_data : list[dict]
        Duration block information. Each block contains:
            - block name
            - dimension names
            - dimension specs
            - inter-event constraints
            - global duration indices
            - block constraint matrix A
            - block right-hand side b
            - current block duration vector x_current
            - block-level Hit-and-Run geometry
    """

    def __init__(self):
        self.instance = None
        self.log_p = None
        self.id = None

        self.kernel_indices = None

        # Block-level duration state and geometry
        self.block_data = None



class JointDimPopUpdater:
    def __init__(self, pop_spec, kernels, pop_spec_prior_unobserved = None):
        self.pop_spec = pop_spec
        self.kernels = kernels
        self.pop_spec_prior_unobserved = pop_spec_prior_unobserved
        

    def instance_to_dataframe(self, instance, id):
        spec = self.pop_spec
        base_cols = [
            "id", "dim_name", "event_name", "event_type",
            "main_start_date", "gap_start_date",
            "duration", "main_duration", "gap_duration"
        ]

        attribute_names = sorted({
            attr.name
            for dim_spec in spec.list_dim_spec
            for ev_spec in dim_spec.list_event_spec
            for attr in (ev_spec.attributes_spec or [])
        })

        rows = []
        spec_id = id

        for dim_spec in spec.list_dim_spec:
            if dim_spec.name == "Existence":
                dim_name = dim_spec.name
                ev_spec = dim_spec.list_event_spec[0]
                ev_name = ev_spec.name
                ev_type = ev_spec.type
                attrs = ev_spec.attributes_spec or []

                dim_instance = instance[dim_name]
                ev_data = dim_instance.get(ev_name)
                row = {
                    "id": spec_id,
                    "dim_name": dim_name,
                    "event_name": ev_name,
                    "event_type": ev_type,
                    "main_start_date": ev_data["main_start_date"],
                    "gap_start_date": ev_data["gap_start_date"],
                    "duration": ev_data["main_duration"] + ev_data["gap_duration"],
                    "main_duration": ev_data["main_duration"],
                    "gap_duration": ev_data["gap_duration"],
                }

                for attr in attrs:
                    row[attr.name] = ev_data.get(attr.name, np.nan)

                rows.append(row)

            if dim_spec.name != "Existence":
                dim_name = dim_spec.name
                dim_instance = instance[dim_name]

                for ev_spec in dim_spec.list_event_spec:
                    base_name = ev_spec.name
                    ev_type = ev_spec.type
                    attrs = ev_spec.attributes_spec or []

                    if ev_spec.max_count == 1:
                        ev_name = base_name
                        ev_data = dim_instance.get(ev_name)

                        if ev_data is None or ev_data["indicator"] != 1:
                            continue

                        row = {
                            "id": spec_id,
                            "dim_name": dim_name,
                            "event_name": ev_name,
                            "event_type": ev_type,
                            "main_start_date": ev_data["main_start_date"],
                            "gap_start_date": ev_data["gap_start_date"],
                            "duration": ev_data["main_duration"] + ev_data["gap_duration"],
                            "main_duration": ev_data["main_duration"],
                            "gap_duration": ev_data["gap_duration"],
                        }

                        for attr in attrs:
                            row[attr.name] = ev_data.get(attr.name, np.nan)

                        rows.append(row)

                    else:
                        for l in range(ev_spec.max_count):
                            ev_name = f"{base_name}_{l+1}"
                            ev_data = dim_instance.get(ev_name)

                            if ev_data is None or ev_data["indicator"] != 1:
                                continue

                            row = {
                                "id": spec_id,
                                "dim_name": dim_name,
                                "event_name": ev_name,
                                "event_type": ev_type,
                                "main_start_date": ev_data["main_start_date"],
                                "gap_start_date": ev_data["gap_start_date"],
                                "duration": ev_data["main_duration"] + ev_data["gap_duration"],
                                "main_duration": ev_data["main_duration"],
                                "gap_duration": ev_data["gap_duration"],
                            }

                            for attr in attrs:
                                row[attr.name] = ev_data.get(attr.name, np.nan)

                            rows.append(row)

        df = pd.DataFrame(rows)

        for col in attribute_names:
            if col not in df.columns:
                df[col] = np.nan

        return df[base_cols + attribute_names]

    def _existence_to_instance(self, existence_trajectory):
        birth = existence_trajectory.list_events[0]
        birth_name = birth.name

        d = {
            "main_duration": birth.main_duration,
            "main_start_date": birth.main_start_date,
            "gap_duration": birth.gap_duration,
            "gap_start_date": birth.gap_start_date,
        }

        if birth.attributes:
            for att in birth.attributes:
                d[att.name] = att.value

        return {birth_name: d}

    # def _get_kernels_alive(self, dob, dod):
    #     return [k for k in self.kernels if dob <= k.t < dod]


    def _get_kernel_indices_alive(self, dob, dod):
        return [i for i, k in enumerate(self.kernels) if dob <= k.t < dod]


    def _compute_loglik_from_cache(self, kernel_caches):
        total = 0.0

        for kernel, cache in zip(self.kernels, kernel_caches):

            S = cache["S"]
            M_sum = cache["M_sum"]

            if np.any(M_sum <= 0):
                return -np.inf

            density = S / M_sum

            if np.any(density <= 0) or np.any(~np.isfinite(density)):
                return -np.inf

            total += np.sum(kernel.weights * np.log(density))

        return float(total)
    
    def _update_kernel_cache_subset(
        self,
        kernel_caches,
        df_old,
        df_new,
        modified_ids,
        it
    ):
        """
        Incrementally update kernel caches after modifying some individuals.

        This version supports observation-specific filters.

        Cache structure per kernel:
            S       : sum of masked contributions, shape (n_obs,)
            M_sum   : number of admissible simulated individuals per obs, shape (n_obs,)
            F_indiv : dict {id -> masked contribution vector}
            M_indiv : dict {id -> admissibility mask vector}
        """
        new_caches = []
        df_sub = df_new[df_new["id"].isin(modified_ids)]

        for kernel, cache in zip(self.kernels, kernel_caches):

            S_new = cache["S"].copy()
            M_sum_new = cache["M_sum"].copy()

            F_indiv_new = cache["F_indiv"].copy()
            M_indiv_new = cache["M_indiv"].copy()

            # -------------------------
            # 1. Remove old contributions
            # -------------------------
            for indiv_id in modified_ids:

                old_F = F_indiv_new.pop(indiv_id, None)
                old_M = M_indiv_new.pop(indiv_id, None)

                if old_F is not None:
                    S_new -= old_F

                if old_M is not None:
                    M_sum_new -= old_M.astype(float)

            # -------------------------
            # 2. Recompute contributions only for modified individuals
            # -------------------------
            if len(df_sub) > 0:

                time_dep_df = kernel.mapping(df_sub, kernel.t)

                if len(time_dep_df) > 0:

                    ids_new, F_new, M_new = kernel.compute_F_and_mask(
                        time_dep_df=time_dep_df,
                        it=it,
                    )

                    for col_idx, indiv_id in enumerate(ids_new):

                        contrib = F_new[:, col_idx].copy()
                        mask_vec = M_new[:, col_idx].copy()

                        F_indiv_new[indiv_id] = contrib
                        M_indiv_new[indiv_id] = mask_vec

                        S_new += contrib
                        M_sum_new += mask_vec.astype(float)

            # -------------------------
            # 3. Keep mapped count for debugging only
            # -------------------------
            time_dep_df_full = kernel.mapping(df_new, kernel.t)
            n_mapped_new = len(time_dep_df_full)

            new_caches.append({
                "S": S_new,
                "M_sum": M_sum_new,
                "F_indiv": F_indiv_new,
                "M_indiv": M_indiv_new,
                "n_mapped": n_mapped_new,
            })

        return new_caches


    def _build_trajectories(self, instance_res, existence_trajectory):
        model = self.pop_spec
        trajs = [existence_trajectory]

        for dim in model.list_dim_spec:
            if dim.name == "Existence":
                continue

            events = []
            indic = []
            inst_dim = instance_res.get(dim.name, {})

            for ev in dim.list_event_spec:
                count = ev.max_count if ev.max_count > 1 else 1

                for m in range(count):
                    name = ev.name if count == 1 else f"{ev.name}_{m+1}"
                    ev_data = inst_dim.get(name)

                    # -----------------------------
                    # Skip if event missing
                    # -----------------------------
                    if ev_data is None:
                        continue

                    # -----------------------------
                    # Indicator check (safe)
                    # -----------------------------
                    indicator = ev_data.get("indicator", 0)
                    if indicator != 1:
                        indic.append(0)
                        continue

                    # -----------------------------
                    # Safe extraction of core fields
                    # -----------------------------
                    try:
                        main_start = ev_data["main_start_date"]["chains"][0]
                        gap_start = ev_data["gap_start_date"]["chains"][0]
                        main_dur = ev_data["main_duration"]["chains"][0]
                        gap_dur = ev_data["gap_duration"]["chains"][0]
                    except KeyError as e:
                        raise KeyError(f"Missing key {e} in event {name}")

                    # -----------------------------
                    # Attributes (safe)
                    # -----------------------------
                    attributes = []
                    if ev.attributes_spec:
                        for attr in ev.attributes_spec:
                            if attr.name in ev_data:
                                val = ev_data[attr.name]["chains"][0]
                            else:
                                val = np.nan  # or "Not applicable" if you prefer

                            attributes.append(
                                Attribute(attr.name, attr.type, val)
                            )

                    # -----------------------------
                    # Build event
                    # -----------------------------
                    events.append(
                        Event(
                            name,
                            ev.type,
                            main_start,
                            gap_start,
                            main_dur + gap_dur,
                            main_dur,
                            gap_dur,
                            attributes,
                        )
                    )

                    indic.append(1)

            trajs.append(Trajectory(dim.name, events, indic))

        return trajs
    
    def _build_trajectories_instance_inside(self, instance, existence_trajectory):
        model = self.pop_spec
        trajs = [existence_trajectory]

        for dim in model.list_dim_spec:
            if dim.name == "Existence":
                continue

            events = []
            indic = []
            inst_dim = instance.get(dim.name, {})

            for ev in dim.list_event_spec:
                count = ev.max_count if ev.max_count > 1 else 1

                for m in range(count):
                    name = ev.name if count == 1 else f"{ev.name}_{m+1}"
                    ev_data = inst_dim.get(name)

                    # -----------------------------
                    # Skip if event missing
                    # -----------------------------
                    if ev_data is None:
                        continue

                    # -----------------------------
                    # Indicator check (safe)
                    # -----------------------------
                    indicator = ev_data.get("indicator", 0)
                    if indicator != 1:
                        indic.append(0)
                        continue

                    # -----------------------------
                    # Safe extraction of core fields
                    # -----------------------------
                    try:
                        main_start = ev_data["main_start_date"]
                        gap_start = ev_data["gap_start_date"]
                        main_dur = ev_data["main_duration"]
                        gap_dur = ev_data["gap_duration"]
                    except KeyError as e:
                        raise KeyError(f"Missing key {e} in event {name}")

                    # -----------------------------
                    # Attributes (safe)
                    # -----------------------------
                    attributes = []
                    if ev.attributes_spec:
                        for attr in ev.attributes_spec:
                            if attr.name in ev_data:
                                val = ev_data[attr.name]
                            else:
                                val = np.nan  # or "Not applicable" if you prefer

                            attributes.append(
                                Attribute(attr.name, attr.type, val)
                            )

                    # -----------------------------
                    # Build event
                    # -----------------------------
                    events.append(
                        Event(
                            name,
                            ev.type,
                            main_start,
                            gap_start,
                            main_dur + gap_dur,
                            main_dur,
                            gap_dur,
                            attributes,
                        )
                    )

                    indic.append(1)

            trajs.append(Trajectory(dim.name, events, indic))

        return trajs
    
    def _get_observed_vs_never_ids(self, df):
        """
        Returns:
        - observed_ids: individuals observed at least once
        - never_observed_ids: individuals never observed
        - observed_kernels: dict {id: [kernel objects where observed]}
        """
        existence_dim = next(
            d for d in self.pop_spec.list_dim_spec if d.name == "Existence"
        )
        existence_event_name = existence_dim.list_event_spec[0].name

        df_exist = df[
            (df["dim_name"] == "Existence") &
            (df["event_name"] == existence_event_name)
        ].copy()

        df_exist["dob"] = df_exist["main_start_date"].values
        df_exist["dod"] = (
            df_exist["main_start_date"].values + df_exist["main_duration"].values + df_exist["gap_duration"].values
        )

        kernels = self.kernels
        kernel_times = np.array([k.t for k in kernels])

        order = np.argsort(kernel_times)
        kernel_times_sorted = kernel_times[order]
        kernels_sorted = [kernels[i] for i in order]

        observed_ids = set()
        observed_kernels = {}

        ids = df_exist["id"].values
        dobs = df_exist["dob"].values
        dods = df_exist["dod"].values

        for i, dob, dod in zip(ids, dobs, dods):
            left = np.searchsorted(kernel_times_sorted, dob, side="left")
            right = np.searchsorted(kernel_times_sorted, dod, side="left")

            if left < right:
                observed_ids.add(i)
                # observed_kernels[i] = kernels_sorted[left:right]
                observed_kernels[i] = order[left:right].tolist()

        all_ids = set(ids)
        never_observed_ids = all_ids - observed_ids

        return observed_ids, never_observed_ids, observed_kernels

    # ==================================================
    # Helper functions for update_indiv_gibbsattr
    # ==================================================

    def _make_indiv_spec(self, indiv_id, bool_posterior):
        if bool_posterior == 1:
            pop_model = self.pop_spec
            indiv_spec = IndividualSpec(
                indiv_id,
                pop_model.list_dim_spec,
                pop_model.joint_dist_duration,
                pop_model.list_inter_constraint_spec,
            )
            return indiv_spec
        else:
            if self.pop_spec_prior_unobserved is None :
                pop_model = self.pop_spec
                indiv_spec = IndividualSpec(
                    indiv_id,
                    pop_model.list_dim_spec,
                    pop_model.joint_dist_duration,
                    pop_model.list_inter_constraint_spec,
                )
                return indiv_spec
            else:
                pop_model = self.pop_spec_prior_unobserved
                indiv_spec = IndividualSpec(
                    indiv_id,
                    pop_model.list_dim_spec,
                    pop_model.joint_dist_duration,
                    pop_model.list_inter_constraint_spec,
                )
                return indiv_spec

    def _sample_prior_individual(
        self,
        indiv_spec,
        indiv_id,
        existence_trajectory,
        n_gibbs_indicators,
        n_samples,
        burnin,
        thin,
        x0,
        seed,
    ):
        indiv_spec.id = indiv_id
        existence_instance = self._existence_to_instance(existence_trajectory)

        instance_res, *_ = (
            all_dim_hit_and_run(
                indiv_spec,
                existence_instance,
                n_samples=n_samples,
                n_gibbs=n_gibbs_indicators,
                burnin=burnin,
                thin=thin,
                x0=x0,
                seed=seed,
            )
        )

        trajs = self._build_trajectories(instance_res, existence_trajectory)
        return Individual(indiv_id, trajs)

    def _initialize_empty_population_states(self, observed_ids, id_to_traj, observed_kernels, indiv_spec):
        population_posterior = {}

        for indiv_id in observed_ids:
            indiv_spec.id = indiv_id
            existence_traj = id_to_traj[indiv_id]
            existence_instance = self._existence_to_instance(existence_traj)

            state = IndividualState()
            state.instance = indiv_spec.create_empty_instance()
            state.instance["Existence"] = existence_instance
            # state.kernels_alive = observed_kernels[indiv_id]
            state.kernel_indices = observed_kernels[indiv_id]
            state.id = indiv_id

            population_posterior[indiv_id] = state

        return population_posterior

    def _initialize_state_indicators_and_attributes(
        self,
        indiv_spec,
        state,
        n_gibbs_indicators,
        rng
    ):
        for _ in range(n_gibbs_indicators):
            indiv_spec.exact_gibbs_update_indicators(state.instance, rng = rng)

        indiv_spec.initialize_instance_attributes(state.instance, rng = rng)

    def _initialize_block_duration_vector(
        self,
        block_data,
        x0_block=None,
        feas_tol=1e-10,
        rng = 0
    ):
        """
        Initialize a feasible duration vector for one duration block.

        If x0_block is provided, it is checked for feasibility.
        Otherwise, a feasible point is created from the block polytope vertices.
        """

        A = np.asarray(block_data["A"], dtype=float)
        b = np.asarray(block_data["b"], dtype=float)

        if x0_block is None:

            vertices = compute_polytope_vertices(A, b)

            # x_block = dirichlet_sample_duration_from_vertices(
            #     vertices,
            #     np.ones(vertices.shape[0]),
            # )

            coefs = rng.dirichlet(np.ones(vertices.shape[0]))
            coefs = np.maximum(coefs, 1e-12)
            x_block = dirichlet_sample_duration_from_vertices(
                vertices,
                coefs
            )


        else:

            x_block = np.asarray(x0_block, dtype=float).copy()

            if not is_feasible(A, b, x_block, tol=feas_tol):
                raise ValueError(
                    f"x0_block is not feasible for {block_data['name']}."
                )

        return x_block
    
    def _prepare_block_geometry(
        self,
        block_data,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
    ):
        """
        Prepare Hit-and-Run geometry for one duration block.

        Parameters
        ----------
        block_data : dict
            Block dictionary containing at least:
                - "A"
                - "b"

        Returns
        -------
        block_data : dict
            Same dictionary enriched with:
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
#########   

    def _initialize_state_geometry(
        self,
        indiv_spec,
        state,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        rng = 0
    ):
        """
        Initialize duration vectors and Hit-and-Run geometry for all duration
        blocks of one individual.

        The block structure is defined by the model specification through
        DurationLinearConstraintsSpec. The same block structure is used for
        all individuals, but the actual block matrices depend on the
        individual's indicators.
        """

        # --------------------------------------------------
        # Build all block matrices for the current individual
        # --------------------------------------------------
        block_data = indiv_spec.build_all_block_constraint_matrices(
            state.instance
        )

        # --------------------------------------------------
        # Initialize each block vector and update the instance
        # --------------------------------------------------
        for block in block_data:

            if x0 is None:
                x0_block = None
            else:
                x0_full = np.asarray(x0, dtype=float)
                x0_block = x0_full[block["global_indices"]]

            x_block = self._initialize_block_duration_vector(
                block_data=block,
                x0_block=x0_block,
                feas_tol=feas_tol,
                rng = rng
            )

            block["x_current"] = x_block

            # Fill only the dimensions belonging to this block
            indiv_spec.update_instance_from_block_vector(
                x_block,
                state.instance,
                block,
            )

            # Prepare block-level Hit-and-Run geometry
            self._prepare_block_geometry(
                block_data=block,
                eq_tol=eq_tol,
                fixed_row_tol=fixed_row_tol,
            )

        state.block_data = block_data


    def _initialize_state_logdensities(self, indiv_spec, state):
        logp_current = float(indiv_spec.joint_dist_duration(state.instance))
        if not np.isfinite(logp_current):
            raise ValueError("Initial log density not finite.")

        state.log_p = logp_current

    def _initialize_one_posterior_state(
        self,
        indiv_spec,
        state,
        n_gibbs_indicators,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        rng = 0,
    ):
        self._initialize_state_indicators_and_attributes(
            indiv_spec=indiv_spec,
            state=state,
            n_gibbs_indicators=n_gibbs_indicators,
            rng = rng
        )

        self._initialize_state_geometry(
            indiv_spec=indiv_spec,
            state=state,
            x0=x0,
            feas_tol=feas_tol,
            eq_tol=eq_tol,
            fixed_row_tol=fixed_row_tol,
            rng = rng
        )

        self._initialize_state_logdensities(
            indiv_spec=indiv_spec,
            state=state,
        )

    def _sample_prior_population(
        self,
        never_observed_ids,
        id_to_traj,
        indiv_spec,
        n_gibbs_indicators,
        n_samples,
        burnin,
        thin,
        x0,
        seed,
    ):
        list_individuals_prior = []

        never_observed_ids = sorted(never_observed_ids)

        for i, indiv_id in enumerate(never_observed_ids):
            indiv_seed = seed + i
            existence_trajectory = id_to_traj[indiv_id]

            indiv = self._sample_prior_individual(
                indiv_spec=indiv_spec,
                indiv_id=indiv_id,
                existence_trajectory=existence_trajectory,
                n_gibbs_indicators=n_gibbs_indicators,
                n_samples=n_samples,
                burnin=burnin,
                thin=thin,
                x0=x0,
                seed=indiv_seed,
            )
            list_individuals_prior.append(indiv)

        return list_individuals_prior
    
    def _sample_prior_population_parallel(
        self,
        never_observed_ids,
        id_to_traj,
        n_gibbs_indicators,
        n_samples,
        burnin,
        thin,
        x0,
        seed,
        n_jobs=None,
        output_dir="posterior_update_unalive"
    ):  
        os.makedirs(output_dir, exist_ok=True)

        never_observed_ids = sorted(never_observed_ids)

        if len(never_observed_ids) == 0:
            return []

        seeds = [seed + i for i in range(len(never_observed_ids))]

        if self.pop_spec_prior_unobserved is None :
            tasks = [
                (
                    self.pop_spec,
                    indiv_id,
                    id_to_traj[indiv_id],
                    n_gibbs_indicators,
                    n_samples,
                    burnin,
                    thin,
                    x0,
                    indiv_seed,
                    output_dir
                )
                for indiv_id, indiv_seed in zip(never_observed_ids, seeds)
            ]
        else:
            tasks = [
                (
                    self.pop_spec_prior_unobserved,
                    indiv_id,
                    id_to_traj[indiv_id],
                    n_gibbs_indicators,
                    n_samples,
                    burnin,
                    thin,
                    x0,
                    indiv_seed,
                    output_dir
                )
                for indiv_id, indiv_seed in zip(never_observed_ids, seeds)
            ]

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            out = list(ex.map(_sample_prior_individual_worker, tasks))

        return out

    def _initialize_posterior_population(
        self,
        observed_ids,
        id_to_traj,
        observed_kernels,
        indiv_spec,
        n_gibbs_indicators,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        seed=0,
    ):
        """
        Returns a dictionary of initialized states for all observed individuals.
        For each id in observed_ids, population_posterior[id] = state.
        """

        population_posterior = self._initialize_empty_population_states(
            observed_ids=observed_ids,
            id_to_traj=id_to_traj,
            observed_kernels=observed_kernels,
            indiv_spec=indiv_spec,
        )

        observed_ids = sorted(observed_ids)

        for i, indiv_id in enumerate(observed_ids):
            indiv_spec.id = indiv_id
            state = population_posterior[indiv_id]

            rng_i = np.random.default_rng(seed + i)

            self._initialize_one_posterior_state(
                indiv_spec=indiv_spec,
                state=state,
                n_gibbs_indicators=n_gibbs_indicators,
                x0=x0,
                feas_tol=feas_tol,
                eq_tol=eq_tol,
                fixed_row_tol=fixed_row_tol,
                rng=rng_i,
            )

        return population_posterior
    
    def _initialize_posterior_population_parallel(
        self,
        observed_ids,
        id_to_traj,
        observed_kernels,
        n_gibbs_indicators,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        seed=0,
        n_jobs=None,
    ):
        ids = list(observed_ids)

        if len(ids) == 0:
            return {}

        seeds = [seed + i for i in range(len(ids))]

        tasks = [
            (
                self.pop_spec,
                indiv_id,
                id_to_traj[indiv_id],
                observed_kernels[indiv_id],
                n_gibbs_indicators,
                x0,
                feas_tol,
                eq_tol,
                fixed_row_tol,
                indiv_seed,
            )
            for indiv_id, indiv_seed in zip(ids, seeds)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_initialize_one_posterior_state_worker, tasks))

        population_posterior = {indiv_id: state for indiv_id, state in results}
        return population_posterior
    
    def _propose_new_state_gibbs_attr(self, indiv_state, indiv_spec, rng):
        indiv_spec.update_instance_attributes_gibbs(indiv_state.instance, rng = rng)
        logp_prop = float(indiv_spec.joint_dist_duration(indiv_state.instance))
        indiv_state.log_p = logp_prop

    def _propose_new_state_mh_attr(self, indiv_state, indiv_spec, rng):
        log_f, log_b = indiv_spec.update_instance_attributes_mh(indiv_state.instance, rng = rng)
        logp_prop = float(indiv_spec.joint_dist_duration(indiv_state.instance))
        indiv_state.log_p = logp_prop
        return(log_f, log_b)
    

    def _two_region_bins(self, t_min, t_max, local_frac):
        """
        Build close/far proposal regions on the hit-and-run chord.

        The current point corresponds to tau = 0.

        The close region is the central fraction of the feasible chord
        around tau = 0:

            close = [local_frac * t_min, local_frac * t_max]

        Since t_min <= 0 <= t_max, this gives an inner region around
        the current point.

        The far region is the remaining feasible interval:

            far_left  = [t_min, local_frac * t_min]
            far_right = [local_frac * t_max, t_max]

        This guarantees that, when 0 < local_frac < 1, the far region is
        non-empty whenever movement is possible.
        """

        if not (0.0 < local_frac <= 1.0):
            raise ValueError("local_frac must satisfy 0 < local_frac <= 1.")

        if not np.isfinite(t_min) or not np.isfinite(t_max):
            raise ValueError("t_min and t_max must be finite.")

        if not (t_max > t_min):
            raise ValueError("Chord length must be strictly positive.")

        # In hit-and-run, tau = 0 is the current point and should be feasible.
        if not (t_min <= 0.0 <= t_max):
            raise ValueError(
                f"The chord interval should contain 0, but got [{t_min}, {t_max}]."
            )

        # -------------------------
        # Close region
        # -------------------------
        close_start = local_frac * t_min
        close_end = local_frac * t_max

        close_parts = []
        if close_end > close_start:
            close_parts.append((close_start, close_end))

        # -------------------------
        # Far region
        # -------------------------
        far_parts = []

        # Left far part
        left_start = t_min
        left_end = close_start

        if left_end > left_start:
            far_parts.append((left_start, left_end))

        # Right far part
        right_start = close_end
        right_end = t_max

        if right_end > right_start:
            far_parts.append((right_start, right_end))

        return [close_parts, far_parts]


    def _bin_length(self, parts):
        """
        Compute the total length of a possibly disconnected bin.
        """

        return float(sum(end - start for start, end in parts))


    def _normalize_two_region_weights(self, bins, p_far):
        """
        Normalize proposal weights for the close/far regions.

        Base weights are:
            close: 1 - p_far
            far:   p_far

        Empty regions get probability zero and the remaining weights
        are renormalized.
        """

        if not (0.0 <= p_far <= 1.0):
            raise ValueError("p_far must be between 0 and 1.")

        lengths = np.array(
            [self._bin_length(parts) for parts in bins],
            dtype=float
        )

        weights = np.array(
            [1.0 - p_far, p_far],
            dtype=float
        )

        # Empty regions cannot be selected
        weights[lengths <= 0.0] = 0.0

        total_weight = weights.sum()

        if total_weight <= 0.0:
            raise ValueError("Both close and far proposal regions are empty.")

        weights /= total_weight

        return weights, lengths


    def _sample_tau_from_two_region_bins(self, bins, weights, lengths, rng):
        """
        Sample tau from the two-region close/far proposal.

        Steps
        -----
        1. Select close or far region using `weights`.
        2. If the selected region has several intervals, select one interval
        with probability proportional to its length.
        3. Sample uniformly inside that interval.
        """

        selected_region = int(rng.choice(2, p=weights))

        parts = bins[selected_region]

        if lengths[selected_region] <= 0.0:
            raise ValueError("Selected an empty proposal region.")

        part_lengths = np.array(
            [end - start for start, end in parts],
            dtype=float
        )

        part_probs = part_lengths / part_lengths.sum()

        selected_part = int(rng.choice(len(parts), p=part_probs))
        start, end = parts[selected_part]

        tau = rng.uniform(start, end)

        return tau, selected_region


    def _log_tau_density_two_region(
        self,
        tau,
        t_min,
        t_max,
        local_frac,
        p_far,
        atol=1e-12,
    ):
        """
        Compute log q(tau) under the close/far proposal.

        If tau belongs to region j, then:

            q(tau) = weight_j / length_j

        where region j is either close or far.
        """

        bins = self._two_region_bins(t_min, t_max, local_frac)
        weights, lengths = self._normalize_two_region_weights(bins, p_far)

        for region_idx, parts in enumerate(bins):

            if weights[region_idx] <= 0.0 or lengths[region_idx] <= 0.0:
                continue

            for start, end in parts:
                if (tau >= start - atol) and (tau <= end + atol):
                    return float(
                        np.log(weights[region_idx]) - np.log(lengths[region_idx])
                    )

        return -np.inf

    def _propose_new_state_duration_block(
        self,
        indiv_state,
        indiv_spec,
        block_index,
        rng,
        dur_local_frac=None,
        dur_p_far=None,
    ):
        """
        Propose a new duration vector for one block of one individual.

        The proposal modifies only the dimensions contained in that block.
        The acceptance ratio is computed outside this function at the
        population level.

        Parameters
        ----------
        indiv_state : IndividualState
            Proposed state to modify in-place.

        indiv_spec : IndividualSpec
            Individual specification.

        block_index : int
            Index of the block in indiv_state.block_data.

        rng : np.random.Generator
            Random generator.

        dur_local_frac, dur_p_far : float or None
            If both are None, classical Hit-and-Run is used.
            If both are provided, the close/far proposal is used.

        Returns
        -------
        log_q_fwd : float
            Forward proposal log-density.

        log_q_back : float
            Backward proposal log-density.

        moved : bool
            Whether a valid proposal was generated.
        """

        use_classical_hr = (dur_local_frac is None) and (dur_p_far is None)
        use_close_far_hr = (dur_local_frac is not None) and (dur_p_far is not None)

        if not use_classical_hr and not use_close_far_hr:
            raise ValueError(
                "Either provide both dur_local_frac and dur_p_far, "
                "or set both to None for classical hit-and-run."
            )

        if use_close_far_hr:
            if not (0.0 < dur_local_frac <= 1.0):
                raise ValueError("dur_local_frac must satisfy 0 < dur_local_frac <= 1.")

            if not (0.0 <= dur_p_far <= 1.0):
                raise ValueError("dur_p_far must satisfy 0 <= dur_p_far <= 1.")

        block = indiv_state.block_data[block_index]

        x_current = block["x_current"]
        N = block["N"]
        k = block["k"]

        log_q_fwd = 0.0
        log_q_back = 0.0

        if k == 0:
            indiv_state.log_p = float(
                indiv_spec.joint_dist_duration(indiv_state.instance)
            )
            return log_q_fwd, log_q_back, False

        # --------------------------------------------------
        # Direction in block nullspace
        # --------------------------------------------------
        v = rng.normal(size=k)
        norm_v = np.linalg.norm(v)

        if norm_v == 0.0:
            indiv_state.log_p = float(
                indiv_spec.joint_dist_duration(indiv_state.instance)
            )
            return log_q_fwd, log_q_back, False

        v /= norm_v
        u = N @ v

        # --------------------------------------------------
        # Feasible chord in the block polytope
        # --------------------------------------------------
        t_min, t_max = chord_interval(
            block["A_ineq"],
            block["b_ineq"],
            x_current,
            u,
        )

        if (
            not np.isfinite(t_min)
            or not np.isfinite(t_max)
            or not (t_max > t_min)
        ):
            indiv_state.log_p = float(
                indiv_spec.joint_dist_duration(indiv_state.instance)
            )
            return log_q_fwd, log_q_back, False

        # --------------------------------------------------
        # Sample tau
        # --------------------------------------------------
        if use_classical_hr:

            tau = rng.uniform(t_min, t_max)

            log_q_fwd = 0.0
            log_q_back = 0.0

        else:

            bins_fwd = self._two_region_bins(
                t_min=t_min,
                t_max=t_max,
                local_frac=dur_local_frac,
            )

            weights_fwd, lengths_fwd = self._normalize_two_region_weights(
                bins=bins_fwd,
                p_far=dur_p_far,
            )

            tau, _ = self._sample_tau_from_two_region_bins(
                bins=bins_fwd,
                weights=weights_fwd,
                lengths=lengths_fwd,
                rng=rng,
            )

            log_q_fwd = self._log_tau_density_two_region(
                tau=tau,
                t_min=t_min,
                t_max=t_max,
                local_frac=dur_local_frac,
                p_far=dur_p_far,
            )

            # Reverse move
            t_min_back = t_min - tau
            t_max_back = t_max - tau
            tau_back = -tau

            log_q_back = self._log_tau_density_two_region(
                tau=tau_back,
                t_min=t_min_back,
                t_max=t_max_back,
                local_frac=dur_local_frac,
                p_far=dur_p_far,
            )

        # --------------------------------------------------
        # Proposed block vector
        # --------------------------------------------------
        x_prop = x_current + tau * u

        fixed_idx = block["fixed_idx"]

        if fixed_idx.size > 0:
            x_prop[fixed_idx] = x_current[fixed_idx]

        # --------------------------------------------------
        # Update only this block in the full individual instance
        # --------------------------------------------------
        indiv_spec.update_instance_from_block_vector(
            x_prop,
            indiv_state.instance,
            block,
        )

        block["x_current"] = x_prop

        # Full individual prior density
        indiv_state.log_p = float(
            indiv_spec.joint_dist_duration(indiv_state.instance)
        )

        return log_q_fwd, log_q_back, True


##################################

    def _count_individuals_with_mapping_change(
        self,
        df_old,
        df_new,
        modified_ids,
        compare_cols=None,
        exclude_cols=("id",),
        atol=1e-10,
        rtol=1e-10,
        return_details=False,
    ):
        """
        Count how many modified individuals have a changed mapped state.

        An individual is counted as changed if, for at least one kernel time t,
        its mapped row before and after the proposal differs.

        Parameters
        ----------
        df_old : pd.DataFrame
            Population dataframe before the proposal.

        df_new : pd.DataFrame
            Population dataframe after the proposal.

        modified_ids : iterable
            Ids of individuals modified by the proposal.

        compare_cols : list[str] or None
            Columns to compare in the mapped dataframe.
            If None, compares all common columns except exclude_cols.

        exclude_cols : tuple[str]
            Columns ignored when comparing mapped values.
            By default, "id" is ignored.

        atol, rtol : float
            Numerical tolerances for comparing numeric columns.

        return_details : bool
            If True, also return a dictionary with detailed changes.

        Returns
        -------
        n_changed : int
            Number of individuals whose mapped state changed in at least one kernel.

        changed_ids : list
            List of individual ids whose mapped state changed.

        details : dict, optional
            Returned only if return_details=True.
            Structure:
                {
                    indiv_id: [
                        {
                            "kernel_index": k_idx,
                            "time": kernel.t,
                            "old": old_row_as_dict_or_None,
                            "new": new_row_as_dict_or_None,
                        },
                        ...
                    ]
                }
        """

        modified_ids = list(modified_ids)

        changed_ids = set()
        details = {indiv_id: [] for indiv_id in modified_ids}

        for k_idx, kernel in enumerate(self.kernels):

            mapped_old = kernel.mapping(df_old, kernel.t)
            mapped_new = kernel.mapping(df_new, kernel.t)

            old_sub = mapped_old[mapped_old["id"].isin(modified_ids)]
            new_sub = mapped_new[mapped_new["id"].isin(modified_ids)]

            for indiv_id in modified_ids:

                old_i = old_sub[old_sub["id"] == indiv_id].copy()
                new_i = new_sub[new_sub["id"] == indiv_id].copy()

                changed = self._mapped_individual_changed(
                    old_i,
                    new_i,
                    compare_cols=compare_cols,
                    exclude_cols=exclude_cols,
                    atol=atol,
                    rtol=rtol,
                )

                if changed:
                    changed_ids.add(indiv_id)

                    if return_details:
                        details[indiv_id].append({
                            "kernel_index": k_idx,
                            "time": kernel.t,
                            "old": None if old_i.empty else old_i.to_dict("records"),
                            "new": None if new_i.empty else new_i.to_dict("records"),
                        })

        changed_ids = sorted(changed_ids)
        n_changed = len(changed_ids)

        if return_details:
            details = {k: v for k, v in details.items() if len(v) > 0}
            return n_changed, changed_ids, details

        return n_changed, changed_ids
    
    def _mapped_individual_changed(
        self,
        old_i,
        new_i,
        compare_cols=None,
        exclude_cols=("id",),
        atol=1e-10,
        rtol=1e-10,
    ):
        """
        Return True if the mapped state of one individual changed.

        NaN handling:
        - NaN vs NaN is treated as unchanged.
        - NaN vs non-NaN is treated as changed.
        """

        # --------------------------------------------------
        # Case 1: individual absent in both mappings
        # --------------------------------------------------
        if old_i.empty and new_i.empty:
            return False

        # --------------------------------------------------
        # Case 2: individual appears/disappears
        # --------------------------------------------------
        if old_i.empty != new_i.empty:
            return True

        # --------------------------------------------------
        # Reset indices to compare row by row
        # --------------------------------------------------
        old_i = old_i.reset_index(drop=True)
        new_i = new_i.reset_index(drop=True)

        # --------------------------------------------------
        # Different number of mapped rows
        # --------------------------------------------------
        if len(old_i) != len(new_i):
            return True

        # --------------------------------------------------
        # Select comparison columns
        # --------------------------------------------------
        if compare_cols is None:
            common_cols = old_i.columns.intersection(new_i.columns)
            compare_cols = [
                c for c in common_cols
                if c not in exclude_cols
            ]
        else:
            compare_cols = [
                c for c in compare_cols
                if c in old_i.columns and c in new_i.columns
            ]

        # --------------------------------------------------
        # Compare column by column
        # --------------------------------------------------
        for col in compare_cols:

            old_vals = old_i[col]
            new_vals = new_i[col]

            # NaN mask
            old_nan = old_vals.isna()
            new_nan = new_vals.isna()

            # If NaN pattern differs, then changed
            if not old_nan.equals(new_nan):
                return True

            # Compare only positions where both are not NaN
            mask = ~old_nan

            if mask.sum() == 0:
                continue

            old_non_nan = old_vals[mask]
            new_non_nan = new_vals[mask]

            # Numeric comparison with tolerance
            if pd.api.types.is_numeric_dtype(old_non_nan) and pd.api.types.is_numeric_dtype(new_non_nan):
                if not np.allclose(
                    old_non_nan.to_numpy(),
                    new_non_nan.to_numpy(),
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                ):
                    return True

            # Non-numeric comparison
            else:
                if not old_non_nan.reset_index(drop=True).equals(
                    new_non_nan.reset_index(drop=True)
                ):
                    return True

        return False


############################
    

    def _population_to_dataframe_with_index(self, population_posterior):
        dfs = []
        id_to_indices = {}
        current_idx = 0

        for indiv_id, state in population_posterior.items():
            df_i = self.instance_to_dataframe(state.instance, indiv_id)
            n_i = len(df_i)

            dfs.append(df_i)

            id_to_indices[indiv_id] = np.arange(current_idx, current_idx + n_i)
            current_idx += n_i

        if not dfs:
            return pd.DataFrame(), {}

        df = pd.concat(dfs, ignore_index=True)
        return df, id_to_indices
    
    def _update_dataframe_subset(
        self,
        df_current,
        id_to_indices,
        population_states,
        modified_ids
    ):
        """
        Update dataframe by replacing rows corresponding to modified_ids.

        Parameters
        ----------
        df_current : pd.DataFrame
        id_to_indices : dict {id: np.ndarray of row indices}
        population_states : dict {id: state}
        modified_ids : list or array

        Returns
        -------
        df_new, new_id_to_indices
        """

        # -------------------------
        # 1. Build mask to KEEP rows
        # -------------------------
        mask = np.ones(len(df_current), dtype=bool)

        for indiv_id in modified_ids:
            if indiv_id in id_to_indices:
                mask[id_to_indices[indiv_id]] = False

        df_kept = df_current.loc[mask].copy()

        # -------------------------
        # 2. Rebuild rows for modified individuals
        # -------------------------
        dfs_new = []
        new_id_to_indices = {}

        current_idx = len(df_kept)

        for indiv_id in modified_ids:
            state = population_states[indiv_id]
            df_i = self.instance_to_dataframe(state.instance, indiv_id)

            n_i = len(df_i)

            dfs_new.append(df_i)
            new_id_to_indices[indiv_id] = np.arange(current_idx, current_idx + n_i)

            current_idx += n_i

        # -------------------------
        # 3. Concatenate
        # -------------------------
        if dfs_new:
            df_new = pd.concat([df_kept] + dfs_new, ignore_index=True)
        else:
            df_new = df_kept

        # -------------------------
        # 4. Update index mapping
        # -------------------------
        # First rebuild mapping for kept rows
        updated_id_to_indices = {}

        # recompute mapping for kept individuals
        grouped = df_kept.groupby("id").indices
        for k, v in grouped.items():
            updated_id_to_indices[k] = np.array(v)

        # overwrite with new ones
        updated_id_to_indices.update(new_id_to_indices)

        return df_new, updated_id_to_indices
    
    def _initialize_kernel_cache(self, df_pop):
        kernel_caches = []

        for kernel in self.kernels:

            time_dep_df = kernel.mapping(df_pop, kernel.t)

            ids, F, M = kernel.compute_F_and_mask(
                time_dep_df=time_dep_df,
                it=0,
            )

            F_indiv = {}
            M_indiv = {}

            for col_idx, indiv_id in enumerate(ids):
                F_indiv[indiv_id] = F[:, col_idx].copy()
                M_indiv[indiv_id] = M[:, col_idx].copy()

            S = F.sum(axis=1)
            M_sum = M.sum(axis=1).astype(float)

            kernel_caches.append({
                "S": S,
                "M_sum": M_sum,
                "F_indiv": F_indiv,
                "M_indiv": M_indiv,
                "n_mapped": len(ids),
            })

        return kernel_caches


    def _debug_print_mapping_changes(
        self,
        df_old,
        df_new,
        modified_ids,
        max_kernels=3
    ):
        """
        Print mapped values before/after for modified individuals.

        Only prints for a subset of kernels to avoid explosion.
        """

        print("\n================ DEBUG: MAPPING CHANGES ================")

        for k_idx, kernel in enumerate(self.kernels[:max_kernels]):
            t = kernel.t

            mapped_old = kernel.mapping(df_old, t)
            mapped_new = kernel.mapping(df_new, t)

            old_sub = mapped_old[mapped_old["id"].isin(modified_ids)]
            new_sub = mapped_new[mapped_new["id"].isin(modified_ids)]

            print(f"\n--- Kernel {k_idx} at time t = {t} ---")

            for indiv_id in modified_ids:
                old_i = old_sub[old_sub["id"] == indiv_id]
                new_i = new_sub[new_sub["id"] == indiv_id]

                if old_i.empty and new_i.empty:
                    continue

                print(f"\nIndividual {indiv_id}:")

                if not old_i.empty:
                    print("  OLD:")
                    print(old_i.to_dict("records")[0])
                else:
                    print("  OLD: not present")

                if not new_i.empty:
                    print("  NEW:")
                    print(new_i.to_dict("records")[0])
                else:
                    print("  NEW: not present")

        print("=======================================================\n")
    
    def sample_prior_indices(
        self,
        indices, 
        id_to_traj,
        n_gibbs_indicators = 1000,
        parallel_prior = 1,
        n_jobs_prior = 8,
        seed = 0,
        n_samples = 1,
        burnin = 5000,
        thin = 1,
        output_dir = "prior_samples",
        x0 = None
    ):
        ## to sample from the prior using the specified indices

        rng = np.random.default_rng(seed)
        indiv_spec_prior = self._make_indiv_spec(indiv_id = None, bool_posterior = 0)

        #---------------- PRIOR ----------------
        if parallel_prior == 1:
            list_individuals_prior = self._sample_prior_population_parallel(
                never_observed_ids=indices,
                id_to_traj=id_to_traj,
                n_gibbs_indicators=n_gibbs_indicators,
                n_samples=n_samples,
                burnin=burnin,
                thin=thin,
                x0=x0,
                seed=seed,
                n_jobs=n_jobs_prior,  
                output_dir=output_dir
            )
        elif parallel_prior == 0:
            list_individuals_prior = self._sample_prior_population(
                never_observed_ids=indices,
                id_to_traj=id_to_traj,
                indiv_spec=indiv_spec_prior,
                n_gibbs_indicators=n_gibbs_indicators,
                n_samples=n_samples,
                burnin=burnin,
                thin=thin,
                x0=x0,
                seed=seed,
            )
        
        else:
            raise ValueError("Invalid value for parallel prior.")


    def _duration_block_population_step(
        self,
        population_posterior,
        time_indep_pop,
        id_to_indices,
        kernel_caches,
        logl_previous,
        selected_individuals_ids,
        indiv_spec_posterior,
        block_index,
        rng,
        it,
        dur_local_frac=None,
        dur_p_far=None,
    ):
        """
        Propose and accept/reject one duration block for several individuals.

        This is the blocked version of the old duration proposal.

        For a fixed block_index:
            1. For each selected individual, propose a new state only for this block.
            2. Build the proposed population.
            3. Update the dataframe only for modified individuals.
            4. Update kernel caches only for modified individuals.
            5. Accept/reject the block proposal at the population level.

        Parameters
        ----------
        population_posterior : dict
            Current posterior states, indexed by individual id.

        time_indep_pop : pd.DataFrame
            Current time-independent population dataframe.

        id_to_indices : dict
            Row indices of each individual in time_indep_pop.

        kernel_caches : list[dict]
            Current likelihood caches.

        logl_previous : float
            Current population log-likelihood.

        selected_individuals_ids : iterable
            Individuals selected at the current MCMC iteration.

        indiv_spec_posterior : IndividualSpec
            Posterior individual specification.

        block_index : int
            Index of the block to update.

        rng : np.random.Generator
            Random generator.

        it : int
            Current MCMC iteration.

        dur_local_frac, dur_p_far : float or None
            Parameters for the close/far Hit-and-Run proposal.
            If both are None, classical Hit-and-Run is used.

        Returns
        -------
        population_posterior : dict
            Updated posterior states.

        time_indep_pop : pd.DataFrame
            Updated dataframe.

        id_to_indices : dict
            Updated row index dictionary.

        kernel_caches : list[dict]
            Updated likelihood caches.

        logl_previous : float
            Updated log-likelihood.

        accepted : int
            1 if the block proposal was accepted, 0 otherwise.

        n_changed_map : int
            Number of individuals whose mapped state changed, counted only
            if the proposal is accepted.

        moved : bool
            Whether at least one individual generated a valid proposal.
        """

        selected_individuals_ids = list(selected_individuals_ids)

        modified_states = {}
        previous_states = {}

        log_f_dur = 0.0
        log_b_dur = 0.0

        moved_ids = []

        # --------------------------------------------------
        # 1. Propose the same block for all selected individuals
        # --------------------------------------------------
        for indiv_id in selected_individuals_ids:

            current_state = population_posterior[indiv_id]

            # Save current state for prior-density cancellation
            previous_states[indiv_id] = copy.deepcopy(current_state)

            # Propose on a copied state
            proposed_state = copy.deepcopy(current_state)

            log_q_f, log_q_b, moved = self._propose_new_state_duration_block(
                indiv_state=proposed_state,
                indiv_spec=indiv_spec_posterior,
                block_index=block_index,
                rng=rng,
                dur_local_frac=dur_local_frac,
                dur_p_far=dur_p_far,
            )

            # If no valid move was possible for this individual/block,
            # leave this individual unchanged for this block proposal.
            if not moved:
                continue

            log_f_dur += log_q_f
            log_b_dur += log_q_b

            modified_states[indiv_id] = proposed_state
            moved_ids.append(indiv_id)

        # Nothing moved in this block
        if len(moved_ids) == 0:
            return (
                population_posterior,
                time_indep_pop,
                id_to_indices,
                kernel_caches,
                logl_previous,
                0,
                0,
                False,
            )

        # --------------------------------------------------
        # 2. Prior terms for modified individuals only
        # --------------------------------------------------
        prior_proposed = sum(
            modified_states[indiv_id].log_p
            for indiv_id in moved_ids
        )

        prior_previous = sum(
            previous_states[indiv_id].log_p
            for indiv_id in moved_ids
        )

        # --------------------------------------------------
        # 3. Build proposed population
        # --------------------------------------------------
        population_proposed = population_posterior.copy()

        for indiv_id in moved_ids:
            population_proposed[indiv_id] = modified_states[indiv_id]

        # --------------------------------------------------
        # 4. Update dataframe only for moved individuals
        # --------------------------------------------------
        time_indep_pop_prop, new_id_to_indices = self._update_dataframe_subset(
            df_current=time_indep_pop,
            id_to_indices=id_to_indices,
            population_states=population_proposed,
            modified_ids=moved_ids,
        )

        # --------------------------------------------------
        # 5. Update likelihood caches only for moved individuals
        # --------------------------------------------------
        kernel_caches_prop = self._update_kernel_cache_subset(
            kernel_caches,
            time_indep_pop,
            time_indep_pop_prop,
            moved_ids,
            it,
        )

        logl_proposed = self._compute_loglik_from_cache(kernel_caches_prop)

        # --------------------------------------------------
        # 6. MH acceptance ratio
        # --------------------------------------------------
        log_alpha = (
            prior_proposed
            + logl_proposed
            + log_b_dur
            - prior_previous
            - logl_previous
            - log_f_dur
        )

        if np.isfinite(log_alpha) and (
            log_alpha >= 0.0
            or np.log(rng.uniform()) < log_alpha
        ):

            n_changed_map, _ = self._count_individuals_with_mapping_change(
                df_old=time_indep_pop,
                df_new=time_indep_pop_prop,
                modified_ids=moved_ids,
            )

            return (
                population_proposed,
                time_indep_pop_prop,
                new_id_to_indices,
                kernel_caches_prop,
                logl_proposed,
                1,
                n_changed_map,
                True,
            )

        # Rejection: keep everything unchanged
        return (
            population_posterior,
            time_indep_pop,
            id_to_indices,
            kernel_caches,
            logl_previous,
            0,
            0,
            True,
        )


    def sample_posterior_alive(
        self,
        id_to_traj,
        indiv_spec_posterior,
        observed_ids,
        observed_kernels,
        n_MH,
        n_prop_function,
        n_prop_min,
        n_gibbs_indicators=1500,
        parallel_posterior_init=1,
        n_jobs_posterior_init=8,
        seed=0,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        dur_local_frac=None,
        dur_p_far=None,
    ):
        
        rng = np.random.default_rng(seed)

        if parallel_posterior_init == 1:
            population_posterior = self._initialize_posterior_population_parallel(
                observed_ids=observed_ids,
                id_to_traj=id_to_traj,
                observed_kernels=observed_kernels,
                n_gibbs_indicators=n_gibbs_indicators,
                x0=x0,
                feas_tol=feas_tol,
                eq_tol=eq_tol,
                fixed_row_tol=fixed_row_tol,
                seed=seed,
                n_jobs=n_jobs_posterior_init,
            )

        elif parallel_posterior_init == 0:
            population_posterior = self._initialize_posterior_population(
                observed_ids=observed_ids,
                id_to_traj=id_to_traj,
                observed_kernels=observed_kernels,
                indiv_spec=indiv_spec_posterior,
                n_gibbs_indicators=n_gibbs_indicators,
                x0=x0,
                feas_tol=feas_tol,
                eq_tol=eq_tol,
                fixed_row_tol=fixed_row_tol,
                seed=seed,
            )

        else:
            raise ValueError("Invalid value for parallel_posterior_init.")

        # -------------------------
        # Check duration proposal settings
        # -------------------------
        use_classical_hr = (dur_local_frac is None) and (dur_p_far is None)
        use_close_far_hr = (dur_local_frac is not None) and (dur_p_far is not None)

        if not use_classical_hr and not use_close_far_hr:
            raise ValueError(
                "Either provide both dur_local_frac and dur_p_far, "
                "or leave both as None for classical hit-and-run."
            )

        if use_close_far_hr:
            if not (0.0 < dur_local_frac <= 1.0):
                raise ValueError("dur_local_frac must satisfy 0 < dur_local_frac <= 1.")

            if not (0.0 <= dur_p_far <= 1.0):
                raise ValueError("dur_p_far must satisfy 0 <= dur_p_far <= 1.")

        # ---------------- MAIN LOOP ----------------          
        accept_rate_dur = []
        accept_rate_attr = []

        n_changed_dur = 0

        time_indep_pop, id_to_indices = self._population_to_dataframe_with_index(
            population_posterior
        )

        kernel_caches = self._initialize_kernel_cache(time_indep_pop)
        logl_previous = self._compute_loglik_from_cache(kernel_caches)

        for it in range(n_MH):
            if it == 1 or it % 500 == 0:
                print(
                    it,
                    np.mean(accept_rate_attr) if len(accept_rate_attr) > 0 else np.nan,
                    np.mean(accept_rate_dur) if len(accept_rate_dur) > 0 else np.nan,
                    n_changed_dur,
                )

            n_prop_ = max(n_prop_min, int(n_prop_function(it)))

            ids = list(population_posterior.keys())

            selected_individuals_ids = rng.choice(
                ids,
                size=min(n_prop_, len(ids)),
                replace=False
            )

            ###################################
            # -------- ATTRIBUTES -------------
            ###################################
            modified_states = {}
            previous_states = {}

            log_f = 0.0
            log_b = 0.0

            for indiv_id in selected_individuals_ids:

                current_state = population_posterior[indiv_id]

                # Gibbs attributes are always accepted
                self._propose_new_state_gibbs_attr(
                    current_state,
                    indiv_spec_posterior,
                    rng=rng
                )

                previous_states[indiv_id] = copy.deepcopy(current_state)

                proposed_state = copy.deepcopy(current_state)

                log_q_f, log_q_b = self._propose_new_state_mh_attr(
                    proposed_state,
                    indiv_spec_posterior,
                    rng=rng
                )

                log_f += log_q_f
                log_b += log_q_b

                modified_states[indiv_id] = proposed_state

            prior_proposed = sum(s.log_p for s in modified_states.values())
            prior_previous = sum(s.log_p for s in previous_states.values())

            population_proposed = population_posterior.copy()

            for indiv_id in selected_individuals_ids:
                population_proposed[indiv_id] = modified_states[indiv_id]

            time_indep_pop_prop, new_id_to_indices = self._update_dataframe_subset(
                df_current=time_indep_pop,
                id_to_indices=id_to_indices,
                population_states=population_proposed,
                modified_ids=selected_individuals_ids
            )

            kernel_caches_prop = self._update_kernel_cache_subset(
                kernel_caches,
                time_indep_pop,
                time_indep_pop_prop,
                selected_individuals_ids,
                it
            )

            logl_proposed = self._compute_loglik_from_cache(kernel_caches_prop)

            log_alpha = (
                prior_proposed + logl_proposed + log_b
                - prior_previous - logl_previous - log_f
            )

            if (log_alpha >= 0.0) or (np.log(rng.uniform()) < log_alpha):
                accept_rate_attr.append(1)

                population_posterior = population_proposed
                time_indep_pop = time_indep_pop_prop
                id_to_indices = new_id_to_indices
                logl_previous = logl_proposed
                kernel_caches = kernel_caches_prop
            else:
                accept_rate_attr.append(0)

            ##################################
            # -------- DURATIONS BY BLOCK ----
            ##################################

            # Number of blocks is the same for all individuals because it is
            # defined by the model specification.
            first_id = selected_individuals_ids[0]
            n_blocks = len(population_posterior[first_id].block_data)

            # Randomize block order to avoid a systematic scan effect.
            block_order = rng.permutation(n_blocks)

            for block_index in block_order:

                (
                    population_posterior,
                    time_indep_pop,
                    id_to_indices,
                    kernel_caches,
                    logl_previous,
                    accepted_block,
                    n_changed_block,
                    moved_block,
                ) = self._duration_block_population_step(
                    population_posterior=population_posterior,
                    time_indep_pop=time_indep_pop,
                    id_to_indices=id_to_indices,
                    kernel_caches=kernel_caches,
                    logl_previous=logl_previous,
                    selected_individuals_ids=selected_individuals_ids,
                    indiv_spec_posterior=indiv_spec_posterior,
                    block_index=int(block_index),
                    rng=rng,
                    it=it,
                    dur_local_frac=dur_local_frac,
                    dur_p_far=dur_p_far,
                )

                # Store an acceptance indicator only if at least one proposal
                # was actually generated for this block.
                if moved_block:
                    accept_rate_dur.append(accepted_block)
                    n_changed_dur += n_changed_block


        list_individuals_posterior = []

        for indiv_id, state in population_posterior.items():

            existence_trajectory = id_to_traj[indiv_id]

            trajs = self._build_trajectories_instance_inside(
                state.instance,
                existence_trajectory
            )

            list_individuals_posterior.append(
                Individual(indiv_id, trajs)
            )

        final_pop = list_individuals_posterior

        return (
            Population(final_pop),
            accept_rate_dur,
            accept_rate_attr,
            n_changed_dur
        )

    def update_pop(
        self,
        df_existence,
        n_gibbs_indicators,
        n_prop_function,
        n_prop_min,
        n_MH,
        parallel_prior=1,
        parallel_posterior_init=1,
        n_jobs_prior=8,
        n_jobs_posterior_init=8,
        seed=0,
        n_samples=1,
        burnin=5000,
        thin=1,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        output_dir_unalive="posterior_update_unalive",
        dur_local_frac=None,
        dur_p_far=None,
    ):
        """
        MAIN MCMC ALGORITHM

        Structure
        ---------
        1. Split population:
            - observed individuals -> posterior
            - unobserved individuals -> prior sampling

        2. Initialize:
            - individual states
            - dataframe representation
            - kernel caches

        3. MCMC loop:
            For each iteration:
                a) Propose attribute updates
                - Gibbs part
                - MH part
                b) Accept/reject attribute proposal
                c) Propose duration updates
                - classical hit-and-run if dur_local_frac=None and dur_p_far=None
                - close/far hit-and-run if both are provided
                d) Accept/reject duration proposal

        4. Rebuild final population

        Duration proposal parameters
        ----------------------------
        dur_local_frac : float or None
            If None, classical hit-and-run is used.

            If provided, it defines the close region as:

                |tau| <= dur_local_frac * (t_max - t_min)

        dur_p_far : float or None
            If None, classical hit-and-run is used.

            If provided, it gives the base probability of selecting the far region.
            Must be provided together with dur_local_frac.
        """

        use_classical_hr = (dur_local_frac is None) and (dur_p_far is None)
        use_close_far_hr = (dur_local_frac is not None) and (dur_p_far is not None)

        if not use_classical_hr and not use_close_far_hr:
            raise ValueError(
                "Either provide both dur_local_frac and dur_p_far, "
                "or leave both as None for classical hit-and-run."
            )

        if use_close_far_hr:
            if not (0.0 < dur_local_frac <= 1.0):
                raise ValueError("dur_local_frac must satisfy 0 < dur_local_frac <= 1.")

            if not (0.0 <= dur_p_far <= 1.0):
                raise ValueError("dur_p_far must satisfy 0 <= dur_p_far <= 1.")

        observed_ids, never_observed_ids, observed_kernels = \
            self._get_observed_vs_never_ids(df_existence)

        observed_ids = sorted(observed_ids)

        ex_dim_spec = self.pop_spec.list_dim_spec[0]
        id_to_traj = dfex_to_trajex(ex_dim_spec, df_existence)

        indiv_spec_posterior = self._make_indiv_spec(
            indiv_id=None,
            bool_posterior=1
        )

        self.sample_prior_indices(
            never_observed_ids,
            id_to_traj,
            n_gibbs_indicators,
            parallel_prior,
            n_jobs_prior,
            seed,
            n_samples,
            burnin,
            thin,
            output_dir_unalive,
            x0
        )

        return self.sample_posterior_alive(
            id_to_traj=id_to_traj,
            indiv_spec_posterior=indiv_spec_posterior,
            observed_ids=observed_ids,
            observed_kernels=observed_kernels,
            n_MH=n_MH,
            n_prop_function=n_prop_function,
            n_prop_min=n_prop_min,
            n_gibbs_indicators=n_gibbs_indicators,
            parallel_posterior_init=parallel_posterior_init,
            n_jobs_posterior_init=n_jobs_posterior_init,
            seed=seed,
            x0=x0,
            feas_tol=feas_tol,
            eq_tol=eq_tol,
            fixed_row_tol=fixed_row_tol,
            dur_local_frac=dur_local_frac,
            dur_p_far=dur_p_far,
        )
    

    def _save_posterior_checkpoint(
        self,
        checkpoint_path,
        it,
        population_posterior,
        time_indep_pop,
        id_to_indices,
        kernel_caches,
        logl_previous,
        accept_rate_dur,
        accept_rate_attr,
        n_changed_dur,
        rng,
    ):
        """
        Save the current posterior MCMC state.

        This checkpoint is sufficient to restart the posterior MCMC loop
        without reinitializing the posterior population.

        The save is atomic:
        - write to a temporary file
        - replace the checkpoint file only when writing succeeds
        """

        checkpoint = {
            "it": it,
            "population_posterior": population_posterior,
            "time_indep_pop": time_indep_pop,
            "id_to_indices": id_to_indices,
            "kernel_caches": kernel_caches,
            "logl_previous": logl_previous,
            "accept_rate_dur": accept_rate_dur,
            "accept_rate_attr": accept_rate_attr,
            "n_changed_dur": n_changed_dur,
            "rng_state": rng.bit_generator.state,
        }

        checkpoint_dir = os.path.dirname(checkpoint_path)

        if checkpoint_dir != "":
            os.makedirs(checkpoint_dir, exist_ok=True)

        tmp_dir = checkpoint_dir if checkpoint_dir != "" else "."

        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=tmp_dir,
        ) as tmp:
            pickle.dump(checkpoint, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path = tmp.name

        os.replace(tmp_path, checkpoint_path)


    def _load_posterior_checkpoint(
        self,
        checkpoint_path,
        rng,
    ):
        """
        Load a posterior MCMC checkpoint and restore the RNG state.
        """

        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        rng.bit_generator.state = checkpoint["rng_state"]

        return checkpoint
    
    def sample_posterior_alive_checkpointed(
        self,
        id_to_traj,
        indiv_spec_posterior,
        observed_ids,
        observed_kernels,
        n_MH,
        n_prop_function,
        n_prop_min,
        n_gibbs_indicators=1500,
        parallel_posterior_init=1,
        n_jobs_posterior_init=8,
        seed=0,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        dur_local_frac=None,
        dur_p_far=None,
        checkpoint_path=None,
        checkpoint_every=500,
        resume=False,
    ):
        """
        Same logic as sample_posterior_alive, but with checkpointing.

        This function does not replace the original sample_posterior_alive.
        It is a separate function.

        Parameters
        ----------
        checkpoint_path : str or None
            Path to the checkpoint file.

        checkpoint_every : int
            Save checkpoint every checkpoint_every iterations.

        resume : bool
            If True, resume from checkpoint_path when it exists.
        """

        rng = np.random.default_rng(seed)

        # -------------------------
        # Check duration proposal settings
        # -------------------------
        use_classical_hr = (dur_local_frac is None) and (dur_p_far is None)
        use_close_far_hr = (dur_local_frac is not None) and (dur_p_far is not None)

        if not use_classical_hr and not use_close_far_hr:
            raise ValueError(
                "Either provide both dur_local_frac and dur_p_far, "
                "or leave both as None for classical hit-and-run."
            )

        if use_close_far_hr:
            if not (0.0 < dur_local_frac <= 1.0):
                raise ValueError("dur_local_frac must satisfy 0 < dur_local_frac <= 1.")

            if not (0.0 <= dur_p_far <= 1.0):
                raise ValueError("dur_p_far must satisfy 0 <= dur_p_far <= 1.")

        # -------------------------
        # Resume from checkpoint
        # -------------------------
        if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):

            checkpoint = self._load_posterior_checkpoint(
                checkpoint_path=checkpoint_path,
                rng=rng,
            )

            start_it = checkpoint["it"] + 1

            population_posterior = checkpoint["population_posterior"]
            time_indep_pop = checkpoint["time_indep_pop"]
            id_to_indices = checkpoint["id_to_indices"]
            kernel_caches = checkpoint["kernel_caches"]
            logl_previous = checkpoint["logl_previous"]
            accept_rate_dur = checkpoint["accept_rate_dur"]
            accept_rate_attr = checkpoint["accept_rate_attr"]
            n_changed_dur = checkpoint["n_changed_dur"]

            print(f"Resuming posterior MCMC from iteration {checkpoint['it']}.")

        # -------------------------
        # Fresh initialization
        # -------------------------
        else:

            start_it = 0

            if parallel_posterior_init == 1:
                population_posterior = self._initialize_posterior_population_parallel(
                    observed_ids=observed_ids,
                    id_to_traj=id_to_traj,
                    observed_kernels=observed_kernels,
                    n_gibbs_indicators=n_gibbs_indicators,
                    x0=x0,
                    feas_tol=feas_tol,
                    eq_tol=eq_tol,
                    fixed_row_tol=fixed_row_tol,
                    seed=seed,
                    n_jobs=n_jobs_posterior_init,
                )

            elif parallel_posterior_init == 0:
                population_posterior = self._initialize_posterior_population(
                    observed_ids=observed_ids,
                    id_to_traj=id_to_traj,
                    observed_kernels=observed_kernels,
                    indiv_spec=indiv_spec_posterior,
                    n_gibbs_indicators=n_gibbs_indicators,
                    x0=x0,
                    feas_tol=feas_tol,
                    eq_tol=eq_tol,
                    fixed_row_tol=fixed_row_tol,
                    seed=seed,
                )

            else:
                raise ValueError("Invalid value for parallel_posterior_init.")

            accept_rate_dur = []
            accept_rate_attr = []
            n_changed_dur = 0

            time_indep_pop, id_to_indices = self._population_to_dataframe_with_index(
                population_posterior
            )

            kernel_caches = self._initialize_kernel_cache(time_indep_pop)
            logl_previous = self._compute_loglik_from_cache(kernel_caches)

        # ==================================================
        # Main MCMC loop
        # ==================================================
        for it in range(start_it, n_MH):

            if it == 1 or it % 500 == 0:
                print(
                    it,
                    np.mean(accept_rate_attr) if len(accept_rate_attr) > 0 else np.nan,
                    np.mean(accept_rate_dur) if len(accept_rate_dur) > 0 else np.nan,
                    n_changed_dur,
                )

            n_prop_ = max(n_prop_min, int(n_prop_function(it)))

            ids = list(population_posterior.keys())

            selected_individuals_ids = rng.choice(
                ids,
                size=min(n_prop_, len(ids)),
                replace=False,
            )

            # ==================================================
            # Attributes
            # ==================================================
            modified_states = {}
            previous_states = {}

            log_f = 0.0
            log_b = 0.0

            for indiv_id in selected_individuals_ids:

                current_state = population_posterior[indiv_id]

                self._propose_new_state_gibbs_attr(
                    current_state,
                    indiv_spec_posterior,
                    rng=rng,
                )

                previous_states[indiv_id] = copy.deepcopy(current_state)

                proposed_state = copy.deepcopy(current_state)

                log_q_f, log_q_b = self._propose_new_state_mh_attr(
                    proposed_state,
                    indiv_spec_posterior,
                    rng=rng,
                )

                log_f += log_q_f
                log_b += log_q_b

                modified_states[indiv_id] = proposed_state

            prior_proposed = sum(s.log_p for s in modified_states.values())
            prior_previous = sum(s.log_p for s in previous_states.values())

            population_proposed = population_posterior.copy()

            for indiv_id in selected_individuals_ids:
                population_proposed[indiv_id] = modified_states[indiv_id]

            time_indep_pop_prop, new_id_to_indices = self._update_dataframe_subset(
                df_current=time_indep_pop,
                id_to_indices=id_to_indices,
                population_states=population_proposed,
                modified_ids=selected_individuals_ids,
            )

            kernel_caches_prop = self._update_kernel_cache_subset(
                kernel_caches,
                time_indep_pop,
                time_indep_pop_prop,
                selected_individuals_ids,
                it,
            )

            logl_proposed = self._compute_loglik_from_cache(kernel_caches_prop)

            log_alpha = (
                prior_proposed + logl_proposed + log_b
                - prior_previous - logl_previous - log_f
            )

            if (log_alpha >= 0.0) or (np.log(rng.uniform()) < log_alpha):

                accept_rate_attr.append(1)

                population_posterior = population_proposed
                time_indep_pop = time_indep_pop_prop
                id_to_indices = new_id_to_indices
                logl_previous = logl_proposed
                kernel_caches = kernel_caches_prop

            else:
                accept_rate_attr.append(0)

            # ==================================================
            # Durations by block
            # ==================================================

            first_id = selected_individuals_ids[0]
            n_blocks = len(population_posterior[first_id].block_data)

            block_order = rng.permutation(n_blocks)

            for block_index in block_order:

                (
                    population_posterior,
                    time_indep_pop,
                    id_to_indices,
                    kernel_caches,
                    logl_previous,
                    accepted_block,
                    n_changed_block,
                    moved_block,
                ) = self._duration_block_population_step(
                    population_posterior=population_posterior,
                    time_indep_pop=time_indep_pop,
                    id_to_indices=id_to_indices,
                    kernel_caches=kernel_caches,
                    logl_previous=logl_previous,
                    selected_individuals_ids=selected_individuals_ids,
                    indiv_spec_posterior=indiv_spec_posterior,
                    block_index=int(block_index),
                    rng=rng,
                    it=it,
                    dur_local_frac=dur_local_frac,
                    dur_p_far=dur_p_far,
                )

                if moved_block:
                    accept_rate_dur.append(accepted_block)
                    n_changed_dur += n_changed_block

            # ==================================================
            # Save checkpoint
            # ==================================================
            if (
                checkpoint_path is not None
                and checkpoint_every is not None
                and checkpoint_every > 0
                and ((it + 1) % checkpoint_every == 0)
            ):
                self._save_posterior_checkpoint(
                    checkpoint_path=checkpoint_path,
                    it=it,
                    population_posterior=population_posterior,
                    time_indep_pop=time_indep_pop,
                    id_to_indices=id_to_indices,
                    kernel_caches=kernel_caches,
                    logl_previous=logl_previous,
                    accept_rate_dur=accept_rate_dur,
                    accept_rate_attr=accept_rate_attr,
                    n_changed_dur=n_changed_dur,
                    rng=rng,
                )

                print(f"Checkpoint saved at iteration {it}.")

        # ==================================================
        # Save final checkpoint
        # ==================================================
        if checkpoint_path is not None:
            self._save_posterior_checkpoint(
                checkpoint_path=checkpoint_path,
                it=n_MH - 1,
                population_posterior=population_posterior,
                time_indep_pop=time_indep_pop,
                id_to_indices=id_to_indices,
                kernel_caches=kernel_caches,
                logl_previous=logl_previous,
                accept_rate_dur=accept_rate_dur,
                accept_rate_attr=accept_rate_attr,
                n_changed_dur=n_changed_dur,
                rng=rng,
            )

        # ==================================================
        # Rebuild final population
        # ==================================================
        list_individuals_posterior = []

        for indiv_id, state in population_posterior.items():

            existence_trajectory = id_to_traj[indiv_id]

            trajs = self._build_trajectories_instance_inside(
                state.instance,
                existence_trajectory,
            )

            list_individuals_posterior.append(
                Individual(indiv_id, trajs)
            )

        final_pop = list_individuals_posterior

        return (
            Population(final_pop),
            accept_rate_dur,
            accept_rate_attr,
            n_changed_dur,
        )
    
    def update_pop_checkpointed(
        self,
        df_existence,
        n_gibbs_indicators,
        n_prop_function,
        n_prop_min,
        n_MH,
        parallel_prior=1,
        parallel_posterior_init=1,
        n_jobs_prior=8,
        n_jobs_posterior_init=8,
        seed=0,
        n_samples=1,
        burnin=5000,
        thin=1,
        x0=None,
        feas_tol=1e-10,
        eq_tol=1e-12,
        fixed_row_tol=1e-14,
        output_dir_unalive="posterior_update_unalive",
        dur_local_frac=None,
        dur_p_far=None,
        checkpoint_path="checkpoints/posterior_alive.pkl",
        checkpoint_every=500,
        resume=False,
    ):
        """
        Same role as update_pop, but calls sample_posterior_alive_checkpointed.

        This function does not replace update_pop.
        It is a separate checkpointed version.
        """

        use_classical_hr = (dur_local_frac is None) and (dur_p_far is None)
        use_close_far_hr = (dur_local_frac is not None) and (dur_p_far is not None)

        if not use_classical_hr and not use_close_far_hr:
            raise ValueError(
                "Either provide both dur_local_frac and dur_p_far, "
                "or leave both as None for classical hit-and-run."
            )

        if use_close_far_hr:
            if not (0.0 < dur_local_frac <= 1.0):
                raise ValueError("dur_local_frac must satisfy 0 < dur_local_frac <= 1.")

            if not (0.0 <= dur_p_far <= 1.0):
                raise ValueError("dur_p_far must satisfy 0 <= dur_p_far <= 1.")

        observed_ids, never_observed_ids, observed_kernels = (
            self._get_observed_vs_never_ids(df_existence)
        )

        observed_ids = sorted(observed_ids)

        ex_dim_spec = self.pop_spec.list_dim_spec[0]
        id_to_traj = dfex_to_trajex(ex_dim_spec, df_existence)

        indiv_spec_posterior = self._make_indiv_spec(
            indiv_id=None,
            bool_posterior=1,
        )

        self.sample_prior_indices(
            never_observed_ids,
            id_to_traj,
            n_gibbs_indicators,
            parallel_prior,
            n_jobs_prior,
            seed,
            n_samples,
            burnin,
            thin,
            output_dir_unalive,
            x0,
        )

        return self.sample_posterior_alive_checkpointed(
            id_to_traj=id_to_traj,
            indiv_spec_posterior=indiv_spec_posterior,
            observed_ids=observed_ids,
            observed_kernels=observed_kernels,
            n_MH=n_MH,
            n_prop_function=n_prop_function,
            n_prop_min=n_prop_min,
            n_gibbs_indicators=n_gibbs_indicators,
            parallel_posterior_init=parallel_posterior_init,
            n_jobs_posterior_init=n_jobs_posterior_init,
            seed=seed,
            x0=x0,
            feas_tol=feas_tol,
            eq_tol=eq_tol,
            fixed_row_tol=fixed_row_tol,
            dur_local_frac=dur_local_frac,
            dur_p_far=dur_p_far,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )




def _sample_prior_individual_worker(args):
    (
        pop_spec,
        indiv_id,
        existence_trajectory,
        n_gibbs_indicators,
        n_samples,
        burnin,
        thin,
        x0,
        seed,
        output_dir
    ) = args
    
    filepath = os.path.join(output_dir, f"individual_{indiv_id}.csv")

    if os.path.exists(filepath):
        return indiv_id

    # Build a fresh spec inside the worker
    indiv_spec = IndividualSpec(
        indiv_id,
        pop_spec.list_dim_spec,
        pop_spec.joint_dist_duration,
        pop_spec.list_inter_constraint_spec,
    )

    # Existence instance
    birth = existence_trajectory.list_events[0]
    birth_name = birth.name
    existence_instance = {
        birth_name: {
            "main_duration": birth.main_duration,
            "main_start_date": birth.main_start_date,
            "gap_duration": birth.gap_duration,
            "gap_start_date": birth.gap_start_date,
            **({att.name: att.value for att in birth.attributes} if birth.attributes else {}),
        }
    }

    instance_res, *_ = all_dim_hit_and_run(
        indiv_spec,
        existence_instance,
        n_samples=n_samples,
        n_gibbs=n_gibbs_indicators,
        burnin=burnin,
        thin=thin,
        x0=x0,
        seed=seed,
    )

    # Rebuild trajectories
    trajs = [existence_trajectory]

    for dim in pop_spec.list_dim_spec:
        if dim.name == "Existence":
            continue

        events = []
        indic = []
        inst_dim = instance_res.get(dim.name, {})

        for ev in dim.list_event_spec:
            count = ev.max_count if ev.max_count > 1 else 1

            for m in range(count):
                name = ev.name if count == 1 else f"{ev.name}_{m+1}"
                ev_data = inst_dim.get(name)

                if ev_data is None:
                    continue

                indicator = ev_data.get("indicator", 0)
                if indicator != 1:
                    indic.append(0)
                    continue

                attributes = []
                if ev.attributes_spec:
                    for attr in ev.attributes_spec:
                        val = None
                        if attr.name in ev_data:
                            attr_data = ev_data[attr.name]
                            if isinstance(attr_data, dict) and "chains" in attr_data:
                                chains = attr_data["chains"]
                                if chains is not None and len(chains) > 0:
                                    val = chains[0]

                        attributes.append(
                            Attribute(attr.name, attr.type, val)
                        )

                main_start = ev_data["main_start_date"]["chains"][0]
                gap_start = ev_data["gap_start_date"]["chains"][0]
                main_dur = ev_data["main_duration"]["chains"][0]
                gap_dur = ev_data["gap_duration"]["chains"][0]

                events.append(
                    Event(
                        name,
                        ev.type,
                        main_start,
                        gap_start,
                        main_dur + gap_dur,
                        main_dur,
                        gap_dur,
                        attributes,
                    )
                )
                indic.append(1)

        trajs.append(Trajectory(dim.name, events, indic))

    indiv = Individual(indiv_id, trajs)

    df = indiv.to_dataframe()
    df.to_csv(filepath, index=False)

    return indiv

    
def _initialize_one_posterior_state_worker(args):
    (
        pop_spec,
        indiv_id,
        existence_trajectory,
        kernels_alive,
        n_gibbs_indicators,
        x0,
        feas_tol,
        eq_tol,
        fixed_row_tol,
        seed,
    ) = args

    rng = np.random.default_rng(seed)

    # Fresh spec inside the worker
    indiv_spec = IndividualSpec(
        indiv_id,
        pop_spec.list_dim_spec,
        pop_spec.joint_dist_duration,
        pop_spec.list_inter_constraint_spec,
    )

    # --------------------------------------------------
    # Existence instance
    # --------------------------------------------------
    birth = existence_trajectory.list_events[0]
    birth_name = birth.name

    existence_instance = {
        birth_name: {
            "main_duration": birth.main_duration,
            "main_start_date": birth.main_start_date,
            "gap_duration": birth.gap_duration,
            "gap_start_date": birth.gap_start_date,
            **(
                {att.name: att.value for att in birth.attributes}
                if birth.attributes
                else {}
            ),
        }
    }

    # --------------------------------------------------
    # Empty state
    # --------------------------------------------------
    state = IndividualState()
    state.instance = indiv_spec.create_empty_instance()
    state.instance["Existence"] = existence_instance
    state.kernel_indices = kernels_alive
    state.id = indiv_id

    # --------------------------------------------------
    # Initialize indicators and attributes
    # --------------------------------------------------
    for _ in range(n_gibbs_indicators):
        indiv_spec.exact_gibbs_update_indicators(
            state.instance,
            rng=rng,
        )

    indiv_spec.initialize_instance_attributes(
        state.instance,
        rng=rng,
    )

    # --------------------------------------------------
    # Build block-level duration geometry
    # --------------------------------------------------
    block_data = indiv_spec.build_all_block_constraint_matrices(
        state.instance
    )

    for block in block_data:

        if x0 is None:
            x0_block = None
        else:
            x0_full = np.asarray(x0, dtype=float)
            x0_block = x0_full[block["global_indices"]]

        A_block = np.asarray(block["A"], dtype=float)
        b_block = np.asarray(block["b"], dtype=float)

        if x0_block is None:

            vertices = compute_polytope_vertices(A_block, b_block)

            # x_block = dirichlet_sample_duration_from_vertices(
            #     vertices,
            #     np.ones(vertices.shape[0]),
            # )

            coefs = rng.dirichlet(np.ones(vertices.shape[0]))
            coefs = np.maximum(coefs, 1e-12)
            x_block = dirichlet_sample_duration_from_vertices(
                vertices,
                coefs
            )

        else:

            x_block = np.asarray(x0_block, dtype=float).copy()

            if not is_feasible(A_block, b_block, x_block, tol=feas_tol):
                raise ValueError(
                    f"x0 not feasible for individual {indiv_id}, "
                    f"{block['name']}."
                )

        block["x_current"] = x_block

        indiv_spec.update_instance_from_block_vector(
            x_block,
            state.instance,
            block,
        )

        # Equality / inequality decomposition
        A_eq, b_eq, keep_mask = extract_equalities_from_opposites(
            A_block,
            b_block,
            tol=eq_tol,
        )

        A_ineq = A_block[keep_mask]
        b_ineq = b_block[keep_mask]

        N = nullspace_basis(A_eq, tol=eq_tol)

        if N is None:
            N = np.eye(A_block.shape[1])

        row_norms = np.linalg.norm(N, axis=1)
        max_row = float(np.max(row_norms)) if row_norms.size > 0 else 0.0

        if max_row > 0:
            fixed_idx = np.where(row_norms <= fixed_row_tol * max_row)[0]
        else:
            fixed_idx = np.arange(A_block.shape[1])

        block["A_eq"] = A_eq
        block["b_eq"] = b_eq
        block["A_ineq"] = A_ineq
        block["b_ineq"] = b_ineq
        block["N"] = N
        block["fixed_idx"] = fixed_idx
        block["k"] = N.shape[1]

    state.block_data = block_data

    # --------------------------------------------------
    # Full individual prior log density
    # --------------------------------------------------
    logp_current = float(indiv_spec.joint_dist_duration(state.instance))

    if not np.isfinite(logp_current):
        logp_current = np.log(1e-300)
        print(
            f"Warning: initial log density not finite "
            f"for individual {indiv_id}."
        )

    state.log_p = logp_current

    return indiv_id, state

