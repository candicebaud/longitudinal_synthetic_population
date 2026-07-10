"""Update birth event using cross-sectional data : make it fast + more general

Baud Candice
Fri July 10 10:33:00 2026
"""

from spec_existence import *
import math

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class FastBirthState:
    dob: np.ndarray               # shape (N,)
    lifespan: np.ndarray          # shape (N,)
    gap_start: np.ndarray         # shape (N,)
    gap_duration: float           # assumed common scalar
    attrs: dict                   # attr_name -> np.ndarray shape (N,)
 

class BirthKernel:
    """
    Generic likelihood kernel for one cross-sectional dataset.

    Incremental version:
    - caches observed arrays once
    - caches one contribution vector per alive simulated individual
    - updates only touched individuals when the state changes

    IMPORTANT:
    This is valid if mapping(df, t) is row-local:
    one input row -> one output row with same id, independently of other rows.
    """

    def __init__(self, df_obs, t, f, mapping, weights=None, normalize_weights=False, sim_filter = None):
        """
        Parameters
        ----------
        df_obs : pandas DataFrame
            observed dataset (alive individuals only)
        t : float
            time at which the data is observed
        f : callable
            vectorized function f(obs_dict, sim_dict, it)
            returning matrix (n_obs, n_sim)
        mapping : function
            maps the time-independent dataframe into a time-dependent dataframe
        """
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
        self.obs_dict = {c: self.df_obs[c].to_numpy() for c in self.df_obs.columns}
        self.n_obs = len(df_obs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _alive_mask_from_time_dep(self, state, time_dep):
        """
        Reproduce the previous alive logic.
        """
        if "alive_status" in time_dep.columns:
            return time_dep["alive_status"].to_numpy() == "Alive"
        else:
            ids = time_dep["id"].to_numpy(dtype=int)
            return (state.dob[ids] <= self.t) & (self.t < state.gap_start[ids])

    def _validate_time_dep(self, time_dep):
        """
        Check that mapping returns one row per id.
        """
        if "id" not in time_dep.columns:
            raise ValueError("Mapped dataframe must contain an 'id' column.")

        ids = time_dep["id"].to_numpy(dtype=int)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                "Incremental kernel update assumes one mapped row per id. "
                "The mapping returned duplicated ids."
            )

    def _compute_contributions_for_time_dep(self, time_dep, it, state):
        """
        Given a mapped dataframe, return contributions for alive individuals.

        If self.sim_filter is provided, it is applied as an observation-specific
        admissibility mask.

        Returns
        -------
        alive_ids : np.ndarray
            ids of alive individuals in time_dep

        F : np.ndarray
            contribution matrix, shape (n_obs, n_alive_subset)

        M : np.ndarray
            admissibility mask, shape (n_obs, n_alive_subset)
        """
        self._validate_time_dep(time_dep)

        alive_mask = self._alive_mask_from_time_dep(state, time_dep)
        sim_alive = time_dep.loc[alive_mask]

        n_alive = len(sim_alive)
        if n_alive == 0:
            return (
                np.array([], dtype=int),
                np.zeros((self.n_obs, 0), dtype=float),
                np.zeros((self.n_obs, 0), dtype=bool),
            )

        sim_dict = {c: sim_alive[c].to_numpy() for c in sim_alive.columns}

        F = self.f(self.obs_dict, sim_dict, it)

        if F.shape != (self.n_obs, n_alive):
            raise ValueError(
                f"Kernel returned wrong shape {F.shape}, expected {(self.n_obs, n_alive)}"
            )

        if self.sim_filter is None:
            M = np.ones((self.n_obs, n_alive), dtype=bool)
        else:
            M = self.sim_filter(self.obs_dict, sim_dict, it)

            if M.shape != (self.n_obs, n_alive):
                raise ValueError(
                    f"sim_filter returned wrong shape {M.shape}, expected {(self.n_obs, n_alive)}"
                )

            M = M.astype(bool)

        # Remove non-admissible contributions
        F = F * M

        alive_ids = sim_alive["id"].to_numpy(dtype=int)

        return alive_ids, F, M

    # ------------------------------------------------------------------
    # Full initialization of the cache
    # ------------------------------------------------------------------
    def initialize_cache(self, state, time_indep_pop, it):
        """
        Build the kernel cache from scratch.

        Cache structure
        ---------------
        {
            "contrib_by_id": dict[id -> vector length n_obs],
            "mask_by_id": dict[id -> boolean vector length n_obs],
            "sum_contrib": np.ndarray shape (n_obs,),
            "sum_mask": np.ndarray shape (n_obs,),
        }
        """
        time_dep = self.mapping(time_indep_pop, self.t)
        alive_ids, F, M = self._compute_contributions_for_time_dep(time_dep, it, state)

        contrib_by_id = {}
        mask_by_id = {}

        for k, i in enumerate(alive_ids):
            contrib_by_id[int(i)] = F[:, k].copy()
            mask_by_id[int(i)] = M[:, k].copy()

        if F.shape[1] == 0:
            sum_contrib = np.zeros(self.n_obs, dtype=float)
            sum_mask = np.zeros(self.n_obs, dtype=float)
        else:
            sum_contrib = F.sum(axis=1)
            sum_mask = M.sum(axis=1).astype(float)

        return {
            "contrib_by_id": contrib_by_id,
            "mask_by_id": mask_by_id,
            "sum_contrib": sum_contrib,
            "sum_mask": sum_mask,
        }

    # ------------------------------------------------------------------
    # Loglik from cache
    # ------------------------------------------------------------------
    def loglik_from_cache(self, cache):
        """
        Compute log-likelihood from the cache.

        For censored / restricted observations, the denominator is observation-specific.
        """
        sum_contrib = cache["sum_contrib"]
        sum_mask = cache["sum_mask"]

        if np.any(sum_mask <= 0):
            return -np.inf

        density_vals = sum_contrib / sum_mask

        if np.any(density_vals <= 0) or np.any(~np.isfinite(density_vals)):
            return -np.inf

        return float(np.sum(self.weights * np.log(density_vals)))

    # ------------------------------------------------------------------
    # Incremental update of the cache
    # ------------------------------------------------------------------
    def update_cache_from_touched(self, state, time_indep_pop, touched, current_cache, it):
        """
        Return a NEW cache obtained by updating only the touched individuals.
        """
        if not touched:
            return {
                "contrib_by_id": current_cache["contrib_by_id"].copy(),
                "mask_by_id": current_cache["mask_by_id"].copy(),
                "sum_contrib": current_cache["sum_contrib"].copy(),
                "sum_mask": current_cache["sum_mask"].copy(),
            }

        contrib_by_id = current_cache["contrib_by_id"].copy()
        mask_by_id = current_cache["mask_by_id"].copy()

        sum_contrib = current_cache["sum_contrib"].copy()
        sum_mask = current_cache["sum_mask"].copy()

        # --------------------------------------------------------------
        # 1. Remove old contributions for touched ids
        # --------------------------------------------------------------
        for i in touched:
            i = int(i)

            old_vec = contrib_by_id.pop(i, None)
            old_mask = mask_by_id.pop(i, None)

            if old_vec is not None:
                sum_contrib -= old_vec

            if old_mask is not None:
                sum_mask -= old_mask.astype(float)

        # --------------------------------------------------------------
        # 2. Recompute mapped rows only for touched ids
        # --------------------------------------------------------------
        df_touched = time_indep_pop.iloc[np.asarray(touched, dtype=int)].copy()
        time_dep_touched = self.mapping(df_touched, self.t)

        alive_ids_new, F_new, M_new = self._compute_contributions_for_time_dep(
            time_dep_touched, it, state
        )

        # --------------------------------------------------------------
        # 3. Add new contributions
        # --------------------------------------------------------------
        for k, i in enumerate(alive_ids_new):
            vec = F_new[:, k].copy()
            mask_vec = M_new[:, k].copy()

            contrib_by_id[int(i)] = vec
            mask_by_id[int(i)] = mask_vec

            sum_contrib += vec
            sum_mask += mask_vec.astype(float)

        return {
            "contrib_by_id": contrib_by_id,
            "mask_by_id": mask_by_id,
            "sum_contrib": sum_contrib,
            "sum_mask": sum_mask,
        }
    # # ------------------------------------------------------------------
    # # Old fallback API (still usable if needed)
    # # ------------------------------------------------------------------
    # def loglik(self, state, time_indep_pop, it):
    #     """
    #     Full recomputation fallback.
    #     """
    #     cache = self.initialize_cache(state, time_indep_pop, it)
    #     return self.loglik_from_cache(cache)



class BirthPopUpdater:    
    def __init__(self, birth_spec, pop_size, target_size_t, kernels):
        self.birth_spec = birth_spec
        self.pop_size = pop_size
        self.target_size_t = target_size_t
        self.kernels = kernels

    # ------------------------------------------------------------------
    # Helpers to extract values from the Gibbs output
    # ------------------------------------------------------------------
    def _extract_vector(self, x, n_expected=None, dtype=float):
        """
        Robust extraction of a length-N vector from several possible formats:
        - array/list
        - dict with key 'chains'
        - scalar (broadcasted if n_expected is given)
        """
        if isinstance(x, dict) and "chains" in x:
            arr = np.asarray(x["chains"], dtype=dtype)
        else:
            arr = np.asarray(x, dtype=dtype)

        if arr.ndim == 0:
            if n_expected is None:
                return np.asarray([arr.item()], dtype=dtype)
            return np.full(n_expected, arr.item(), dtype=dtype)

        return arr

    def _extract_scalar(self, x, default_cast=float):
        """
        Robust extraction of a scalar from several possible formats.
        """
        if isinstance(x, dict) and "chains" in x:
            arr = np.asarray(x["chains"])
            if arr.ndim == 0:
                return default_cast(arr.item())
            if arr.size == 1:
                return default_cast(arr.ravel()[0])
            raise ValueError("Expected a scalar, got a vector with several values.")
        arr = np.asarray(x)
        if arr.ndim == 0:
            return default_cast(arr.item())
        if arr.size == 1:
            return default_cast(arr.ravel()[0])
        raise ValueError("Expected a scalar, got a vector with several values.")

    # ------------------------------------------------------------------
    # Fast internal initialization
    # ------------------------------------------------------------------
    def initialize_fast_state(self, x0 = None, burnin = 1000, seed = 0, thin = 1):
        n_draws = self.pop_size
        res_sample = self.birth_spec.sample(
            n_samples = n_draws, burnin = burnin, seed = seed, x0 = x0, thin = thin
        )

        ev_name = self.birth_spec.name
        instance_res = res_sample["instance"]
        ev_data = instance_res[ev_name]

        dob = self._extract_vector(ev_data["main_start_date"], n_expected=n_draws, dtype=float)
        lifespan = self._extract_vector(ev_data["main_duration"], n_expected=n_draws, dtype=float)

        # gap_duration is often a scalar in your code
        gap_duration = self._extract_scalar(ev_data["gap_duration"], default_cast=float)
        gap_start = dob + lifespan

        attrs = {}
        if self.birth_spec.attributes_spec is not None:
            for attr_spec in self.birth_spec.attributes_spec:
                raw_attr = ev_data[attr_spec.name]
                # attributes can be numeric or objects/categories -> use dtype=object by default
                vals = self._extract_vector(raw_attr, n_expected=n_draws, dtype=object)
                attrs[attr_spec.name] = vals.copy()

        state = FastBirthState(
            dob=dob.copy(),
            lifespan=lifespan.copy(),
            gap_start=gap_start.copy(),
            gap_duration=gap_duration,
            attrs=attrs,
        )

        return {
            "state": state,
            "instance": instance_res,
            "A": res_sample["A"],
            "b": res_sample["b"],
        } 
   

    # ------------------------------------------------------------------
    # Optional conversion back to your original Population objects
    # ------------------------------------------------------------------
    def fast_state_to_population(self, state):
        indivs = []
        ev_name = self.birth_spec.name

        for i in range(state.dob.shape[0]):
            attr = []
            if self.birth_spec.attributes_spec is not None:
                for attr_spec in self.birth_spec.attributes_spec:
                    attr.append(Attribute(attr_spec.name, attr_spec.type, state.attrs[attr_spec.name][i]))

            ev = Event(
                ev_name,
                "Event",
                float(state.dob[i]),
                float(state.gap_start[i]),
                float(state.lifespan[i] + state.gap_duration),
                float(state.lifespan[i]),
                float(state.gap_duration),
                attr,
            )
            existence_traj = Trajectory("Existence", [ev], [1])
            indivs.append(Individual(i, [existence_traj]))

        return Population(indivs)

    # Keep old initializer if you still need it externally
    def initialize_attr(self, x0=None, burnin=1000, seed=0, thin = 1):
        out = self.initialize_fast_state(x0=x0, burnin=burnin, seed=seed, thin = thin)
        pop = self.fast_state_to_population(out["state"])
        return {
            "population": pop,
            "instance": out["instance"],
            "A": out["A"],
            "b": out["b"],
        }

    # ------------------------------------------------------------------
    # Dictionary view for ONE individual only
    # This keeps birth_spec.log_density(instance) unchanged
    # ------------------------------------------------------------------
    def individual_instance_from_state(self, state, i):
        birth = {
            "indicator": 1,
            "main_start_date": float(state.dob[i]),
            "gap_start_date": float(state.gap_start[i]),
            "main_duration": float(state.lifespan[i]),
            "gap_duration": float(state.gap_duration),
        }

        if self.birth_spec.attributes_spec is not None:
            for attr_spec in self.birth_spec.attributes_spec:
                birth[attr_spec.name] = state.attrs[attr_spec.name][i]

        return {self.birth_spec.name: birth}

    # ------------------------------------------------------------------
    # DataFrame passed to mapping(...)
    # Adjust here if your mapping expects very specific column names
    # ------------------------------------------------------------------
    def _build_mapping_dataframe(self, state):
        """
        Fast replacement for Population.to_dataframe() when the state only contains
        the realized birth/existence event for each individual.

        It mimics the structure of Individual.to_dataframe():
            id, dim_name, event_name, event_type,
            main_start_date, gap_start_date, duration, main_duration, gap_duration,
            <attribute columns...>

        One row is created per individual, since in the fast state each individual
        has exactly one active realized event.
        """
        n = state.dob.shape[0]

        # Attribute column names in a stable order
        if self.birth_spec.attributes_spec is not None:
            attribute_names = [attr_spec.name for attr_spec in self.birth_spec.attributes_spec]
        else:
            attribute_names = []

        data = {
            "id": np.arange(n, dtype=int),
            "dim_name": np.full(n, "Existence", dtype=object),
            "event_name": np.full(n, self.birth_spec.name, dtype=object),
            "event_type": np.full(n, "Event", dtype=object),
            "main_start_date": state.dob,
            "gap_start_date": state.gap_start,
            "duration": state.lifespan + state.gap_duration,
            "main_duration": state.lifespan,
            "gap_duration": np.full(n, state.gap_duration, dtype=float),
        }

        # Attributes
        for attr_name in attribute_names:
            vals = state.attrs[attr_name]

            # Convert None -> np.nan (vectorized, no Python loop)
            col = np.array(vals, dtype=object)
            mask_none = col == None
            if mask_none.any():
                col[mask_none] = np.nan

            data[attr_name] = col

        # Keep column order consistent
        col_names = [
            "id", "dim_name", "event_name", "event_type",
            "main_start_date", "gap_start_date", "duration",
            "main_duration", "gap_duration"
        ] + attribute_names

        return pd.DataFrame(data, columns=col_names)
    
    def _update_mapping_dataframe(self, df, state, touched):
        """
        Incrementally update only the rows corresponding to `touched` individuals.

        Parameters
        ----------
        df : pandas DataFrame
            Existing dataframe (will be modified in-place)
        state : FastBirthState
            Current state
        touched : list[int]
            Indices of individuals that changed
        """

        if not touched:
            return df

        # ----------------------------
        # Core columns (vectorized)
        # ----------------------------
        idx = np.asarray(touched, dtype=int)

        df.iloc[idx, df.columns.get_loc("main_start_date")] = state.dob[idx]
        df.iloc[idx, df.columns.get_loc("gap_start_date")] = state.gap_start[idx]
        df.iloc[idx, df.columns.get_loc("main_duration")] = state.lifespan[idx]
        df.iloc[idx, df.columns.get_loc("duration")] = state.lifespan[idx] + state.gap_duration

        # gap_duration is constant → no need to update

        # ----------------------------
        # Attributes
        # ----------------------------
        if self.birth_spec.attributes_spec is not None:
            for attr_spec in self.birth_spec.attributes_spec:
                col_idx = df.columns.get_loc(attr_spec.name)

                vals = np.asarray(state.attrs[attr_spec.name])[idx]

                # Replace None → np.nan (vectorized, no strings)
                mask_none = vals == None
                if mask_none.any():
                    vals = vals.copy()  # avoid modifying original state
                    vals[mask_none] = np.nan

                df.iloc[idx, col_idx] = vals

        return df

    # ------------------------------------------------------------------
    # Prior cache
    # ------------------------------------------------------------------
    def compute_prior_terms(self, state):
        n = state.dob.shape[0]
        out = np.empty(n, dtype=float)
        for i in range(n):
            instance = self.individual_instance_from_state(state, i)
            out[i] = self.birth_spec.log_density(instance)
        return out

    def compute_prior(self, state):
        return float(self.compute_prior_terms(state).sum())

    # ------------------------------------------------------------------
    # Proposal for dob/lifespan, directly on arrays
    # ------------------------------------------------------------------
    def propose_dob_lifespan_fast(self, state, ids, A, b, rng):
        j = int(rng.integers(0, len(ids)))
        i = ids[j]

        x_current = np.array([state.dob[i], state.lifespan[i]], dtype=float)

        u = rng.normal(size=2)
        u_norm = np.linalg.norm(u)
        if u_norm == 0:
            u = np.array([1.0, 0.0])
        else:
            u = u / u_norm

        t_min, t_max = chord_interval(A, b, x_current, u)
        t_prop = rng.uniform(t_min, t_max)
        x_prop = x_current + t_prop * u

        dob_new = float(x_prop[0])
        lifespan_new = float(x_prop[1])
        gap_start_new = dob_new + lifespan_new

        return i, dob_new, lifespan_new, gap_start_new

    # ------------------------------------------------------------------
    # Gibbs update for attributes for ONE individual only
    # ------------------------------------------------------------------

    def propose_new_attr_gibbs_fast(self, state, i, rng):
        """
        Propose attributes with gibbs for those specified with gibbs
        """

        instance = self.individual_instance_from_state(state, i)

        new_attrs = {}
        if self.birth_spec.attributes_spec is not None:
            for att in self.birth_spec.attributes_spec:
                if att.prob_model.method == "gibbs":
                    val = att.next_exact_conditional(instance = instance, index = None, rng = rng)
                    instance[self.birth_spec.name][att.name] = val
                    new_attrs[att.name] = val
                elif att.prob_model.method == "mh":
                    continue
                else:
                    raise ValueError ("Undefined method to sample the attributes.")
                
        
        return {
            "instance": instance,
            "new_attrs": new_attrs,}

    
    def propose_new_attr_mh_fast(self, state, i, rng):
        """
        Propose MH attributes jointly (independent proposals) for one individual (i),
        return log_q_forward and log_q_backward.
        """
        log_q_forward = 0.0
        log_q_backward = 0.0

        new_attrs = {}

        # Extract current values
        instance = self.individual_instance_from_state(state, i)
        b_spec = self.birth_spec

        if b_spec.attributes_spec is None:
            return {
                "instance": instance,
                "new_attrs": new_attrs,
            }

        # --------------------------------------
        # Select only MH attributes
        # --------------------------------------
        mh_atts = [
            att for att in b_spec.attributes_spec
            if getattr(att.prob_model, "method", None) == "mh"
        ]

        if not mh_atts:
            return {
                "instance": instance,
                "new_attrs": new_attrs,
                "log_q_forward": 0.0,
                "log_q_backward": 0.0,
            }

        # --------------------------------------
        # Extract current values
        # --------------------------------------
        attributes = [att.name for att in mh_atts]
        current_values = [instance[b_spec.name][att.name] for att in mh_atts]

        # --------------------------------------
        # Propose
        # --------------------------------------
        proposals = [
            att.next_proposal(val, instance, index = None, rng = rng)
            for att, val in zip(mh_atts, current_values)
        ]

        new_values, log_q_f, log_q_b = zip(*proposals)

        # --------------------------------------
        # Update instance
        # --------------------------------------
        for att_name, new_value in zip(attributes, new_values):
            instance[b_spec.name][att_name] = new_value
            new_attrs[att_name] = new_value

        # --------------------------------------
        # Sum logs
        # --------------------------------------
        log_q_forward = sum(log_q_f)
        log_q_backward = sum(log_q_b)

        return {
            "instance": instance,
            "new_attrs": new_attrs,
            "log_q_forward": log_q_forward,
            "log_q_backward": log_q_backward,
        }


    def compute_cross_section_likelihood(self, state, time_indep_pop, iter):
        """
        Sum likelihood contributions of all kernels.
        """
        total = 0.0
        for kernel in self.kernels:
            total += kernel.loglik(state, time_indep_pop, iter)
        return total




    def _add_event_change(self, event_changes, t, delta, tol=1e-15):
        """
        Add delta to the event change at time t.
        Remove the key if the resulting value is numerically zero.
        """
        t = float(t)
        new_val = event_changes.get(t, 0.0) + float(delta)

        if abs(new_val) <= tol:
            event_changes.pop(t, None)
        else:
            event_changes[t] = new_val

    def _event_changes_to_stair(self, event_changes):
        """
        Convert a dictionary {time: net_change} into (times, alive_values),
        where alive_values[k] is the alive count immediately after processing
        all events at times[k].

        Times are unique and sorted.
        """
        if not event_changes:
            return np.array([], dtype=float), np.array([], dtype=float)

        times = np.array(sorted(event_changes.keys()), dtype=float)
        deltas = np.array([event_changes[t] for t in times], dtype=float)
        values = np.cumsum(deltas)

        return times, values
    
    def build_stair_cache(self, state):
        """
        Build the full stair cache from scratch from the current state.

        Returns
        -------
        dict with:
            - "event_changes": dict {time: net_change}
            - "times": sorted unique times
            - "values": cumulative alive counts after each event time
        """
        event_changes = {}

        for b, d in zip(state.dob, state.gap_start):
            self._add_event_change(event_changes, b, +1.0)
            self._add_event_change(event_changes, d, -1.0)

        times, values = self._event_changes_to_stair(event_changes)

        return {
            "event_changes": event_changes,
            "times": times,
            "values": values,
        }
    
    def stair_function_alive(self, state):
        """
        Wrapper returning only (times, values), for compatibility.
        """
        stair_cache = self.build_stair_cache(state)
        return stair_cache["times"], stair_cache["values"]
    
    def update_stair_cache(self, stair_cache, backup, state, touched):
        """
        Incrementally update the stair cache after modifying touched individuals.

        Parameters
        ----------
        stair_cache : dict
            Current cache with keys:
                - event_changes
                - times
                - values
        backup : dict
            Backup of the OLD values for touched individuals, built before proposal.
            Must contain for each touched i:
                backup[i]["dob"]
                backup[i]["gap_start"]
        state : FastBirthState
            Current state AFTER proposal (contains NEW values).
        touched : list[int]
            Individuals whose dob / gap_start changed.

        Returns
        -------
        new_cache : dict
            Same structure as stair_cache
        """
        new_event_changes = stair_cache["event_changes"].copy()

        for i in touched:
            old_b = float(backup[i]["dob"])
            old_d = float(backup[i]["gap_start"])

            new_b = float(state.dob[i])
            new_d = float(state.gap_start[i])

            # remove old contribution
            self._add_event_change(new_event_changes, old_b, -1.0)
            self._add_event_change(new_event_changes, old_d, +1.0)

            # add new contribution
            self._add_event_change(new_event_changes, new_b, +1.0)
            self._add_event_change(new_event_changes, new_d, -1.0)

        times, values = self._event_changes_to_stair(new_event_changes)

        return {
            "event_changes": new_event_changes,
            "times": times,
            "values": values,
        }
    
    def likelihood_size_pop(
        self,
        target_size_t,
        t_eval_min,
        t_eval_max,
        sigma,
        times,
        values,
        n_eval_time=200,
        c=1.0,
    ):
        """
        Continuous-time quadratic penalty:
            - (1/(2 sigma^2)) * average_t [ (log(alive(t)+c)-log(target(t)+c))^2 ]

        Parameters
        ----------
        times : np.ndarray
            Sorted unique event times
        values : np.ndarray
            Alive counts immediately after each corresponding event time
        """
        if times.size == 0:
            return 0.0

        def alive_at(t):
            if t < times[0] or t > times[-1]:
                return 0.0
            idx = np.searchsorted(times, t, side="right") - 1
            if idx < 0:
                return 0.0
            return float(values[idx])

        K = max(2, int(n_eval_time))
        T = t_eval_max - t_eval_min
        dt = T / (K - 1)

        sse_int = 0.0

        t_prev = t_eval_min
        a_prev = alive_at(t_prev)
        y_prev = float(target_size_t(t_prev))

        e_prev = math.log(a_prev + c) - math.log(y_prev + c)
        f_prev = e_prev * e_prev

        for k in range(1, K):
            t_cur = t_eval_min + k * dt

            a_cur = alive_at(t_cur)
            y_cur = float(target_size_t(t_cur))

            e_cur = math.log(a_cur + c) - math.log(y_cur + c)
            f_cur = e_cur * e_cur

            sse_int += 0.5 * (f_prev + f_cur) * dt

            f_prev = f_cur

        sse_int /= T
        return -(1.0 / (2.0 * sigma * sigma)) * sse_int


    def initialize_kernel_caches(self, state, time_indep_pop, it):
        """
        Initialize one cache per kernel.
        """
        return [
            kernel.initialize_cache(state, time_indep_pop, it)
            for kernel in self.kernels
        ]


    def cross_section_loglik_from_caches(self, kernel_caches):
        """
        Sum log-likelihood contributions from all kernel caches.
        """
        total = 0.0
        for kernel, cache in zip(self.kernels, kernel_caches):
            total += kernel.loglik_from_cache(cache)
        return total


    def update_kernel_caches_from_touched(self, state, time_indep_pop, touched, kernel_caches, it):
        """
        Build proposed kernel caches by updating only touched individuals.
        """
        return [
            kernel.update_cache_from_touched(state, time_indep_pop, touched, cache, it)
            for kernel, cache in zip(self.kernels, kernel_caches)
        ]


    # ------------------------------------------------------------------
    # MH accept/reject
    # ------------------------------------------------------------------
    def accept_reject_move(self, prior_prop, prior_curr, likelihood_prop, likelihood_curr, rng, log_f = 1, log_b = 1,):
        ratio = (prior_prop + likelihood_prop + log_b) - (prior_curr + likelihood_curr + log_f)
        u = rng.uniform()
        return 1 if (np.log(u) <= min(0.0, ratio)) else 0

    def update_pop(
        self,
        n_MH,
        t_eval_min,
        t_eval_max,
        n_prop_function,
        sigma_size=0.1,
        n_eval_time=200,
        burnin_init=1000,
        n_prop_min=1,
        seed=0,
    ):

        initial = self.initialize_fast_state(seed=seed, burnin=burnin_init)
        state = initial["state"]
        current_A = initial["A"]
        current_B = initial["b"]

        ids = np.arange(self.pop_size, dtype=int)
        rng = np.random.default_rng(seed)

        posterior_array = []
        accept_rate_dur = []
        accept_rate_attrs = []

        # ----------------------------
        # PRIOR
        # ----------------------------
        prior_terms = self.compute_prior_terms(state)
        prev_prior = float(prior_terms.sum())

        # ----------------------------
        # CROSS-SECTION LIKELIHOOD
        # ----------------------------
        # time_indep_pop = self._build_mapping_dataframe(state)
        # prev_l_cross = self.compute_cross_section_likelihood(state, time_indep_pop, 0)

        time_indep_pop = self._build_mapping_dataframe(state)
        current_kernel_caches = self.initialize_kernel_caches(
            state, time_indep_pop, it=0
        )
        prev_l_cross = self.cross_section_loglik_from_caches(current_kernel_caches)

        # ----------------------------
        # SIZE LIKELIHOOD + STAIR CACHE
        # ----------------------------
        current_stair_cache = self.build_stair_cache(state)

        prev_l_size0 = self.likelihood_size_pop(
            self.target_size_t,
            t_eval_min,
            t_eval_max,
            sigma_size,
            current_stair_cache["times"],
            current_stair_cache["values"],
            n_eval_time=n_eval_time,
        )

        posterior_array.append(prev_l_cross + prev_l_size0 + prev_prior)
        l_c_size = prev_l_size0

        # ==========================================================
        # MCMC LOOP
        # ==========================================================
        for it in range(n_MH):
            if it == 1:
                print(it)
                
            if it % 100 == 0:
                print(it)
            n_prop_ = max(n_prop_min, int(n_prop_function(it)))

            likelihood_current = prev_l_cross + l_c_size

            # ----------------------------
            # BACKUP
            # ----------------------------
            backup = {}
            touched = set()

            for _ in range(n_prop_):

                i, dob_new, lifespan_new, gap_start_new = \
                    self.propose_dob_lifespan_fast(state, ids, current_A, current_B, rng)

                if i not in backup:
                    backup[i] = {
                        "dob": state.dob[i],
                        "lifespan": state.lifespan[i],
                        "gap_start": state.gap_start[i],
                        "attrs": {name: state.attrs[name][i] for name in state.attrs} if state.attrs else {},
                    }

                state.dob[i] = dob_new
                state.lifespan[i] = lifespan_new
                state.gap_start[i] = gap_start_new

                touched.add(i)

            touched = sorted(touched)

            # ----------------------------
            # PRIOR
            # ----------------------------
            old_prior_sum = prior_terms[touched].sum() if touched else 0.0

            new_prior_terms = np.empty(len(touched))
            for k, i in enumerate(touched):
                instance_i = self.individual_instance_from_state(state, i)
                new_prior_terms[k] = self.birth_spec.log_density(instance_i)

            prior_proposed = prev_prior - old_prior_sum + new_prior_terms.sum()

            # ----------------------------
            # LIKELIHOOD (duration)
            # ----------------------------
            # self._update_mapping_dataframe(time_indep_pop, state, touched)
            # l_p_cross = self.compute_cross_section_likelihood(state, time_indep_pop, it)

            self._update_mapping_dataframe(time_indep_pop, state, touched)
            proposed_kernel_caches = self.update_kernel_caches_from_touched(
                state, time_indep_pop, touched, current_kernel_caches, it
            )
            l_p_cross = self.cross_section_loglik_from_caches(proposed_kernel_caches)

            proposed_stair_cache = self.update_stair_cache(
                current_stair_cache,
                backup,
                state,
                touched,
            )

            l_p_size = self.likelihood_size_pop(
                self.target_size_t,
                t_eval_min,
                t_eval_max,
                sigma_size,
                proposed_stair_cache["times"],
                proposed_stair_cache["values"],
                n_eval_time=n_eval_time,
            )

            likelihood_proposed = l_p_cross + l_p_size

            # ----------------------------
            # ACCEPT / REJECT (duration)
            # ----------------------------
            accepted = self.accept_reject_move(
                prior_proposed,
                prev_prior,
                likelihood_proposed,
                likelihood_current,
                rng=rng,
            )

            if accepted:
                prev_prior = prior_proposed
                prev_l_cross = l_p_cross
                l_c_size = l_p_size
                current_stair_cache = proposed_stair_cache
                current_kernel_caches = proposed_kernel_caches

                for k, i in enumerate(touched):
                    prior_terms[i] = new_prior_terms[k]

                accept_rate_dur.append(1)

            else:
                for i in touched:
                    state.dob[i] = backup[i]["dob"]
                    state.lifespan[i] = backup[i]["lifespan"]
                    state.gap_start[i] = backup[i]["gap_start"]

                accept_rate_dur.append(0)
                self._update_mapping_dataframe(time_indep_pop, state, touched)
 
            if self.birth_spec.attributes_spec is not None :
                # ==========================================================
                # STEP 2: GIBBS (always accepted)
                # ==========================================================
                for i in touched:
                    res_gibbs = self.propose_new_attr_gibbs_fast(state, i, rng)
                    for attr_name, val in res_gibbs["new_attrs"].items():
                        state.attrs[attr_name][i] = val
                        backup[i]["attrs"][attr_name] = val

                # ==========================================================
                # recompute CURRENT after Gibbs
                # ==========================================================
                # self._update_mapping_dataframe(time_indep_pop, state, touched)
                # prev_l_cross = self.compute_cross_section_likelihood(state, time_indep_pop, it)

                self._update_mapping_dataframe(time_indep_pop, state, touched)
                current_kernel_caches = self.update_kernel_caches_from_touched(
                    state, time_indep_pop, touched, current_kernel_caches, it
                )
                prev_l_cross = self.cross_section_loglik_from_caches(current_kernel_caches)

                old_prior_sum = prior_terms[touched].sum() if touched else 0.0
                new_prior_terms_current = np.empty(len(touched))
                for k, i in enumerate(touched):
                    instance_i = self.individual_instance_from_state(state, i)
                    new_prior_terms_current[k] = self.birth_spec.log_density(instance_i)

                prev_prior = prev_prior - old_prior_sum + new_prior_terms_current.sum()
                for k, i in enumerate(touched):
                    prior_terms[i] = new_prior_terms_current[k]

                # ==========================================================
                # STEP 3: MH attributes
                # ==========================================================
                log_f = 0.0  
                log_b = 0.0

                for i in touched:
                    prop = self.propose_new_attr_mh_fast(state, i, rng)
                    log_f += prop["log_q_forward"]
                    log_b += prop["log_q_backward"]

                    for attr_name, val in prop["new_attrs"].items():
                        state.attrs[attr_name][i] = val

                # PRIOR (proposed)
                old_prior_sum = prior_terms[touched].sum() if touched else 0.0

                new_prior_terms = np.empty(len(touched))
                for k, i in enumerate(touched):
                    instance_i = self.individual_instance_from_state(state, i)
                    new_prior_terms[k] = self.birth_spec.log_density(instance_i)

                prior_proposed = prev_prior - old_prior_sum + new_prior_terms.sum()

                # LIKELIHOOD (proposed)
                # self._update_mapping_dataframe(time_indep_pop, state, touched)
                # l_p_cross = self.compute_cross_section_likelihood(state, time_indep_pop, it)
                self._update_mapping_dataframe(time_indep_pop, state, touched)
                proposed_kernel_caches = self.update_kernel_caches_from_touched(
                    state, time_indep_pop, touched, current_kernel_caches, it
                )
                l_p_cross = self.cross_section_loglik_from_caches(proposed_kernel_caches)

                likelihood_current = prev_l_cross + l_c_size 
                likelihood_proposed = l_p_cross + l_c_size

                # ACCEPT
                accepted = self.accept_reject_move(
                    prior_proposed, prev_prior,
                    likelihood_proposed, likelihood_current,
                    rng=rng,
                    log_f=log_f,
                    log_b=log_b,
                )

                if accepted:
                    accept_rate_attrs.append(1)
                    prev_prior = prior_proposed
                    prev_l_cross = l_p_cross
                    current_kernel_caches = proposed_kernel_caches
                    for k, i in enumerate(touched):
                        prior_terms[i] = new_prior_terms[k]
                    posterior_array.append(prior_proposed + likelihood_proposed)
                else:
                    accept_rate_attrs.append(0)
                    for i in touched:
                        for attr_name, val in backup[i]["attrs"].items():
                            state.attrs[attr_name][i] = val
                    posterior_array.append(prev_prior + likelihood_current)
                    self._update_mapping_dataframe(time_indep_pop, state, touched)

            else:
                posterior_array.append(prev_prior + prev_l_cross + l_c_size)

        final_pop = self.fast_state_to_population(state)
        return final_pop, accept_rate_dur, accept_rate_attrs, posterior_array


