"""Existence constraints 

Baud Candice
Fri July 10 10:33:00 2026
"""

import numpy as np

from har_existence import *
from constraints_existence import *
from classes_data import *

# sample : for single chain sampling
# sample_diagnostics : for multi-chain diagnostics
# sample_from_prior : for returning one Trajectory 

class BirthConstraintSpec:
    """
    Specification of temporal constraints for the birth (existence) event.

    This class defines admissible bounds on the birth time and lifespan,
    and may optionally enforce a warmup period before the nominal lower bound.
    """

    def __init__(self, start_low, start_high, duration_high, warmup_time=0):
        self.start_high = start_high
        self.duration_high = duration_high
        self.warmup_time, self.start_low = self.validate(warmup_time, start_low)

    def validate(self, warmup_time, start_low):
        if warmup_time is None:
            return 0, start_low

        if warmup_time < 0:
            raise ValueError("The warmup time must be greater or equal than 0.")

        return warmup_time, start_low - warmup_time


class ExistenceEventSpec:
    """
    Specification of the existence event (birth).

    This class contains:
    - the event definition
    - the associated attributes
    - the prior log density
    - the constraint specification
    - the sampling interface
    """

    def __init__(
        self,
        attributes_spec,
        log_density,
        name="Birth",
        type="Event",
        constraints=BirthConstraintSpec(1950, 2050, 110, 0),
    ):
        self.name = name
        self.attributes_spec = attributes_spec
        self.constraints = constraints
        self.log_density = log_density
        self.type = type

    # ---------------------------------------------------------------------
    # Instance construction utilities
    # ---------------------------------------------------------------------

    def create_empty_instance(self):
        birth = {
            "indicator": 1,
            "main_start_date": None,
            "gap_start_date": None,
            "main_duration": None,
            "gap_duration": 0,
        }

        if self.attributes_spec is not None:
            for att in self.attributes_spec:
                birth[att.name] = None

        return {self.name: birth}

    def update_instance_from_vector(self, x, instance):
        instance[self.name]["main_start_date"] = x[0]
        instance[self.name]["main_duration"] = x[1]
        instance[self.name]["gap_start_date"] = x[0] + x[1]

    def create_empty_output_container(self, n_samples):
        birth = {
            "indicator": 1,
            "main_start_date": {"chains": np.zeros(n_samples)},
            "main_duration": {"chains": np.zeros(n_samples)},
            "gap_start_date": {"chains": np.zeros(n_samples)},
            "gap_duration": 0,
        }

        if self.attributes_spec is not None:
            for att in self.attributes_spec:
                birth[att.name] = {"chains": np.zeros(n_samples)}

        return {self.name: birth}

    def store_sample(self, output_instance, inside_instance, x, idx):
        output_instance[self.name]["main_start_date"]["chains"][idx] = x[0]
        output_instance[self.name]["main_duration"]["chains"][idx] = x[1]
        output_instance[self.name]["gap_start_date"]["chains"][idx] = x[0] + x[1]

        if self.attributes_spec is not None:
            for att in self.attributes_spec:
                output_instance[self.name][att.name]["chains"][idx] = inside_instance[self.name][att.name]

    # ---------------------------------------------------------------------
    # Attribute update utilities
    # ---------------------------------------------------------------------

    def initialize_instance_attributes(self, instance, rng):
        if self.attributes_spec is not None:
            for att in self.attributes_spec:
                instance[self.name][att.name] = att.initialize(index = None, rng = rng)

    def update_instance_attributes_gibbs(self, instance, rng):
        if self.attributes_spec is not None :
            for att in self.attributes_spec:
                if att.prob_model.method == "gibbs":
                    instance[self.name][att.name] = att.next_exact_conditional(instance = instance, index = None, rng = rng)
                elif att.prob_model.method == "mh":
                    continue
                else:
                    raise ValueError("Invalid method to update the attributes.")
        
    def update_instance_attributes_mh(self, instance, rng):
        log_q_forward = 0.0
        log_q_backward = 0.0

        if self.attributes_spec is None:
            return log_q_forward, log_q_backward
        
        for att in self.attributes_spec:
            if att.prob_model.method == "gibbs":
                continue
            elif att.prob_model.method == "mh":
                current_value = instance[self.name][att.name]
                new_value, log_q_f, log_q_b = att.next_proposal(current_value = current_value, instance = instance, index = None, rng = rng)

                instance[self.name][att.name] = new_value
                log_q_forward += log_q_f
                log_q_backward += log_q_b
            else:
                raise ValueError("Invalid method to update the attributes.")
        
        return log_q_forward, log_q_backward

    # ---------------------------------------------------------------------
    # Internal helpers for sampling
    # ---------------------------------------------------------------------

    def _build_constraints(self):
        return build_birth_constraint_matrix(self.constraints)

    def sample(
        self,
        n_samples=1,
        burnin=1000,
        thin=1,
        seed=0,
        x0=None,
    ):
        """
        Generic single-chain sampling interface.

        Parameters
        ----------
        n_samples : int
            Number of retained samples.
        burnin : int
            Burn-in iterations.
        thin : int
            Thinning interval.
        seed : int
            Random seed.
        x0 : optional
            Initial point.

        Returns
        -------
        dict
            Dictionary with:
            - 'instance'
            - 'accept_rate'
            - 'A'
            - 'b'
        """
        A, b = self._build_constraints()

        output, accept_rate_dur, accept_rate_attr = hit_and_run_existence(
           model=self,
                A=A,
                b=b,
                n_samples=n_samples,
                burnin=burnin,
                thin=thin,
                seed=seed,
                x0=x0,
            ) 

        return {
            "instance": output,
            "accept_rate_dur": accept_rate_dur,
            "accept_rate_attr" : accept_rate_attr,
            "A": A,
            "b": b,
        }


    def sample_diagnostics(
        self,
        n_samples=100,
        burnin=1000,
        n_chains=4,
        thin=1,
        x0s=None,
    ):
        """
        Generic multi-chain sampling interface for diagnostics.
        """
        A, b = self._build_constraints()

        result = run_multiple_chains_hitandrun_existence(
                model=self,
                A=A,
                b=b,
                n_samples=n_samples,
                burnin=burnin,
                thin=thin,
                n_chains=n_chains,
                x0s=x0s,
            )
        
        return result



class ExistenceDimensionSpec:
    """
    Specification of the existence dimension.

    This class wraps the Birth event and exposes a high-level API:
    - sample : returns raw samples (delegates to event)
    - sample_diagnostics : multi-chain diagnostics
    - sample_from_prior : returns a Trajectory
    """

    def __init__(
        self,
        attributes_spec,
        log_density,
        constraints=BirthConstraintSpec(1950, 2050, 110, 0),
    ):
        self.name = "Existence"

        self.birth = ExistenceEventSpec(
            attributes_spec=attributes_spec,
            log_density=log_density,
            name="Birth",
            constraints=constraints,
        )

        self.list_event_spec = [self.birth]

    # ---------------------------------------------------------------------
    # Internal helper
    # ---------------------------------------------------------------------

    def _build_trajectory_from_sample(self, sampled):
        """
        Convert EventSpec output into a Trajectory object.
        """
        sampled_instance = sampled["instance"]
        ev_name = self.birth.name

        attributes = []
        if self.birth.attributes_spec is not None:
            for attr in self.birth.attributes_spec:
                attributes.append(
                    Attribute(
                        attr.name,
                        attr.type,
                        sampled_instance[ev_name][attr.name]["chains"][0],
                    )
                )

        main_start = sampled_instance[ev_name]["main_start_date"]["chains"][0]
        gap_start = sampled_instance[ev_name]["gap_start_date"]["chains"][0]
        main_dur = sampled_instance[ev_name]["main_duration"]["chains"][0]
        gap_dur = sampled_instance[ev_name]["gap_duration"]
        dur = main_dur + gap_dur

        birth_event = Event(
            ev_name,
            self.birth.type,
            main_start,
            gap_start,
            dur,
            main_dur,
            gap_dur,
            attributes,
        )

        return Trajectory(self.name, [birth_event], [1])

    # ---------------------------------------------------------------------
    # Public API (same style as EventSpec)
    # ---------------------------------------------------------------------

    def sample(self, **kwargs):
        """
        Delegates to EventSpec.sample()

        Returns raw sampling output (not a trajectory).
        """
        return self.birth.sample(**kwargs)

    def sample_diagnostics(self, **kwargs):
        """
        Delegates to EventSpec.sample_diagnostics()
        """
        return self.birth.sample_diagnostics(**kwargs)

    def sample_from_prior(self, **kwargs):
        """
        Sample one trajectory from the prior.

        This is the main high-level method users will call.
        """
        sampled = self.birth.sample(n_samples=1, **kwargs)
        return self._build_trajectory_from_sample(sampled)
