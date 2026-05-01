"""Specifications

Baud Candice
Mon March 30 17:42:00 2026
"""

from scipy.linalg import block_diag
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os

from pre_coded_function import *
from classes_data import *
from constraints_events import *
from spec_existence import *
from constraints import *
from har_inside_dimension import *
from har_generic_code import *
from har_individual_level import *


class AttributeSpec:
    def __init__(self, name, type, prob_model):
        """Initializes the characterisitcs of an attribute
        
        Parameters
        ----------
        name : name
        type : Single/Multiple, corresponds to if the Event has a max_count > 1 or = 1
        prob_model : probabilistic model used to sample from it
        """
        self.name = name
        self.type = type
        self.prob_model = prob_model


    @classmethod
    def SingleAttr(
        cls,
        name,
        prob_model
    ):
        """
        Single event attribute : don't need to know the "position" = which event of the attribute to sample
        """
        return cls(
            name=name,
            type="Single",
            prob_model = prob_model
        )

    @classmethod
    def MultipleAttr(
        cls,
        name,
        prob_model
    ):
        """
        Multiple event attribute : need to know the "position" of the attribute to sample
        """
        return cls(
            name=name,
            type="Multiple",
            prob_model = prob_model
        )
    

    def initialize(self, index = None, rng = 0):
        """
        Samples the attribute value from the initial distribution.

        Parameters
        ----------
        index : Optional, if the event has multiple spells to sample the "good" event

        Returns
        -------
        Attribute
            Instantiated attribute with a sampled value.
        """

        if self.type == "Single":
            value = self.prob_model.sampler_initial(rng = rng)
        elif self.type == "Multiple":
            value = self.prob_model.sampler_initial(index = index, rng = rng)
        else :
            raise ValueError("There are only two types of Attributes : Single and Multiple.")
        return value
    
    def next_exact_conditional(self, instance, index = None, rng = 0):
        """
        Samples the attribute value from its marginal distribution.

        Parameters
        ----------
        instance : dict
            Current instance dictionary.

        Returns
        -------
        Attribute
            Instantiated attribute with a sampled value.
        """
        # sample the new value conditional on the instance  
        if self.type == "Single":
            value = self.prob_model.sampler_full_conditional(instance = instance, rng = rng)
        elif self.type == "Multiple":
            value = self.prob_model.sampler_full_conditional(instance = instance, index = index, rng = rng)
        else :
            raise ValueError("There are only two types of Attributes : Single and Multiple.")
        return value

    def next_proposal(self, current_value, instance, index = None, rng = 0):
        """
        Samples the attribute value from a proposal and returns the proposal density.

        Parameters
        ----------
        instance : dict
            Current instance dictionary.

        Returns
        -------
        Attribute
            Instantiated attribute with a sampled value.
        """
        # sample the new value conditional on the instance  
        if self.type == "Single":
            value = self.prob_model.sampler_proposal(current_value = current_value, instance = instance, rng = rng)
            prop_forward = self.prob_model.evaluation_proposal(value, current_value, instance)  
            prop_backward = self.prob_model.evaluation_proposal(current_value, value, instance)  
        elif self.type == "Multiple":
            value = self.prob_model.sampler_proposal(current_value = current_value, instance = instance, index = index, rng = rng)
            prop_forward = self.prob_model.evaluation_proposal(value, current_value, instance, index) 
            prop_backward = self.prob_model.evaluation_proposal(current_value, value, instance, index) 
        else :
            raise ValueError("There are only two types of Attributes : Single and Multiple.")
        return value, prop_forward, prop_backward
    

class EventSpec:
    def __init__(self, name, type, max_count, indic_prob_model, attributes_spec, constraint_spec):
        """Event Spec defines everything related to the event
        
        Parameters
        ---------
        name 
        type : Event/NoEvent
        max_count : between 1 and M maximal value
        prob_model : probabilistic model for the INDICATORS
        attributes_spec
        constraint_spec : specification of constraints tied to the event (not to other events)
        """
        self.name = name
        self.type, self.max_count = self._check_type_count(type, max_count)
        self.indic_prob_model = indic_prob_model
        self.attributes_spec = attributes_spec
        self.constraint_spec = constraint_spec

        if type == "Event":
            self.constraint_spec = self._check_constraints(constraint_spec)
        else:
            self.constraint_spec = None
    
    def _check_type_count(self, type, max_count):
        """
        Validates event type and occurrence specifications.
        """
        if max_count<0:
            raise ValueError("The maximal number of occurences of the event must be positive.")
        if type == "No Event":
            if max_count != 1:
                raise ValueError("The No Event must have a max count equal to 1.")
        if type != "Event" and type != "No Event":
            raise ValueError("Events can either be of the type Event or No Event")
        
        return(type, max_count)
    
    def _check_constraints(self, constraint_spec):
        """
        Checks consistency of temporal constraints for the event.
        """
        if constraint_spec == []:
           constraint_spec = None
    
        if constraint_spec is not None :
            main_start_age_min = constraint_spec.main_start_age_min
            main_start_age_max = constraint_spec.main_start_age_max

            main_min_duration = constraint_spec.main_min_duration
            gap_min_duration = constraint_spec.gap_min_duration

            main_max_duration = constraint_spec.main_max_duration
            gap_start_age_min = constraint_spec.gap_start_age_min
            gap_start_age_max = constraint_spec.gap_start_age_max
            gap_max_duration = constraint_spec.gap_max_duration
            max_duration = constraint_spec.max_duration

            # --- Main start age consistency ---
            if main_start_age_min is not None and main_start_age_max is not None:
                if main_start_age_min > main_start_age_max:
                    raise ValueError(
                        f"CONSTRAINT ERROR: Minimal starting age must be ≤ maximal starting age in {self.name}, main"
                    )

            # --- Gap start age consistency ---
            if gap_start_age_min is not None and gap_start_age_max is not None:
                if gap_start_age_min > gap_start_age_max:
                    raise ValueError(
                        f"CONSTRAINT ERROR: Minimal starting age must be ≤ maximal starting age in {self.name}, gap"
                    )

            # --- Max durations must be non-negative ---
            if max_duration is not None:
                if max_duration < 0:
                    raise ValueError(f"Maximal duration must be non-negative in {self.name}, total duration")

            if main_max_duration is not None:
                if main_max_duration < 0:
                    raise ValueError(f"Maximal duration must be non-negative in {self.name}, main duration")

            if gap_max_duration is not None:
                if gap_max_duration < 0:
                    raise ValueError(f"Maximal duration must be non-negative in {self.name}, gap duration")

            # --- Main duration min consistency ---
            if main_min_duration is not None:
                if main_min_duration < 0:
                    raise ValueError(
                        f"Minimal duration must be non-negative in {self.name}, main duration"
                    )
                
            # --- Gap duration min consistency ---
            if gap_min_duration is not None:
                if gap_min_duration < 0:
                    raise ValueError(
                        f"Minimal duration must be non-negative in {self.name}, gap duration"
                    )
                
        return(constraint_spec)
    
    
    @classmethod
    def NoEvent(
        cls,
        name,
        constraint_spec=None,
    ):
        """
        Specification of a dimension with no event.

        - max_count is fixed to 1
        - attributes are not allowed
        - constraint_spec can be specified but by default is None
        """
        return cls(
            name=name,
            type="No Event",
            max_count=1,
            indic_prob_model = indic_prob_model_no_event,
            constraint_spec=constraint_spec,
            attributes_spec=None,
        )
    
    @classmethod
    def SingleTimeEvent(
        cls,
        name,
        indic_prob_model,
        constraint_spec=None,
        attributes_spec=None,
    ):
        """
        Specification of an event that can occur at most once.

        - type is fixed to "Event"
        - max_count is fixed to 1
        - indic_prob_model must be provided (to know how to sample the indicator)
        - constraints and attributes are optional
        """
        return cls(
            name=name,
            type="Event",
            max_count=1,
            indic_prob_model = indic_prob_model,
            constraint_spec=constraint_spec,
            attributes_spec=attributes_spec,
        )
    
    @classmethod
    def MultipleTimeEvent(
        cls,
        name,
        max_count,
        indic_prob_model,
        constraint_spec=None,
        attributes_spec=None,
    ):
        """
        Specification of an event that can occur multiple times.

        - type is fixed to "Event"
        - max_count must be provided by the user
        - indic_prob_model must be provided (to know how to sample the indicators)
        - constraints and attributes are optional
        """
        return cls(
            name=name,
            type="Event",
            max_count=max_count,
            indic_prob_model = indic_prob_model,
            constraint_spec=constraint_spec,
            attributes_spec=attributes_spec,
        )


class DimensionSpec:
    def __init__(self, name, list_event_spec, log_density):
        self.name = name
        self.list_event_spec = self._check_validity(list_event_spec)
        self.log_density = log_density

    def _check_validity(self, list_event_spec):
        """
        Validates the event structure of the dimension.

        Ensures that the dimension contains exactly one 'No Event',
        that it appears in the first position, and that its occurrence
        constraints are consistent.
        """
        if self.name != "Existence" and self.name != "Residence":
            # Check if list_event_spec is a valid, non-empty list
            if not isinstance(list_event_spec, list) or len(list_event_spec) == 0:
                raise ValueError("The dimension must contain a non-empty list of events.")
            
            # Track how many "No Event" types exist
            noevent_count = 0
            no_event_index = None
            for idx, e in enumerate(list_event_spec):
                # Check the type of each event and modify if necessary
                if e.type == "No Event":
                    noevent_count += 1
                    no_event_index = idx  # Track the index of the "No Event"
                    if e.max_count != 1:
                        e.max_count = 1  # Correct the max_count value to 1 if it's not correct
            
            # Validation for "No Event" conditions
            if noevent_count == 0:
                raise ValueError("There must be a 'No Event' specified in each dimension.")
                #here we could also print the error and create a no event of duration 0 or something like that 
            elif noevent_count > 1:
                raise ValueError("There must be only one 'No Event' specified in each dimension.")

            # Ensure "No Event" is the first event in the list
            if no_event_index != 0:
                print(f"Warning: 'No Event' of {self.name} is not in the first position. Moving it to the first position.")
                # Move the "No Event" to the first position
                no_event = list_event_spec.pop(no_event_index)
                list_event_spec.insert(0, no_event)
        
        # Return the (possibly modified) list of events
        return list_event_spec
    

    def create_empty_instance(self):
        instance = {}

        def build_event_dict(indicator, attributes_spec):
            ev_dict = {
                "indicator": indicator,
                "main_start_date": None,
                "gap_start_date": None,
                "main_duration": None,
                "gap_duration": None,
            }
            if attributes_spec is not None:
                for att in attributes_spec:
                    ev_dict[att.name] = None
            return ev_dict

        for idx, ev in enumerate(self.list_event_spec):

            if ev.max_count <= 0:
                raise ValueError(
                    f"The maximum number of occurrences of an event must be positive, "
                    f"not {ev.max_count} in {ev.name}"
                )

            # --- Special rule for non-Residence dimensions ---
            if self.name != "Residence" and idx == 0:
                if ev.type != "No Event":
                    raise ValueError("All dimensions must start with a No Event")
                instance[ev.name] = build_event_dict(1, ev.attributes_spec)
                continue

            # --- Residence first event special indicator ---
            if self.name == "Residence" and idx == 0 and ev.max_count >= 1:
                first_indicator = 1
            else:
                first_indicator = None

            # --- Single occurrence ---
            if ev.max_count == 1:
                indicator = first_indicator if idx == 0 else None
                instance[ev.name] = build_event_dict(indicator, ev.attributes_spec)

            # --- Multiple occurrences ---
            else:
                for m in range(ev.max_count):
                    name = f"{ev.name}_{m+1}"
                    indicator = first_indicator if (idx == 0 and m == 0) else None
                    instance[name] = build_event_dict(indicator, ev.attributes_spec)

        return instance


    def create_empty_output_container(self, n_samples, instance_gibbs):
        ## creates empty container for the chains with indicators obtained by the gibbs sampler
        instance = {}

        def build_event_dict(indicator, attributes_spec):
            ev_dict = {
                "indicator": indicator,  # keep same indicator as Gibbs instance
                "main_start_date": {"chains": np.zeros(n_samples)},
                "gap_start_date":  {"chains": np.zeros(n_samples)},
                "main_duration":   {"chains": np.zeros(n_samples)},
                "gap_duration":    {"chains": np.zeros(n_samples)},
            }

            if attributes_spec is not None:
                for att in attributes_spec:
                    ev_dict[att.name] = {"chains": np.empty(n_samples, dtype=object)}

            return ev_dict

        for idx, ev in enumerate(self.list_event_spec):

            if ev.max_count <= 0:
                raise ValueError(
                    f"The maximum number of occurrences of an event must be positive, "
                    f"not {ev.max_count} in {ev.name}"
                )

            # --- Single occurrence ---
            if ev.max_count == 1:
                name = ev.name
                indicator = instance_gibbs[name]["indicator"]
                instance[name] = build_event_dict(indicator, ev.attributes_spec)

            # --- Multiple occurrences ---
            else:
                for m in range(ev.max_count):
                    name = f"{ev.name}_{m+1}"
                    indicator = instance_gibbs[name]["indicator"]
                    instance[name] = build_event_dict(indicator, ev.attributes_spec)

        return instance

    def initialize_indicators(self, none_instance, lifespan, rng):
        # first event must always happen (at least first spell if max_count > 1)
        ev0 = self.list_event_spec[0]
        ev0_max_count = ev0.max_count
        n_ones_sampled = ev0.indic_prob_model.sampler_initial(rng = rng)
        n_ones = max(1, min(n_ones_sampled, ev0_max_count)) # actual number of spells
        if ev0_max_count == 1:
            ev0_name = ev0.name
            none_instance[ev0_name]["indicator"] = 1
        elif ev0_max_count > 1 :
            for i in range (ev0_max_count):
                ev0_name = f"{ev0.name}_{i + 1}"
                if i < n_ones :
                    none_instance[ev0_name]["indicator"] = 1
                else :
                    none_instance[ev0_name]["indicator"] = 0
        
        # other events
        if len(self.list_event_spec) > 1:
            for j in range (1, len(self.list_event_spec)):
                curr_event_spec = self.list_event_spec[j]
                prev_event_spec = self.list_event_spec[j-1]

                curr_event_name = curr_event_spec.name # base name
                curr_event_max_count = curr_event_spec.max_count
                prev_event_max_count = prev_event_spec.max_count
                
                if prev_event_max_count == 1:
                    prev_ev_name = prev_event_spec.name
                elif prev_event_max_count > 1:
                    prev_ev_name = f"{prev_event_spec.name}_{1}"

                if none_instance[prev_ev_name]["indicator"] == 1: #if previous event happens
                    if curr_event_spec.constraint_spec is None :                      
                        n_ones_sampled = curr_event_spec.indic_prob_model.sampler_initial(rng = rng)
                        n_ones = min(n_ones_sampled, curr_event_max_count)
                        if curr_event_max_count > 1:
                            for m in range(curr_event_max_count):
                                event_name_with_index = f"{curr_event_name}_{m + 1}"
                                # If the indicator is within the sampled count, set it to 1, else 0
                                none_instance[event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                        elif curr_event_max_count == 1:
                            none_instance[curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                    else:
                        if curr_event_spec.constraint_spec.main_start_age_min is None : 
                            n_ones_sampled = curr_event_spec.indic_prob_model.sampler_initial(rng = rng)
                            n_ones = min(n_ones_sampled, curr_event_max_count)
                            if curr_event_max_count > 1:
                                for m in range(curr_event_max_count):
                                    event_name_with_index = f"{curr_event_name}_{m + 1}"
                                    # If the indicator is within the sampled count, set it to 1, else 0
                                    none_instance[event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                            elif curr_event_max_count == 1:
                                none_instance[curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                        else:
                            if lifespan > curr_event_spec.constraint_spec.main_start_age_min:
                                n_ones_sampled = curr_event_spec.indic_prob_model.sampler_initial(rng = rng)
                                n_ones = min(n_ones_sampled, curr_event_max_count)
                                if curr_event_max_count > 1:
                                    for m in range(curr_event_max_count):
                                        event_name_with_index = f"{curr_event_name}_{m + 1}"
                                        # If the indicator is within the sampled count, set it to 1, else 0
                                        none_instance[event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                                elif curr_event_max_count == 1:
                                    none_instance[curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                            else:
                                n_ones_sampled = 0
                                if curr_event_max_count > 1:
                                    for m in range(curr_event_max_count):
                                        event_name_with_index = f"{curr_event_name}_{m + 1}"
                                        # If the indicator is within the sampled count, set it to 1, else 0
                                        none_instance[event_name_with_index]["indicator"] = 1 if m < n_ones_sampled else 0
                                elif curr_event_max_count == 1:
                                    none_instance[curr_event_name]["indicator"] = min(1, n_ones_sampled)
                else:
                    n_ones_sampled = 0
                    if curr_event_max_count > 1:
                        for m in range(curr_event_max_count):
                            event_name_with_index = f"{curr_event_name}_{m + 1}"
                            # If the indicator is within the sampled count, set it to 1, else 0
                            none_instance[event_name_with_index]["indicator"] = 1 if m < n_ones_sampled else 0
                    elif curr_event_max_count == 1:
                        none_instance[curr_event_name]["indicator"] = min(1, n_ones_sampled)

    
    ## gibbs sampler
    def exact_gibbs_update_indicators(self, instance_inside, lifespan, rng):
        # keep the first event as it is, only iterate for the other ones
        if len(self.list_event_spec) > 1:
            for j in range (1, len(self.list_event_spec)):
                curr_event_spec = self.list_event_spec[j]
                curr_event_name = curr_event_spec.name # base name
                curr_event_max_count = curr_event_spec.max_count
                prev_event_spec = self.list_event_spec[j-1]
                prev_event_max_count = prev_event_spec.max_count
                
                if prev_event_max_count == 1:
                    prev_ev_name = prev_event_spec.name
                elif prev_event_max_count > 1:
                    prev_ev_name = f"{prev_event_spec.name}_{1}"
                
                if instance_inside[prev_ev_name]["indicator"] == 1: #if previous event happens
                    if curr_event_spec.constraint_spec is None :
                        n_ones_sampled = curr_event_spec.indic_prob_model.sampler_full_conditional(instance_inside, rng = rng)
                        n_ones = min(n_ones_sampled, curr_event_max_count)
                        if curr_event_max_count > 1:
                            for m in range(curr_event_max_count):
                                event_name_with_index = f"{curr_event_name}_{m + 1}"
                                # If the indicator is within the sampled count, set it to 1, else 0
                                instance_inside[event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                        elif curr_event_max_count == 1:
                            instance_inside[curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                    else:
                        if curr_event_spec.constraint_spec.main_start_age_min is None : 
                            n_ones_sampled = curr_event_spec.indic_prob_model.sampler_full_conditional(instance_inside,rng = rng)
                            n_ones = min(n_ones_sampled, curr_event_max_count)
                            if curr_event_max_count > 1:
                                for m in range(curr_event_max_count):
                                    event_name_with_index = f"{curr_event_name}_{m + 1}"
                                    # If the indicator is within the sampled count, set it to 1, else 0
                                    instance_inside[event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                            elif curr_event_max_count == 1:
                                instance_inside[curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                        else:
                            if lifespan > curr_event_spec.constraint_spec.main_start_age_min:
                                n_ones_sampled = curr_event_spec.indic_prob_model.sampler_full_conditional(instance_inside, rng = rng)
                                n_ones = min(n_ones_sampled, curr_event_max_count)
                                if curr_event_max_count > 1:
                                    for m in range(curr_event_max_count):
                                        event_name_with_index = f"{curr_event_name}_{m + 1}"
                                        # If the indicator is within the sampled count, set it to 1, else 0
                                        instance_inside[event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                                elif curr_event_max_count == 1:
                                    instance_inside[curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                            else:
                                n_ones_sampled = 0
                                if curr_event_max_count > 1:
                                    for m in range(curr_event_max_count):
                                        event_name_with_index = f"{curr_event_name}_{m + 1}"
                                        # If the indicator is within the sampled count, set it to 1, else 0
                                        instance_inside[event_name_with_index]["indicator"] = 1 if m < n_ones_sampled else 0
                                elif curr_event_max_count == 1:
                                    instance_inside[curr_event_name]["indicator"] = min(1, n_ones_sampled)
                else:
                    n_ones_sampled = 0
                    if curr_event_max_count > 1:
                        for m in range(curr_event_max_count):
                            event_name_with_index = f"{curr_event_name}_{m + 1}"
                            # If the indicator is within the sampled count, set it to 1, else 0
                            instance_inside[event_name_with_index]["indicator"] = 1 if m < n_ones_sampled else 0
                    elif curr_event_max_count == 1:
                        instance_inside[curr_event_name]["indicator"] = min(1, n_ones_sampled)

   
    ### build constraint matrix and vector for the dimension
    def build_dimension_constraint_matrix(self, instance, lifespan):
        indicator_list = extract_event_indicators_inside_dimension(instance)
        
        n_events_dim = 0
        for k, ev in enumerate(self.list_event_spec):
            n_events_dim += ev.max_count  # to take into account the multiple events

        dim_matrix_rows = []
        dim_vec_rows = []

        cumulative_event_index = 0  # Initialize the cumulative event index

        # iterate events
        for k, ev in enumerate(self.list_event_spec):
            for l in range(ev.max_count):  # Iterate over events (whether they are unique or multiple)
                # Use cumulative index for event
                ev_matrix, ev_b = build_event_constraint_rows(
                    ev.constraint_spec,
                    n_events_dim,
                    cumulative_event_index,  # Use the cumulative index here
                    indicator_list,
                    lifespan,
                )
                
                if ev_matrix is not None:
                    dim_matrix_rows.append(ev_matrix)
                    dim_vec_rows.append(ev_b)
                cumulative_event_index += 1  # Update the cumulative index after processing each event
        
        # infer number of columns once
        n_cols = dim_matrix_rows[0].shape[1]

        # --- dimension-level constraints ---
        # sum of durations <= lifespan
        dim_matrix_rows.append(np.ones((1, n_cols)))
        dim_vec_rows.append(np.array([lifespan]))

        # -sum of durations <= -lifespan (i.e., sum >= lifespan)
        dim_matrix_rows.append(-np.ones((1, n_cols)))
        dim_vec_rows.append(np.array([-lifespan]))  #

        # gap duration of the "No event"
        no_event_row = np.zeros((1, n_cols))
        no_event_row[0, 1] = 1  # second coefficient is 1
        dim_matrix_rows.append(no_event_row)
        dim_vec_rows.append(np.array([0]))

        # stack the event-level rows for this dimension
        dim_matrix = np.vstack(dim_matrix_rows)
        dim_vec = np.hstack(dim_vec_rows)

        return(dim_matrix, dim_vec)
    
    def initialize_instance_attributes(self, instance_inside, rng):
        for k, ev in enumerate(self.list_event_spec):
            if ev.attributes_spec is not None:
                if ev.max_count == 1:
                    ev_name = ev.name
                    if instance_inside[ev_name]["indicator"] == 1:
                        for att in ev.attributes_spec :
                            instance_inside[ev.name][att.name] = att.initialize(index = None, rng = rng)
                elif ev.max_count > 1 :
                    for m in range (ev.max_count):
                        ev_name = f'{ev.name}_{m+1}'
                        if instance_inside[ev_name]["indicator"] == 1:
                            for att in ev.attributes_spec :
                                instance_inside[ev_name][att.name] = att.initialize(index = m, rng = rng)

    
    def fill_durations_instance_in_dimension(self, duration_vector, instance):
        vector_position = 0 
        for k,ev in enumerate(self.list_event_spec):
            if ev.max_count == 1:
                instance[ev.name]["main_duration"] = duration_vector[vector_position]
                instance[ev.name]["gap_duration"] = duration_vector[vector_position + 1]
                vector_position = vector_position + 2 #even if not present, we need to put +2 because otherwise index problem
            elif ev.max_count > 1:
                for z in range(ev.max_count):
                    ev_name = f"{ev.name}_{z + 1}"
                    instance[ev_name]["main_duration"] = duration_vector[vector_position]
                    instance[ev_name]["gap_duration"] = duration_vector[vector_position + 1]
                    
                    vector_position = vector_position + 2
    
    def fill_terminal_event_duration(self, instance):
        for k, ev in enumerate(self.list_event_spec):

            if ev.constraint_spec is not None:

                # ==========================================================
                # CASE 1 — SINGLE EVENT (max_count == 1)
                # ==========================================================
                if ev.max_count == 1:

                    ev_current_name = ev.name

                    # ------------------ MAIN DURATION ------------------
                    terminal_bool = 0
                    if (
                        ev.constraint_spec.main_min_duration is not None
                        and instance[ev_current_name]["indicator"] == 1
                    ):

                        # Check if terminal event
                        if k + 1 >= len(self.list_event_spec):
                            # Last event of the spec
                            terminal_bool = 1
                        else:
                            next_event = self.list_event_spec[k + 1]

                            if next_event.max_count == 1:
                                if instance[next_event.name]["indicator"] == 0:
                                    terminal_bool = 1
                            else:
                                next_name = f"{next_event.name}_1"
                                if instance[next_name]["indicator"] == 0:
                                    terminal_bool = 1

                        if terminal_bool == 1:
                            if (
                                instance[ev.name]["main_duration"]
                                != ev.constraint_spec.main_min_duration
                            ):
                                total = (
                                    instance[ev.name]["main_duration"]
                                    + instance[ev.name]["gap_duration"]
                                )

                                instance[ev.name]["main_duration"] = min(
                                    ev.constraint_spec.main_min_duration, total
                                )
                                instance[ev.name]["gap_duration"] = (
                                    total - instance[ev.name]["main_duration"]
                                )

                    # ------------------ GAP DURATION ------------------
                    terminal_bool = 0
                    if (
                        ev.constraint_spec.gap_min_duration is not None
                        and instance[ev_current_name]["indicator"] == 1
                    ):

                        if k + 1 >= len(self.list_event_spec):
                            terminal_bool = 1
                        else:
                            next_event = self.list_event_spec[k + 1]

                            if next_event.max_count == 1:
                                if instance[next_event.name]["indicator"] == 0:
                                    terminal_bool = 1
                            else:
                                next_name = f"{next_event.name}_1"
                                if instance[next_name]["indicator"] == 0:
                                    terminal_bool = 1

                        if terminal_bool == 1:
                            if (
                                instance[ev.name]["gap_duration"]
                                != ev.constraint_spec.gap_min_duration
                            ):
                                total = (
                                    instance[ev.name]["main_duration"]
                                    + instance[ev.name]["gap_duration"]
                                )

                                instance[ev.name]["gap_duration"] = min(
                                    ev.constraint_spec.gap_min_duration, total
                                )
                                instance[ev.name]["main_duration"] = (
                                    total - instance[ev.name]["gap_duration"]
                                )

                # ==========================================================
                # CASE 2 — MULTIPLE EVENTS (max_count > 1)
                # ==========================================================
                elif ev.max_count > 1:

                    # ------------------ MAIN DURATION ------------------
                    if ev.constraint_spec.main_min_duration is not None:

                        for z in range(ev.max_count):

                            ev_name = f"{ev.name}_{z + 1}"

                            if instance[ev_name]["indicator"] == 1:

                                terminal_bool = 0

                                # If last occurrence
                                if z + 1 >= ev.max_count:

                                    # And last event of spec
                                    if k + 1 >= len(self.list_event_spec):
                                        terminal_bool = 1

                                else:
                                    next_event_name = f"{ev.name}_{z + 2}"
                                    if instance[next_event_name]["indicator"] == 0:
                                        terminal_bool = 1

                                if terminal_bool == 1:

                                    if (
                                        instance[ev_name]["main_duration"]
                                        != ev.constraint_spec.main_min_duration
                                    ):
                                        total = (
                                            instance[ev_name]["main_duration"]
                                            + instance[ev_name]["gap_duration"]
                                        )

                                        instance[ev_name]["main_duration"] = min(
                                            ev.constraint_spec.main_min_duration, total
                                        )
                                        instance[ev_name]["gap_duration"] = (
                                            total
                                            - instance[ev_name]["main_duration"]
                                        )

                    # ------------------ GAP DURATION ------------------
                    if ev.constraint_spec.gap_min_duration is not None:

                        for z in range(ev.max_count):

                            ev_name = f"{ev.name}_{z + 1}"

                            if instance[ev_name]["indicator"] == 1:

                                terminal_bool = 0

                                if z + 1 >= ev.max_count:

                                    if k + 1 >= len(self.list_event_spec):
                                        terminal_bool = 1

                                else:
                                    next_event_name = f"{ev.name}_{z + 2}"
                                    if instance[next_event_name]["indicator"] == 0:
                                        terminal_bool = 1

                                if terminal_bool == 1:

                                    if (
                                        instance[ev_name]["gap_duration"]
                                        != ev.constraint_spec.gap_min_duration
                                    ):
                                        total = (
                                            instance[ev_name]["main_duration"]
                                            + instance[ev_name]["gap_duration"]
                                        )

                                        instance[ev_name]["gap_duration"] = min(
                                            ev.constraint_spec.gap_min_duration, total
                                        )
                                        instance[ev_name]["main_duration"] = (
                                            total
                                            - instance[ev_name]["gap_duration"]
                                        )

    def fill_starting_dates_in_dimension(self, instance, dob):
        list_durations = []
        for k,ev in enumerate(self.list_event_spec):
            if ev.max_count == 1:
                ev_name = ev.name
                ev_main_dur = instance[ev_name]["main_duration"]
                ev_gap_dur = instance[ev_name]["gap_duration"]
                ev_dur = ev_main_dur + ev_gap_dur
                if k == 0:
                    ev_start_date = dob
                    ev_gap_start_date = ev_start_date + ev_main_dur
                    instance[ev_name]["main_start_date"] = ev_start_date
                    instance[ev_name]["gap_start_date"] = ev_gap_start_date
                else:
                    ev_start_date = dob + np.sum(list_durations)
                    ev_gap_start_date = ev_start_date + ev_main_dur
                    instance[ev_name]["main_start_date"] = ev_start_date
                    instance[ev_name]["gap_start_date"] = ev_gap_start_date

                list_durations.append(ev_dur)

            elif ev.max_count > 1 :
                for z in range(ev.max_count):
                    ev_name = f"{ev.name}_{z + 1}"
                    ev_main_dur = instance[ev_name]["main_duration"]
                    ev_gap_dur = instance[ev_name]["gap_duration"]
                    ev_dur = ev_main_dur + ev_gap_dur
                    if k == 0:
                        ev_start_date = dob
                        ev_gap_start_date = ev_start_date + ev_main_dur
                        instance[ev_name]["main_start_date"] = ev_start_date
                        instance[ev_name]["gap_start_date"] = ev_gap_start_date
                    else:
                        ev_start_date = dob + np.sum(list_durations)
                        ev_gap_start_date = ev_start_date + ev_main_dur
                        instance[ev_name]["main_start_date"] = ev_start_date
                        instance[ev_name]["gap_start_date"] = ev_gap_start_date
                    
                    list_durations.append(ev_dur)


    def update_instance_from_vector(self, x_current, instance_inside, dob):
        self.fill_durations_instance_in_dimension(x_current, instance_inside)
        self.fill_terminal_event_duration(instance_inside)
        self.fill_starting_dates_in_dimension(instance_inside, dob)

    def update_instance_attributes_gibbs(self, instance_inside, rng):
        for ev in self.list_event_spec:
            if ev.attributes_spec is not None:
                if ev.max_count == 1:
                    ev_name = ev.name
                    if instance_inside[ev_name]["indicator"] == 1:
                        for att in ev.attributes_spec:
                            if att.prob_model.method == "gibbs":
                                instance_inside[ev_name][att.name] = att.next_exact_conditional(instance = instance_inside, index = None, rng = rng)
                            elif att.prob_model.method == "mh":
                                continue
                            else:
                                raise ValueError ("Invalid method to update attributes, single time attribute.")

                elif ev.max_count > 1:
                    for m in range(ev.max_count):
                        ev_name = f"{ev.name}_{m+1}"
                        if instance_inside[ev_name]["indicator"] == 1:
                            for att in ev.attributes_spec:
                                if att.prob_model.method == "gibbs":
                                    instance_inside[ev_name][att.name] = att.next_exact_conditional(instance = instance_inside, index = m, rng = rng)
                                elif att.prob_model.method == "mh":
                                    continue
                                else:
                                    raise ValueError ("Invalid method to update attributes, multiple time attribute.")
                
                else:
                    raise ValueError(f"Invalid max_count={ev.max_count} for event {ev.name}")
    
    def update_instance_attributes_mh(self, instance_inside, rng):
        log_q_forward = 0.0
        log_q_backward = 0.0

        for ev in self.list_event_spec:
            if ev.attributes_spec is not None:
                if ev.max_count == 1:
                    ev_name = ev.name
                    if instance_inside[ev_name]["indicator"] == 1:
                        for att in ev.attributes_spec:
                            if att.prob_model.method == "gibbs":
                                continue
                            elif att.prob_model.method == "mh":
                                current_value = instance_inside[ev_name][att.name]
                                new_value, log_q_f, log_q_b = att.next_proposal(current_value = current_value, instance = instance_inside, index = None, rng = rng)
                                instance_inside[ev_name][att.name] = new_value
                                log_q_forward += float(log_q_f)
                                log_q_backward += float(log_q_b)
                            else:
                                raise ValueError ("Invalid method to update attributes, single time attribute.")

                elif ev.max_count > 1:
                    for m in range(ev.max_count):
                        ev_name = f"{ev.name}_{m+1}"
                        if instance_inside[ev_name]["indicator"] == 1:
                            for att in ev.attributes_spec:
                                if att.prob_model.method == "gibbs":
                                    continue
                                elif att.prob_model.method == "mh":
                                    current_value = instance_inside[ev_name][att.name]
                                    new_value, log_q_f, log_q_b = att.next_proposal(current_value = current_value, instance = instance_inside, index = m, rng = rng)
                                    instance_inside[ev_name][att.name] = new_value
                                    log_q_forward += float(log_q_f)
                                    log_q_backward += float(log_q_b)
                                else:
                                    raise ValueError ("Invalid method to update attributes, multiple time attribute.")
                
                else:
                    raise ValueError(f"Invalid max_count={ev.max_count} for event {ev.name}")
                
        return log_q_forward, log_q_backward


    def store_sample(self, output_instance, instance_inside, idx):
        for k, ev in enumerate (self.list_event_spec):
            if ev.max_count == 1:
                ev_name = ev.name
                output_instance[ev_name]["main_start_date"]["chains"][idx] = instance_inside[ev_name]["main_start_date"]
                output_instance[ev_name]["main_duration"]["chains"][idx] = instance_inside[ev_name]["main_duration"]
                output_instance[ev_name]["gap_start_date"]["chains"][idx] = instance_inside[ev_name]["gap_start_date"]
                output_instance[ev_name]["gap_duration"]["chains"][idx] = instance_inside[ev_name]["gap_duration"]
                if ev.attributes_spec is not None : 
                    for att in ev.attributes_spec:
                        output_instance[ev_name][att.name]["chains"][idx] = instance_inside[ev_name][att.name]

            elif ev.max_count > 1 :
                for m in range (ev.max_count):
                    ev_name = f"{ev.name}_{m+1}"
                    output_instance[ev_name]["main_start_date"]["chains"][idx] = instance_inside[ev_name]["main_start_date"]
                    output_instance[ev_name]["main_duration"]["chains"][idx] = instance_inside[ev_name]["main_duration"]
                    output_instance[ev_name]["gap_start_date"]["chains"][idx] = instance_inside[ev_name]["gap_start_date"]
                    output_instance[ev_name]["gap_duration"]["chains"][idx] = instance_inside[ev_name]["gap_duration"]
                    if ev.attributes_spec is not None : 
                        for att in ev.attributes_spec:
                            output_instance[ev_name][att.name]["chains"][idx] = instance_inside[ev_name][att.name]

    def _run_single_chain_sampler(
        self,
        dob,
        lifespan,
        n_samples,
        n_gibbs=1000,
        burnin=1000,
        thin=1,
        seed=0,
        x0=None,
    ):
        output, accept_rate_dur, accept_rate_attr, A, b, fixed_idx = inside_dimension_hit_and_run(
                model=self,
                dob=dob,
                lifespan=lifespan,
                n_samples=n_samples,
                n_gibbs=n_gibbs,
                burnin=burnin,
                thin=thin,
                seed=seed,
                x0=x0,
            ) 

        return {
            "instance": output,
            "accept_rate_dur": accept_rate_dur,
            "accept_rate_attr": accept_rate_attr,
            "A": A,
            "b": b,
            "fixed_idx": fixed_idx,
        }
    
    def _run_multi_chain_sampler(
        self,
        dob,
        lifespan,
        n_samples,
        burnin,
        n_chains,
        n_gibbs=1000,
        thin=1,
        x0s=None,
    ):  
        return(run_multiple_inside_dimension(
            model=self,
            dob=dob,
            lifespan=lifespan,
            n_samples=n_samples,
            burnin=burnin,
            thin=thin,
            n_chains=n_chains,
            x0s=x0s,
            n_gibbs=n_gibbs,
        ))


    def sample(
        self,
        dob,
        lifespan,
        n_samples=1,
        n_gibbs=1000,
        burnin=1000,
        thin=1,
        seed=0,
        x0=None,
    ):
        """
        Generic sampling for a dimension conditional on existence.
        """

        if dob is None or lifespan is None:
            raise ValueError("Dimension sampling requires dob and lifespan from existence.")
        
        return self._run_single_chain_sampler(
            dob=dob,
            lifespan=lifespan,
            n_samples=n_samples,
            n_gibbs=n_gibbs,
            burnin=burnin,
            thin=thin,
            seed=seed,
            x0=x0,
        )
    
    def sample_diagnostics(
        self,
        dob,
        lifespan,
        n_samples=100,
        burnin=1000,
        n_chains=4,
        n_gibbs=1000,
        thin=1,
        x0s=None,
    ):
        
        if dob is None or lifespan is None:
            raise ValueError("Dimension sampling requires dob and lifespan from existence.")
        
        return self._run_multi_chain_sampler(
            dob=dob,
            lifespan=lifespan,
            n_samples=n_samples,
            burnin=burnin,
            n_chains=n_chains,
            n_gibbs=n_gibbs,
            thin=thin,
            x0s=x0s,
        )
    

class IndividualSpec:
    """
    Specification of an individual life-course model.

    This class defines the full generative specification for an
    individual, including dimension specifications, inter-event
    constraints, and the joint distribution governing durations.
    """
    def __init__(self, id, list_dim_spec, joint_dist_duration, list_inter_constraint_spec):
        self.id = id
        self.list_dim_spec = self._validate(list_dim_spec)
        self.joint_dist_duration = joint_dist_duration
        self.list_inter_constraint_spec = list_inter_constraint_spec

    def _validate(self, list_dim_spec):
        """
        Validates and completes the list of dimension specifications.

        Ensures that exactly one Existence dimension is present, that
        it is correctly typed, and that it contains a single Birth event.
        """
        # -------------------------------------------------------------
        # Case 1: dim_specs is None or empty
        # -------------------------------------------------------------
        if not list_dim_spec:
            print("WARNING: No dimension specified. Existence was automatically created.")
            list_dim_spec = [ExistenceDimensionSpec([], existence_density)]
        
        # -------------------------------------------------------------
        # Case 2: Find Existence dimensions
        # -------------------------------------------------------------
        existence_dims = [d for d in list_dim_spec if d.name == "Existence"]

        if len(existence_dims) == 0:
            print("WARNING: Existence was not specified. It has been created automatically.")
            existence = ExistenceDimensionSpec([], existence_density)
            list_dim_spec.append(existence)
            existence_dims = [existence]

        if len(existence_dims) > 1:
            raise ValueError("Existence dimension is specified multiple times.")
        
        # -------------------------------------------------------------
        # Case 3: Ensure correct type
        # -------------------------------------------------------------
        existence = existence_dims[0]

        if not isinstance(existence, ExistenceDimensionSpec):
            print(
                "WARNING: Existence must be specified with ExistenceDimensionSpec. "
                "It was automatically replaced."
            )
            list_dim_spec = [d for d in list_dim_spec if d is not existence]
            existence = ExistenceDimensionSpec([], existence_density)
            list_dim_spec.append(existence)

        # -------------------------------------------------------------
        # Case 4: Ensure exactly one Birth event
        # -------------------------------------------------------------
        events = existence.list_event_spec

        if len(events) == 0:
            print("WARNING: Existence has no event. Birth was created automatically.")
            events.append(ExistenceEventSpec([], existence_density))
        elif len(events) > 1:
            raise ValueError("Existence must contain exactly one event (Birth).")
        else : 
            if getattr(events[0], "name", None) != "Birth":
                print(
                    f"WARNING: Existence must contain Birth as its only event. "
                    f"'{events[0].name}' was specified instead. "
                    "The dimension specification has been automatically modified."
                )
                existence.list_event_spec = [ExistenceEventSpec([], existence_density)]
                
        return list_dim_spec
    
    def create_empty_instance(self):
        # create instance inside
        instance = {}
        for i, dim in enumerate (self.list_dim_spec):
            if dim.name != "Existence":
                instance[dim.name] = dim.create_empty_instance()
            else:
                birth_ev = dim.list_event_spec[0]
                birth_instance = birth_ev.create_empty_instance()
                instance[dim.name] = birth_instance
        
        return(instance)
    
    def create_empty_output_container(self, n_samples, instance_init):
        # create instance outputed
        instance = {}
        for i, dim in enumerate (self.list_dim_spec):
            if dim.name != "Existence":
                instance[dim.name] = dim.create_empty_output_container(n_samples, instance_init[dim.name])
            else:
                birth_ev = dim.list_event_spec[0]
                birth_instance = birth_ev.create_empty_output_container(n_samples)
                instance[dim.name] = birth_instance
        return(instance)
    
    def initialize_indicators(self, instance_birth_sampled, rng):
        lifespan = instance_birth_sampled["Existence"]["Birth"]["main_duration"]
        for i, dim in enumerate (self.list_dim_spec):
            if dim.name != "Existence" : #already sampled in the None instance so only sample the rest
                dim.initialize_indicators(instance_birth_sampled[dim.name], lifespan, rng = rng)

    def build_constraint_matrix(self, instance):
        BigM = None
        BigB = None
        lifespan = instance["Existence"]["Birth"]["main_duration"]

        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                A_dim, b_dim = dim.build_dimension_constraint_matrix(instance[dim.name], lifespan)
                if BigM is None:
                        BigM = A_dim
                        BigB = b_dim
                else:
                    BigM = block_diag(BigM, A_dim)
                    BigB = np.concatenate([BigB, b_dim])

        if self.list_inter_constraint_spec is not None :
            # append the rows as well : to think how to get the indices 
            for i in range (len(self.list_inter_constraint_spec)):
                # to do : check that this part works, but it should
                BigM, BigB = add_inter_event_constraints(BigM, BigB, self.list_inter_constraint_spec[i], self.list_dim_spec, instance) 

        return BigM, BigB
                
    def initialize_instance_attributes(self, instance_inside, rng):
        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                dim.initialize_instance_attributes(instance_inside[dim.name], rng = rng)

    def fill_durations_instance(self, duration_vector, instance):
        vector_position = 0 
        for i, dim in enumerate(self.list_dim_spec):
            current_dim = dim
            current_dim_name = dim.name
            if current_dim_name != "Existence":
                for k,ev in enumerate(current_dim.list_event_spec):
                    if ev.max_count == 1:
                        instance[current_dim_name][ev.name]["main_duration"] = duration_vector[vector_position]
                        instance[current_dim_name][ev.name]["gap_duration"] = duration_vector[vector_position + 1]
                        vector_position = vector_position + 2 #even if not present, we need to put +2 because otherwise index problem
                    elif ev.max_count > 1:
                        for z in range(ev.max_count):
                            ev_name = f"{ev.name}_{z + 1}"
                            instance[current_dim_name][ev_name]["main_duration"] = duration_vector[vector_position]
                            instance[current_dim_name][ev_name]["gap_duration"] = duration_vector[vector_position + 1]
                            
                            vector_position = vector_position + 2
        
        return(instance)
    
    def fill_terminal_event_duration(self, instance):
        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                dim.fill_terminal_event_duration(instance[dim.name])
    
    def fill_starting_dates(self, instance, dob):
        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                dim.fill_starting_dates_in_dimension(instance[dim.name], dob) 

    def update_instance_from_vector(self, x_current, instance_inside):
        dob = instance_inside["Existence"]["Birth"]["main_start_date"]
        self.fill_durations_instance(x_current, instance_inside)
        self.fill_terminal_event_duration(instance_inside)
        self.fill_starting_dates(instance_inside, dob)

    def exact_gibbs_update_indicators(self, instance_inside, rng):
        lifespan = instance_inside["Existence"]["Birth"]["main_duration"]
        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                if len(dim.list_event_spec) > 1:
                    for j in range (1, len(dim.list_event_spec)):
                        curr_event_spec = dim.list_event_spec[j]
                        curr_event_name = curr_event_spec.name # base name
                        curr_event_max_count = curr_event_spec.max_count
                        prev_event_spec = dim.list_event_spec[j-1]
                        prev_event_max_count = prev_event_spec.max_count
                        
                        if prev_event_max_count == 1:
                            prev_ev_name = prev_event_spec.name
                        elif prev_event_max_count > 1:
                            prev_ev_name = f"{prev_event_spec.name}_{1}"
                        
                        if instance_inside[dim.name][prev_ev_name]["indicator"] == 1: #if previous event happens
                            if curr_event_spec.constraint_spec is None :
                                n_ones_sampled = curr_event_spec.indic_prob_model.sampler_full_conditional(instance_inside, rng = rng)
                                n_ones = min(n_ones_sampled, curr_event_max_count)
                                if curr_event_max_count > 1:
                                    for m in range(curr_event_max_count):
                                        event_name_with_index = f"{curr_event_name}_{m + 1}"
                                        # If the indicator is within the sampled count, set it to 1, else 0
                                        instance_inside[dim.name][event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                                elif curr_event_max_count == 1:
                                    instance_inside[dim.name][curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                            else:
                                if curr_event_spec.constraint_spec.main_start_age_min is None : 
                                    n_ones_sampled = curr_event_spec.indic_prob_model.sampler_full_conditional(instance_inside, rng = rng)
                                    n_ones = min(n_ones_sampled, curr_event_max_count)
                                    if curr_event_max_count > 1:
                                        for m in range(curr_event_max_count):
                                            event_name_with_index = f"{curr_event_name}_{m + 1}"
                                            # If the indicator is within the sampled count, set it to 1, else 0
                                            instance_inside[dim.name][event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                                    elif curr_event_max_count == 1:
                                        instance_inside[dim.name][curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                                else:
                                    if lifespan > curr_event_spec.constraint_spec.main_start_age_min:
                                        n_ones_sampled = curr_event_spec.indic_prob_model.sampler_full_conditional(instance_inside, rng = rng)
                                        n_ones = min(n_ones_sampled, curr_event_max_count)
                                        if curr_event_max_count > 1:
                                            for m in range(curr_event_max_count):
                                                event_name_with_index = f"{curr_event_name}_{m + 1}"
                                                # If the indicator is within the sampled count, set it to 1, else 0
                                                instance_inside[dim.name][event_name_with_index]["indicator"] = 1 if m < n_ones else 0
                                        elif curr_event_max_count == 1:
                                            instance_inside[dim.name][curr_event_name]["indicator"] = min(1, n_ones) #ensures it doesn't go above 1
                                    else:
                                        n_ones_sampled = 0
                                        if curr_event_max_count > 1:
                                            for m in range(curr_event_max_count):
                                                event_name_with_index = f"{curr_event_name}_{m + 1}"
                                                # If the indicator is within the sampled count, set it to 1, else 0
                                                instance_inside[dim.name][event_name_with_index]["indicator"] = 1 if m < n_ones_sampled else 0
                                        elif curr_event_max_count == 1:
                                            instance_inside[dim.name][curr_event_name]["indicator"] = min(1, n_ones_sampled)
                        else:
                            n_ones_sampled = 0
                            if curr_event_max_count > 1:
                                for m in range(curr_event_max_count):
                                    event_name_with_index = f"{curr_event_name}_{m + 1}"
                                    # If the indicator is within the sampled count, set it to 1, else 0
                                    instance_inside[dim.name][event_name_with_index]["indicator"] = 1 if m < n_ones_sampled else 0
                            elif curr_event_max_count == 1:
                                instance_inside[dim.name][curr_event_name]["indicator"] = min(1, n_ones_sampled)


    def update_instance_attributes_gibbs(self, instance, rng):
        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                for ev in dim.list_event_spec:
                    if ev.attributes_spec is not None:
                        if ev.max_count == 1:
                            ev_name = ev.name
                            if instance[dim.name][ev_name]["indicator"] == 1:
                                for att in ev.attributes_spec :
                                    if att.prob_model.method == "gibbs":
                                        instance[dim.name][ev.name][att.name] = att.next_exact_conditional(instance = instance, index = None, rng = rng) 
                                    elif att.prob_model.method == "mh":
                                        continue
                                    else:
                                        raise ValueError("Invalid method for attributes sampling, single time attribute.")
                        elif ev.max_count > 1 :
                            for m in range (ev.max_count):
                                ev_name = f'{ev.name}_{m+1}'
                                if instance[dim.name][ev_name]["indicator"] == 1:
                                    for att in ev.attributes_spec :
                                        if att.prob_model.method == "gibbs" :
                                            instance[dim.name][ev_name][att.name] = att.next_exact_conditional(instance = instance, index = m, rng = rng)
                                        elif att.prob_model.method == "mh":
                                            continue
                                        else:
                                            raise ValueError("Invalid method for attributes sampling, multiple time attribute.")
                        else:
                            raise ValueError(f"Invalid max_count={ev.max_count} for event {ev.name}")
                        
    def update_instance_attributes_mh(self, instance, rng):
        log_q_forward = 0.0
        log_q_backward = 0.0

        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                for ev in dim.list_event_spec:
                    if ev.attributes_spec is not None:
                        if ev.max_count == 1:
                            ev_name = ev.name
                            if instance[dim.name][ev_name]["indicator"] == 1:
                                for att in ev.attributes_spec :
                                    if att.prob_model.method == "gibbs":
                                        continue
                                    elif att.prob_model.method == "mh":
                                        current_value = instance[dim.name][ev_name][att.name]
                                        new_value, log_q_f, log_q_b = att.next_proposal(current_value = current_value, instance = instance, index = None, rng = rng)
                                        instance[dim.name][ev_name][att.name] = new_value
                                        log_q_forward += float(log_q_f)
                                        log_q_backward += float(log_q_b)
                                    else:
                                        raise ValueError("Invalid method for attributes sampling, single time attribute.")
                        elif ev.max_count > 1 :
                            for m in range (ev.max_count):
                                ev_name = f'{ev.name}_{m+1}'
                                if instance[dim.name][ev_name]["indicator"] == 1:
                                    for att in ev.attributes_spec :
                                        if att.prob_model.method == "gibbs" :
                                            continue
                                        elif att.prob_model.method == "mh":
                                            current_value = instance[dim.name][ev_name][att.name]
                                            new_value, log_q_f, log_q_b = att.next_proposal(current_value = current_value, instance = instance, index = m, rng = rng)
                                            instance[dim.name][ev_name][att.name] = new_value
                                            log_q_forward += float(log_q_f)
                                            log_q_backward += float(log_q_b)
                                        else:
                                            raise ValueError("Invalid method for attributes sampling, multiple time attribute.")
                        else:
                            raise ValueError(f"Invalid max_count={ev.max_count} for event {ev.name}")
        
        return log_q_forward, log_q_backward

            
    def store_sample(self, output_instance, instance_inside, idx):
        for i, dim in enumerate(self.list_dim_spec):
            if dim.name != "Existence":
                for k, ev in enumerate (dim.list_event_spec):
                    if ev.max_count == 1:
                        ev_name = ev.name
                        output_instance[dim.name][ev_name]["main_start_date"]["chains"][idx] = instance_inside[dim.name][ev_name]["main_start_date"]
                        output_instance[dim.name][ev_name]["main_duration"]["chains"][idx] = instance_inside[dim.name][ev_name]["main_duration"]
                        output_instance[dim.name][ev_name]["gap_start_date"]["chains"][idx] = instance_inside[dim.name][ev_name]["gap_start_date"]
                        output_instance[dim.name][ev_name]["gap_duration"]["chains"][idx] = instance_inside[dim.name][ev_name]["gap_duration"]
                        if ev.attributes_spec is not None : 
                            for att in ev.attributes_spec:
                                output_instance[dim.name][ev_name][att.name]["chains"][idx] = instance_inside[dim.name][ev_name][att.name]

                    elif ev.max_count > 1 :
                        for m in range (ev.max_count):
                            ev_name = f"{ev.name}_{m+1}"
                            output_instance[dim.name][ev_name]["main_start_date"]["chains"][idx] = instance_inside[dim.name][ev_name]["main_start_date"]
                            output_instance[dim.name][ev_name]["main_duration"]["chains"][idx] = instance_inside[dim.name][ev_name]["main_duration"]
                            output_instance[dim.name][ev_name]["gap_start_date"]["chains"][idx] = instance_inside[dim.name][ev_name]["gap_start_date"]
                            output_instance[dim.name][ev_name]["gap_duration"]["chains"][idx] = instance_inside[dim.name][ev_name]["gap_duration"]
                            if ev.attributes_spec is not None : 
                                for att in ev.attributes_spec:
                                    output_instance[dim.name][ev_name][att.name]["chains"][idx] = instance_inside[dim.name][ev_name][att.name]
   
    
    def _build_trajectory_from_instance(self, dim, instance_res):
        list_events = []
        list_indic = []

        for ev in dim.list_event_spec:

            if ev.max_count == 1:
                names = [ev.name]
            else:
                names = [f"{ev.name}_{i+1}" for i in range(ev.max_count)]

            for ev_name in names:

                indicator = instance_res[ev_name]["indicator"]

                main_start = instance_res[ev_name]["main_start_date"]["chains"][0]
                gap_start = instance_res[ev_name]["gap_start_date"]["chains"][0]
                main_dur = instance_res[ev_name]["main_duration"]["chains"][0]
                gap_dur = instance_res[ev_name]["gap_duration"]["chains"][0]

                attributes = []
                if ev.attributes_spec is not None:
                    for attr in ev.attributes_spec:
                        attributes.append(
                            Attribute(
                                attr.name,
                                attr.type,
                                instance_res[ev_name][attr.name]["chains"][0],
                            )
                        )

                list_events.append(
                    Event(
                        ev_name,
                        ev.type,
                        main_start,
                        gap_start,
                        main_dur + gap_dur,
                        main_dur,
                        gap_dur,
                        attributes,
                    )
                )

                list_indic.append(indicator)

        return Trajectory(dim.name, list_events, list_indic)

    def sample(
        self,
        dimension_mode="joint",   # "independent" or "joint"
        n_gibbs=1000,
        burnin=1000,
        burnin_existence=1000,
        thin=1,
        seed=0,
        x0=None,
    ):
        """
        Unified sampling interface for an individual.

        Parameters
        ----------
        dimension_mode : str
        """

        trajectories = []

        # --------------------------------------------------
        # 1. SAMPLE EXISTENCE
        # --------------------------------------------------
        d0 = next((d for d in self.list_dim_spec if d.name == "Existence"), None)

        existence_trajectory = d0.sample_from_prior(
            burnin=burnin_existence,
            seed=seed,
        )

        dob = existence_trajectory.list_events[0].main_start_date
        lifespan = existence_trajectory.list_events[0].main_duration

        trajectories.append(existence_trajectory)

        # --------------------------------------------------
        # 2. SAMPLE OTHER DIMENSIONS
        # --------------------------------------------------
        if dimension_mode == "independent":
            return self._sample_dimensions_independent(
                trajectories,
                dob,
                lifespan,
                n_gibbs,
                burnin,
                seed,
                x0,
            )

        elif dimension_mode == "joint":
            return self._sample_dimensions_joint(
                trajectories,
                existence_trajectory,
                n_gibbs,
                burnin,
                thin,
                seed,
                x0,
            )

        else:
            raise ValueError(f"Unknown dimension_mode: {dimension_mode}")
        
    def _sample_dimensions_independent(
        self,
        trajectories,
        dob,
        lifespan,
        n_gibbs,
        burnin,
        seed,
        x0,
    ):

        for dim in self.list_dim_spec:
            if dim.name == "Existence":
                continue

            dim_res = dim.sample(
                dob=dob,
                lifespan=lifespan,
                n_gibbs=n_gibbs,
                burnin=burnin,
                seed=seed,
                x0=x0,
            )

            trajectory = self._build_trajectory_from_instance(
                dim,
                dim_res["instance"]
            )

            trajectories.append(trajectory)

        return Individual(self.id, trajectories)
    
    def _sample_dimensions_joint(
        self,
        trajectories,
        existence_trajectory,
        n_gibbs,
        burnin,
        thin,
        seed,
        x0,
    ):

        birth = existence_trajectory.list_events[0]

        instance_birth = {
            "Birth": {
                "main_duration": birth.main_duration,
                "main_start_date": birth.main_start_date,
                "gap_duration": birth.gap_duration,
                "gap_start_date": birth.gap_start_date,
            }
        }

        # add attributes if any
        if birth.attributes is not None:
            for att in birth.attributes:
                instance_birth["Birth"][att.name] = att.value

        instance_res, *_ = all_dim_hit_and_run(
            self,
            instance_birth,
            n_samples=1,
            n_gibbs=n_gibbs,
            burnin=burnin,
            thin=thin,
            seed=seed,
            x0=x0,
        )

        for dim in self.list_dim_spec:
            if dim.name == "Existence":
                continue

            trajectory = self._build_trajectory_from_instance(
                dim,
                instance_res[dim.name]
            )

            trajectories.append(trajectory)

        return Individual(self.id, trajectories)
    

class PopulationSpec:

    def __init__(self, size, list_dim_spec, joint_dist_duration, list_inter_constraint_spec):
        self.size = size
        self.list_dim_spec = list_dim_spec
        self.joint_dist_duration = joint_dist_duration
        self.list_inter_constraint_spec = list_inter_constraint_spec

    # --------------------------------------------------
    # Core sampling
    # --------------------------------------------------

    def sample(
        self,
        dimension_mode="joint",
        n_gibbs=1000,
        burnin=1000,
        burnin_existence=1000,
        thin=1,
        seed=0,
        x0=None,
    ):
        """
        Sample a full population.

        Each individual is sampled independently.
        """

        pop = []

        indiv_spec  = IndividualSpec(
                None,
                self.list_dim_spec,
                self.joint_dist_duration,
                self.list_inter_constraint_spec
            )


        for i in range(self.size):

            indiv_spec.id = i
            
            indiv = indiv_spec.sample(
                dimension_mode=dimension_mode,
                n_gibbs=n_gibbs,
                burnin=burnin,
                burnin_existence=burnin_existence,
                thin=thin,
                seed=seed + i,   # IMPORTANT: avoid identical chains
                x0=x0,
            )

            pop.append(indiv)

        return Population(pop)

    def sample_parallel(
        self,
        dimension_mode="joint",
        n_gibbs=1000,
        burnin=1000,
        burnin_existence=1000,
        thin=1,
        seed=0,
        x0=None,
        n_jobs=1,
    ):
        args_list = [
            (
                i,
                self.list_dim_spec,
                self.joint_dist_duration,
                self.list_inter_constraint_spec,
                dimension_mode,
                n_gibbs,
                burnin,
                burnin_existence,
                thin,
                seed + i,
                x0,
            )
            for i in range(self.size)
        ]

        if n_jobs == 1:
            pop = [_sample_one_individual(args) for args in args_list]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                pop = list(ex.map(_sample_one_individual, args_list))

        return Population(pop)
    
    
    def sample_parallel_to_disk(
        self,
        dimension_mode="joint",
        n_gibbs=1000,
        burnin=1000,
        burnin_existence=1000,
        thin=1,
        seed=0,
        x0=None,
        n_workers=None,
        output_dir="population_output",
    ):
        """
        SCITAS-friendly population sampling.

        - No population stored in memory
        - Each individual saved as CSV
        - Restartable
        """

        if n_workers is None:
            n_workers = os.cpu_count()

        os.makedirs(output_dir, exist_ok=True)

        args_list = [
            (
                i,
                self.list_dim_spec,
                self.joint_dist_duration,
                self.list_inter_constraint_spec,
                dimension_mode,
                n_gibbs,
                burnin,
                burnin_existence,
                thin,
                seed + i,
                x0,
                output_dir,
            )
            for i in range(self.size)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for _ in ex.map(_sample_and_save_one, args_list):
                pass



def _sample_and_save_one(args):
    (
        i,
        list_dim_spec,
        joint_dist_duration,
        list_inter_constraint_spec,
        dimension_mode,
        n_gibbs,
        burnin,
        burnin_existence,
        thin,
        seed,
        x0,
        output_dir,
    ) = args

    filepath = os.path.join(output_dir, f"individual_{i}.csv")

    # --------------------------------------------------
    # Skip if already computed (CRUCIAL for SCITAS)
    # --------------------------------------------------
    if os.path.exists(filepath):
        return i

    indiv_spec = IndividualSpec(
        id=i,
        list_dim_spec=list_dim_spec,
        joint_dist_duration=joint_dist_duration,
        list_inter_constraint_spec=list_inter_constraint_spec,
    )

    indiv = indiv_spec.sample(
        dimension_mode=dimension_mode,
        n_gibbs=n_gibbs,
        burnin=burnin,
        burnin_existence=burnin_existence,
        thin=thin,
        seed=seed,
        x0=x0,
    )

    df = indiv.to_dataframe()

    df.to_csv(filepath, index=False)

    return i





def _sample_one_individual(args):
    (
        i,
        list_dim_spec,
        joint_dist_duration,
        list_inter_constraint_spec,
        dimension_mode,
        n_gibbs,
        burnin,
        burnin_existence,
        thin,
        seed,
        x0,
    ) = args

    indiv_spec = IndividualSpec(
        id=i,
        list_dim_spec=list_dim_spec,
        joint_dist_duration=joint_dist_duration,
        list_inter_constraint_spec=list_inter_constraint_spec,
    )

    indiv = indiv_spec.sample(
        dimension_mode=dimension_mode,
        n_gibbs=n_gibbs,
        burnin=burnin,
        burnin_existence=burnin_existence,
        thin=thin,
        seed=seed,
        x0=x0,
    )

    return indiv