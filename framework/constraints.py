"""Building the constraints

Baud Candice
Fri Feb 13 17:42:00 2026
"""

import numpy as np


def extract_event_indicators_inside_dimension(instance):
    """
    Extracts the indicator vector for a given dimension from an instance.

    The indicator specifies which events in the dimension are active
    and should therefore be considered in the constraint system.
    """
    res = []
    for event, event_data in instance.items():
        res.append(event_data["indicator"])
    return res


def build_event_constraint_rows(constraint_spec, n_events_dim, event_number, indicator_list, lifespan):
    """
    Builds the constraint rows associated with a single event occurrence.

    This function generates linear inequality constraints for the main
    and gap durations of an event, conditional on its activation indicator
    and user-specified temporal constraints.
    """

    # if the event has indicator zero : we constraint durations to zero
    # list to store rows
    A_matrix = []
    b = []

    if indicator_list[event_number] == 0:
        #constrain the duration to zero (upper)
        row = np.zeros(2*n_events_dim)
        row[2*event_number] = 1 #main
        A_matrix.append(row.copy())
        b.append(0) # main constrained to zero

        row = np.zeros(2*n_events_dim)
        row[2*event_number+1] = 1 #gap
        A_matrix.append(row.copy())
        b.append(0) #gap constrained to zero
    else :
        # ------------------------------------------------------------------
        # 1. Lifespan constraints (duration bounds)
        # ------------------------------------------------------------------
        row = np.zeros(2*n_events_dim)
        row[2*event_number] = 1 #main : should be one
        A_matrix.append(row.copy())
        b.append(lifespan)

        row = np.zeros(2*n_events_dim)
        row[2*event_number+1] = 1 #gap : should be one as well
        A_matrix.append(row.copy())
        b.append(lifespan)

    # ------------------------------------------------------------------
    # 2. Main durations must be positive
    # ------------------------------------------------------------------
    row = np.zeros(2*n_events_dim)
    row[2*event_number] = -1 #duration positive
    A_matrix.append(row.copy())
    b.append(0)

    # ------------------------------------------------------------------
    # 2. Gap durations must be positive
    # ------------------------------------------------------------------
    row = np.zeros(2*n_events_dim)
    row[2*event_number+1] = -1 #gap positive
    A_matrix.append(row.copy())
    b.append(0)

    if indicator_list[event_number] == 0: #if event doesn't happen : no more constraints
        return(A_matrix, b)  
    else : #else : add more constraints
        # ------------------------------------------------------------------
        # 3. User-specified constraints: MAIN EVENT
        # ------------------------------------------------------------------
        if constraint_spec is not None : 
            if constraint_spec.main_start_age_min is not None:
                row = np.zeros(2*n_events_dim)
                for i in range(event_number):
                    if indicator_list[i] == 1:
                        row[2*i]   = -indicator_list[i]
                        row[2*i+1] = -indicator_list[i]
                A_matrix.append(row.copy())
                b.append(-constraint_spec.main_start_age_min)

            if constraint_spec.main_start_age_max is not None:
                row = np.zeros(2*n_events_dim)
                for i in range(event_number):
                    if indicator_list[i] != 0:
                        row[2*i]   = indicator_list[i]
                        row[2*i+1] = indicator_list[i]
                A_matrix.append(row.copy())
                b.append(constraint_spec.main_start_age_max)

            if constraint_spec.main_max_duration is not None:
                row = np.zeros(2*n_events_dim)
                if indicator_list[event_number]!=0:
                    row[2*event_number] = indicator_list[event_number]
                A_matrix.append(row.copy())
                b.append(constraint_spec.main_max_duration)

            # ------------------------------------------------------------------
            # 4. GAP constraints
            # ------------------------------------------------------------------
            if constraint_spec.gap_start_age_min is not None:
                row = np.zeros(2*n_events_dim)
                row[0] = -1
                row[1] = -1
                for i in range (1, event_number):
                    if indicator_list[i] == 1:
                        row[2*i] = -1
                        row[2*i + 1] = -1
                row[2*event_number] = -1
                A_matrix.append(row.copy())
                b.append(-constraint_spec.gap_start_age_min)

            if constraint_spec.gap_start_age_max is not None:
                row = np.zeros(2*n_events_dim)
                row[0] = 1
                row[1] = 1
                for i in range (1, event_number):
                    if indicator_list[i] == 1:
                        row[2*i] = 1
                        row[2*i + 1] = 1
                row[2*event_number] = 1
                A_matrix.append(row.copy())
                b.append(constraint_spec.gap_start_age_max)

            if constraint_spec.gap_max_duration is not None:
                row = np.zeros(2*n_events_dim)
                if indicator_list[event_number]!=0:
                    row[2*event_number+1] = indicator_list[event_number]
                A_matrix.append(row.copy())
                b.append(constraint_spec.gap_max_duration)

            # ------------------------------------------------------------------
            # 5. TOTAL duration constraint
            # ------------------------------------------------------------------
            if constraint_spec.max_duration is not None:
                row = np.zeros(2*n_events_dim)
                if indicator_list[event_number]!=0:
                    row[2*event_number]   = indicator_list[event_number]
                    row[2*event_number+1] = indicator_list[event_number]
                A_matrix.append(row.copy())
                b.append(constraint_spec.max_duration)
            
            # ------------------------------------------------------------------
            # 5. Saturation constraint : main
            # ------------------------------------------------------------------
            if constraint_spec.main_min_duration is not None : #if there is a saturation constraint on the main
                # check if the event is a terminal event or not : 
                # either last possible event,
                # either other events after have an indicator zero 
                terminal_bool = 0 
                if event_number == n_events_dim - 1:
                    terminal_bool = 1
                else :
                    if indicator_list[event_number + 1] == 0:
                        terminal_bool = 1

                # if not a terminal event : saturate the duration 
                if terminal_bool == 0 :
                    row = np.zeros(2*n_events_dim)
                    if indicator_list[event_number]!=0:
                        row[2*event_number]   = - indicator_list[event_number]
                    A_matrix.append(row.copy())
                    b.append(- constraint_spec.main_min_duration)

            # ------------------------------------------------------------------
            # 5. Saturation constraint : gap
            # ------------------------------------------------------------------
            if constraint_spec.gap_min_duration is not None : #if there is a saturation constraint on the gap
                # check if the event is a terminal event or not : 
                # either last possible event,
                # either other events after have an indicator zero 
                terminal_bool = 0 
                if event_number == n_events_dim - 1:
                    terminal_bool = 1
                else :
                    if indicator_list[event_number + 1] == 0:
                        terminal_bool = 1

                # if not a terminal event : saturate the duration 
                if terminal_bool == 0 :
                    row = np.zeros(2*n_events_dim)
                    if indicator_list[event_number]!=0:
                        row[2*event_number+1]   = - indicator_list[event_number]
                    A_matrix.append(row.copy())
                    b.append(- constraint_spec.gap_min_duration)

        # convert to numpy
        A_matrix = np.vstack(A_matrix)
        b = np.array(b)

        return A_matrix, b


def get_duration_variable_index(path, list_dim_specs):
    """
    Computes the column index of an event duration variable in the global matrix.

    This function maps a (dimension, event, duration type, occurrence index)
    path to its corresponding position in the full constraint matrix.
    """
    index_in_dim, index_dim = get_index_in_dim(path, list_dim_specs) # gives the index of the event in the dimension

    n_before_dim = 0
    for k in range (index_dim):
        current_dim = list_dim_specs[k]
        if current_dim.name != "Existence":
            curr_ev_list = current_dim.list_event_spec
            for i in range(len(curr_ev_list)):
                curr_ev = curr_ev_list[i]
                n_before_dim = n_before_dim + 2*curr_ev.max_count

    ev_number_total = index_in_dim + n_before_dim
    return(ev_number_total)


def get_index_in_dim(path, list_dim_specs):
    """
    Computes the index of an event duration variable in its dimension matrix. 
    """
    # FUNCTION TESTED, no problems found
    if path.dimension_name == "Existence":
        raise ValueError("It is not possible to constrain Existence duration to other events durations.") 
    else :
        path_no_event_bool = path.no_event  # if = 1 : no event, else : event
        if path_no_event_bool == 1:
            # no event
            dim_name = path.dimension_name
            index_dim = next(
                i for i, dim in enumerate(list_dim_specs)
                if dim.name == dim_name
                )
            index_in_dim = 0 
            return(index_in_dim, index_dim) #it can only be the main duration that is constrained 
        elif path_no_event_bool == 0:
            # event
            dim_name = path.dimension_name
            ev_base_name = path.event_name
            ev_number = path.event_number
            duration_type = path.duration_type
            
            # index of dimension
            index_dim = next(
                i for i, dim in enumerate(list_dim_specs)
                if dim.name == dim_name
                )

            # index of the event in the dimension
            index_event = next(
                i for i, ev in enumerate(list_dim_specs[index_dim].list_event_spec)
                if ev.name == ev_base_name
                )
            
            # compute the number of previous events 
            count = 0
            for i in range (index_event):
                ev_curr_spec = list_dim_specs[index_dim].list_event_spec[i]
                ev_max_count = ev_curr_spec.max_count
                count = count + ev_max_count

            # if the event of interest is a multiple occuring event, need to add the previous occurences
            ev_spec = list_dim_specs[index_dim].list_event_spec[index_event]
            max_count_ev = ev_spec.max_count

            if max_count_ev > 1:
                # need to account for it 
                if ev_number is None:
                    raise ValueError("For a multiple time event, the event number must be specified.")
                elif ev_number < 1:
                    raise ValueError("Event number must be greater than 0.")
                else:
                    count = count + ev_number - 1 #à checker si c'est - 1 ou pas 
            
            count = count*2
            if duration_type == "gap_duration":
                count += 1
            else :
                if duration_type != "main_duration":
                    raise ValueError(f"The duration can only be a main duration or a gap duration but {duration_type} was specified.")

            return(count, index_dim)

        else: 
            raise ValueError(f"The NoEvent bool can only take values 0 or 1, not {path_no_event_bool}.")



def add_inter_event_constraints(A, b, linear_constraintspec, list_dim_specs, instance):
    """
    Adds inter-event linear constraints to the global constraint system.

    These constraints link durations of events across different dimensions
    or trajectories, and are only enforced if the constrained event is active.

    Parameters
    ----------
    A : ndarray
        matrix containing the constraints for each event
    b : ndarray
        corresponding vector for constraints
    linear_constraintspec : 
        specified constraint between durations 
    list_dim_specs:
        list of dimensions specifications
    instance:
        current state of the individual
    """

    path_constrained = linear_constraintspec.path_1
    path_constrained_dim_name = path_constrained.dimension_name
    path_constrained_ev_base_name = path_constrained.event_name
    # if no event : event_number will be None
    # if single time event : event_number will be None 
    # if multiple time event : event_number will be specified 
    if path_constrained.event_number is None :
        path_constrained_ev_name = path_constrained_ev_base_name
    else:
        if path_constrained.event_number < 1 :
            raise ValueError("Event number must be greater than 0.")
        else :
            path_constrained_ev_name = f"{path_constrained_ev_base_name}_{path_constrained.event_number}"

    # if the constrained event doesn't happen, don't put any constraint
    if instance[path_constrained_dim_name][path_constrained_ev_name]["indicator"] == 0:
        return(A,b)
    
    # otherwise put the constraint 
    else :
        # constrained quantity index in the big matrix
        index_path_constrained = get_duration_variable_index(path_constrained, list_dim_specs)

        # indices of other quantities constrained in the big matrix 
        constraining_paths = linear_constraintspec.list_paths
        if len(constraining_paths) == 0:
            # if there are no constraining paths given
            print("WARNING : No constraining path specified, no constraint will be added.")
            return(A,b)
        else:
            list_constraining_indices = []
            n_constraining = len(constraining_paths)
            for i in range (n_constraining):
                index_constraining = get_duration_variable_index(constraining_paths[i], list_dim_specs)
                list_constraining_indices.append(index_constraining)
            
            a_list = linear_constraintspec.a_list
            b_coef = linear_constraintspec.b

            row = np.zeros(A.shape[1])

            row[index_path_constrained] = 1
            for k in range (n_constraining):
                row[list_constraining_indices[k]] = - a_list[k]
            A = np.vstack([A, row])
            b = np.append(b, b_coef)

            if linear_constraintspec.eq_bool == 1 :
                #if equality : add a second row
                row = np.zeros(A.shape[1])
                row[index_path_constrained] = -1
                for k in range (n_constraining):
                    row[list_constraining_indices[k]] = a_list[k]
            
                A = np.vstack([A, row])
                b = np.append(b, -b_coef)

            return(A,b)  
        