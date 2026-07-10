"""Specifications of constraints

Baud Candice
Fri July 10 10:33:00 2026
"""


### Event Constraint
class EventTemporalConstraintsSpec:
    """
    Specification of temporal constraints for an event.

    This class defines lower and upper bounds on start ages and
    durations for both main and gap components of an event.
    A value set to None indicates that no constraint is imposed.
    A minimal duration must be understood as a minimal duration to access the next state
    Ex : in the case of education : cannot access certain education levels if the previous ones haven't been "completed"
    """
    def __init__(self, main_start_age_min, main_start_age_max, 
                main_min_duration, main_max_duration,
                gap_start_age_min, gap_start_age_max, 
                gap_min_duration, gap_max_duration, max_duration):
        self.main_start_age_min = main_start_age_min
        self.main_start_age_max = main_start_age_max
        self.main_min_duration = main_min_duration
        self.gap_min_duration = gap_min_duration
        self.main_max_duration = main_max_duration
        self.gap_start_age_min = gap_start_age_min
        self.gap_start_age_max = gap_start_age_max
        self.gap_max_duration = gap_max_duration
        self.max_duration = max_duration


#### Inter event constraints
class DurationPath:
    """
    Class to specify the path to a duration.
    """
    def __init__(self, dimension_name, event_name, event_number, duration_type, no_event):
        self.dimension_name = dimension_name
        self.event_name = event_name
        self.event_number = event_number
        self.duration_type = duration_type
        self.no_event = no_event # bool

    @classmethod
    def NoEvent(
        cls,
        dimension_name
    ):
        """
        Specification of the no event path

        - dimension_name is specified
        - event_name doesn't even have to be specified because there is only one No Event per dimension
        - event_number is none because the no event can not be a multiple happening event
        - duration_type is none because for the no event the total duration is the main duration, gap duration is zero
        - no_event = 1 by default bool
        """
        return cls(
            dimension_name=dimension_name,
            event_name=None,
            event_number=None,
            duration_type=None,
            no_event=1
        )
    
    @classmethod
    def MultipleTimeEvent(
        cls,
        dimension_name,
        event_name,
        event_number,
        duration_type
    ):
        """
        Specification of a multiple event path

        - dimension_name is specified
        - event_name is specified
        - event_number is specified for multiple happening events
        - duration_type is specified because it could be gap or duration
        - no_event = 0 by default bool
        """
        return cls(
            dimension_name=dimension_name,
            event_name=event_name,
            event_number=event_number,
            duration_type=duration_type,
            no_event=0
        )

    @classmethod
    def SingleTimeEvent(
        cls,
        dimension_name,
        event_name,
        duration_type
    ):
        """
        Specification of a single time event path

        - dimension_name is specified
        - event_name is specified
        - event_number is fixed at None, because for single time events it's always 1 so no need to specify
        - duration_type is specified because it could be gap or duration
        - no_event = 0 by default bool
        """
        return cls(
            dimension_name=dimension_name,
            event_name=event_name,
            event_number=None,
            duration_type=duration_type,
            no_event=0
        )


class DurationLinearConstraintsSpec:
    """
    Specification of linear constraints between event durations.

    This class defines linear relationships between durations of
    different events, possibly across dimensions, of the form

        path_1 <= sum_k a_k · path_k + b

    or, if equality is enforced, path_1 = sum_k a_k · path_k + b.
    """
    def __init__(self, path_1, list_paths, a_list, b, eq_bool):
        """
        Parameters
        ----------
        path_1 : DurationPath
            Target duration path defined as
            dimension, event_name, event_number, duration type, bool no event.
        list_paths : array of DurationPath 
            List of duration paths appearing in the linear expression.
        a_list : list
            Coefficients associated with each path in list_paths.
        b : float
            Constant term of the linear constraint.
        eq_bool : int
            If 0, the constraint is an inequality (<=).
            If 1, the constraint is an equality encoded as two inequalities.
        """
        self.path_1 = path_1
        self.list_paths, self.a_list = self._validate(list_paths, a_list)
        self.b = b
        self.eq_bool = eq_bool

    def _validate(self, list_paths, a_list):
        """
        Validates consistency between paths and coefficients.
        """
        if list_paths is not None and a_list is None:
            raise ValueError("Coefficients must be provided for all paths.")
        if a_list is not None and list_paths is None:
            raise ValueError("Paths must be provided for all coefficients.")
        if list_paths is not None and len(a_list) != len(list_paths):
            raise ValueError(
                "Each path in the linear constraint must have a corresponding coefficient."
            )
        return list_paths, a_list
