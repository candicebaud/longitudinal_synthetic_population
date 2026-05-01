"""Data classes

Baud Candice
Fri Feb 13 17:42:00 2026
"""


import pandas as pd 
import numpy as np

#### ProbabilisticModel
class ProbabilisticModel:
    def __init__(self, sampler_full_conditional, sampler_proposal, evaluation_proposal, sampler_initial, method):
        """Definition of what is needed to sample each variable
        
        Parameters
        ----------
        sampler_full_conditional : samples from the exact conditional (exact gibbs sampler)
        sampler_proposal : proposal to sample a value, NOT necessarily from exact conditional (MH)
        evaluation_proposal : evaluate the proposal density for MH
        sampler_initial : samples an initial value in absence of previous value
        method : how to sample ("gibbs", "mh")

        Notes 
        -----
        Note that the user doesn't need to define all the quantities. If the user
        knows the exact conditional, doesn't need to specify a proposal. However, if he
        doesn't know the exact conditional, he can specify how to sample from a proposal.
        This enables to use different sampling techniques. 
        """
        self.sampler_full_conditional = sampler_full_conditional
        self.sampler_proposal = sampler_proposal
        self.evaluation_proposal = evaluation_proposal
        self.sampler_initial = sampler_initial
        self.method = method

    @classmethod
    def Gibbs(
        cls,
        sampler_full_conditional,
        sampler_initial
    ):
        
        return cls(
            sampler_full_conditional=sampler_full_conditional,
            sampler_initial = sampler_initial,
            sampler_proposal = None,
            evaluation_proposal = None,
            method = "gibbs"
        )
    
    @classmethod
    def Proposal(
        cls,
        sampler_proposal,
        evaluation_proposal,
        sampler_initial
    ):
        
        return cls(
            sampler_full_conditional=None,
            sampler_initial = sampler_initial,
            sampler_proposal = sampler_proposal,
            evaluation_proposal = evaluation_proposal,
            method = "mh"
        )
    

class Attribute:
    """
    Represents an attribute attached to an event.

    Attributes are used to store additional information describing
    events, such as categorical, spatial, or temporal properties.
    """
    def __init__(self, name, type, value):
        self.name = name
        self.type = type  # categorical, spatial, temporal, etc.
        self.value = value

class Event:
    """
    Represents an event occurring along a trajectory.

    An event is characterized by its timing, duration, and a set
    of associated attributes.
    """
    def __init__(self, name, type, main_start_date, gap_start_date, duration, main_duration, gap_duration, attributes):
        self.name = name
        self.type = type  # e.g., No Event or Event, used for trajectory consistency checks
        self.main_start_date = main_start_date
        self.gap_start_date = gap_start_date
        self.duration = duration
        self.main_duration = main_duration
        self.gap_duration = gap_duration
        self.attributes = attributes

class Trajectory:
    """
    Represents the evolution of events along a given dimension.

    A trajectory contains an ordered list of events and corresponding
    indicators specifying which events effectively occur.
    """
    def __init__(self, dim_name, list_events, list_indic):
        self.dim_name = dim_name
        self.list_events = list_events
        self.list_indic = list_indic


class Individual:
    """
    Represents an individual described by multiple trajectories.

    Each trajectory corresponds to a specific dimension of the
    individual's life course.
    """
    def __init__(self, id, trajectories):
        self.id = id
        self.trajectories = trajectories


    def to_dataframe(self):
        """
        Converts the individual's realized events into a pandas DataFrame.
        Only events with an active indicator are included.
        """

        base_cols = [
            "id", "dim_name", "event_name", "event_type",
            "main_start_date", "gap_start_date",
            "duration", "main_duration", "gap_duration"
        ]

        attribute_names = sorted({
            attr.name
            for traj in self.trajectories
            for ev in traj.list_events
            if ev.attributes
            for attr in ev.attributes
        })

        rows = []

        for traj in self.trajectories:
            dim_name = traj.dim_name

            for ev, indic in zip(traj.list_events, traj.list_indic):

                if indic != 1:
                    continue

                row = {
                    "id": self.id,
                    "dim_name": dim_name,
                    "event_name": ev.name,
                    "event_type": ev.type,
                    "main_start_date": ev.main_start_date,
                    "gap_start_date": ev.gap_start_date,
                    "duration": ev.duration,
                    "main_duration": ev.main_duration,
                    "gap_duration": ev.gap_duration,
                }

                if ev.attributes:
                    for attr in ev.attributes:
                        row[attr.name] = attr.value

                rows.append(row)

        df = pd.DataFrame(rows)

        # ensure attribute columns exist
        for col in attribute_names:
            if col not in df.columns:
                df[col] = np.nan

        return df[base_cols + attribute_names]


class Population:
    """
    Represents a collection of individuals.

    Provides utilities to aggregate individual-level results
    at the population level.
    """
    def __init__(self, list_individuals):
        self.list_individuals = list_individuals

    def to_dataframe(self):
        """
        Aggregates all individuals into a single pandas DataFrame.
        """
        return pd.concat(
            (ind.to_dataframe() for ind in self.list_individuals),
            ignore_index=True
        )