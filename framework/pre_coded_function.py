"""Pre coded functions

Baud Candice
Fri July 10 10:33:00 2026
"""

from classes_data import *
import numpy as np


#### No Events
def sampler_indic_no_event(instance = None, rng = 0):
    return(1)

indic_prob_model_no_event = ProbabilisticModel(sampler_indic_no_event, sampler_indic_no_event, sampler_indic_no_event, sampler_indic_no_event, method = None) # no need to define a method as there are no attributes for no events

