"""Existence constraints 

Baud Candice
Fri July 10 10:33:00 2026
"""

import numpy as np

def build_birth_constraint_matrix(alive_constraint_spec):
    """
    Builds constraints A x <= b for x = (start_date, duration).
    """
    # time = alive_constraint_spec.time
    start_low = alive_constraint_spec.start_low
    start_high = alive_constraint_spec.start_high
    duration_high = alive_constraint_spec.duration_high
    warmup = alive_constraint_spec.warmup_time

    A_matrix = []
    b = []

    row = np.zeros(2); row[0] = -1
    A_matrix.append(row.copy()); b.append(-start_low)

    row = np.zeros(2); row[0] = 1
    A_matrix.append(row.copy()); b.append(start_high)

    row = np.zeros(2); row[1] = -1
    A_matrix.append(row.copy()); b.append(0)

    row = np.zeros(2); row[1] = 1
    A_matrix.append(row.copy()); b.append(duration_high)


    # add constraint to ensure only alive at T0 (don't generate dead)
    row = -np.ones(2) 
    A_matrix.append(row.copy()); b.append(-(start_low + warmup))

    A_matrix = np.vstack(A_matrix)
    b = np.array(b, float)
    return A_matrix, b