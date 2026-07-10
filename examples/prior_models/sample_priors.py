"""Baud Candice
Fri July 10 10:33:00 2026"""
import numpy as np
from scipy.stats import norm, uniform, expon, gamma, lognorm, beta, poisson
import math
from scipy.special import gammaln, logsumexp
from pre_coded_function import *
import random

import os

from model_priors import *

# --------------------------------------------------
# MAIN ENTRY POINT (required for multiprocessing)
# --------------------------------------------------
if __name__ == "__main__":

    # VERY IMPORTANT on HPC
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------
    OUTPUT_DIR = "population_output_har"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------
    # Build population using SCITAS parallel version
    # --------------------------------------------------
    pop_spec_prior.sample_parallel_to_disk(
        dimension_mode="joint",
        x0=None,
        n_gibbs=1000,
        burnin=6500,
        burnin_existence=1500,
        seed=12345,
        n_workers=55,          # number of CPUs
        output_dir=OUTPUT_DIR,
    )

    print("Population generation finished.")