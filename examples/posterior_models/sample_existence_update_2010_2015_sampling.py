""""
everything that needs to be specified to update the existence dimension
Baud Candice
Fri July 10 10:33:00 2026
"""


import numpy as np
import pandas as pd
from numba import njit, prange
from update_existence import *
from spec_other_dim import *

def weibull_logpdf(x, lam, k):
    """
    Numeric Weibull log-pdf.
    Returns a float.
    """
    if x < 0:
        return -np.inf

    return (
        np.log(k / lam)
        + (k - 1) * np.log(x / lam)
        - (x / lam) ** k
    )

def existence_density_prior(instance):
    log_p = weibull_logpdf(instance["Birth"]["main_duration"], 85, 3)
    return float(log_p)

dob_min = 1900
dob_max = 2050
lifespan_max = 110
warmup = 110 

def sex_proposal(current_value, instance, rng):
    return(rng.binomial(1, 0.5))

def sex_proposal_init(rng):
    return(rng.binomial(1, 0.5))

def sex_proposal_evaluation(value, current_value, instance):
    return(np.log(0.5))

p_a0 = ProbabilisticModel.Proposal(sex_proposal, sex_proposal_evaluation, sex_proposal_init)
a0 = AttributeSpec.SingleAttr("Sex", p_a0)
e0 = ExistenceEventSpec([a0], existence_density_prior, constraints = BirthConstraintSpec(dob_min, dob_max, lifespan_max, warmup))

## load data
df_observed_2010 = pd.read_csv("2010_sample.csv")
df_observed_2015 = pd.read_csv("2015_sample.csv")

## mapping
def mapping_existence(df_time_indep, t):
    # Keep only Birth events
    df_birth = df_time_indep[df_time_indep["event_name"] == "Birth"]
    
    # Select relevant columns and rename for clarity
    df_birth = df_birth[["id", "main_start_date", "main_duration", "Sex"]].copy()
    df_birth.rename(columns={"main_start_date": "birth_date", "main_duration": "lifespan"}, inplace=True)
    
    # Map gender directly 
    df_birth["gender"] = df_birth["Sex"]
    
    # Compute death time (vectorized operation)
    df_birth["death_time"] = df_birth["birth_date"] + df_birth["lifespan"]
    
    # Create an "alive_status" column based on conditions
    df_birth["alive_status"] = "Dead"
    df_birth["alive_status"] = np.where(df_birth["birth_date"] > t, "Not born", df_birth["alive_status"])
    df_birth["alive_status"] = np.where(
        (df_birth["birth_date"] <= t) & (t < df_birth["death_time"]),
        "Alive",
        df_birth["alive_status"]
    )
    
    # Calculate age only for alive individuals
    df_birth["age"] = np.where(df_birth["alive_status"] == "Alive", t - df_birth["birth_date"], np.nan)
    
    # Final dataset with selected columns
    df_time_dep = df_birth[["id", "alive_status", "age", "gender"]]
    
    return df_time_dep

## target size
n_samples = 5000 # size of population = 5000
t_eval_min = 1900
t_eval_max = 2050

def target_size(t):
    t = np.asarray(t)
    return np.where((t >= t_eval_min) & (t <= t_eval_max), (n_samples*85)/(t_eval_max - t_eval_min), 0)

sigma_age = 1
misclassification_rate = 0.05

@njit(parallel=True)
def f_age_gender_matrix(obs_age, sim_age,
                       obs_gender, sim_gender,
                       sigma, misclassification_rate):

    n_obs = len(obs_age)
    n_sim = len(sim_age)

    out = np.empty((n_obs, n_sim))

    # Precompute constants
    log_norm_const = -np.log(sigma * np.sqrt(2.0 * np.pi))
    p_same = 1.0 - misclassification_rate
    p_diff = misclassification_rate

    for i in prange(n_obs):
        a_i = obs_age[i]
        g_i = obs_gender[i]

        for j in range(n_sim):
            # ------------------
            # AGE contribution
            # ------------------
            diff = a_i - sim_age[j]
            val = -0.5 * (diff / sigma) ** 2 + log_norm_const
            age_val = np.exp(val)

            # ------------------
            # GENDER contribution
            # ------------------
            if g_i == sim_gender[j]:
                gender_val = p_same
            else:
                gender_val = p_diff

            # ------------------
            # COMBINED
            # ------------------
            out[i, j] = age_val * gender_val

    return out

def f_age_gender(obs, sim, it):
    res = f_age_gender_matrix(
        obs["age"].astype(np.float64),
        sim["age"].astype(np.float64),
        obs["gender"].astype(np.int32),
        sim["gender"].astype(np.int32),
        sigma_age,
        misclassification_rate)

    return(res)


###### filter for the sampling
def filter_sim_age_above_7(obs_dict, sim_dict, it):
    sim_ok = sim_dict["age"] > 7
    return np.broadcast_to(
        sim_ok[None, :],
        (len(obs_dict["age"]), len(sim_dict["age"]))
    )

## kernel
t1 = 2010
kernel_2010 = BirthKernel(
    df_obs=df_observed_2010,
    t=t1,
    f=f_age_gender,
    mapping=mapping_existence,
    weights = None,
    normalize_weights = True,
    sim_filter=filter_sim_age_above_7
)


t2 = 2015
kernel_2015 = BirthKernel(
    df_obs=df_observed_2015,
    t=t2,
    f=f_age_gender,
    mapping=mapping_existence,
    weights = None,
    normalize_weights = True,
    sim_filter=filter_sim_age_above_7
)

updater = BirthPopUpdater(e0, n_samples, target_size, [kernel_2010, kernel_2015])

n_init = 5000 #initial number of individuals to propose
def n_prop_function(i): # decreasing number of individuals to propose
    return(int(n_init/np.sqrt(i+2)))

burnin = 1000 # for birth
n_iter = 50000  # Number of iterations for MC
sigma_size = 0.1

if __name__ == "__main__":
    # Update the population
    updated_pop, accept_rate_dur, accept_rate_attr, likelihood = updater.update_pop(n_iter, t_eval_min, t_eval_max, n_prop_function, sigma_size, burnin_init=burnin)
    
    # Convert the updated population to DataFrame
    updated_pop_df = updated_pop.to_dataframe()
    
    # Save the updated population to CSV
    updated_pop_df.to_csv("updated_population_2010_2015_correctionsampling.csv", index=False)
    
    # Save accept_rate_dur, accept_rate_attr, and likelihood to CSV
    accept_rate_dur_df = pd.DataFrame({"accept_rate_dur": accept_rate_dur})
    accept_rate_dur_df.to_csv("accept_rate_dur_2010_2015_correctionsampling.csv", index=False)

    accept_rate_attr_df = pd.DataFrame({"accept_rate_attr": accept_rate_attr})
    accept_rate_attr_df.to_csv("accept_rate_attr_2010_2015_correctionsampling.csv", index=False)

    likelihood_df = pd.DataFrame({"likelihood": likelihood})
    likelihood_df.to_csv("likelihood_2010_2015_correctionsampling.csv", index=False)
