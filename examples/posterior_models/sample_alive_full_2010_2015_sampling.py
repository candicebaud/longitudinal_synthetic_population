"""sample from the posterior other dim
Baud Candice
Fri July 10 10:33:00 2026
"""


import numpy as np
from scipy.stats import norm, uniform, expon, gamma, lognorm, beta, poisson
import math
from scipy.special import gammaln, logsumexp
import random
from numba import njit, prange
import os

from pre_coded_function import *
from spec_other_dim import *
from update_other_dim import *


##########################################
######### import prior model for unalive to run faster
#########################################

from model_priors import *

#############################################
###### Define the model (prior with MH to go fast)
#############################################

# functions 
def weibull_logpdf(x, lam, k):
    """
    Numeric Weibull log-pdf.
    Returns a float.
    """
    if x < 0:
        return np.log(1e-300)

    return (
        np.log(k / lam)
        + (k - 1) * np.log(x / lam)
        - (x / lam) ** k
    )

def log_indicator_function(a, b, tol=0.1):
    if abs(a - b) <= tol:
        return 0.0  # log(1)
    else:
        return np.log(1e-300)  # log(0)
    
def truncated_lognorm_logpdf(x, mu, sigma, a, b):
    if x < a or x > b:
        return np.log(1e-300)
    
    dist = lognorm(s=sigma, scale=np.exp(mu))
    Z = dist.cdf(b) - dist.cdf(a)

    if Z <= 0:
        return np.log(1e-300)

    return dist.logpdf(x) - np.log(Z)

def truncated_normal_pdf(x, mu, sigma2, a, b):
    """
    PDF of N(mu, sigma2) truncated to [a, b]
    """
    sigma = np.sqrt(sigma2)
    
    if x < a or x > b:
        return 0.0
    
    Z = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)
    return norm.pdf(x, mu, sigma) / Z

def truncated_exponential_pdf(x, lambd, a, b):
    """
    PDF of Exp(lambd) truncated to [a, b]
    """
    if x < a or x > b:
        return 0.0
    
    lambd = 1/lambd
    Z = np.exp(-lambd * a) - np.exp(-lambd * b)
    return lambd * np.exp(-lambd * x) / Z

def log_truncated_exponential_pdf(x, lambd, a, b):
    if x < a or x > b:
        return np.log(1e-300)

    rate = 1 / lambd

    log_num = np.log(rate) - rate * x

    A = -rate * a
    B = -rate * b  # note: B < A

    # log(exp(A) - exp(B)) in a stable way
    log_Z = A + np.log1p(-np.exp(B - A))

    return log_num - log_Z

def truncated_lognorm_pdf(x, mu, sigma, a, b):
    if x < a or x > b:
        return 0.0
    
    dist = lognorm(s=sigma, scale=np.exp(mu))
    Z = dist.cdf(b) - dist.cdf(a)

    pdf = dist.pdf(x) / Z
    return np.where((x >= a) & (x <= b), pdf, 0.0)

def weibull_cdf(x, l, k):
    if x < 0 :
        return(0)
    else:
        return(1 - np.exp(-((x/l)**k)))
    
def weibull_pdf(x, l, k):
    if x < 0:
        return(0)
    else:
        return ((k/l)*(x/l)**(k-1)*np.exp(-(x/l)**k))
    
def weibull_pdf_truncated(x, l, k, a, b):
    if x < a or x > b:
        return 0.0
    Z = weibull_cdf(b, l, k) - weibull_cdf(a, l, k)
    return weibull_pdf(x, l, k) / Z

def log_from_pdf(pdf_val):
    """Convert a (nonnegative) pdf value to log-pdf safely."""
    if pdf_val <= 0.0 or not np.isfinite(pdf_val):
        return np.log(1e-300)
    return float(np.log(pdf_val))

def log_truncated_normal_pdf(x, mu, sigma2, a, b):
    """
    Log-PDF of N(mu, sigma2) truncated to [a, b]
    """
    if x < a or x > b:
        return np.log(1e-300)

    sigma = np.sqrt(sigma2)

    # log pdf of normal
    log_pdf = norm.logpdf(x, mu, sigma)

    # normalization constant
    Z = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)

    if Z <= 0:
        return np.log(1e-300)

    return log_pdf - np.log(Z)

def log_weibull_pdf_truncated(x, l, k, a, b):
    if x < a or x > b:
        return np.log(1e-300)

    # log pdf
    log_pdf = (
        np.log(k) - np.log(l)
        + (k - 1) * np.log(x / l)
        - (x / l) ** k
    )

    A = - (a / l) ** k
    B = - (b / l) ** k  # B < A

    # log(exp(A) - exp(B)) safely
    log_Z = A + np.log1p(-np.exp(B - A))

    return log_pdf - log_Z


# define the dimensions 
#### Existence
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
d0 = ExistenceDimensionSpec([a0], existence_density_prior, constraints=BirthConstraintSpec(dob_min, dob_max, lifespan_max, warmup))


#### DL
def ndl_log_density(instance):
    logp = 0.0

    indic = instance["DL"]["NoDL"]["indicator"]
    main_start = instance["DL"]["NoDL"]["main_start_date"]
    gap_start = instance["DL"]["NoDL"]["gap_start_date"]
    main_duration = instance["DL"]["NoDL"]["main_duration"]
    gap_duration = instance["DL"]["NoDL"]["gap_duration"]

    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    logp += log_indicator_function(indic, 1)
    logp += log_indicator_function(main_start, dob)
    logp += log_indicator_function(gap_start, dob + main_duration)
    logp += log_indicator_function(gap_duration, 0)

    if lifespan <= 18:
        logp += log_indicator_function(main_duration, lifespan)
        return logp
    else:
        if instance["DL"]["DL"]["indicator"] == 1:
            logp += truncated_lognorm_logpdf(
                main_duration,
                math.log(20.5),
                0.15,
                18,
                70
            )
            return logp
        else:
            logp += log_indicator_function(main_duration, lifespan)
            return logp

def dl_log_density(instance):
    logp = 0.0

    indic = instance["DL"]["DL"]["indicator"]
    main_start = instance["DL"]["DL"]["main_start_date"]
    gap_start = instance["DL"]["DL"]["gap_start_date"]
    main_duration = instance["DL"]["DL"]["main_duration"]
    gap_duration = instance["DL"]["DL"]["gap_duration"]

    start_ndl = instance["DL"]["NoDL"]["main_start_date"]
    d_ndl = instance["DL"]["NoDL"]["main_duration"]

    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    logp += log_indicator_function(main_start, start_ndl + d_ndl)
    logp += log_indicator_function(gap_start, start_ndl + d_ndl + main_duration)

    if indic == 1:
        logp += log_indicator_function(main_duration, lifespan - d_ndl)
        logp += log_indicator_function(gap_duration, 0)
        return logp
    else:
        logp += log_indicator_function(main_duration, 0)
        logp += log_indicator_function(gap_duration, 0)
        return logp

def driving_license_log_density(instance):
    logp = 0.0
    logp += ndl_log_density(instance)
    logp += dl_log_density(instance)
    return logp

def dl_initialize_indic(rng):
    return(1)

def dl_gibbs_indic(instance, rng):
    return(rng.binomial(1, 0.85)) 


e1_0 = EventSpec.NoEvent("NoDL")
p_e1_1 = ProbabilisticModel.Gibbs(dl_gibbs_indic, dl_initialize_indic)
c_e1_1 = EventTemporalConstraintsSpec(18, 70, None, None, None, None, None, 0, None)
e1_1 = EventSpec.SingleTimeEvent("DL", p_e1_1, c_e1_1, None)
d1 = DimensionSpec("DL", [e1_0, e1_1], None)


#### Education
def log_ned_density(instance):
    indic = instance["Education"]["NoEducation"]["indicator"]
    main_start = instance["Education"]["NoEducation"]["main_start_date"]
    gap_start = instance["Education"]["NoEducation"]["gap_start_date"]
    gap_duration = instance["Education"]["NoEducation"]["gap_duration"]
    main_duration = instance["Education"]["NoEducation"]["main_duration"]

    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    # If "NoEducation" not active: contributes factor 1
    if indic == 0:
        return 0.0

    lp = 0.0
    lp += log_indicator_function(main_start, dob)
    lp += log_indicator_function(gap_duration, 0)
    lp += log_indicator_function(main_duration, min(lifespan, 6))
    lp += log_indicator_function(gap_start, main_start + main_duration)
    return lp


def log_me_density(instance):
    indic = instance["Education"]["MandatoryEduc"]["indicator"]
    main_start = instance["Education"]["MandatoryEduc"]["main_start_date"]
    gap_start = instance["Education"]["MandatoryEduc"]["gap_start_date"]
    main_duration = instance["Education"]["MandatoryEduc"]["main_duration"]
    gap_duration = instance["Education"]["MandatoryEduc"]["gap_duration"]

    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    start_ned = instance["Education"]["NoEducation"]["main_start_date"]
    d_ned = instance["Education"]["NoEducation"]["main_duration"] + instance["Education"]["NoEducation"]["gap_duration"]

    lp = 0.0

    if lifespan <= 6:
        lp += log_indicator_function(indic, 0)
        lp += log_indicator_function(main_duration, 0)
        lp += log_indicator_function(gap_duration, 0)
        return lp

    lp += log_indicator_function(indic, 1)
    lp += log_indicator_function(main_start, start_ned + d_ned)
    lp += log_indicator_function(main_duration, min(lifespan - 6, 9))
    lp += log_indicator_function(gap_start, main_start + main_duration)

    if lifespan <= 15:
        lp += log_indicator_function(gap_duration, 0)
        return lp

    if instance["Education"]["SecondaryEduc"]["indicator"] == 1:
        lp += log_indicator_function(gap_duration, 0)
        return lp

    lp += log_indicator_function(gap_duration, lifespan - 15)
    return lp


def log_se_density(instance):
    indic = instance["Education"]["SecondaryEduc"]["indicator"]
    main_start = instance["Education"]["SecondaryEduc"]["main_start_date"]
    gap_start = instance["Education"]["SecondaryEduc"]["gap_start_date"]
    main_duration = instance["Education"]["SecondaryEduc"]["main_duration"]
    gap_duration = instance["Education"]["SecondaryEduc"]["gap_duration"]

    start_me = instance["Education"]["MandatoryEduc"]["main_start_date"]
    d_me = instance["Education"]["MandatoryEduc"]["main_duration"] + instance["Education"]["MandatoryEduc"]["gap_duration"]

    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    lp = 0.0
    lp += log_indicator_function(main_start, start_me + d_me)
    lp += log_indicator_function(gap_start, main_start + main_duration)

    if lifespan <= 15:
        lp += log_indicator_function(indic, 0)
        lp += log_indicator_function(main_duration, 0)
        lp += log_indicator_function(gap_duration, 0)
        return lp

    if indic == 1:

        if lifespan <= 18:
            lp += log_indicator_function(main_duration, lifespan - 15)
            lp += log_indicator_function(gap_duration, 0)
            return lp

        # lifespan > 18
        if instance["Education"]["TertiaryEduc"]["indicator"] == 1:
            lp += log_truncated_normal_pdf(main_duration, 4, 0.5, 3, 6)
            lp += log_truncated_exponential_pdf(gap_duration, 1, 0, 25 - 15 - main_duration)
            return lp

        lp += log_indicator_function(main_duration, 3)
        lp += log_indicator_function(gap_duration, lifespan - 15 - main_duration)
        return lp

    # indic == 0
    lp += log_indicator_function(main_duration, 0)
    lp += log_indicator_function(gap_duration,0)
    return lp


def log_te_density(instance):
    indic = instance["Education"]["TertiaryEduc"]["indicator"]
    main_start = instance["Education"]["TertiaryEduc"]["main_start_date"]
    gap_start = instance["Education"]["TertiaryEduc"]["gap_start_date"]
    main_duration = instance["Education"]["TertiaryEduc"]["main_duration"]
    gap_duration = instance["Education"]["TertiaryEduc"]["gap_duration"]

    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    start_se = instance["Education"]["SecondaryEduc"]["main_start_date"]
    d_se = instance["Education"]["SecondaryEduc"]["main_duration"] + instance["Education"]["SecondaryEduc"]["gap_duration"]

    lp = 0.0
    lp += log_indicator_function(main_start, start_se + d_se)
    lp += log_indicator_function(gap_start, main_start + main_duration)

    if lifespan <= 18:
        # If not enough life to have tertiary but indicator says yes => impossible
        if indic == 1:
            return np.log(1e-300)
        lp += log_indicator_function(main_duration, 0)
        lp += log_indicator_function(gap_duration, 0)
        return lp

    # lifespan > 18
    if indic == 1:
        lp += log_weibull_pdf_truncated(
            main_duration,
            4.5, 3.5,
            0,
            min(15, lifespan - (15 + d_se)),
        )
        lp += log_indicator_function(
            gap_duration,
            dob + lifespan - (main_start + main_duration),
            
        )
        return lp

    lp += log_indicator_function(main_duration, 0)
    lp += log_indicator_function(gap_duration, 0)
    return lp


def education_log_density(instance):
    lp = 0.0

    ln = log_ned_density(instance)
    if not np.isfinite(ln):
        return np.log(1e-300)
    lp += ln

    lm = log_me_density(instance)
    if not np.isfinite(lm):
        return np.log(1e-300)
    lp += lm

    ls = log_se_density(instance)
    if not np.isfinite(ls):
        return np.log(1e-300)
    lp += ls

    lt = log_te_density(instance)
    if not np.isfinite(lt):
        return np.log(1e-300)
    lp += lt

    return lp

def marginal_indicator_mandatory(instance, rng):
    return(1)

def marginal_indicator_secondary(instance, rng):
    bool = rng.binomial(1, 0.95)
    return(bool)

def marginal_indicator_tertiary(instance, rng):
    bool = rng.binomial(1, 0.6)
    return(bool)

def initialize_mandatory(rng):
    return(1)

def initialize_secondary(rng):
    return(rng.binomial(1, 0.95))

def initialize_tertiary(rng):
    return(rng.binomial(1, 0.6))

e0 = EventSpec.NoEvent("NoEducation")

#mandatory
p_e1 = ProbabilisticModel.Gibbs(marginal_indicator_mandatory, initialize_mandatory)
c_e1 = EventTemporalConstraintsSpec(6, 6, 9, 9, None, 15, None, None, None)
e1 = EventSpec.SingleTimeEvent("MandatoryEduc", p_e1, c_e1, None)

#secondary
p_e2 = ProbabilisticModel.Gibbs(marginal_indicator_secondary, initialize_secondary)
c_e2 = EventTemporalConstraintsSpec(15, 15, 3, 6, None, 21, None, None, None)
e2 = EventSpec.SingleTimeEvent("SecondaryEduc", p_e2, c_e2, None)

#tertiary
p_e3 = ProbabilisticModel.Gibbs(marginal_indicator_tertiary, initialize_tertiary)
c_e3 = EventTemporalConstraintsSpec(18, 25, None, 15, None, 40, None, None, None)
e3 = EventSpec.SingleTimeEvent("TertiaryEduc", p_e3, c_e3, None)

d2 = DimensionSpec("Education", [e0, e1, e2, e3], None)


#### Employment
def indicator_function(a,b):
    # a bit more flexible than pure equality
    if abs(a-b) <= 0.1:
        return(1)
    else:
        return(0)
    
def count_pdf_work(instance, rng):
    lifespan = instance["Existence"]["Birth"]["main_duration"]
    ret_age = 64
    lambda_value = math.sqrt(min(lifespan, ret_age) - 15)
    sample = rng.poisson(lambda_value)
    while sample >11:
        sample = rng.poisson(lambda_value)
    return(sample)

def count_pdf_work_init(rng):
    sample = rng.poisson(6)
    return(sample)

def nem_density(instance):
    indic = instance["Employment"]["NoWork"]["indicator"]
    main_start = instance["Employment"]["NoWork"]["main_start_date"]
    gap_start = instance["Employment"]["NoWork"]["gap_start_date"]
    main_duration = instance["Employment"]["NoWork"]["main_duration"]
    gap_duration = instance["Employment"]["NoWork"]["gap_duration"]
    dob = instance["Existence"]["Birth"]["main_start_date"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]

    p = 0
    p = p + log_indicator_function(indic, 1)
    p = p + log_indicator_function(main_start, dob)
    p = p + log_indicator_function(gap_start, dob + main_duration)
    p = p + log_indicator_function(gap_duration, 0)

    if lifespan <= 15:
        p = p + log_indicator_function(main_duration, lifespan)
    else:
        if instance["Employment"]["Work_1"]["indicator"] == 1:
            # there is at least one spell
            if instance["Education"]["SecondaryEduc"]["indicator"] == 0 :
                p = p+ log_truncated_normal_pdf(main_duration, 16, 2, 15, lifespan)
            else:
                if instance["Education"]["TertiaryEduc"]["indicator"] == 0: 
                    p = p + log_truncated_normal_pdf(main_duration, 18, 4, 15, lifespan)
                else:
                    p = p + log_truncated_normal_pdf(main_duration, 21, 7, 15, lifespan)
        else:
            p = p + log_indicator_function(main_duration, lifespan)
    
    return(p)

def lambda_age_work(age):
    if age < 25:
        return(2.5)
    elif 25 <= age and age < 35:
        return(3.5)
    elif 35 <= age and age < 45:
        return(4.5)
    elif 45 <= age and age < 55:
        return(5.5)
    elif age >= 55:
        return(9)

def logIncomeinitialdist(index, rng):
    mean = 5000
    std = 1500
    sample = rng.normal(mean, std)
    while sample < 0 :
        sample = rng.normal(mean, std)
    return(np.log(sample))

def logIncome_proposal(current_value, instance, index, rng):
    proposed_income = rng.normal(current_value, 1)
    return proposed_income

def logIncome_evaluation_proposal(value, current_value, instance, index):
    # return(np.log(truncated_normal_pdf(value, current_value, 1, 0, np.inf)))
    return(norm.logpdf(current_value, loc = value, scale = 1)) 

def place_initial_dist_work(index, rng):
    # initialization : sample at random based only on the actual proportion of inhabitants in Swiss cantons
    # same function for work and for residence
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions)
    )

    return(canton)

def pow_proposal(current_value, instance, index, rng):
    log_inc = instance["Employment"][f"Work_{index + 1}"]["LogIncome"]
    inc = np.exp(log_inc)

    cantons = list(canton_proportions.keys())

    if inc > gamma:
        new_proportions = list(canton_proportions_boost.values())
    else:
        new_proportions = list(canton_proportions.values())

    canton = rng.choice(
    cantons,
    p=new_proportions/np.sum(new_proportions)
    )
    return(canton)

def pow_proposal_evaluation(value, current_value, instance, index):
    log_inc = instance["Employment"][f"Work_{index + 1}"]["LogIncome"]
    inc = np.exp(log_inc)
    if inc > gamma :
        return (np.log(canton_proportions_boost[value]))
    else:
        return(np.log(canton_proportions[value]))

beta_0 = 7.8
beta_1 = 0.025
beta_2 = -0.0004
rho = 0.1
sigma_eps2 = 0.3

def work_density(instance):
    p = 0
    d_nem = instance["Employment"]["NoWork"]["main_duration"]
    lifespan = instance["Existence"]["Birth"]["main_duration"]
    dob = instance["Existence"]["Birth"]["main_start_date"]

    # count the number of work spells
    count = 0
    for i in range (11):
        count = count + instance["Employment"][f"Work_{i+1}"]["indicator"]
    
    if lifespan <= 15:
        if count > 0:
            return(0)
        else :
            p = 0
    else:
        # probability
        
        if count == 0:
            for i in range (11):
                p = p + log_indicator_function(instance["Employment"][f"Work_{i+1}"]["main_duration"], 0)
                p = p + log_indicator_function(instance["Employment"][f"Work_{i+1}"]["gap_duration"], 0)
        else:
            indic_zero = 11
            
            age_at_start = d_nem
            curr_indic = instance["Employment"]["Work_1"]["indicator"]
            next_indic = instance["Employment"]["Work_2"]["indicator"]
            main_duration = instance["Employment"]["Work_1"]["main_duration"]
            gap_duration = instance["Employment"]["Work_1"]["gap_duration"]      
            cumulated_duration = 0

            if curr_indic == 1:
                p = p + log_truncated_exponential_pdf(main_duration, lambda_age_work(age_at_start), 0, min(70 - d_nem - cumulated_duration, lifespan - d_nem - cumulated_duration))
                if next_indic == 1:
                    p = p + log_truncated_exponential_pdf(gap_duration, 1, 0, min(70 - cumulated_duration - d_nem - main_duration, lifespan - cumulated_duration - d_nem - main_duration))
                else:
                    indic_zero = 2
                    p = p + log_indicator_function(gap_duration, lifespan - cumulated_duration - d_nem - main_duration)
            else:
                p = p + log_indicator_function(main_duration, 0)
                p = p + log_indicator_function(gap_duration, 0)
            
            cumulated_duration = cumulated_duration + main_duration + gap_duration
            age_at_start = age_at_start + main_duration + gap_duration

            for i in range (2, 11):
                curr_indic = instance["Employment"][f"Work_{i}"]["indicator"]
                next_indic = instance["Employment"][f"Work_{i+1}"]["indicator"]
                main_duration = instance["Employment"][f"Work_{i}"]["main_duration"]
                gap_duration = instance["Employment"][f"Work_{i}"]["gap_duration"]     
                if curr_indic == 1:
                    l = lambda_age_work(age_at_start)
                    p = p + log_truncated_exponential_pdf(main_duration, l, 0, min(70 - d_nem - cumulated_duration, lifespan - d_nem - cumulated_duration))
                    if next_indic == 1:                        
                        p = p + log_truncated_exponential_pdf(gap_duration, 1, 0, min(70 - cumulated_duration - d_nem - main_duration, lifespan - cumulated_duration - d_nem - main_duration))
                    else:
                        indic_zero = i
                        p = p + log_indicator_function(gap_duration, lifespan - cumulated_duration - d_nem - main_duration)
                else:
                    p = p + log_indicator_function(main_duration, 0)
                    p = p + log_indicator_function(gap_duration, 0)

                cumulated_duration = cumulated_duration + main_duration + gap_duration
                age_at_start = age_at_start + main_duration + gap_duration

            # last employment spell
            last_indic = instance["Employment"]["Work_11"]["indicator"]
            main_duration = instance["Employment"]["Work_11"]["main_duration"]
            gap_duration = instance["Employment"]["Work_11"]["gap_duration"]      
            if last_indic == 1:
                p = p + log_truncated_exponential_pdf(main_duration, lambda_age_work(age_at_start), 0, min(70 - d_nem - cumulated_duration, lifespan - d_nem - cumulated_duration))
                p = p + log_indicator_function(gap_duration, lifespan - cumulated_duration - d_nem - main_duration) 
            else:
                p = p + log_indicator_function(main_duration, 0)
                p = p + log_indicator_function(gap_duration, 0)

            # last actually done employment spell, density 
            # if lifespan > 64 :
            #     gap_start_date = instance["Employment"][f"Work_{indic_zero}"]["gap_start_date"]
            #     p = p*truncated_normal_pdf(gap_start_date, 65 + dob, 1, 64 + dob, 70 + dob)
    
    ### income, pow
    log_prior_income = 0
    log_prior_pow = 0
    d_ne = instance["Education"]["NoEducation"]["main_duration"]
    d_me_e = instance["Education"]["MandatoryEduc"]["main_duration"] 
    d_me_g = instance["Education"]["MandatoryEduc"]["gap_duration"]
    d_me = d_me_e + d_me_g
    d_se_e = instance["Education"]["SecondaryEduc"]["main_duration"]
    d_se_g = instance["Education"]["SecondaryEduc"]["gap_duration"]
    d_se = d_se_e + d_se_g
    d_te_e = instance["Education"]["TertiaryEduc"]["main_duration"]

    indic_me = instance["Education"]["MandatoryEduc"]["indicator"]
    indic_se = instance["Education"]["SecondaryEduc"]["indicator"]
    indic_te = instance["Education"]["TertiaryEduc"]["indicator"]

    for index in range (11):
        emp = instance["Employment"][f"Work_{index+1}"]
        work = 0 
        if emp["indicator"] == 1:
            age_start_work_spell = instance["Employment"]["NoWork"]["main_duration"] + instance["Employment"]["NoWork"]["gap_duration"]
            for i in range (index):
                work = work + instance["Employment"][f"Work_{i+1}"]["main_duration"]
                age_start_work_spell = age_start_work_spell + instance["Employment"][f"Work_{i+1}"]["main_duration"] + instance["Employment"][f"Work_{i+1}"]["gap_duration"]

            if indic_me == 0:
                # doesn't do mandatory : no education
                edu = 0
            else:
                # does mandatory
                if age_start_work_spell <= d_ne + d_me_e :
                    # but hasn't finished mandatory education
                    edu = 0
                else:
                    # has finished me
                    if indic_se == 0 :
                        # doesn't do secondary 
                        edu = d_me_e
                    else:
                        # does secondary
                        if age_start_work_spell <= d_ne + d_me + d_se_e:
                            # but hasn't finished
                            edu = d_me_e
                        else:
                            # has finished se 
                            if indic_te == 0:
                                # doesn't do tertiary
                                edu = d_me_e + d_se_e
                            else:
                                # does tertiary
                                if age_start_work_spell <= d_ne + d_me + d_se + d_te_e:
                                    # but hasn't finished
                                    edu = d_me_e + d_se_e
                                else:
                                    # has finished
                                    edu = d_me_e + d_se_e + d_te_e
            
            mean = beta_0 + beta_1*work + beta_2*work**2 + rho*edu
            log_prior_income = log_prior_income + norm.logpdf(emp['LogIncome'], loc = mean, scale = sigma_eps2)

            if np.exp(emp['LogIncome'])>gamma:
                log_prior_pow = log_prior_pow + np.log(canton_proportions_boost[emp["PlaceWork"]])
            else:
                log_prior_pow = log_prior_pow + np.log(canton_proportions[emp["PlaceWork"]])
        
    p = p + log_prior_income + log_prior_pow
    return(p)


e3_0 = EventSpec.NoEvent("NoWork")

p_a_3_1_1 = ProbabilisticModel.Proposal(logIncome_proposal, logIncome_evaluation_proposal, logIncomeinitialdist)
a_3_1_1 = AttributeSpec.MultipleAttr("LogIncome", p_a_3_1_1)
p_a_3_1_2 = ProbabilisticModel.Proposal(pow_proposal, pow_proposal_evaluation, place_initial_dist_work)
a_3_1_2 = AttributeSpec.MultipleAttr("PlaceWork", p_a_3_1_2)

p_e3_1 = ProbabilisticModel.Gibbs(count_pdf_work, count_pdf_work_init)
c_e3_1 = EventTemporalConstraintsSpec(15, 64, None, None, None, 70, None, None, None)
e3_1 = EventSpec.MultipleTimeEvent("Work", 11, p_e3_1, c_e3_1, [a_3_1_1, a_3_1_2])

d3 = DimensionSpec("Employment", [e3_0, e3_1], None)

def log_employment_density(instance):
    return(nem_density(instance) + work_density(instance))




#### Residence 
def count_pdf_residence_init(rng):
    return(1)

def count_pdf_residence(instance, rng):
    total = 0
    for k in range (11):
        ev_name = f"Work_{k+1}"
        total = total + instance["Employment"][ev_name]["indicator"]
    return(total)

def place_initial_dist_residence(index, rng):
    # initialization : sample at random based only on the actual proportion of inhabitants in Swiss cantons
    # same function for work and for residence
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions),
    )

    return(canton)

def place_initial_dist_0(rng):
    # initialization : sample at random based only on the actual proportion of inhabitants in Swiss cantons
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions)
    )

    return(canton)

def por_distribution_0(instance, rng):
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions),
    )

    return(canton)

p_w = 0.05
def place_residence_proposal(current_value, instance, index, rng):
    canton_work = instance["Employment"][f"Work_{index + 1}"]["PlaceWork"] 
    
    # sample uniformly a neighbouring canton with probability 0.05
    bool = rng.binomial(1, p_w)
    if bool == 1 :
        canton_residence = rng.choice(neighbouring_cantons[canton_work])
    else:
        canton_residence = canton_work

    return(canton_residence)

def place_residence_proposal_0(current_value, instance, rng):
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions)
    )

    return(canton)

def place_residence_evaluation_proposal(value, current_value, instance, index):
    canton_work = instance["Employment"][f"Work_{index + 1}"]["PlaceWork"] 
    if value == canton_work:
        return(np.log(0.95))
    elif value in neighbouring_cantons[canton_work]:
        return(np.log(0.05))
    else:
        return(np.log(1e-300))

def place_residence_evaluation_proposal_0(value, current_value, instance):
    return(canton_proportions[value])

def car_av_initial_dist(index, rng):
    return(0) # initialize everyone at 0 so there is no error

def car_av_initial_dist_0(rng):
    return(0) # for Home_0 as well return 0 

def car_av_proposal(current_value, instance, index, rng):
    dl_indic = instance["DL"]["DL"]["indicator"]
    if dl_indic == 0:
        # if no driving license, no car availability
        return(0)
    
    age_at_dl = instance["DL"]["NoDL"]["main_duration"]    
    age_at_start_res = instance["Residence"]["Home0"]["main_duration"]
    for k in range (index):
        age_at_start_res = age_at_start_res + instance["Residence"][f"Home_{k + 1}"]["main_duration"]
    
    if age_at_start_res <= age_at_dl:
        return(0)

    return(rng.binomial(1, 0.76))

def car_av_proposal_0(current_value, instance, rng):
    return(rng.binomial(1, 0.5)) 

def car_av_proposal_evaluation(value, current_value, instance, index):
    dl_indic = instance["DL"]["DL"]["indicator"]
    if dl_indic == 0:
        if value == 1:
            return(np.log(1e-300))
        else:
            return(np.log(1))
    age_at_dl = instance["DL"]["NoDL"]["main_duration"]    
    age_at_start_res = instance["Residence"]["Home0"]["main_duration"]
    for k in range (index):
        age_at_start_res = age_at_start_res + instance["Residence"][f"Home_{k + 1}"]["main_duration"]
    
    if age_at_start_res <= age_at_dl:
        if value == 1:
            return(np.log(1e-300))
        else:
            return(np.log(1))
    
    if value == 1:
        return(np.log(0.76))
    else:
        return(np.log(0.24))
    
def car_av_proposal_evaluation_0(value, current_value, instance):
    return(np.log(0.5))

def ag_av_initial_dist(index, rng):
    return(rng.binomial(1, 0.1))

def ag_av_initial_dist_0(rng):
    return(rng.binomial(1, 0.1))

def ag_av_proposal(current_value, index, instance, rng):
    return(rng.binomial(1, 0.1))

def ag_av_proposal_0(current_value, instance, rng):
    return(rng.binomial(1, 0.1))

def ag_av_proposal_evaluation(value, current_value, instance, index):
    if value == 1:
        return(np.log(0.1))
    else:
        return(np.log(0.9))
    
def ag_av_proposal_evaluation_0(value, current_value, instance):
    if value == 1:
        return(np.log(0.1))
    else:
        return(np.log(0.9))

def urban_initial_dist_0(rng):
    return(rng.binomial(1, 0.85)) 

def urban_initial_dist(index, rng):
    return(rng.binomial(1, 0.85))


urban_prop = {
    "Zurich": 1.00,
    "Bern": 0.75,
    "Vaud": 0.90,
    "Aargau": 0.85,
    "St_Gallen": 0.85,
    "Geneva": 1.00,
    "Lucerne": 0.65,
    "Valais": 0.75,
    "Ticino": 0.90,
    "Fribourg": 0.75,
    "Basel_Landschaft": 0.95,
    "Thurgau": 0.70,
    "Solothurn": 0.85,
    "Graubunden": 0.45,
    "Basel_Stadt": 1.00,
    "Neuchatel": 0.90,
    "Schwyz": 0.80,
    "Zug": 1.00,
    "Schaffhausen": 0.90,
    "Jura": 0.50,
    "Appenzell_Ausserrhoden": 0.75,
    "Nidwalden": 0.50,
    "Glarus": 0.75,
    "Obwalden": 0.30,
    "Uri": 0.85,
    "Appenzell_Innerrhoden": 0.00
}

def urban_proposal(current_value, instance, index, rng):
    canton_of_res = instance["Residence"][f"Home_{index + 1}"]["PlaceResidence"]
    prop = urban_prop[canton_of_res]
    return rng.binomial(1, prop)

def urban_proposal_0(current_value, instance, rng):
    canton_of_res = instance["Residence"]["Home0"]["PlaceResidence"]
    prop = urban_prop[canton_of_res]
    return rng.binomial(1, prop)

def urban_proposal_evaluation(value, current_value, instance, index):
    canton_of_res = instance["Residence"][f"Home_{index + 1}"]["PlaceResidence"]
    if value == 1:
        return(np.log(max(1e-300, urban_prop[canton_of_res])))
    else:
        return(np.log(max(1e-300, 1-urban_prop[canton_of_res])))

def urban_proposal_evaluation_0(value, current_value, instance):
    canton_of_res = instance["Residence"][f"Home0"]["PlaceResidence"]
    if value == 1:
        return(np.log(max(1e-300, urban_prop[canton_of_res])))
    else:
        return(np.log(max(1e-300, 1-urban_prop[canton_of_res])))


p_a4_0 = ProbabilisticModel.Proposal(place_residence_proposal_0, place_residence_evaluation_proposal_0, place_initial_dist_0)
a4_0 = AttributeSpec.SingleAttr("PlaceResidence", p_a4_0)
p_a4_0_1 = ProbabilisticModel.Proposal(car_av_proposal_0, car_av_proposal_evaluation_0, car_av_initial_dist_0)
a4_0_1 = AttributeSpec.SingleAttr("Car_availability", p_a4_0_1)
p_a4_0_2 = ProbabilisticModel.Proposal(ag_av_proposal_0, ag_av_proposal_evaluation_0, ag_av_initial_dist_0)
a4_0_2 = AttributeSpec.SingleAttr("AG_availability", p_a4_0_2)
p_a4_0_3 = ProbabilisticModel.Proposal(urban_proposal_0, urban_proposal_evaluation_0, urban_initial_dist_0)
a4_0_3 = AttributeSpec.SingleAttr("Urban", p_a4_0_3)

p_e4_0 = indic_prob_model_no_event
c_e4_0 = EventTemporalConstraintsSpec(None, None, None, None, None, None, None, 0, None)
e4_0 = EventSpec.SingleTimeEvent("Home0", p_e4_0, c_e4_0, [a4_0, a4_0_1, a4_0_2, a4_0_3])

p_a4_1 = ProbabilisticModel.Proposal(place_residence_proposal, place_residence_evaluation_proposal, place_initial_dist_residence)
a4_1 = AttributeSpec.MultipleAttr("PlaceResidence", p_a4_1)
p_a4_1_1 = ProbabilisticModel.Proposal(car_av_proposal, car_av_proposal_evaluation, car_av_initial_dist)
a4_1_1 = AttributeSpec.MultipleAttr("Car_availability", p_a4_1_1)
p_a4_1_2 = ProbabilisticModel.Proposal(ag_av_proposal, ag_av_proposal_evaluation, ag_av_initial_dist)
a4_1_2 = AttributeSpec.MultipleAttr("AG_availability", p_a4_1_2)
p_a4_1_3 = ProbabilisticModel.Proposal(urban_proposal, urban_proposal_evaluation, urban_initial_dist)
a4_1_3 = AttributeSpec.MultipleAttr("Urban", p_a4_1_3)

p_e4_1 = ProbabilisticModel.Gibbs(count_pdf_residence, count_pdf_residence_init)
c_e4_1 = EventTemporalConstraintsSpec(15, None, None, None, None, None, None, 0, None)
e4_1 = EventSpec.MultipleTimeEvent("Home", 11, p_e4_1, c_e4_1, [a4_1, a4_1_1, a4_1_2, a4_1_3])

d4 = DimensionSpec("Residence", [e4_0, e4_1], None)



def residence_0_density(instance):
    home0 = instance['Residence']["Home0"]

    canton = home0["PlaceResidence"]
    car_av = home0["Car_availability"]
    ag_av = home0["AG_availability"]
    urban = home0["Urban"]

    logp = 0.0

    # ----------------------
    # 1. Canton 
    # ----------------------
    logp += np.log(canton_proportions[canton])

    # ----------------------
    # 2. Car availability (hard constraint)
    # ----------------------
    if car_av != 0:
        return np.log(1e-300)

    # ----------------------
    # 3. Urban 
    # ----------------------
    if urban == 1:
        logp += np.log(max(urban_prop[canton],1e-300))
    elif urban == 0:
        logp += np.log(max(1-urban_prop[canton],1e-300))
    else:
        return np.log(1e-300)

    # ----------------------
    # 4. AG availability
    # ----------------------
    if ag_av == 1:
        logp += np.log(0.1)
    elif ag_av == 0:
        logp += np.log(0.9)
    else:
        return np.log(1e-300)

    return logp

def residence_density(instance):
    logp = 0

    for index in range (11):
        res = instance["Residence"][f"Home_{index+1}"]
        if res["indicator"] == 1:
            canton = res["PlaceResidence"]
            car_av = res["Car_availability"]
            ag_av = res["AG_availability"]
            urban = res["Urban"]

            canton_work = instance["Employment"][f"Work_{index + 1}"]["PlaceWork"]
            if canton_work == canton:
                logp = logp + np.log(0.95)
            elif canton in neighbouring_cantons[canton_work]:
                logp = logp + np.log(0.05)
            else:
                logp = logp + np.log(1e-300) #not 0 but very small

            dl_indic = instance["DL"]["DL"]["indicator"]
            if dl_indic == 0:
                if car_av == 1:
                    logp = logp + np.log(1e-300) #not 0 but very small
            else:
                age_at_dl = instance["DL"]["NoDL"]["main_duration"]    
                dob = instance["Residence"]["Home0"]["main_start_date"]
                age_at_start_res = res["main_start_date"] - dob
                if age_at_start_res <= age_at_dl:
                    if car_av == 1:
                        logp = logp + np.log(1e-300) #not 0 but very small
                else:
                    if car_av == 1:
                        logp = logp + np.log(0.76)
                    else:
                        logp = logp + np.log(0.24)
                
            if ag_av == 1:
                logp = logp + np.log(0.1)
            elif ag_av == 0:
                logp = logp + np.log(0.9)
            else:
                logp = logp + np.log(1e-300) #not 0 but very small

            urban_prop_canton = urban_prop[canton]
            if urban == 1:
                logp = logp + np.log(max(urban_prop_canton,1e-300))
            elif urban == 0:
                logp = logp + np.log(max(1-urban_prop_canton, 1e-300))
            else:
                logp = logp + np.log(1e-300) #not 0 but very small

    return(logp)

def log_residence_density(instance):
    return(residence_0_density(instance) + residence_density(instance))

home_0_main = DurationPath.SingleTimeEvent("Residence", "Home0", "main_duration")
home_1_main = DurationPath.MultipleTimeEvent("Residence", "Home", 1, "main_duration")
home_2_main = DurationPath.MultipleTimeEvent("Residence", "Home", 2, "main_duration")
home_3_main = DurationPath.MultipleTimeEvent("Residence", "Home", 3, "main_duration")
home_4_main = DurationPath.MultipleTimeEvent("Residence", "Home", 4, "main_duration")
home_5_main = DurationPath.MultipleTimeEvent("Residence", "Home", 5, "main_duration")
home_6_main = DurationPath.MultipleTimeEvent("Residence", "Home", 6, "main_duration")
home_7_main = DurationPath.MultipleTimeEvent("Residence", "Home", 7, "main_duration")
home_8_main = DurationPath.MultipleTimeEvent("Residence", "Home", 8, "main_duration")
home_9_main = DurationPath.MultipleTimeEvent("Residence", "Home", 9, "main_duration")
home_10_main = DurationPath.MultipleTimeEvent("Residence", "Home", 10, "main_duration")
home_11_main = DurationPath.MultipleTimeEvent("Residence", "Home", 11, "main_duration")

no_work = DurationPath.NoEvent("Employment")
work_1_main = DurationPath.MultipleTimeEvent("Employment", "Work", 1, "main_duration")
work_2_main = DurationPath.MultipleTimeEvent("Employment", "Work", 2, "main_duration")
work_3_main = DurationPath.MultipleTimeEvent("Employment", "Work", 3, "main_duration")
work_4_main = DurationPath.MultipleTimeEvent("Employment", "Work", 4, "main_duration")
work_5_main = DurationPath.MultipleTimeEvent("Employment", "Work", 5, "main_duration")
work_6_main = DurationPath.MultipleTimeEvent("Employment", "Work", 6, "main_duration")
work_7_main = DurationPath.MultipleTimeEvent("Employment", "Work", 7, "main_duration")
work_8_main = DurationPath.MultipleTimeEvent("Employment", "Work", 8, "main_duration")
work_9_main = DurationPath.MultipleTimeEvent("Employment", "Work", 9, "main_duration")
work_10_main = DurationPath.MultipleTimeEvent("Employment", "Work", 10, "main_duration")
work_11_main = DurationPath.MultipleTimeEvent("Employment", "Work", 11, "main_duration")


work_1_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 1, "gap_duration")
work_2_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 2, "gap_duration")
work_3_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 3, "gap_duration")
work_4_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 4, "gap_duration")
work_5_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 5, "gap_duration")
work_6_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 6, "gap_duration")
work_7_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 7, "gap_duration")
work_8_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 8, "gap_duration")
work_9_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 9, "gap_duration")
work_10_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 10, "gap_duration")
work_11_gap = DurationPath.MultipleTimeEvent("Employment", "Work", 11, "gap_duration")

ic0 = DurationLinearConstraintsSpec(home_0_main, [no_work], [1], 0, 1)
ic1 = DurationLinearConstraintsSpec(home_1_main, [work_1_main, work_1_gap], [1, 1], 0, 1)
ic2 = DurationLinearConstraintsSpec(home_2_main, [work_2_main, work_2_gap], [1, 1], 0, 1)
ic3 = DurationLinearConstraintsSpec(home_3_main, [work_3_main, work_3_gap], [1, 1], 0, 1)
ic4 = DurationLinearConstraintsSpec(home_4_main, [work_4_main, work_4_gap], [1, 1], 0, 1)
ic5 = DurationLinearConstraintsSpec(home_5_main, [work_5_main, work_5_gap], [1, 1], 0, 1)
ic6 = DurationLinearConstraintsSpec(home_6_main, [work_6_main, work_6_gap], [1, 1], 0, 1)
ic7 = DurationLinearConstraintsSpec(home_7_main, [work_7_main, work_7_gap], [1, 1], 0, 1)
ic8 = DurationLinearConstraintsSpec(home_8_main, [work_8_main, work_8_gap], [1, 1], 0, 1)
ic9 = DurationLinearConstraintsSpec(home_9_main, [work_9_main, work_9_gap], [1, 1], 0, 1)
ic10 = DurationLinearConstraintsSpec(home_10_main, [work_10_main, work_10_gap], [1, 1], 0, 1)
ic11 = DurationLinearConstraintsSpec(home_11_main, [work_11_main, work_11_gap], [1, 1], 0, 1)

inter_constraints = [ic0, ic1, ic2, ic3, ic4, ic5, ic6, ic7, ic8, ic9, ic10, ic11]

def joint_duration(instance):
    return(log_employment_density(instance) + driving_license_log_density(instance) + education_log_density(instance) + log_residence_density(instance))



#####################################
##### load data #####################
#####################################

df_observed_2010 = pd.read_csv("2010_sample.csv")
df_observed_2015 = pd.read_csv("2015_sample.csv")
df_existence_2010_2015 = pd.read_csv("updated_sample_2010_2015_correctionsampling.csv")


##########################################
####### Define the mapping with the data
#########################################
canton_to_number = {
    "Zurich": 1,
    "Bern": 2,
    "Lucerne": 3,
    "Uri": 4,
    "Schwyz": 5,
    "Obwalden": 6,
    "Nidwalden": 7,
    "Glarus": 8,
    "Zug": 9,
    "Fribourg": 10,
    "Solothurn": 11,
    "Basel_Stadt": 12,
    "Basel_Landschaft": 13,
    "Schaffhausen": 14,
    "Appenzell_Ausserrhoden": 15,
    "Appenzell_Innerrhoden": 16,
    "St_Gallen": 17,
    "Graubunden": 18,
    "Aargau": 19,
    "Thurgau": 20,
    "Ticino": 21,
    "Vaud": 22,
    "Valais": 23,
    "Neuchatel": 24,
    "Geneva": 25,
    "Jura": 26
}

code_to_canton = {v: k for k, v in canton_to_number.items()}

neighbours_1  = [19, 14, 20, 17, 5, 9]                 # Zurich
neighbours_2  = [10, 22, 24, 26, 11, 19, 3, 6]         # Bern
neighbours_3  = [19, 9, 5, 6, 7, 2]                    # Lucerne
neighbours_4  = [5, 7, 6, 21, 18]                      # Uri
neighbours_5  = [1, 9, 3, 7, 4, 8, 17]                 # Schwyz
neighbours_6  = [3, 7, 4, 2]                           # Obwalden
neighbours_7  = [3, 6, 4, 5]                           # Nidwalden
neighbours_8  = [5, 17, 18]                            # Glarus
neighbours_9  = [1, 19, 3, 5]                          # Zug
neighbours_10 = [22, 2]                                # Fribourg
neighbours_11 = [13, 12, 26, 2, 19]                    # Solothurn
neighbours_12 = [13]                                   # Basel_Stadt
neighbours_13 = [12, 19, 11, 26]                       # Basel_Landschaft
neighbours_14 = [1, 20]                                # Schaffhausen
neighbours_15 = [17]                                   # Appenzell_Ausserrhoden
neighbours_16 = [17]                                   # Appenzell_Innerrhoden
neighbours_17 = [1, 20, 15, 16, 5, 8, 18]              # St_Gallen
neighbours_18 = [17, 8, 4, 21]                         # Graubunden
neighbours_19 = [1, 9, 3, 2, 11, 13]                   # Aargau
neighbours_20 = [1, 17, 14]                            # Thurgau
neighbours_21 = [4, 18, 23]                            # Ticino
neighbours_22 = [25, 23, 10, 2, 24]                    # Vaud
neighbours_23 = [22, 2, 4, 21]                         # Valais
neighbours_24 = [22, 2, 26]                            # Neuchatel
neighbours_25 = [22]                                   # Geneva
neighbours_26 = [13, 11, 2, 24]                        # Jura

# 26 x 26 matrix
is_neighbour = np.zeros((27, 27), dtype=np.int32)

# fill it
pairs = [
    (1, neighbours_1), (2, neighbours_2), (3, neighbours_3),
    (4, neighbours_4), (5, neighbours_5), (6, neighbours_6),
    (7, neighbours_7), (8, neighbours_8), (9, neighbours_9),
    (10, neighbours_10), (11, neighbours_11), (12, neighbours_12),
    (13, neighbours_13), (14, neighbours_14), (15, neighbours_15),
    (16, neighbours_16), (17, neighbours_17), (18, neighbours_18),
    (19, neighbours_19), (20, neighbours_20), (21, neighbours_21),
    (22, neighbours_22), (23, neighbours_23), (24, neighbours_24),
    (25, neighbours_25), (26, neighbours_26)
]

for i, neighs in pairs:
    for j in neighs:
        is_neighbour[i, j] = 1

neigh_count = np.zeros(27, dtype=np.int32)
for i in range(1, 27):
    neigh_count[i] = is_neighbour[i].sum()

bins_income = [
    -np.inf,   # for safety (shouldn't happen but robust)
    2000,
    4000,
    6000,
    8000,
    10000,
    12000,
    14000,
    16000,
    np.inf
]

labels_income = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def build_income_kernel(sigma_inc):
    K = np.empty((10, 10), dtype=np.float64)  # use indices 1..9

    for y in range(1, 10):
        denom = 0.0
        for k in range(1, 10):
            denom += np.exp(-((k - y) ** 2) / (2.0 * sigma_inc ** 2))

        for y_obs in range(1, 10):
            K[y_obs, y] = np.exp(-((y_obs - y) ** 2) / (2.0 * sigma_inc ** 2)) / denom

    return K

P_inc = build_income_kernel(1.5)


def active_events_at_time(df: pd.DataFrame, t: float) -> pd.DataFrame:
    """maps active event of each dimension at time t and keeps attributes"""
    df = df.copy()

    # --- compute end dates
    main_end = df["main_start_date"] + df["main_duration"]
    gap_end  = df["gap_start_date"] + df["gap_duration"]

    # --- activity indicators
    df["main"] = ((df["main_start_date"] <= t) & (t < main_end)).astype(int)
    df["gap"]  = ((df["gap_start_date"] <= t) & (t < gap_end)).astype(int)

    # --- keep only active rows
    df_active = df[(df["main"] == 1) | (df["gap"] == 1)].copy()

    # --- resolve conflicts: one event per (id, dim_name)
    df_active["priority"] = df_active["main"]  # main > gap

    df_active = (
        df_active
        .sort_values(["id", "dim_name", "priority"], ascending=[True, True, False])
        .drop_duplicates(subset=["id", "dim_name"], keep="first")
        .drop(columns="priority")
    )

    # --- IMPORTANT: keep ALL columns
    # Just reorder so main/gap are visible
    cols_front = ["id", "dim_name", "event_name", "main", "gap"]
    other_cols = [c for c in df_active.columns if c not in cols_front]

    result = df_active[cols_front + other_cols].reset_index(drop=True)

    return result

def add_employment_status(df_active: pd.DataFrame, t: float) -> pd.DataFrame:
    df = df_active.copy()

    # --------------------------------------------------
    # 1. Get age from Existence
    # --------------------------------------------------
    existence = df[df["dim_name"] == "Existence"][["id", "main_start_date"]].copy()
    existence["age"] = t - existence["main_start_date"]

    # --------------------------------------------------
    # 2. Get Birth end date
    # --------------------------------------------------
    birth = df[df["event_name"] == "Birth"][["id", "main_start_date", "duration"]].copy()
    birth["birth_end"] = birth["main_start_date"] + birth["duration"]
    birth = birth[["id", "birth_end"]]

    # --------------------------------------------------
    # 3. Merge into Employment rows only
    # --------------------------------------------------
    emp = df[df["dim_name"] == "Employment"].copy()

    emp = emp.merge(existence[["id", "age"]], on="id", how="left")
    emp = emp.merge(birth, on="id", how="left")

    # compute end date of employment event
    emp["event_end"] = emp["main_start_date"] + emp["duration"]

    # --------------------------------------------------
    # 4. Initialize
    # --------------------------------------------------
    emp["employment_status"] = np.nan

    # --------------------------------------------------
    # 5. RULES
    # --------------------------------------------------

    # --- CASE 1: main = 1
    mask_main = emp["main"] == 1

    # 1a: employed (not NoWork)
    emp.loc[mask_main & (emp["event_name"] != "NoWork"), "employment_status"] = 1

    # 1b: NoWork
    mask_nowork = mask_main & (emp["event_name"] == "NoWork")
    emp.loc[mask_nowork & (emp["age"] < 15), "employment_status"] = 4
    emp.loc[mask_nowork & (emp["age"] >= 15), "employment_status"] = 2

    # --- CASE 2: gap = 1
    mask_gap = emp["gap"] == 1

    # default unemployed
    emp.loc[mask_gap, "employment_status"] = 2

    # retirement condition
    tol = 1e-6 
    mask_retired = (
        mask_gap
        & (emp["age"] >= 64)
        & np.isclose(emp["event_end"], emp["birth_end"], atol=tol)
    )
    emp.loc[mask_retired, "employment_status"] = 5

    # kids override (safety)
    emp.loc[emp["age"] < 15, "employment_status"] = 4

    # --------------------------------------------------
    # 6. Put back into dataframe
    # --------------------------------------------------
    df = df.merge(
        emp[["id", "dim_name", "event_name", "employment_status"]],
        on=["id", "dim_name", "event_name"],
        how="left"
    )

    return df

def add_education_status(df_active: pd.DataFrame) -> pd.DataFrame:
    df = df_active.copy()

    # --- select education rows
    edu = df[df["dim_name"] == "Education"].copy()

    # --- initialize
    edu["education_status"] = 0  # default = 0, finished / not in education

    # --- NoEducation corresponds to age < 6
    mask_no_education = edu["event_name"] == "NoEducation"
    edu.loc[mask_no_education, "education_status"] = 2

    # --- main education event, excluding NoEducation -> 1
    mask_main = (edu["main"] == 1) & (edu["event_name"] != "NoEducation")
    edu.loc[mask_main, "education_status"] = 1

    # --- merge back
    df = df.merge(
        edu[["id", "dim_name", "event_name", "education_status"]],
        on=["id", "dim_name", "event_name"],
        how="left"
    )

    return df

def collapse_to_individual(df_active: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "id",
        "canton_residence",
        "urban",
        "canton_work",
        "Income",
        "Income_cat",
        "employment_status",
        "education_status",
        "has_license",
        "car_available",
        "has_ga",
        "age",
        "gender"
        
    ]
    
    cols = [c for c in cols if c in df_active.columns]

    return (
        df_active[cols]
        .groupby("id", sort=False, as_index=False)
        .first()
    )


def mapping(df_time_indep, t):
    # map active event in each dimension
    df_active_events = active_events_at_time(df_time_indep, t)

    # map variables to compare with df
    df_active_events["canton_residence"] = df_active_events["PlaceResidence"].map(canton_to_number)
    df_active_events["canton_work"] = df_active_events["PlaceWork"].map(canton_to_number)
    df_active_events["Income"] = np.exp(df_active_events["LogIncome"].astype(np.float64))
    df_active_events["Income_cat"] = pd.cut(
        df_active_events["Income"],
        bins=bins_income,
        labels=labels_income,
        right=True,   # intervals like (a, b]
    ).astype("Int64")

    df_active_events = add_employment_status(df_active_events, t)
    df_active_events = add_education_status(df_active_events)

    #driving license
    mask_dl = (df_active_events["event_name"] == "DL") & (df_active_events["main"] == 1)

    df_active_events["has_license"] = (
        mask_dl.groupby(df_active_events["id"]).transform("any").astype(int)
    )
    
    mask_res = df_active_events["dim_name"] == "Residence"

    df_active_events.loc[mask_res, "car_available"] = (
        df_active_events.loc[mask_res, "has_license"] *
        df_active_events.loc[mask_res, "Car_availability"].fillna(0)
    ).astype(int)

    df_active_events["has_ga"] = df_active_events["AG_availability"]
    df_active_events["urban"] = df_active_events["Urban"]

    # select existence rows (already unique per id after dedup)
    existence = df_active_events[df_active_events["dim_name"] == "Existence"].copy()

    # compute age
    existence["age"] = t - existence["main_start_date"]

    # gender = Sex
    existence["gender"] = existence["Sex"]

    # keep only needed columns
    existence = existence[["id", "age", "gender"]]

    # merge back to all rows
    df_active_events = df_active_events.merge(existence, on="id", how="left")

    df_final = collapse_to_individual(df_active_events)
    return(df_final)



#########################
##### likelihood ######
#########################
P = np.array([
    [0.85, 0.60, 0.10, 0.03],
    [0.05, 0.30, 0.02, 0.02],
    [0.08, 0.05, 0.78, 0.10],
    [0.02, 0.05, 0.10, 0.85]
], dtype=np.float64)


P_emp = np.array([
    [0.90, 0.08, 0.00, 0.02],  # observed = 1
    [0.08, 0.90, 0.00, 0.02],  # observed = 2
    [0.00, 0.00, 1.00, 0.00],  # observed = 4
    [0.02, 0.02, 0.00, 0.96]   # observed = 5
], dtype=np.float64)

@njit
def map_emp(e):
    if e == 1:
        return 0
    elif e == 2:
        return 1
    elif e == 4:
        return 2
    else:
        return 3
    

sigma_age = 3
misclassification_rate = 0.05

@njit(parallel=True)
def f_all(
    obs_canton_res, sim_canton_res,
    obs_urban, sim_urban,
    obs_has_ga, sim_has_ga,
    obs_educ_status, sim_educ_status,
    obs_has_license, sim_has_license,
    obs_car_av, sim_car_av,
    obs_emp_status, sim_emp_status,
    obs_canton_work, sim_canton_work,
    obs_income, sim_income,
    obs_age, sim_age, obs_gender, sim_gender
):

    n_obs = len(obs_canton_res)
    n_sim = len(sim_canton_res)

    out = np.empty((n_obs, n_sim), dtype=np.float64)

    log_norm_const = -np.log(sigma_age * np.sqrt(2.0 * np.pi))
    p_same = 1.0 - misclassification_rate
    p_diff = misclassification_rate

    for i in prange(n_obs):
        # --------------------
        # observed values
        # --------------------
        obs_l = obs_has_license[i]
        obs_ca = obs_car_av[i]
        obs_car_idx = obs_l * 2 + obs_ca

        a_i = obs_age[i]
        g_i = obs_gender[i]
        obs_e = obs_educ_status[i]

        obs_c_res = obs_canton_res[i]
        obs_u = obs_urban[i]
        obs_ga = obs_has_ga[i]

        obs_emp = obs_emp_status[i]
        obs_emp_idx = map_emp(obs_emp)

        obs_work = obs_canton_work[i]
        obs_inc = obs_income[i]

        work_missing_obs = np.isnan(obs_work)
        inc_missing_obs = np.isnan(obs_inc)

        n_neigh_res = neigh_count[obs_c_res]

        for j in range(n_sim):
            val = 1.0

            sim_u = sim_urban[j]
            if sim_u == obs_u:
                p_urban = 0.7
            else:
                p_urban = 0.3

            sim_c_res = sim_canton_res[j]
            if obs_c_res == sim_c_res:
                p_res_canton = 0.8
            elif is_neighbour[obs_c_res, sim_c_res] == 1:
                p_res_canton = 0.15 / n_neigh_res
            else:
                p_res_canton = 0.05 / (26 - n_neigh_res - 1)


            sim_ga = sim_has_ga[j]
            if sim_ga == obs_ga:
                p_ga = 0.9
            else:
                p_ga = 0.1

            val *= p_urban * p_res_canton * p_ga

            # ==================================================
            # 2. EDUCATION
            # ==================================================
            sim_e = sim_educ_status[j]
            if sim_e == obs_e:
                val *= 0.8
            else:
                val *= 0.2

            # ==================================================
            # 3. LICENSE + CAR
            # ==================================================
            sim_l = sim_has_license[j]
            sim_ca = sim_car_av[j]
            sim_car_idx = sim_l * 2 + sim_ca

            val *= P[obs_car_idx, sim_car_idx]

           ###### age gender
            diff = a_i - sim_age[j]
            value_age = -0.5 * (diff / sigma_age) ** 2 + log_norm_const
            age_val = np.exp(value_age)
            val = val*age_val

            if g_i == sim_gender[j]:
                gender_val = p_same
            else:
                gender_val = p_diff

            val = val*gender_val

            # ==================================================
            # 4. EMPLOYMENT + CONDITIONAL VARIABLES
            # ==================================================
            sim_emp = sim_emp_status[j]
            sim_emp_idx = map_emp(sim_emp)

            p_emp = P_emp[obs_emp_idx, sim_emp_idx]

            # observed not employed
            if obs_emp != 1:
                val *= p_emp
                out[i, j] = val
                continue

            # simulated not employed
            if sim_emp != 1:
                val *= p_emp
                out[i, j] = val
                continue

            # both employed
            sim_work = sim_canton_work[j]
            sim_inc = sim_income[j]

            work_missing_sim = np.isnan(sim_work)
            inc_missing_sim = np.isnan(sim_inc)

            has_work = (not work_missing_obs) and (not work_missing_sim)
            has_inc = (not inc_missing_obs) and (not inc_missing_sim)

            if (not has_work) and (not has_inc):
                val *= p_emp
                out[i, j] = val
                continue

            # work part
            if has_work:
                obs_c_work = int(obs_work)
                sim_c_work = int(sim_work)

                n_neigh_work = neigh_count[obs_c_work]

                if obs_c_work == sim_c_work:
                    p_work = 0.8
                elif is_neighbour[obs_c_work, sim_c_work] == 1:
                    p_work = 0.15 / n_neigh_work
                else:
                    p_work = 0.05 / (26 - n_neigh_work - 1)
            else:
                p_work = 1.0

            # income part
            if has_inc:
                obs_y = int(obs_inc)
                sim_y = int(sim_inc)
                p_inc = P_inc[obs_y, sim_y]
            else:
                p_inc = 1.0

            # combine exactly as in your original code
            if has_work and has_inc:
                val *= (p_emp * p_work * p_inc) ** (1.0 / 3.0)
            elif has_work:
                val *= (p_emp * p_work) ** 0.5
            else:
                val *= (p_emp * p_inc) ** 0.5

            out[i, j] = val

    return(out)


def f(obs, sim, it):
    res= f_all(
        obs["canton_residence"].astype(np.int32),
        sim["canton_residence"].astype(np.int32),

        obs["urban"].astype(np.int32),
        sim["urban"].astype(np.int32),

        obs["has_ga"].astype(np.int32),
        sim["has_ga"].astype(np.int32),

        obs["education_status"].astype(np.int32),
        sim["education_status"].astype(np.int32),

        obs["has_license"].astype(np.int32),
        sim["has_license"].astype(np.int32),

        obs["car_available"].astype(np.int32),
        sim["car_available"].astype(np.int32),

        obs["employment_status"].astype(np.int32),
        sim["employment_status"].astype(np.int32),

        np.asarray(obs["canton_work"], dtype=np.float64),
        np.asarray(sim["canton_work"], dtype=np.float64),

        np.asarray(obs["hh_income"], dtype=np.float64),
        np.asarray(sim["Income_cat"], dtype=np.float64),

        obs["age"].astype(np.float64),
        sim["age"].astype(np.float64),
        obs["gender"].astype(np.int32),
        sim["gender"].astype(np.int32)
    )

    return(res)


def filter_sim_age_above_7(obs_dict, sim_dict, it):
    sim_ok = sim_dict["age"] > 7

    return np.broadcast_to(
        sim_ok,
        (len(next(iter(obs_dict.values()))), len(sim_ok))
    )


######################
############# kernel and update
#######################
n_samples = 5000
pop_spec = PopulationSpec(n_samples, [d0, d1, d2, d3, d4], joint_duration, inter_constraints)
pop_spec_prior = PopulationSpec(n_samples, [d0_prior, d1_prior, d2_prior, d3_prior, d4_prior], joint_duration_prior, inter_constraints)

t1 = 2010
kernel_2010 = DimKernel(
    df_observed_2010, 
    t1, 
    f, 
    mapping, 
    weights = None, 
    normalize_weights = True,
    sim_filter=filter_sim_age_above_7)

t2 = 2015
kernel_2015 = DimKernel(
    df_observed_2015, 
    t2, 
    f, 
    mapping, 
    weights = None, 
    normalize_weights = True,
    sim_filter=filter_sim_age_above_7)

updater = JointDimPopUpdater(pop_spec, [kernel_2010, kernel_2015], pop_spec_prior)

n_init = 2500
def n_prop_function(i):
    return(int(n_init/np.sqrt(i+1)))


n_iter = 100000  # Number of iterations for MC
n_gibbs_indic = 1000
n_prop_min = 1

if __name__ == "__main__":
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    observed_ids, never_observed_ids, observed_kernels = updater._get_observed_vs_never_ids(df_existence_2010_2015)

    observed_ids = sorted(observed_ids)

    ex_dim_spec = updater.pop_spec.list_dim_spec[0]
    id_to_traj = dfex_to_trajex(ex_dim_spec, df_existence_2010_2015)

    indiv_spec_posterior = updater._make_indiv_spec(indiv_id=None, bool_posterior = 1)


    # Update the population
    updated_pop, accept_rate_dur, accept_rate_attr, n_changed_dur = updater.sample_posterior_alive_checkpointed(id_to_traj = id_to_traj, 
                                                                                                indiv_spec_posterior = indiv_spec_posterior,
                                                                                                observed_ids= observed_ids, 
                                                                                                observed_kernels=observed_kernels, 
                                                                                                n_MH = n_iter,
                                                                                                n_prop_function=n_prop_function,
                                                                                                n_prop_min=n_prop_min,
                                                                                                n_gibbs_indicators=n_gibbs_indic,
                                                                                                n_jobs_posterior_init=n_workers,
                                                                                                dur_local_frac=0.8,
                                                                                                dur_p_far=0.5,
                                                                                                seed = 0,
                                                                                                checkpoint_every=500,
                                                                                                checkpoint_path="updated_alive_full_2010_2015_checkpoint_sampling",
                                                                                                resume = True)
    
    # Convert the updated population to DataFrame
    updated_pop_df = updated_pop.to_dataframe()
    
    # Save the updated population to CSV
    updated_pop_df.to_csv("updated_alive_2010_2015_sampling.csv", index=False)
    
    # Save accept_rate_dur, accept_rate_attr, and likelihood to CSV
    accept_rate_dur_df = pd.DataFrame({"accept_rate_dur": accept_rate_dur})
    accept_rate_dur_df.to_csv("accept_rate_dur_2010_2015_sampling.csv", index=False)

    accept_rate_attr_df = pd.DataFrame({"accept_rate_attr": accept_rate_attr})
    accept_rate_attr_df.to_csv("accept_rate_attr_2010_2015_sampling.csv", index=False)

