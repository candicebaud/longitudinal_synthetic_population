""" Model definition (work included)"""

import numpy as np
from scipy.stats import norm, uniform, expon, gamma, lognorm, beta, poisson
import math
from scipy.special import gammaln, logsumexp
import random

from pre_coded_function import *
from spec_other_dim import *
from spec_existence import *

#### Densities
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

def log_indicator_function(a, b, tol=0.1):
    if abs(a - b) <= tol:
        return 0.0  # log(1)
    else:
        return -np.inf  # log(0)
    
def truncated_lognorm_logpdf(x, mu, sigma, a, b):
    if x < a or x > b:
        return -np.inf
    
    dist = lognorm(s=sigma, scale=np.exp(mu))
    Z = dist.cdf(b) - dist.cdf(a)

    if Z <= 0:
        return -np.inf

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
        return -np.inf

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
        return -np.inf
    return float(np.log(pdf_val))

def log_truncated_normal_pdf(x, mu, sigma2, a, b):
    """
    Log-PDF of N(mu, sigma2) truncated to [a, b]
    """
    if x < a or x > b:
        return -np.inf

    sigma = np.sqrt(sigma2)

    # log pdf of normal
    log_pdf = norm.logpdf(x, mu, sigma)

    # normalization constant
    Z = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)

    if Z <= 0:
        return -np.inf

    return log_pdf - np.log(Z)

def log_weibull_pdf_truncated(x, l, k, a, b):
    if x < a or x > b:
        return -np.inf

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

# def log_truncated_normal_pdf(x, mu, sigma2, a, b):
#     return log_from_pdf(truncated_normal_pdf(x, mu, sigma2, a, b))

# def log_truncated_exponential_pdf(x, lambd, a, b):
#     return log_from_pdf(truncated_exponential_pdf(x, lambd, a, b))

# def log_weibull_pdf_truncated(x, k, lam, a, b):
#     """Weibull(k, lam) truncated to [a,b]. Uses your existing weibull_pdf_truncated if you have it."""
#     return log_from_pdf(weibull_pdf_truncated(x, k, lam, a, b))


#### Existence
def existence_density_prior(instance):
    log_p = weibull_logpdf(instance["Birth"]["main_duration"], 85, 3)
    return float(log_p)

def sex_sampler_prior(instance, rng):
    return(rng.binomial(1, 0.5))
    
def sex_sampler_initial_prior(rng):
    return(rng.binomial(1, 0.5))

p_a0_prior = ProbabilisticModel.Gibbs(sex_sampler_prior, sex_sampler_initial_prior)
a0_prior = AttributeSpec.SingleAttr("Sex", p_a0_prior)

d0_prior = ExistenceDimensionSpec([a0_prior], existence_density_prior, constraints=BirthConstraintSpec(1900, 2050, 110, 0))

#### DL
def ndl_log_density_prior(instance):
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

def dl_log_density_prior(instance):
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

def driving_license_log_density_prior(instance):
    logp = 0.0
    logp += ndl_log_density_prior(instance)
    logp += dl_log_density_prior(instance)
    return logp

def dl_initialize_indic_prior(rng):
    return(1)

def dl_gibbs_indic_prior(instance, rng):
    return(rng.binomial(1, 0.85)) 

e1_0_prior = EventSpec.NoEvent("NoDL")
p_e1_1_prior = ProbabilisticModel.Gibbs(dl_gibbs_indic_prior, dl_initialize_indic_prior)
c_e1_1_prior = EventTemporalConstraintsSpec(18, 70, None, None, None, None, None, 0, None)
e1_1_prior = EventSpec.SingleTimeEvent("DL", p_e1_1_prior, c_e1_1_prior, None)
d1_prior = DimensionSpec("DL", [e1_0_prior, e1_1_prior], None)

#### Education
def log_ned_density_prior(instance):
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


def log_me_density_prior(instance):
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


def log_se_density_prior(instance):
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


def log_te_density_prior(instance):
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
            return -np.inf
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


def education_log_density_prior(instance):
    lp = 0.0

    ln = log_ned_density_prior(instance)
    if not np.isfinite(ln):
        return -np.inf
    lp += ln

    lm = log_me_density_prior(instance)
    if not np.isfinite(lm):
        return -np.inf
    lp += lm

    ls = log_se_density_prior(instance)
    if not np.isfinite(ls):
        return -np.inf
    lp += ls

    lt = log_te_density_prior(instance)
    if not np.isfinite(lt):
        return -np.inf
    lp += lt

    return lp

def marginal_indicator_mandatory_prior(instance, rng):
    return(1)

def marginal_indicator_secondary_prior(instance, rng):
    bool = rng.binomial(1, 0.95)
    return(bool)

def marginal_indicator_tertiary_prior(instance, rng):
    bool = rng.binomial(1, 0.6)
    return(bool)

def initialize_mandatory_prior(rng):
    return(1)

def initialize_secondary_prior(rng):
    return(rng.binomial(1, 0.95))

def initialize_tertiary_prior(rng):
    return(rng.binomial(1, 0.6))

e0_prior = EventSpec.NoEvent("NoEducation")

#mandatory
p_e1_prior = ProbabilisticModel.Gibbs(marginal_indicator_mandatory_prior, initialize_mandatory_prior)
c_e1_prior = EventTemporalConstraintsSpec(6, 6, 9, 9, None, 15, None, None, None)
e1_prior = EventSpec.SingleTimeEvent("MandatoryEduc", p_e1_prior, c_e1_prior, None)

#secondary
p_e2_prior = ProbabilisticModel.Gibbs(marginal_indicator_secondary_prior, initialize_secondary_prior)
c_e2_prior = EventTemporalConstraintsSpec(15, 15, 3, 6, None, 21, None, None, None)
e2_prior = EventSpec.SingleTimeEvent("SecondaryEduc", p_e2_prior, c_e2_prior, None)

#tertiary
p_e3_prior = ProbabilisticModel.Gibbs(marginal_indicator_tertiary_prior, initialize_tertiary_prior)
c_e3_prior = EventTemporalConstraintsSpec(18, 25, None, 15, None, 40, None, None, None)
e3_prior = EventSpec.SingleTimeEvent("TertiaryEduc", p_e3_prior, c_e3_prior, None)

d2_prior = DimensionSpec("Education", [e0_prior, e1_prior, e2_prior, e3_prior], None)


#### Employment
def indicator_function(a,b):
    # a bit more flexible than pure equality
    if abs(a-b) <= 0.1:
        return(1)
    else:
        return(0)
    
def count_pdf_work_prior(instance, rng):
    lifespan = instance["Existence"]["Birth"]["main_duration"]
    ret_age = 64
    lambda_value = math.sqrt(min(lifespan, ret_age) - 15)
    sample = rng.poisson(lambda_value)
    while sample >11:
        sample = rng.poisson(lambda_value)
    return(sample)

def count_pdf_work_init_prior(rng):
    sample = rng.poisson(6)
    return(sample)

def nem_density_prior(instance):
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

def lambda_age_work_prior(age):
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
    
def work_density_prior(instance):
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
                p = p + log_truncated_exponential_pdf(main_duration, lambda_age_work_prior(age_at_start), 0, min(70 - d_nem - cumulated_duration, lifespan - d_nem - cumulated_duration))
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
                    l = lambda_age_work_prior(age_at_start)
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
                p = p + log_truncated_exponential_pdf(main_duration, lambda_age_work_prior(age_at_start), 0, min(70 - d_nem - cumulated_duration, lifespan - d_nem - cumulated_duration))
                p = p + log_indicator_function(gap_duration, lifespan - cumulated_duration - d_nem - main_duration) 
            else:
                p = p + log_indicator_function(main_duration, 0)
                p = p + log_indicator_function(gap_duration, 0)

            # last actually done employment spell, density 
            # if lifespan > 64 :
            #     gap_start_date = instance["Employment"][f"Work_{indic_zero}"]["gap_start_date"]
            #     p = p*truncated_normal_pdf(gap_start_date, 65 + dob, 1, 64 + dob, 70 + dob)
    
    return(p)

def log_employment_density_prior(instance):
    return(nem_density_prior(instance) + work_density_prior(instance))


# attributes
urb_cantons = ["Zurich", "Vaud", "Geneva", "Basel_Stadt"]
neighbouring_cantons = {
    "Zurich": [
        "Aargau", "Schaffhausen", "Thurgau",
        "St_Gallen", "Schwyz", "Zug"
    ],

    "Bern": [
        "Fribourg", "Vaud", "Neuchatel", "Jura",
        "Solothurn", "Aargau", "Lucerne", "Obwalden"
    ],

    "Lucerne": [
        "Aargau", "Zug", "Schwyz",
        "Obwalden", "Nidwalden", "Bern"
    ],

    "Uri": [
        "Schwyz", "Nidwalden", "Obwalden",
        "Ticino", "Graubunden"
    ],

    "Schwyz": [
        "Zurich", "Zug", "Lucerne",
        "Nidwalden", "Uri", "Glarus", "St_Gallen"
    ],

    "Obwalden": [
        "Lucerne", "Nidwalden", "Uri", "Bern"
    ],

    "Nidwalden": [
        "Lucerne", "Obwalden", "Uri", "Schwyz"
    ],

    "Glarus": [
        "Schwyz", "St_Gallen", "Graubunden"
    ],

    "Zug": [
        "Zurich", "Aargau", "Lucerne", "Schwyz"
    ],

    "Fribourg": [
        "Vaud", "Bern"
    ],

    "Solothurn": [
        "Basel_Landschaft", "Basel_Stadt",
        "Jura", "Bern", "Aargau"
    ],

    "Basel_Stadt": [
        "Basel_Landschaft"
    ],

    "Basel_Landschaft": [
        "Basel_Stadt", "Aargau", "Solothurn", "Jura"
    ],

    "Schaffhausen": [
        "Zurich", "Thurgau"
    ],

    "Appenzell_Ausserrhoden": [
        "St_Gallen"
    ],

    "Appenzell_Innerrhoden": [
        "St_Gallen"
    ],

    "St_Gallen": [
        "Zurich", "Thurgau",
        "Appenzell_Ausserrhoden", "Appenzell_Innerrhoden",
        "Schwyz", "Glarus", "Graubunden"
    ],

    "Graubunden": [
        "St_Gallen", "Glarus", "Uri", "Ticino"
    ],

    "Aargau": [
        "Zurich", "Zug", "Lucerne",
        "Bern", "Solothurn", "Basel_Landschaft"
    ],

    "Thurgau": [
        "Zurich", "St_Gallen", "Schaffhausen"
    ],

    "Ticino": [
        "Uri", "Graubunden", "Valais"
    ],

    "Vaud": [
        "Geneva", "Valais",
        "Fribourg", "Bern", "Neuchatel"
    ],

    "Valais": [
        "Vaud", "Bern", "Uri", "Ticino"
    ],

    "Neuchatel": [
        "Vaud", "Bern", "Jura"
    ],

    "Geneva": [
        "Vaud"
    ],

    "Jura": [
        "Basel_Landschaft", "Solothurn", "Bern", "Neuchatel"
    ]
}

canton_proportions = {
    "Zurich": 0.1791,
    "Bern": 0.1187,
    "Vaud": 0.0944,
    "Aargau": 0.0811,
    "St_Gallen": 0.0597,
    "Geneva": 0.0585,
    "Lucerne": 0.0483,
    "Valais": 0.0408,
    "Ticino": 0.0399,
    "Fribourg": 0.0381,
    "Basel_Landschaft": 0.0333,
    "Thurgau": 0.0329,
    "Solothurn": 0.0320,
    "Graubunden": 0.0229,
    "Basel_Stadt": 0.0223,
    "Neuchatel": 0.0199,
    "Schwyz": 0.0187,
    "Zug": 0.0148,
    "Schaffhausen": 0.0097,
    "Jura": 0.0083,
    "Appenzell_Ausserrhoden": 0.0063,
    "Nidwalden": 0.0050,
    "Glarus": 0.0047,
    "Obwalden": 0.0044,
    "Uri": 0.0042,
    "Appenzell_Innerrhoden": 0.0019
}

cantons = list(canton_proportions.keys())
proportions = list(canton_proportions.values())
delta = 1.5
gamma = 6000

def boost_and_renormalize_prior(cantons, proportions, delta, boosted_cantons):
    """
    Boost selected cantons by a factor delta and renormalize so proportions sum to 1.

    Parameters
    ----------
    cantons : list of str
        Canton names (same order as proportions).
    proportions : list or array of float
        Original proportions (must sum to ~1).
    delta : float
        Boosting factor (> 0).
    boosted_cantons : set or list of str
        Cantons whose proportions are multiplied by delta.

    Returns
    -------
    new_proportions : np.ndarray
        Renormalized proportions summing exactly to 1.
    """
    p = np.array(proportions, dtype=float)

    # Apply boost
    for i, c in enumerate(cantons):
        if c in boosted_cantons:
            p[i] *= delta

    # Renormalize
    p /= p.sum()

    return dict(zip(cantons, p))

canton_proportions_boost = boost_and_renormalize_prior(cantons, proportions, delta, urb_cantons)

def logIncomeinitialdist_prior(index, rng):
    mean = 5000
    std = 1500
    sample = rng.normal(mean, std)
    while sample < 0 :
        sample = rng.normal(mean, std)
    return(np.log(sample))

def logIncomedist_prior(instance, index, rng):
    beta_0 = 7.8
    beta_1 = 0.025
    beta_2 = -0.0004
    rho = 0.1
    sigma_eps2 = 0.3

    work = 0 
    age_start_work_spell = instance["Employment"]["NoWork"]["main_duration"] + instance["Employment"]["NoWork"]["gap_duration"]

    # if index == 0:
    #     work = 0
    #     age_start_work_spell = instance["Employment"]["NoWork"]["main_duration"] + instance["Employment"]["NoWork"]["gap_duration"]

    for i in range (index):
        work = work + instance["Employment"][f"Work_{i+1}"]["main_duration"]
        age_start_work_spell = age_start_work_spell + instance["Employment"][f"Work_{i+1}"]["main_duration"] + instance["Employment"][f"Work_{i+1}"]["gap_duration"]

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
    value = rng.normal(mean, sigma_eps2)
    return(value)

def place_initial_dist_work_prior(index, rng):
    # initialization : sample at random based only on the actual proportion of inhabitants in Swiss cantons
    # same function for work and for residence
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions)
    )

    return(canton)

def pow_distribution_prior(instance, index, rng):
    log_inc = instance["Employment"][f"Work_{index + 1}"]["LogIncome"]
    inc = np.exp(log_inc)

    cantons = list(canton_proportions.keys())
    proportions = list(canton_proportions.values())

    if inc > gamma:
        new_proportions = list(canton_proportions_boost.values())
    else:
        new_proportions = proportions

    canton = rng.choice(
    cantons,
    p=new_proportions/np.sum(new_proportions)
    )

    return(canton)
    

e3_0_prior = EventSpec.NoEvent("NoWork")

p_a_3_1_1_prior = ProbabilisticModel.Gibbs(logIncomedist_prior, logIncomeinitialdist_prior)
a_3_1_1_prior = AttributeSpec.MultipleAttr("LogIncome", p_a_3_1_1_prior)
p_a_3_1_2_prior = ProbabilisticModel.Gibbs(pow_distribution_prior, place_initial_dist_work_prior)
a_3_1_2_prior = AttributeSpec.MultipleAttr("PlaceWork", p_a_3_1_2_prior)

p_e3_1_prior = ProbabilisticModel.Gibbs(count_pdf_work_prior, count_pdf_work_init_prior)
c_e3_1_prior = EventTemporalConstraintsSpec(15, 64, None, None, None, 70, None, None, None)
e3_1_prior = EventSpec.MultipleTimeEvent("Work", 11, p_e3_1_prior, c_e3_1_prior, [a_3_1_1_prior, a_3_1_2_prior])

d3_prior = DimensionSpec("Employment", [e3_0_prior, e3_1_prior], None)

#### Residence 
def count_pdf_residence_init_prior(rng):
    return(1)

def count_pdf_residence_prior(instance, rng):
    total = 0
    for k in range (11):
        ev_name = f"Work_{k+1}"
        total = total + instance["Employment"][ev_name]["indicator"]
    return(total)

def place_initial_dist_residence_prior(index, rng):
    # initialization : sample at random based only on the actual proportion of inhabitants in Swiss cantons
    # same function for work and for residence
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions),
    )

    return(canton)

def place_initial_dist_0_prior(rng):
    # initialization : sample at random based only on the actual proportion of inhabitants in Swiss cantons
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions)
    )

    return(canton)

def por_distribution_0_prior(instance, rng):
    canton = rng.choice(
    cantons,
    p=proportions/np.sum(proportions),
    )

    return(canton)

p_w = 0.05
def por_distribution_prior(instance, index, rng):
    canton_work = instance["Employment"][f"Work_{index+1}"]["PlaceWork"] 
    
    # sample uniformly a neighbouring canton with probability 0.05
    bool = rng.binomial(1, p_w)
    if bool == 1 :
        canton_residence = rng.choice(neighbouring_cantons[canton_work])
    else:
        canton_residence = canton_work

    return(canton_residence)
    
def car_av_initial_dist_prior(index, rng):
    return(0) # initialize everyone at 0 so there is no error

def car_av_initial_dist_0_prior(rng):
    return(0) # for Home_0 as well return 0 

def car_distribution_prior(instance, index, rng):
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

def car_distribution_0_prior(instance, rng):
    return(0) # starts at birth, so never has a car availability with this spell

def ag_av_initial_dist_0_prior(rng):
    return(rng.binomial(1, 0.1))

def ag_av_initial_dist_prior(index, rng):
    return(rng.binomial(1, 0.1))

def ag_distribution_prior(instance, index, rng):
    return(rng.binomial(1, 0.1))

def ag_distribution_0_prior(instance, rng):
    return(rng.binomial(1, 0.1))

def urban_initial_dist_0_prior(rng):
    return(rng.binomial(1, 0.85)) 

def urban_initial_dist_prior(index, rng):
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

def urban_distribution_prior(instance, index, rng):
    canton_of_res = instance["Residence"][f"Home_{index + 1}"]["PlaceResidence"]
    prop = urban_prop[canton_of_res]
    return rng.binomial(1, prop)

def urban_distribution_0_prior(instance, rng):
    canton_of_res = instance["Residence"]["Home0"]["PlaceResidence"]
    prop = urban_prop[canton_of_res]
    return rng.binomial(1, prop)

    
p_a4_0_prior = ProbabilisticModel.Gibbs(por_distribution_0_prior, place_initial_dist_0_prior)
a4_0_prior = AttributeSpec.SingleAttr("PlaceResidence", p_a4_0_prior)
p_a4_0_1_prior = ProbabilisticModel.Gibbs(car_distribution_0_prior, car_av_initial_dist_0_prior)
a4_0_1_prior = AttributeSpec.SingleAttr("Car_availability", p_a4_0_1_prior)
p_a4_0_2_prior = ProbabilisticModel.Gibbs(ag_distribution_0_prior, ag_av_initial_dist_0_prior)
a4_0_2_prior = AttributeSpec.SingleAttr("AG_availability", p_a4_0_2_prior)
p_a4_0_3_prior = ProbabilisticModel.Gibbs(urban_distribution_0_prior, urban_initial_dist_0_prior)
a4_0_3_prior = AttributeSpec.SingleAttr("Urban", p_a4_0_3_prior)

p_e4_0_prior = indic_prob_model_no_event
c_e4_0_prior = EventTemporalConstraintsSpec(None, None, None, None, None, None, None, 0, None)
e4_0_prior = EventSpec.SingleTimeEvent("Home0", p_e4_0_prior, c_e4_0_prior, [a4_0_prior, a4_0_1_prior, a4_0_2_prior, a4_0_3_prior])

p_a4_1_prior = ProbabilisticModel.Gibbs(por_distribution_prior, place_initial_dist_residence_prior)
a4_1_prior = AttributeSpec.MultipleAttr("PlaceResidence", p_a4_1_prior)
p_a4_1_1_prior = ProbabilisticModel.Gibbs(car_distribution_prior, car_av_initial_dist_prior)
a4_1_1_prior = AttributeSpec.MultipleAttr("Car_availability", p_a4_1_1_prior)
p_a4_1_2_prior = ProbabilisticModel.Gibbs(ag_distribution_prior, ag_av_initial_dist_prior)
a4_1_2_prior = AttributeSpec.MultipleAttr("AG_availability", p_a4_1_2_prior)
p_a4_1_3_prior = ProbabilisticModel.Gibbs(urban_distribution_prior, urban_initial_dist_prior)
a4_1_3_prior = AttributeSpec.MultipleAttr("Urban", p_a4_1_3_prior)

p_e4_1_prior = ProbabilisticModel.Gibbs(count_pdf_residence_prior, count_pdf_residence_init_prior)
c_e4_1_prior = EventTemporalConstraintsSpec(15, None, None, None, None, None, None, 0, None)
e4_1_prior = EventSpec.MultipleTimeEvent("Home", 11, p_e4_1_prior, c_e4_1_prior, [a4_1_prior, a4_1_1_prior, a4_1_2_prior, a4_1_3_prior])

d4_prior = DimensionSpec("Residence", [e4_0_prior, e4_1_prior], None)


#####################################
################# MODEL #############
#####################################
# let's assume employment phases are the same as residence phases
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

def joint_duration_prior(instance):
    return(log_employment_density_prior(instance) + driving_license_log_density_prior(instance) + education_log_density_prior(instance))

pop_spec_prior = PopulationSpec(50000, [d0_prior, d1_prior, d2_prior, d3_prior, d4_prior], joint_duration_prior, inter_constraints)