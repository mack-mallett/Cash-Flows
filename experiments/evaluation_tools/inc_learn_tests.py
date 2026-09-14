"""
In the context of the sklearn stuff 'differences' is the difference between mean model performance. In my case it will be the difference between
either the static model and the incremental model on test set t or between model_a and model_b on test set t.
"""
#Boilerplate
from dataclasses import dataclass, field
from typing import Any
#Basics
import numpy as np
import pandas as pd
#Specific
from itertools import combinations
from scipy.stats import t
from math import factorial
# from sklearn.model_selection import StratifiedKFold

@dataclass
class TestVariables():
    results: pd.DataFrame
    # cv: StratifiedKFold
    #post init
    model_scores:pd.DataFrame = field(init=False)
    num_tests:int = field(init=False)

    def __post_init__(self):
        # self.model_scores = pd.DataFrame(self.results, columns=['accuracy'])#.filter(regex=r"split\d*_test_score")
        self.model_scores = self.results.T
        self.num_tests = self.model_scores.shape[0] - 1

def dmtest(model_1_loss:np.typing.NDArray, model_2_loss:np.typing.NDArray, h:int=1):
    """Function to calculate the Diebold-Mariano test statistic (1995). Based on a MATLAB Implementation at: https://www.mathworks.com/matlabcentral/fileexchange/33979-diebold-mariano-test-statistic/files/dmtest.m
    From the original:
    Retrieves the Diebold-Mariano test statistic (1995) for the equality of forecast accuracy of two forecasts under general assumptions.

   DM = dmtest(e1, e2, ...) calculates the D-M test statistic on the base of the loss differential which is defined as the difference of the squared forecast errors.

   In particular, with the DM statistic one can test the null hypothesis: 
   H0: E(d) = 0. The Diebold-Mariano test assumes that the loss 
   differential process 'd' is stationary and defines the statistic as:
   DM = mean(d) / sqrt[ (1/T) * VAR(d) ]  ~ N(0,1),
   where VAR(d) is an estimate of the unconditional variance of 'd'.

   This function also corrects for the autocorrelation that multi-period 
   forecast errors usually exhibit. Note that an efficient h-period 
   forecast will have forecast errors following MA(h-1) processes. 
   Diebold-Mariano use a Newey-West type estimator for sample variance of
   the loss differential to account for this concern.

   'e1' is a 'T1-by-1' vector of the forecast errors from the first model
   'e2' is a 'T2-by-1' vector of the forecast errors from the second model

   It should hold that T1 = T2 = T.

   DM = DMTEST(e1, e2, 'h') allows you to specify an additional parameter 
   value 'h' to account for the autocorrelation in the loss differential 
   for multi-period ahead forecasts.   
       'h'         the forecast horizon, initially set equal to 1

   DM = DMTEST(...) returns a constant:
       'DM'      the Diebold-Mariano (1995) test statistic

  Semin Ibisevic (2011)
  $Date: 11/29/2011 $
    """
    e1_arr = np.asarray(model_1_loss, dtype=np.float64).ravel()
    e2_arr = np.asarray(model_2_loss, dtype=np.float64).ravel()
    assert e1_arr.size == e2_arr.size, f"Size of e1 ({e1_arr.size} does not equal the size of e2 ({e2_arr.size}))"
    #Initialize T
    T = e1_arr.size
    #Define the loss differential
    d = e1_arr - e2_arr
    #Recalculate the variance of the loss differential, taking into account autocorrelation
    d_mean = np.mean(d)
    gamma_0 = d.var(ddof=1)
    if h > 1:
        gamma = np.zeros(h-1, dtype=np.float64)
        for i in range(1,h):
            gamma[i-1] = np.dot(d[i:T], d[0:T-i]) / T
        var_d = gamma_0 + 2 * gamma.sum()
    else:
        var_d = gamma_0
    var_d = max(0.0, float(var_d))
    return d_mean, float(np.sqrt((1/T)*var_d))

def compute_Diebold_Mariano_ttest(num_tests:int, model_1_loss:np.ndarray[Any], model_2_loss:np.ndarray[Any], h:int=1):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    #these should be fed in as a vector, not as a scaler
    mean, std = dmtest(model_1_loss=model_1_loss, model_2_loss=model_2_loss, h=h)
    if std == 0 or np.isnan(std):
        print(f"std is 0 or NaN for pair")
        t_stat = 0.0
    else:
        t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), num_tests)  # right-tailed t-test
    return t_stat, p_val

#need to change the scores to loss
def pairwise_freq(model_scores:pd.DataFrame, num_tests:int, pairwise_comp_df:pd.DataFrame | None):
    """Null Hypothesis: The two models have equal predictive accuracy"""
    n_comparisons = factorial(len(model_scores)) / (
        factorial(2) * factorial(len(model_scores) - 2)
    )
    pairwise_t_test = []
    for model_i, model_k in combinations(range(len(model_scores)), 2):
        
        model_i_error = (1 - model_scores.iloc[model_i].to_numpy()) ** 2
        model_k_error = (1 - model_scores.iloc[model_k].to_numpy()) ** 2
        t_stat, p_val = compute_Diebold_Mariano_ttest(num_tests, model_i_error, model_k_error)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        if pairwise_comp_df is None:
            pairwise_t_test.append(
                [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
            )
        else:
            pairwise_t_test.append(
                [t_stat, p_val]
            )
    if pairwise_comp_df is None:
        pairwise_comp_df = pd.DataFrame(
            pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
        ).round(3)
        return pairwise_comp_df
    else:
        pairwise_freq_df = pd.DataFrame(
            pairwise_t_test, columns=["t_stat", "p_val"]
        ).round(3)
        pairwise_comp_df = pairwise_comp_df.join(pairwise_freq_df)
        return pairwise_comp_df

def pairwise_bayesian(model_scores:pd.DataFrame, rope_interval:list, num_tests:int, h:int, pairwise_comp_df:pd.DataFrame | None):
    """Probability that Model A is Worse, Better, or the same (over the ROPE) as model B"""
    pairwise_bayesian = []
    # model_scores = model_scores.T
    for model_i, model_k in combinations(range(len(model_scores)), 2):
        model_i_error = (1 - model_scores.iloc[model_i].to_numpy()) ** 2
        model_k_error = (1 - model_scores.iloc[model_k].to_numpy()) ** 2
        mean, std = dmtest(model_1_loss=model_i_error, model_2_loss=model_k_error, h=h)
        #Protect against 0 division errors, which could occur if the error is equal
        if std == 0 or np.isnan(std):
            print(f"std is 0 or NaN for pair {model_scores.index[model_i]} and {model_scores.index[model_k]}")
            better_prob = 1.0 if mean < rope_interval[0] else 0.0
            worse_prob = 1.0 if mean > rope_interval[1] else 0.0
            rope_prob = 1.0 if rope_interval[0] <= mean <= rope_interval[1] else 0.0
        else:
            t_post = t(
                num_tests, loc=mean, scale=std) #replace scale with dmtest
            better_prob = t_post.cdf(rope_interval[0])
            worse_prob = 1 - t_post.cdf(rope_interval[1])
            rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
        if pairwise_comp_df is None:
            pairwise_bayesian.append([model_scores.index[model_i], model_scores.index[model_k], worse_prob, better_prob, rope_prob])
        else:
            pairwise_bayesian.append([worse_prob, better_prob, rope_prob])


    if pairwise_comp_df is None:
        pairwise_bayesian_df = pd.DataFrame(
                    pairwise_bayesian, columns=["model_1", "model_2", "worse_prob", "better_prob", "rope_prob"]
                ).round(3)
        return pairwise_bayesian_df
    else:
        pairwise_bayesian_df = pd.DataFrame(
            pairwise_bayesian, columns=["worse_prob", "better_prob", "rope_prob"]
        ).round(3)
        pairwise_comp_df = pairwise_comp_df.join(pairwise_bayesian_df)
        return pairwise_comp_df
