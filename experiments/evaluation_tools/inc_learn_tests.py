"""
In the context of the sklearn stuff 'differences' is the difference between mean model performance. In my case it will be the difference between
either the static model and the incremental model on test set stats.t or between model_a and model_b on test set stats.t.
"""
#Boilerplate
from dataclasses import dataclass, field
from typing import Any
#Basics
import numpy as np
import pandas as pd
#Specific
from itertools import combinations
from scipy import stats
from math import factorial
import statsmodels.api as sm
# from sklearn.model_selection import StratifiedKFold

@dataclass
class TestVariables():
    results: pd.DataFrame
    num_test:int
    # cv: StratifiedKFold
    #post init
    model_scores:pd.DataFrame = field(init=False)
    num_batch:int = field(init=False)

    def __post_init__(self):
        # self.model_scores = pd.DataFrame(self.results, columns=['accuracy'])#.filter(regex=r"split\d*_test_score")
        self.model_scores = self.results.T
        self.num_batch = int(self.model_scores.shape[1] / self.num_test)

def get_mean_var(array:np.typing.NDArray, num_batches:int, num_tests:int):
    """Calculate the mean and variance(1 degree of freedom) of each batch in a set of repeates batch tests"""
    grid = array.reshape(num_tests,num_batches)
    means = grid.mean(axis=0)
    vars = grid.var(axis=0, ddof=1)
    return means, vars

# def dmtest(model_1_loss:np.typing.NDArray, model_2_loss:np.typing.NDArray, num_batches:int, num_tests:int, h:int=1):
#     """Function to calculate the Diebold-Mariano test statistic (1995). Based on a MATLAB Implementation at: https://www.mathworks.com/matlabcentral/fileexchange/33979-diebold-mariano-test-statistic/files/dmtest.m"""

#     e1_arr = np.asarray(model_1_loss, dtype=np.float64).ravel()
#     e2_arr = np.asarray(model_2_loss, dtype=np.float64).ravel()
#     assert e1_arr.size == e2_arr.size, f"Size of e1 ({e1_arr.size} does not equal the size of e2 ({e2_arr.size}))"

#     #Define the loss differential
#     d = e1_arr - e2_arr

#     print(f"\nLength d: {len(d)}")

#     #Initialize n
#     n = int(e1_arr.size)

#     d_mean = np.mean(d)

#     gamma_0 = d.var(ddof=1) + 1e-10
#     if h > 1:
#         gamma = np.zeros(h-1, dtype=np.float64)
#         for i in range(1,h):
#             gamma[i-1] = np.dot(d[i:n], d[0:n-i]) / n
#         var_d = gamma_0 + 2 * gamma.sum()
#     else:
#         var_d = gamma_0
#     var_d = max(0.0, float(var_d))
#     return d_mean, float(np.sqrt((1/n)*var_d))
import numpy as np
from scipy import stats

def dmtest(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """
    Diebold-Mariano test adapted for M independent test runs of length T.
    
    e1, e2 : ndarray of shape (num_tests, horizon) e.g., (100, 5)
    h      : forecast horizon (lags to adjust = h - 1)
    """
    e1_arr = np.asarray(e1, dtype=np.float64)
    e2_arr = np.asarray(e2, dtype=np.float64)
    
    # Ensure 2D shape: (M trials, T horizon)
    if e1_arr.ndim == 1:
        raise ValueError("Reshape error inputs to (num_tests, forecast_horizon)")
        
    d = e1_arr - e2_arr  # Shape: (M, T)
    M, T = d.shape
    N = M * T  # Total pooled sample size
    
    d_flat = d.ravel()
    d_mean = np.mean(d_flat)
    
    # Variance at lag 0 across all pooled data
    gamma_0 = np.var(d_flat, ddof=1)
    
    # Calculate autocovariances ONLY within individual test runs (no cross-trial bleeding)
    gamma_sum = 0.0
    if h > 1:
        for k in range(1, min(h, T)):
            # Sum inner products strictly within each trial row
            lag_cov = np.sum((d[:, k:] - d_mean) * (d[:, :-k] - d_mean)) / N
            # Applying uniform Bartlett weight (or standard DM unweighted lag sum)
            gamma_sum += lag_cov

    var_d = gamma_0 + 2 * gamma_sum
    var_d = max(1e-12, float(var_d))
    
    # Standard error of the mean differential
    se_d = np.sqrt(var_d / N)
    return d_mean, se_d
    DM_stat = d_mean / se_d
    p_val = 2 * stats.norm.sf(np.abs(DM_stat))  # Two-tailed standard normal p-value
    
    return DM_stat, p_val

#need to change the scores to loss
def pairwise_freq(model_scores:pd.DataFrame, pairwise_comp_df:pd.DataFrame | None, num_batches:int, num_tests:int):
    """Null Hypothesis: The two models have equal predictive accuracy"""
    num_models = len(model_scores)
    n_comparisons = factorial(num_models) / (
        factorial(2) * factorial(num_models - 2)
    )
    pairwise_t_test = []
    for model_i, model_k in combinations(range(num_models), 2):
        scores_i = model_scores.iloc[model_i].to_numpy()
        scores_i = scores_i.reshape(num_batches, num_tests)
        scores_k = model_scores.iloc[model_k].to_numpy()
        scores_k = scores_k.reshape(num_batches, num_tests)
        model_i_error = (1 - scores_i) ** 2
        model_k_error = (1 - scores_k) ** 2
        h = int(round(((num_tests ** (1/3)) + 1), 0))
        mean, std = dmtest(e1=model_i_error, e2=model_k_error, h=h)#, num_tests=num_tests, num_batches=num_batches)

        if std == 0 or np.isnan(std):
            print(f"std is 0 or NaN for pair {model_scores.index[model_i]} and {model_scores.index[model_k]}")
            t_stat = 0.0
        else:
            t_stat = mean / std

        # p_val = stats.t.sf(np.abs(t_stat), num_batches)  # right-tailed stats.t-test
        p_val = 2 * stats.norm.sf(np.abs(t_stat))
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

def pairwise_bayesian(model_scores:pd.DataFrame, rope_interval:list, num_batches:int, num_tests:int, h:int, pairwise_comp_df:pd.DataFrame | None):
    """Probability that Model A is Worse, Better, or the same (over the ROPE) as model B"""
    pairwise_bayesian = []

    num_models = len(model_scores)
    for model_i, model_k in combinations(range(num_models), 2):
        scores_i = model_scores.iloc[model_i].to_numpy()
        scores_i = scores_i.reshape(num_batches, num_tests)
        scores_k = model_scores.iloc[model_k].to_numpy()
        scores_k = scores_k.reshape(num_batches, num_tests)
        model_i_error = (1 - scores_i) ** 2
        model_k_error = (1 - scores_k) ** 2
        h = int(round(((num_tests ** (1/3)) + 1), 0))
        mean, std = dmtest(e1=model_i_error, e2=model_k_error, h=h)
        #Protect against 0 division errors, which could occur if the error is equal
        if std == 0 or np.isnan(std):
            print(f"std is 0 or NaN for pair {model_scores.index[model_i]} and {model_scores.index[model_k]}")
            better_prob = 1.0 if mean < rope_interval[0] else 0.0
            worse_prob = 1.0 if mean > rope_interval[1] else 0.0
            rope_prob = 1.0 if rope_interval[0] <= mean <= rope_interval[1] else 0.0
        else:
            t_post = stats.t(
                num_batches, loc=mean, scale=std) #replace scale with dmtest
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