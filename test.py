#%%
from param_def import chi,sigma,prop_gp,capacities_rate,prop_all_g_prefer,sigma_i,sigma_ii,cor_i,cor_ii,lambdas
import numpy as np
from util import *
from itertools import compress
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import numpy as np
from scipy.optimize import least_squares
from pybads import BADS
import pandas as pd

#%%
grade_estimated_gr = []
for i in range(len(chi)):
    grade_estimated_gr.append(np.load(f'grade_estimated_gr{i+1}.npy'))
print(grade_estimated_gr[0].shape)
print(grade_estimated_gr[1].shape)
grade_estimated = grades_gr_to_grades_col(grade_estimated_gr)
#%%
stud_pref = []
for i in range(len(chi)):
    stud_pref.append(np.load(f'stud_pref_gr{i+1}.npy').tolist())
#%%
stud_pref
#%%
grade_estimated
# %%
def objective_bads(params):
    Pa,Pb = params
    f1,f2 = market_clear_noise_corr(Pa, Pb, grade_estimated, stud_pref, prop_gp[0], capacities_rate[0], capacities_rate[1], prop_all_g_prefer[0], prop_all_g_prefer[1], sigma_i, sigma_ii, cor_i, cor_ii,chi,sigma,lambdas,bayes='right_partial')
    # return np.abs(f1) + np.abs(f2)
    return f1**2+f2**2
# %%
lower_bounds = np.array([-10, -10])
upper_bounds = np.array([10, 10])
plausible_lower_bounds = np.array([-5, -5])
plausible_upper_bounds = np.array([5, 5])
x0 = np.array([0, 0]);        # Starting point
# %%
bads = BADS(objective_bads, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds)
optimize_result = bads.optimize()

# %%
x_min = optimize_result['x']
fval = optimize_result['fval']
print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
print(f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s")

# %%

