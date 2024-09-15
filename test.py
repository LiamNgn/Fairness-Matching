#%%
from param_def import chi,sigma,prop_gp,capacities_rate,prop_all_g_prefer,sigma_i,sigma_ii,cor_i,cor_ii
import numpy as np
from util import market_clear,grades_gr_to_grades_col,bayes_update_grade
from itertools import compress
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import numpy as np



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
#%%
def objective(params):
    Pa = params['x']
    Pb = params['y']
    f1,f2 = market_clear(Pa, Pb, grade_estimated, prop_gp[0], capacities_rate[0], capacities_rate[1], prop_all_g_prefer, prop_all_g_prefer, sigma_i, sigma_ii, cor_i, cor_ii,chi,sigma)
    return np.abs(f1) + np.abs(f2)


# %%
space = {
    'x': hp.uniform('x', -6, 6),
    'y': hp.uniform('y', -6, 6)
}
# %%
best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm (representative TPE)
    max_evals=1000 # Number of optimization attempts
)
print(best)

# %%
def objective(params):
    Pa,Pb = params
    f1,f2 = market_clear(Pa, Pb, grade_estimated, prop_gp[0], capacities_rate[0], capacities_rate[1], prop_all_g_prefer, prop_all_g_prefer, sigma_i, sigma_ii, cor_i, cor_ii,chi,sigma)
    return f1**2 + f2**2
# %%
cutoff_values = [best['x'],best['y']]
cutoff_values
# %%
objective(best)