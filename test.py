#%%
from param_def import chi,sigma,prop_gp,capacities_rate,prop_all_g_prefer,sigma_i,sigma_ii,cor_i,cor_ii
import numpy as np
from util import market_clear,grades_gr_to_grades_col,bayes_update_grade
from itertools import compress
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import numpy as np
from BayesEst import bayesian_est


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
    return f1**2 + f2**2


# %%
space = {
    'x': hp.uniform('x', 1, 7),
    'y': hp.uniform('y', 1, 7)
}
# %%
best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm (representative TPE)
    max_evals=100 # Number of optimization attempts
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
n_groups = len(grade_estimated[0])
n_groups
# %%
for i in range(n_groups):
    all_choices = []
    no_choices = []
    for j in zip(grade_estimated[0][i],grade_estimated[1][i]):
        all_choices.append(all(np.array(j) > cutoff_values))
        no_choices.append(all(np.array(j) < cutoff_values))
    one_choice = [not(a|b) for a,b in zip(all_choices,no_choices)]
    #one_choice_pref = list(compress(stud_pref[i],one_choice))
    one_choice_index = list(compress(range(len(one_choice)),one_choice))
    
    second_choice = []

    for idx in one_choice_index:
        stud_first_choice = stud_pref[i][idx].index(0)
        second_choice.append(grade_estimated[stud_first_choice][i][idx] < cutoff_values[stud_first_choice])
    print(f'Proportion of students in group {i} with no offer {sum(no_choices)/len(stud_pref[i])}')
    print(f'Proportion of students in group {i} with only a second preference offer {sum(second_choice)/len(stud_pref[i])}')
    print(f'Proportion of students in group {i} with first choice offer {(len(stud_pref[i]) - sum(no_choices) - sum(second_choice))/len(stud_pref[i])}')    

# %%
bayes_update_grade(cutoff_values[0],cutoff_values[1],grade_estimated,chi,sigma)
# %%
