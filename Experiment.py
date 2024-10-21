from util import *
from data_generator import *
import numpy as np
from itertools import compress
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import numpy as np
from BayesEst import bayesian_est,anal_cond_exp
from scipy.optimize import fsolve
from scipy.stats import norm, multivariate_normal
from param_def import chi,sigma,lambdas
import pandas as pd
from pybads import BADS



def market_clear_default(Pa, Pb, prop, capA, capB, prefi, prefii, sigmai, sigmaii, cori, corii):
    f1 = prop*prefi*(1 - cdf(Pa, sigmai)) + (1 - prop)*prefii*(1 - cdf(Pa, sigmaii)) + prop*(1 - prefi)*cdfmsi(Pa, Pb, sigmai, cori) + (1 -prop)*(1 - prefii)*cdfmsi(Pa, Pb, sigmaii, corii) - capA
    f2 = prop*(1 - prefi)*(1 - cdf(Pb, sigmai)) + (1 - prop)*(1 - prefii)*(1 - cdf(Pb, sigmaii)) + prop*prefi*cdfmis(Pa, Pb, sigmai, cori) + (1 -prop)*prefii*cdfmis(Pa, Pb, sigmaii, corii) - capB
    return f1, f2





def experiment(n_stud,prop_gp,mean_gp,std_gp,n_col,noise_mean,sigma, lambdas,capacities_rate):
    data_generator(n_stud,prop_gp,mean_gp,std_gp,n_col,noise_mean,sigma, lambdas)
    #print(capacities_rate)
    capacities = [int(r * n_stud) for r in capacities_rate]
    print('Capacity of each college:',capacities)
    grade_estimated_gr = []
    for i in range(len(chi)):
        grade_estimated_gr.append(np.load(f'grade_estimated_gr{i+1}.npy'))
    grade_estimated = grades_gr_to_grades_col(grade_estimated_gr)
    stud_pref = []
    for i in range(len(chi)):
        stud_pref.append(np.load(f'stud_pref_gr{i+1}.npy').tolist())
    sigma_i = np.sqrt(chi[0]**2 + sigma[0]**2)
    sigma_ii = np.sqrt(chi[1]**2 + sigma[1]**2)
    cor_i = (chi[0]**2 + lambdas[0]**2)/sigma_i**2
    cor_ii = (chi[1]**2 + lambdas[1]**2)/sigma_ii**2
    ## No update matching
    from pybads import BADS

    lower_bounds = np.array([-10, -10])
    upper_bounds = np.array([10, 10])
    plausible_lower_bounds = np.array([-5, -5])
    plausible_upper_bounds = np.array([5, 5])
    options = {'display': 'off',}
    x0 = np.array([0, 0]);        # Starting point
    def objective_bads(params):
        Pa,Pb = params
        f1,f2 = market_clear_default(Pa,Pb,prop_gp[0],capacities_rate[0],capacities_rate[1],
                  prop_all_g_prefer[0],prop_all_g_prefer[1],sigma_i,sigma_ii,cor_i,cor_ii)
        # return np.abs(f1) + np.abs(f2)
        return f1**2+f2**2
    bads = BADS(objective_bads, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds,options = options)
    optimize_result = bads.optimize()
    x_min = optimize_result['x']
    fval = optimize_result['fval']
    print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
    print(f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s")
    cutoff_values = list(x_min)
    print('For matching without any update, the welfares are')
    welfare_metrics(cutoff_values,grade_estimated,stud_pref)
    student_by_col(cutoff_values,grade_estimated,stud_pref)
    utility_by_col(cutoff_values,grade_estimated,stud_pref)
    welfare_by_col(cutoff_values,grade_estimated,stud_pref)

    ##Partially update matching
    df1 = pd.DataFrame({'A':grade_estimated[0][0],'B':grade_estimated[1][0],'pref':np.array(stud_pref[0]).T[0]})
    df2 = pd.DataFrame({'A':grade_estimated[0][1],'B':grade_estimated[1][1],'pref':np.array(stud_pref[1]).T[0]})

    ##Solve for best cutoff
    def objective_bads(params):
        Pa,Pb = params
        f1,f2 = market_clear_noise_corr(Pa, Pb, grade_estimated, stud_pref, prop_gp[0], capacities_rate[0], capacities_rate[1], prop_all_g_prefer[0], prop_all_g_prefer[1], sigma_i, sigma_ii, cor_i, cor_ii,chi,sigma,lambdas,bayes='right_partial')
        return np.abs(f1) + np.abs(f2)
        # return f1**2+f2**2
 
    lower_bounds = np.array([-10, -10])
    upper_bounds = np.array([10, 10])
    plausible_lower_bounds = np.array([-5, -5])
    plausible_upper_bounds = np.array([5, 5])
    x0 = np.array([0, 0]);        # Starting point

    options = {'display': 'off',}
    bads = BADS(objective_bads, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds,options = options)
    optimize_result = bads.optimize()

   
    x_min = optimize_result['x']
    fval = optimize_result['fval']
    print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
    print(f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s")
    new_cutoff_values = list(x_min)
    updated_grade_estimated = bayes_update_grade(new_cutoff_values[0],new_cutoff_values[1],grade_estimated,stud_pref,chi,sigma,lambdas,bayes_type='right_partial')

    welfare_metrics(new_cutoff_values,updated_grade_estimated,stud_pref)
    student_by_col(new_cutoff_values,updated_grade_estimated,stud_pref)
    utility_by_col(new_cutoff_values,updated_grade_estimated,stud_pref)
    welfare_by_col(new_cutoff_values,updated_grade_estimated,stud_pref)