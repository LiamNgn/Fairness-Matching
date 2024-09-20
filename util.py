import numpy as np
import scipy.stats as st
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.misc import derivative
import pandas as pd
import random
import time
from param_def import chi,sigma,prop_gp,capacities_rate,prop_all_g_prefer,sigma_i,sigma_ii,cor_i,cor_ii
from BayesEst import anal_cond_exp
from itertools import compress
import copy

def create_students(n_stud: int, prop_gp: list, mean_gp: list, std_gp: list):
    n_gp = prop_gp.__len__()
    n_gp1 = mean_gp.__len__()
    n_gp2 = std_gp.__len__()
    if ((n_gp==n_gp1) and (n_gp==n_gp2)) == False:
        print("group arguments of different sizes")
        return 0
    s = 0
    for i in range(n_gp):
        if ((prop_gp[i] < 0) or (prop_gp[i] > 1)):
            print("wrong proportions")
            return 0
        s = s + prop_gp[i]
    if ((s<0.99) or (s>1.01)):
        print("wrong proportions")
        return 0
    nb_stud_gp = [int(x*n_stud) for x in prop_gp]
    students = []
    for i in range(n_gp):
        stud_gp = np.random.normal(mean_gp[i], std_gp[i], nb_stud_gp[i])
        students.append(stud_gp)
    return students


def create_col_estim(n_col,students, noise_mean, noise_std):
    n_gp = noise_mean.__len__()
    n_gp1 = noise_std.__len__()
    n_gp2 = students.__len__()
    if ((n_gp==n_gp1) and (n_gp==n_gp2)) == False:
        print("group arguments of different sizes")
        return 0
    col_estim = []
    for i in range (n_col):
        estim = students.copy()
        for j in range(n_gp):
            m = students[j].size
            noise = np.random.normal(noise_mean[j], noise_std[j], m)
            estim[j] = students[j] + noise
        col_estim.append(estim)
    return col_estim


def create_stud_pref_2(students, prop_0):
    stud_pref=[]
    for i in range(students.__len__()):
        group = []
        for j in range(students[i].size):
            rand = np.random.rand()
            if rand < prop_0[i]:
                group.append([0,1])
            else:
                group.append([1,0])
        stud_pref.append(group)
    return stud_pref


def _rank2(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if N == 1:
        return 0
    else:
        idx = np.argpartition(points[:,0], N2)
        idxA_ = idx[:N2]
        idxA = np.zeros(N, dtype=bool)
        idxA[idxA_] = True
        if mask is not None:
            NAm = np.sum(idxA & mask)
            points_reduced = np.vstack((points[idxA & mask], points[~idxA & ~mask]))
        else:
            NAm = np.sum(idxA)
            points_reduced = np.vstack((points[idxA], points[~idxA]))
        count_points = np.zeros(points_reduced.shape[0], dtype=bool)
        count_points[:NAm] = True
        idxY = np.argsort(points_reduced[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        count_points = count_points[idxY]
        numA = np.cumsum(count_points)[idxYr]
        rank = np.zeros(N, dtype=int)
        if mask is not None:
            rank[idxA] = _rank2(points[idxA], mask[idxA])
            rank[~idxA] = _rank2(points[~idxA], mask[~idxA])
            rank[~idxA & ~mask] += numA[NAm:]
        else:
            rank[idxA] = _rank2(points[idxA])
            rank[~idxA] = _rank2(points[~idxA])
            rank[~idxA] += numA[NAm:]
        return rank
    

def rankn(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if mask is None:
        mask = np.ones(N, dtype=bool)
        first_call = True
    else:
        first_call = False
    if N == 1:
        return 0
    if points.shape[1] == 2:
        if first_call:
            return _rank2(points)
        else:
            return _rank2(points, mask)
    idx = np.argpartition(points[:,0], N2)
    idxA_ = idx[:N2]
    idxA = np.zeros(N, dtype=bool)
    idxA[idxA_] = True
    rank = np.zeros(N, dtype=int)
    rank[idxA] = rankn(points[idxA], mask[idxA])
    rank[~idxA] = rankn(points[~idxA], mask[~idxA]) + rankn(points[:,1:], idxA*mask)[~idxA]
    return rank

def grades_col_to_grades_gr(grade_estimated):
    lst = []
    for i in range(len(grade_estimated[0])):
        qualities_gr = []
        for j in range(len(grade_estimated)):
            qualities_gr.append(grade_estimated[j][i])
        qualities_gr = np.array(qualities_gr).transpose()
        lst.append(qualities_gr)
    return lst

def grades_gr_to_grades_col(grade_estimated_gr):
    lst = []
    for i in range(grade_estimated_gr[0].shape[1]):
        col_grades = []
        for j in range(len(grade_estimated_gr)):
            gr_grades = []
            for t in range(grade_estimated_gr[j].shape[0]):
                gr_grades.append(grade_estimated_gr[j][t][i].item())
            col_grades.append(gr_grades)
        lst.append(col_grades)
    return lst



def multivariate_ecdf(vector_points,value):
    list_values = np.vstack([value,vector_points])
    rank = rankn(list_values)
    return rank[0]/len(rank)


def sampling_ecdf(grade_estimated,Pa,Pb,chi,sigma,type = 'both'):
    type_bayes = ('right','left','both')
    if type not in type_bayes:
        raise ValueError(f'bayes_update must be one of {type_bayes}')
    
    if type == 'right':
        updated_grade_estimate_1 = [anal_cond_exp(i,Pa,chi[0],sigma[0]) for i in grade_estimated[1][0]]
        updated_grade_estimate_2 = [anal_cond_exp(i,Pa,chi[1],sigma[1]) for i in grade_estimated[1][1]]
        res1 = st.ecdf(updated_grade_estimate_1)
        res2 = st.ecdf(updated_grade_estimate_2)
        updated_grade_estimated = copy.deepcopy(grade_estimated)
        updated_grade_estimated[1][0] = updated_grade_estimate_1
        updated_grade_estimated[1][1] = updated_grade_estimate_2
        grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
        #ECDF
        multi_ecdf_gr1 = multivariate_ecdf(grade_estimated_gr[0],[Pa,Pb]).item()
        multi_ecdf_gr2 = multivariate_ecdf(grade_estimated_gr[1],[Pa,Pb]).item()
        gr1_ecdf_pb = res1.cdf.evaluate(Pb).item()
        gr2_ecdf_pb = res2.cdf.evaluate(Pb).item()

        return gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2

    if type == 'both':
        updated_grade_estimate_1B = [anal_cond_exp(i,Pa,chi[0],sigma[0]) for i in grade_estimated[1][0]]
        updated_grade_estimate_2B = [anal_cond_exp(i,Pa,chi[1],sigma[1]) for i in grade_estimated[1][1]]
        updated_grade_estimate_1A = [anal_cond_exp(i,Pb,chi[0],sigma[0]) for i in grade_estimated[0][0]]
        updated_grade_estimate_2A = [anal_cond_exp(i,Pb,chi[1],sigma[1]) for i in grade_estimated[0][1]]
        res1B = st.ecdf(updated_grade_estimate_1B)
        res2B = st.ecdf(updated_grade_estimate_2B)
        res1A = st.ecdf(updated_grade_estimate_1A)
        res2A = st.ecdf(updated_grade_estimate_2A)
        updated_grade_estimated[1][0] = updated_grade_estimate_1B
        updated_grade_estimated[1][1] = updated_grade_estimate_2B
        updated_grade_estimated[0][0] = updated_grade_estimate_1A
        updated_grade_estimated[0][1] = updated_grade_estimate_2A
        grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
        #ECDF
        multi_ecdf_gr1 = multivariate_ecdf(grade_estimated_gr[0],[Pa,Pb]).item()
        multi_ecdf_gr2 = multivariate_ecdf(grade_estimated_gr[1],[Pa,Pb]).item()
        gr1_ecdf_pb = res1B.cdf.evaluate(Pb).item()
        gr2_ecdf_pb = res2B.cdf.evaluate(Pb).item()
        gr1_ecdf_pa = res1A.cdf.evaluate(Pa).item()
        gr2_ecdf_pa = res2A.cdf.evaluate(Pa).item()

        return gr1_ecdf_pa,gr2_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2

def bayes_update_grade(Pa,Pb,grade_estimated,chi,sigma,bayes_type='right'):
    type_bayes = ('right','left','both')
    if bayes_type not in type_bayes:
        raise ValueError(f'bayes_update must be one of {type_bayes}')
    if bayes_type == 'right':
        updated_grade_estimate_1 = [anal_cond_exp(i,Pa,chi[0],sigma[0]) for i in grade_estimated[1][0]]
        updated_grade_estimate_2 = [anal_cond_exp(i,Pa,chi[1],sigma[1]) for i in grade_estimated[1][1]]
        grade_estimated[1][0] = updated_grade_estimate_1
        grade_estimated[1][1] = updated_grade_estimate_2
    elif bayes_type == 'left':
        updated_grade_estimate_1 = [anal_cond_exp(i,Pb,chi[0],sigma[0]) for i in grade_estimated[0][0]]
        updated_grade_estimate_2 = [anal_cond_exp(i,Pb,chi[1],sigma[1]) for i in grade_estimated[0][1]]
        grade_estimated[0][0] = updated_grade_estimate_1
        grade_estimated[0][1] = updated_grade_estimate_2 
    elif bayes_type == 'both':
        updated_grade_estimate_1 = [anal_cond_exp(i,Pa,chi[0],sigma[0]) for i in grade_estimated[1][0]]
        updated_grade_estimate_2 = [anal_cond_exp(i,Pa,chi[1],sigma[1]) for i in grade_estimated[1][1]]
        grade_estimated[1][0] = updated_grade_estimate_1
        grade_estimated[1][1] = updated_grade_estimate_2
        updated_grade_estimate_1 = [anal_cond_exp(i,Pb,chi[0],sigma[0]) for i in grade_estimated[0][0]]
        updated_grade_estimate_2 = [anal_cond_exp(i,Pb,chi[1],sigma[1]) for i in grade_estimated[0][1]]
        grade_estimated[0][0] = updated_grade_estimate_1
        grade_estimated[0][1] = updated_grade_estimate_2 
    return grade_estimated      

def welfare_metrics(cutoff_values,estimated_grade,stud_pref):
    
    n_groups = len(stud_pref)
    
    for i in range(n_groups):
        all_choices = []
        no_choices = []
        for j in zip(estimated_grade[0][i],estimated_grade[1][i]):
            all_choices.append(all(np.array(j) > cutoff_values))
            no_choices.append(all(np.array(j) < cutoff_values))
        one_choice = [not(a|b) for a,b in zip(all_choices,no_choices)]
        #one_choice_pref = list(compress(stud_pref[i],one_choice))
        one_choice_index = list(compress(range(len(one_choice)),one_choice))
        
        second_choice = []

        for idx in one_choice_index:
            stud_first_choice = stud_pref[i][idx].index(0)
            second_choice.append(estimated_grade[stud_first_choice][i][idx] < cutoff_values[stud_first_choice])

        print(f'Proportion of students in group {i} with no offer {sum(no_choices)/len(stud_pref[i]):.2f}')
        print(f'Proportion of students in group {i} with only a second preference offer {sum(second_choice)/len(stud_pref[i]):.2f}')
        print(f'Proportion of students in group {i} with first choice offer {(len(stud_pref[i]) - sum(no_choices) - sum(second_choice))/len(stud_pref[i]):.2f}')    

def student_by_col(cutoff_values,estimated_grade,stud_pref):
    
    n_groups = len(stud_pref)
    no_col_0 = 0
    no_col_1 = 0
    for i in range(n_groups):
        all_choices = []
        no_choices = []
        for j in zip(estimated_grade[0][i],estimated_grade[1][i]):
            all_choices.append(all(np.array(j) > cutoff_values))
            no_choices.append(all(np.array(j) < cutoff_values))
        all_choices_index = list(compress(range(len(all_choices)),all_choices))
        one_choice = [not(a|b) for a,b in zip(all_choices,no_choices)]
        #one_choice_pref = list(compress(stud_pref[i],one_choice))
        one_choice_index = list(compress(range(len(one_choice)),one_choice))
        
        col_0 = 0
        col_1 = 0
        pref_all_choices = np.array(stud_pref[i])[all_choices_index]
        for pref in pref_all_choices:
            if np.where(pref==0)[0][0] == 0:
                col_0 += 1
            elif np.where(pref==0)[0][0] == 1:
                col_1 += 1

        second_choice = []
        
        for idx in one_choice_index:
            if estimated_grade[0][i][idx] < cutoff_values[0]:
                col_1 += 1
            else:
                col_0 += 1
        print(f'Number of student admitted to college 1 in group {i}:',col_0)
        print(f'Number of student admitted to college 2 in group {i}:',col_1)
        no_col_0 += col_0
        no_col_1 += col_1

    return no_col_0,no_col_1



def cdf(x, sigma):
    return norm.cdf(x, scale = sigma)



def cdfmsi(x, y, sigma, rho):
    return norm.cdf(y, scale = sigma) - multivariate_normal([0, 0], [[sigma**2, sigma**2 * rho], [sigma**2 * rho, sigma**2]]).cdf(np.array([x,y]))



def cdfmis(x, y, sigma, rho):
    return norm.cdf(x, scale = sigma) - multivariate_normal([0, 0], [[sigma**2, sigma**2 * rho], [sigma**2 * rho, sigma**2]]).cdf(np.array([x,y]))


def market_clear(Pa, Pb, grade_estimated, prop, capA, capB, prefi, prefii, sigmai, sigmaii, cori, corii, chi, sigma,bayes = 'none'):
    type_bayes = ('right','left','both','none')
    if bayes not in type_bayes:
        raise ValueError(f'bayes_update must be one of {type_bayes}')
    if bayes == 'both':
        gr1_ecdf_pa,gr2_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = sampling_ecdf(grade_estimated,Pa,Pb,chi,sigma,type=bayes)    
        f1 = prop*prefi*(1 - gr1_ecdf_pa) + (1 - prop)*prefii*(1 - gr2_ecdf_pa) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
        f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(gr1_ecdf_pa - multi_ecdf_gr1) + (1 -prop)*prefii*(gr2_ecdf_pa - multi_ecdf_gr2) - capB
    elif bayes == 'right':
        gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = sampling_ecdf(grade_estimated,Pa,Pb,chi,sigma,type=bayes)    
        f1 = prop*prefi*(1 - cdf(Pa, sigmai)) + (1 - prop)*prefii*(1 - cdf(Pa, sigmaii)) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
        f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(cdf(Pa,sigmai) - multi_ecdf_gr1) + (1 -prop)*prefii*(cdf(Pa,sigmaii) - multi_ecdf_gr2) - capB   
    return f1, f2


# def objective(params):
#     Pa,Pb = params
#     f1,f2 = market_clear(Pa, Pb, grade_estimated, prop_gp[0], capacities_rate[0], capacities_rate[1], prop_all_g_prefer, prop_all_g_prefer, sigma_i, sigma_ii, cor_i, cor_ii,chi,sigma)
#     return f1**2 + f2**2