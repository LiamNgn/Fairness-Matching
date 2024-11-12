import numpy as np
import scipy.stats as st
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.misc import derivative
import pandas as pd
import random
import time
from param_def import chi,sigma,prop_gp,capacities_rate,prop_all_g_prefer,sigma_i,sigma_ii,cor_i,cor_ii,capacities
from BayesEst import anal_cond_exp
from itertools import compress
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pybads import BADS

@dataclass
class simulation:
    n_stud: int
    prop_gp: List
    mean_gp: List
    std_gp: List
    n_col: int
    noise_mean: List
    noise_std: List
    prop_all_g_prefer: List
    capacities_rate: List
    noise_cov: Optional[List] = field(default_factory=([0,0]))
    students: Optional[List] = None
    grade_estimate_gr: Optional[List] = None
    grade_estimate_col: Optional[List] = None
    stud_pref: Optional[List] = None
    cutoff_value: Optional[List] = None
    updated_grade_estimate_gr: Optional[List] = None

    @property
    def estimate_std_i(self):
        return  np.sqrt(self.std_gp[0]**2 + self.noise_std[0]**2) 
    
    @property
    def estimate_std_ii(self):
        return np.sqrt(self.std_gp[1]**2 + self.noise_std[1]**2)

    @property
    def estimate_cov_i(self):
        return (self.std_gp[0]**2 + self.noise_cov[0]**2)/self.estimate_std_i**2
    
    @property
    def estimate_cov_ii(self):
        return (self.std_gp[1]**2 + self.noise_cov[1]**2)/self.estimate_std_ii**2



    def create_students(self):
        n_gp = self.prop_gp.__len__()
        n_gp1 = self.mean_gp.__len__()
        n_gp2 = self.std_gp.__len__()
        if ((n_gp==n_gp1) and (n_gp==n_gp2)) == False:
            print("group arguments of different sizes")
            return 0
        s = 0
        for i in range(n_gp):
            if ((self.prop_gp[i] < 0) or (self.prop_gp[i] > 1)):
                print("wrong proportions")
                return 0
            s = s + self.prop_gp[i]
        if ((s<0.99) or (s>1.01)):
            print("wrong proportions")
            return 0
        nb_stud_gp = [int(x*self.n_stud) for x in self.prop_gp]
        students = []
        for i in range(n_gp):
            stud_gp = np.random.normal(self.mean_gp[i], self.std_gp[i], nb_stud_gp[i])
            students.append(stud_gp)
        self.students = students
        return students


    def create_col_estim(self):
        n_gp = self.noise_mean.__len__()
        n_gp1 = self.noise_std.__len__()
        n_gp2 = self.students.__len__()
        if ((n_gp==n_gp1) and (n_gp==n_gp2)) == False:
            print("group arguments of different sizes")
            return 0
        col_estim = []
        for i in range (self.n_col):
            estim = self.students.copy()
            for j in range(n_gp):
                m = self.students[j].size
                noise = np.random.normal(self.noise_mean[j], self.noise_std[j], m)
                estim[j] = self.students[j] + noise
            col_estim.append(estim)
        self.grade_estimate_group = col_estim
        return col_estim

    def create_col_estim_corr_nois(self):
        n_gp = self.noise_mean.__len__()
        n_gp1 = self.noise_std.__len__()
        n_gp2 = self.students.__len__()
        n_gp3 = self.noise_cov.__len__()
        if ((n_gp==n_gp1) and (n_gp==n_gp2) and (n_gp3 == n_gp)) == False:
            print("group arguments of different sizes")
            return 0
        gr_col_estim = []
        for i in range(n_gp):
            estim = self.students[i].copy()
            noise_means = [self.noise_mean[i],self.noise_mean[i]]
            noise_covs = [[self.noise_std[i]**2,self.noise_cov[i]**2],[self.noise_cov[i]**2,self.noise_std[i]**2]]
            noise = np.random.multivariate_normal(noise_means,noise_covs,len(estim)).T
            estim = estim + noise
            gr_col_estim.append(estim.T)
        self.grade_estimate_gr = gr_col_estim
        return gr_col_estim


    def create_stud_pref_2(self):
        stud_pref=[]
        for i in range(self.students.__len__()):
            group = []
            for j in range(self.students[i].size):
                rand = np.random.rand()
                if rand < self.prop_all_g_prefer[i]:
                    group.append([0,1])
                else:
                    group.append([1,0])
            stud_pref.append(group)
        self.stud_pref = stud_pref
        return stud_pref
    
    def grades_gr_to_grades_col(self):
        lst = []
        for i in range(self.grade_estimate_gr[0].shape[1]):
            col_grades = []
            for j in range(len(self.grade_estimate_gr)):
                gr_grades = []
                for t in range(self.grade_estimate_gr[j].shape[0]):
                    gr_grades.append(self.grade_estimate_gr[j][t][i].item())
                col_grades.append(gr_grades)
            lst.append(col_grades)
        self.grade_estimate_col = lst
        return lst


    def sampling_ecdf(self ,type = 'right_partial'):
        type_bayes = ('right_all','left','both','right_partial')
        Pa = self.cutoff_value[0]
        Pb = self.cutoff_value[1]
        if type not in type_bayes:
            raise ValueError(f'bayes_update must be one of {type_bayes}')
        updated_grade_estimated = copy.deepcopy(self.grade_estimate_col)
        if type == 'right_all':
            updated_grade_estimate_1 = [anal_cond_exp(i,Pa,self.std_gp[0],self.noise_std[0],self.noise_cov[0]) for i in self.grade_estimated[1][0]]
            updated_grade_estimate_2 = [anal_cond_exp(i,Pa,self.std_gp[1],self.noise_std[1], self.noise_cov[1]) for i in self.grade_estimated[1][1]]
            res1 = st.ecdf(updated_grade_estimate_1)
            res2 = st.ecdf(updated_grade_estimate_2)
            updated_grade_estimated[1][0] = updated_grade_estimate_1
            updated_grade_estimated[1][1] = updated_grade_estimate_2
            grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
            #ECDF
            multi_ecdf_gr1 = multivariate_ecdf(grade_estimated_gr[0],[Pa,Pb]).item()
            multi_ecdf_gr2 = multivariate_ecdf(grade_estimated_gr[1],[Pa,Pb]).item()
            gr1_ecdf_pb = res1.cdf.evaluate(Pb).item()
            gr2_ecdf_pb = res2.cdf.evaluate(Pb).item()

            return gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2

        if type == 'right_partial':
            # df1 = pd.DataFrame({'A':grade_estimated[0][0],'B':grade_estimated[1][0],'pref':stud_pref[0]}) #gr1
            # df2 = pd.DataFrame({'A':grade_estimated[0][1],'B':grade_estimated[1][1],'pref':stud_pref[1]}) #gr2
            df1 = pd.DataFrame({'A':self.grade_estimate_col[0][0],'B':self.grade_estimate_col[1][0],'pref':np.array(self.stud_pref[0]).T[0]})
            df2 = pd.DataFrame({'A':self.grade_estimate_col[0][1],'B':self.grade_estimate_col[1][1],'pref':np.array(self.stud_pref[1]).T[0]})
            df1.loc[df1['pref']==1,'pref_name'] = 'B'
            df1.loc[df1['pref']==0,'pref_name'] = 'A'
            df2.loc[df2['pref']==1,'pref_name'] = 'B'
            df2.loc[df2['pref']==0,'pref_name'] = 'A'
            df1['bayes_B'] = np.where(df1['pref_name']=='B',anal_cond_exp(df1['B'],Pa,self.std_gp[0],self.noise_std[0],self.noise_cov[0]),df1['B'])
            df2['bayes_B'] = np.where(df2['pref_name']=='B',anal_cond_exp(df2['B'],Pa,self.std_gp[1],self.noise_std[1],self.noise_cov[1]),df2['B'])
            res1A = st.ecdf(df1.loc[df1['pref'] == 0,'A'])
            res2A = st.ecdf(df2.loc[df2['pref'] == 0,'A'])
            res1B = st.ecdf(df1.loc[df1['pref'] == 1, 'bayes_B'])
            res2B = st.ecdf(df2.loc[df2['pref'] == 1, 'bayes_B'])
            # updated_grade_estimated[1][0] = df1['bayes_B']
            # updated_grade_estimated[1][1] = df2['bayes_B']
            # grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
            #ECDF
            multi_ecdf_gr1_A = multivariate_ecdf(df1.loc[df1['pref']==0,['A','bayes_B']],[Pa,Pb]).item()
            multi_ecdf_gr1_B = multivariate_ecdf(df1.loc[df1['pref']==1,['A','bayes_B']],[Pa,Pb]).item()
            multi_ecdf_gr2_A = multivariate_ecdf(df2.loc[df2['pref']==0,['A','bayes_B']],[Pa,Pb]).item()
            multi_ecdf_gr2_B = multivariate_ecdf(df2.loc[df2['pref']==1,['A','bayes_B']],[Pa,Pb]).item()
            gr1_ecdf_pb = res1B.cdf.evaluate(Pb).item()
            gr2_ecdf_pb = res2B.cdf.evaluate(Pb).item()
            gr1_ecdf_pa = res1A.cdf.evaluate(Pa).item()
            gr2_ecdf_pa = res2A.cdf.evaluate(Pa).item()
            return [gr1_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pa,gr2_ecdf_pb,multi_ecdf_gr1_A,multi_ecdf_gr1_B,multi_ecdf_gr2_A,multi_ecdf_gr2_B]

        if type == 'both':
            updated_grade_estimate_1B = [anal_cond_exp(i,Pa,self.std_gp[0],self.noise_std[0],self.noise_cov[0]) for i in self.grade_estimated[1][0]]
            updated_grade_estimate_2B = [anal_cond_exp(i,Pa,self.std_gp[1],self.noise_std[1],self.noise_cov[1]) for i in self.grade_estimated[1][1]]
            updated_grade_estimate_1A = [anal_cond_exp(i,Pb,self.std_gp[0],self.noise_std[0],self.noise_cov[0]) for i in self.grade_estimated[0][0]]
            updated_grade_estimate_2A = [anal_cond_exp(i,Pb,self.std_gp[1],self.noise_std[1],self.noise_cov[1]) for i in self.grade_estimated[0][1]]
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

    def bayes_update_grade(self,bayes_type='right_partial'):
        Pa,Pb = self.cutoff_value
        type_bayes = ('right_all','right_partial','left','both')
        if bayes_type not in type_bayes:
            raise ValueError(f'bayes_update must be one of {type_bayes}')
        updated_grade_estimated = copy.deepcopy(self.grade_estimated_gr)
        if bayes_type == 'right_all':
            updated_grade_estimate_1 = [anal_cond_exp(i,Pa,self.std_gp[0],self.noise_std[0],self.noise_cov[0]) for i in self.grade_estimated_gr[1][0]]
            updated_grade_estimate_2 = [anal_cond_exp(i,Pa,self.std_gp[1],self.noise_std[1],self.noise_cov[1]) for i in self.grade_estimated_gr[1][1]]
            updated_grade_estimated[1][0] = updated_grade_estimate_1
            updated_grade_estimated[1][1] = updated_grade_estimate_2
        elif bayes_type == 'right_partial':
            df1 = pd.DataFrame({'A':self.grade_estimate_col[0][0],'B':self.grade_estimate_cl[1][0],'pref':np.array(self.stud_pref[0]).T[0]})
            df2 = pd.DataFrame({'A':self.grade_estimate_col[0][1],'B':self.grade_estimate_cl[1][1],'pref':np.array(self.stud_pref[1]).T[0]})
            df1.loc[df1['pref']==1,'pref_name'] = 'B'
            df1.loc[df1['pref']==0,'pref_name'] = 'A'
            df2.loc[df2['pref']==1,'pref_name'] = 'B'
            df2.loc[df2['pref']==0,'pref_name'] = 'A'
            df1['bayes_B'] = np.where(df1['pref_name']=='B',anal_cond_exp(df1['B'],Pa,self.std_gp[0],self.noise_std[0],self.noise_cov[0]),df1['B'])
            df2['bayes_B'] = np.where(df2['pref_name']=='B',anal_cond_exp(df2['B'],Pa,self.std_gp[1],self.noise_std[1],self.noise_cov[1]),df2['B'])
            updated_grade_estimated[1][0] = df1['bayes_B']
            updated_grade_estimated[1][1] = df2['bayes_B']
        elif bayes_type == 'left':
            updated_grade_estimate_1 = [anal_cond_exp(i,Pb,self.std_gp[0],self.noise_std[0],self.noise_cov[0]) for i in self.grade_estimated_gr[0][0]]
            updated_grade_estimate_2 = [anal_cond_exp(i,Pb,self.std_gp[1],self.noise_std[1],self.noise_cov[1]) for i in self.grade_estimated_gr[0][1]]
            updated_grade_estimated[0][0] = updated_grade_estimate_1
            updated_grade_estimated[0][1] = updated_grade_estimate_2 
        elif bayes_type == 'both':
            updated_grade_estimate_1 = [anal_cond_exp(i,Pa,self.std_gp[0],self.noise_std[0],self.noies_cov[0]) for i in self.grade_estimated_gr[1][0]]
            updated_grade_estimate_2 = [anal_cond_exp(i,Pa,self.std_gp[1],self.noise_std[1],self.noise_cov[1]) for i in self.grade_estimated_gr[1][1]]
            updated_grade_estimated[1][0] = updated_grade_estimate_1
            updated_grade_estimated[1][1] = updated_grade_estimate_2
            updated_grade_estimate_1 = [anal_cond_exp(i,Pb,self.std_gp[0],self.noise_std[0],self.noise_cov[0]) for i in self.grade_estimated_gr[0][0]]
            updated_grade_estimate_2 = [anal_cond_exp(i,Pb,self.std_gp[1],self.noise_std[1],self.noise_cov[1]) for i in self.grade_estimated_gr[0][1]]
            updated_grade_estimated[0][0] = updated_grade_estimate_1
            updated_grade_estimated[0][1] = updated_grade_estimate_2 
        self.updated_grade_estimate_col = updated_grade_estimated
        return updated_grade_estimated   


    def market_clear_noise_corr(self,Pa,Pb, bayes = 'none'):
        type_bayes = ('right_all','left','both','none','right_partial')
        # Pa = self.cutoff_value[0]
        # Pb = self.cutoff_value[1]
        self.cutoff_value = [Pa,Pb]
        prop = self.prop_gp[0]
        capA,capB = self.capacities_rate
        prefi, prefii = self.prop_all_g_prefer
        sigmai, sigmaii = self.estimate_std_i, self.estimate_std_ii
        cori, corii = self.estimate_cov_i, self.estimate_cov_ii
        chi = self.std_gp
        sigma = self.noise_std
        lambdas = self.noise_cov
        if bayes not in type_bayes:
            raise ValueError(f'bayes_update must be one of {type_bayes}')
        if bayes == 'both':
            gr1_ecdf_pa,gr2_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = self.sampling_ecdf(type=bayes)    
            f1 = prop*prefi*(1 - gr1_ecdf_pa) + (1 - prop)*prefii*(1 - gr2_ecdf_pa) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
            f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(gr1_ecdf_pa - multi_ecdf_gr1) + (1 -prop)*prefii*(gr2_ecdf_pa - multi_ecdf_gr2) - capB
        elif (bayes == 'right_all') :
            gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = self.sampling_ecdf(type=bayes)    
            f1 = prop*prefi*(1 - cdf(Pa, sigmai)) + (1 - prop)*prefii*(1 - cdf(Pa, sigmaii)) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
            f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(cdf(Pa,sigmai) - multi_ecdf_gr1) + (1 -prop)*prefii*(cdf(Pa,sigmaii) - multi_ecdf_gr2) - capB   
        elif (bayes == 'right_partial'):
            gr1_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pa,gr2_ecdf_pb,multi_ecdf_gr1_A,multi_ecdf_gr1_B,multi_ecdf_gr2_A,multi_ecdf_gr2_B = self.sampling_ecdf(type=bayes)
            f1 = prop*prefi*(1 - gr1_ecdf_pa) + (1 - prop)*prefii*(1 - gr2_ecdf_pa) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1_B) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2_B) - capA
            f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(gr1_ecdf_pa - multi_ecdf_gr1_A) + (1 -prop)*prefii*(gr2_ecdf_pa - multi_ecdf_gr2_A) - capB   
        return f1, f2

    def objective_bads(self,params):
        Pa,Pb = params
        f1,f2 = self.market_clear_noise_corr(Pa,Pb,bayes='right_partial')
        return np.abs(f1) + np.abs(f2)
        # return f1**2+f2**2


@dataclass
class solver:
    lower_bounds: Optional[List] = None
    upper_bounds: Optional[List] = None
    plausible_lower_bounds: Optional[List] = None
    plausible_upper_bounds: Optional[List] = None
    x0: Optional[List] = None     # Starting point  
    optimize_result: Optional[List] = None

    def __post_init__(self):
        if self.lower_bounds is None:
            self.lower_bounds = np.array([-10,-10])
        if self.upper_bounds is None:
            self.upper_bounds = np.array([10, 10])
        if self.plausible_lower_bounds is None:
            self.plausible_lower_bounds = np.array([-5, -5])
        if self.plausible_upper_bounds is None:
            self.plausible_upper_bounds = np.array([5, 5])
        if self.x0 is None:
            self.x0 = np.array([0, 0])

    def cutoff_solver(self, objective_bads: Callable):
        bads = BADS(objective_bads, self.x0, self.lower_bounds, self.upper_bounds, self.plausible_lower_bounds, self.plausible_upper_bounds)
        optimize_result = bads.optimize()
        self.optimize_result = optimize_result
        return optimize_result

    def print_result(self):
        x_min = self.optimize_result['x']
        fval = self.optimize_result['fval']
        print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
        print(f"total f-count: {self.optimize_result['func_count']}, time: {round(self.optimize_result['total_time'], 2)} s")
        return x_min,fval

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

# def grades_gr_to_grades_col(grade_estimated_gr):
#     lst = []
#     for i in range(grade_estimated_gr[0].shape[1]):
#         col_grades = []
#         for j in range(len(grade_estimated_gr)):
#             gr_grades = []
#             for t in range(grade_estimated_gr[j].shape[0]):
#                 gr_grades.append(grade_estimated_gr[j][t][i].item())
#             col_grades.append(gr_grades)
#         lst.append(col_grades)
#     return lst



def multivariate_ecdf(vector_points,value):
    list_values = np.vstack([value,vector_points])
    rank = rankn(list_values)
    return rank[0]/len(rank)


# def sampling_ecdf(grade_estimated,stud_pref,Pa,Pb,chi,sigma,lambdas = [0,0] ,type = 'both'):
#     type_bayes = ('right_all','left','both','right_partial')
#     if type not in type_bayes:
#         raise ValueError(f'bayes_update must be one of {type_bayes}')
#     updated_grade_estimated = copy.deepcopy(grade_estimated)
#     if type == 'right_all':
#         updated_grade_estimate_1 = [anal_cond_exp(i,Pa,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[1][0]]
#         updated_grade_estimate_2 = [anal_cond_exp(i,Pa,chi[1],sigma[1], lambdas[1]) for i in grade_estimated[1][1]]
#         res1 = st.ecdf(updated_grade_estimate_1)
#         res2 = st.ecdf(updated_grade_estimate_2)
#         updated_grade_estimated[1][0] = updated_grade_estimate_1
#         updated_grade_estimated[1][1] = updated_grade_estimate_2
#         grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
#         #ECDF
#         multi_ecdf_gr1 = multivariate_ecdf(grade_estimated_gr[0],[Pa,Pb]).item()
#         multi_ecdf_gr2 = multivariate_ecdf(grade_estimated_gr[1],[Pa,Pb]).item()
#         gr1_ecdf_pb = res1.cdf.evaluate(Pb).item()
#         gr2_ecdf_pb = res2.cdf.evaluate(Pb).item()

#         return gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2

#     if type == 'right_partial':
#         # df1 = pd.DataFrame({'A':grade_estimated[0][0],'B':grade_estimated[1][0],'pref':stud_pref[0]}) #gr1
#         # df2 = pd.DataFrame({'A':grade_estimated[0][1],'B':grade_estimated[1][1],'pref':stud_pref[1]}) #gr2
#         df1 = pd.DataFrame({'A':grade_estimated[0][0],'B':grade_estimated[1][0],'pref':np.array(stud_pref[0]).T[0]})
#         df2 = pd.DataFrame({'A':grade_estimated[0][1],'B':grade_estimated[1][1],'pref':np.array(stud_pref[1]).T[0]})
#         df1.loc[df1['pref']==1,'pref_name'] = 'B'
#         df1.loc[df1['pref']==0,'pref_name'] = 'A'
#         df2.loc[df2['pref']==1,'pref_name'] = 'B'
#         df2.loc[df2['pref']==0,'pref_name'] = 'A'
#         df1['bayes_B'] = np.where(df1['pref_name']=='B',anal_cond_exp(df1['B'],Pa,chi[0],sigma[0],lambdas[0]),df1['B'])
#         df2['bayes_B'] = np.where(df2['pref_name']=='B',anal_cond_exp(df2['B'],Pa,chi[1],sigma[1],lambdas[1]),df2['B'])
#         res1A = st.ecdf(df1.loc[df1['pref'] == 0,'A'])
#         res2A = st.ecdf(df2.loc[df2['pref'] == 0,'A'])
#         res1B = st.ecdf(df1.loc[df1['pref'] == 1, 'bayes_B'])
#         res2B = st.ecdf(df2.loc[df2['pref'] == 1, 'bayes_B'])
#         # updated_grade_estimated[1][0] = df1['bayes_B']
#         # updated_grade_estimated[1][1] = df2['bayes_B']
#         # grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
#         #ECDF
#         multi_ecdf_gr1_A = multivariate_ecdf(df1.loc[df1['pref']==0,['A','bayes_B']],[Pa,Pb]).item()
#         multi_ecdf_gr1_B = multivariate_ecdf(df1.loc[df1['pref']==1,['A','bayes_B']],[Pa,Pb]).item()
#         multi_ecdf_gr2_A = multivariate_ecdf(df2.loc[df2['pref']==0,['A','bayes_B']],[Pa,Pb]).item()
#         multi_ecdf_gr2_B = multivariate_ecdf(df2.loc[df2['pref']==1,['A','bayes_B']],[Pa,Pb]).item()
#         gr1_ecdf_pb = res1B.cdf.evaluate(Pb).item()
#         gr2_ecdf_pb = res2B.cdf.evaluate(Pb).item()
#         gr1_ecdf_pa = res1A.cdf.evaluate(Pa).item()
#         gr2_ecdf_pa = res2A.cdf.evaluate(Pa).item()
#         return [gr1_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pa,gr2_ecdf_pb,multi_ecdf_gr1_A,multi_ecdf_gr1_B,multi_ecdf_gr2_A,multi_ecdf_gr2_B]

#     if type == 'both':
#         updated_grade_estimate_1B = [anal_cond_exp(i,Pa,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[1][0]]
#         updated_grade_estimate_2B = [anal_cond_exp(i,Pa,chi[1],sigma[1],lambdas[1]) for i in grade_estimated[1][1]]
#         updated_grade_estimate_1A = [anal_cond_exp(i,Pb,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[0][0]]
#         updated_grade_estimate_2A = [anal_cond_exp(i,Pb,chi[1],sigma[1],lambdas[1]) for i in grade_estimated[0][1]]
#         res1B = st.ecdf(updated_grade_estimate_1B)
#         res2B = st.ecdf(updated_grade_estimate_2B)
#         res1A = st.ecdf(updated_grade_estimate_1A)
#         res2A = st.ecdf(updated_grade_estimate_2A)
#         updated_grade_estimated[1][0] = updated_grade_estimate_1B
#         updated_grade_estimated[1][1] = updated_grade_estimate_2B
#         updated_grade_estimated[0][0] = updated_grade_estimate_1A
#         updated_grade_estimated[0][1] = updated_grade_estimate_2A
#         grade_estimated_gr = grades_col_to_grades_gr(updated_grade_estimated)
#         #ECDF
#         multi_ecdf_gr1 = multivariate_ecdf(grade_estimated_gr[0],[Pa,Pb]).item()
#         multi_ecdf_gr2 = multivariate_ecdf(grade_estimated_gr[1],[Pa,Pb]).item()
#         gr1_ecdf_pb = res1B.cdf.evaluate(Pb).item()
#         gr2_ecdf_pb = res2B.cdf.evaluate(Pb).item()
#         gr1_ecdf_pa = res1A.cdf.evaluate(Pa).item()
#         gr2_ecdf_pa = res2A.cdf.evaluate(Pa).item()

#         return gr1_ecdf_pa,gr2_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2

# def bayes_update_grade(Pa,Pb,grade_estimated,stud_pref,chi,sigma,lambdas = [0.0],bayes_type='right_partial'):
#     type_bayes = ('right_all','right_partial','left','both')
#     if bayes_type not in type_bayes:
#         raise ValueError(f'bayes_update must be one of {type_bayes}')
#     updated_grade_estimated = copy.deepcopy(grade_estimated)
#     if bayes_type == 'right_all':
#         updated_grade_estimate_1 = [anal_cond_exp(i,Pa,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[1][0]]
#         updated_grade_estimate_2 = [anal_cond_exp(i,Pa,chi[1],sigma[1],lambdas[1]) for i in grade_estimated[1][1]]
#         updated_grade_estimated[1][0] = updated_grade_estimate_1
#         updated_grade_estimated[1][1] = updated_grade_estimate_2
#     elif bayes_type == 'right_partial':
#         df1 = pd.DataFrame({'A':grade_estimated[0][0],'B':grade_estimated[1][0],'pref':np.array(stud_pref[0]).T[0]})
#         df2 = pd.DataFrame({'A':grade_estimated[0][1],'B':grade_estimated[1][1],'pref':np.array(stud_pref[1]).T[0]})
#         df1.loc[df1['pref']==1,'pref_name'] = 'B'
#         df1.loc[df1['pref']==0,'pref_name'] = 'A'
#         df2.loc[df2['pref']==1,'pref_name'] = 'B'
#         df2.loc[df2['pref']==0,'pref_name'] = 'A'
#         df1['bayes_B'] = np.where(df1['pref_name']=='B',anal_cond_exp(df1['B'],Pa,chi[0],sigma[0],lambdas[0]),df1['B'])
#         df2['bayes_B'] = np.where(df2['pref_name']=='B',anal_cond_exp(df2['B'],Pa,chi[1],sigma[1],lambdas[1]),df2['B'])
#         updated_grade_estimated[1][0] = df1['bayes_B']
#         updated_grade_estimated[1][1] = df2['bayes_B']
#     elif bayes_type == 'left':
#         updated_grade_estimate_1 = [anal_cond_exp(i,Pb,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[0][0]]
#         updated_grade_estimate_2 = [anal_cond_exp(i,Pb,chi[1],sigma[1],lambdas[1]) for i in grade_estimated[0][1]]
#         updated_grade_estimated[0][0] = updated_grade_estimate_1
#         updated_grade_estimated[0][1] = updated_grade_estimate_2 
#     elif bayes_type == 'both':
#         updated_grade_estimate_1 = [anal_cond_exp(i,Pa,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[1][0]]
#         updated_grade_estimate_2 = [anal_cond_exp(i,Pa,chi[1],sigma[1],lambdas[1]) for i in grade_estimated[1][1]]
#         updated_grade_estimated[1][0] = updated_grade_estimate_1
#         updated_grade_estimated[1][1] = updated_grade_estimate_2
#         updated_grade_estimate_1 = [anal_cond_exp(i,Pb,chi[0],sigma[0],lambdas[0]) for i in grade_estimated[0][0]]
#         updated_grade_estimate_2 = [anal_cond_exp(i,Pb,chi[1],sigma[1],lambdas[1]) for i in grade_estimated[0][1]]
#         updated_grade_estimated[0][0] = updated_grade_estimate_1
#         updated_grade_estimated[0][1] = updated_grade_estimate_2 
#     return updated_grade_estimated   

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
    print(no_col_0,no_col_1)
    return no_col_0,no_col_1

def utility_by_col(cutoff_values,estimated_grade,stud_pref):
    df1 = pd.DataFrame({'A':estimated_grade[0][0],'B':estimated_grade[1][0],'pref':np.array(stud_pref[0]).T[0]})
    df2 = pd.DataFrame({'A':estimated_grade[0][1],'B':estimated_grade[1][1],'pref':np.array(stud_pref[1]).T[0]})
    A_utility_gr1_first_choice = sum(df1.loc[(df1['pref'] == 0)&(df1['A'] > cutoff_values[0]),'A'])
    A_utility_gr2_first_choice = sum(df2.loc[(df2['pref'] == 0)&(df2['A'] > cutoff_values[0]),'A'])
    A_utility_gr1_second_choice = sum(df1.loc[(df1['pref'] == 1)&(df1['A'] > cutoff_values[0])&(df1['B']<cutoff_values[1]),'A'])
    A_utility_gr2_second_choice = sum(df2.loc[(df2['pref'] == 1)&(df2['A'] > cutoff_values[0])&(df2['B']<cutoff_values[1]),'A'])
    B_utility_gr1_first_choice = sum(df1.loc[(df1['pref'] == 1)&(df1['B'] > cutoff_values[1]),'B'])
    B_utility_gr2_first_choice = sum(df2.loc[(df2['pref'] == 1)&(df2['B'] > cutoff_values[1]),'B'])
    B_utility_gr1_second_choice = sum(df1.loc[(df1['pref'] == 0)&(df1['A'] < cutoff_values[0])&(df1['B'] > cutoff_values[1]),'B'])
    B_utility_gr2_second_choice = sum(df2.loc[(df2['pref'] == 0)&(df2['A'] < cutoff_values[0])&(df2['B'] > cutoff_values[1]),'B'])
    print('Total A utility:', A_utility_gr1_first_choice + A_utility_gr2_first_choice + A_utility_gr1_second_choice + A_utility_gr2_second_choice)
    print('A utility for student with first choice:',A_utility_gr1_first_choice + A_utility_gr2_first_choice)
    print('A utility for student with second choice:',A_utility_gr1_second_choice + A_utility_gr2_second_choice)
    print('Total B utility:', B_utility_gr1_first_choice + B_utility_gr2_first_choice + B_utility_gr1_second_choice + B_utility_gr2_second_choice)
    print('B utility for student with first choice:',B_utility_gr1_first_choice + B_utility_gr2_first_choice)
    print('B utility for student with second choice:',B_utility_gr1_second_choice + B_utility_gr2_second_choice)
    return True

def welfare_by_col(cutoff_values,estimated_grade,stud_pref):
    df1 = pd.DataFrame({'A':estimated_grade[0][0],'B':estimated_grade[1][0],'pref':np.array(stud_pref[0]).T[0]})
    df2 = pd.DataFrame({'A':estimated_grade[0][1],'B':estimated_grade[1][1],'pref':np.array(stud_pref[1]).T[0]})
    A_first_choice_admit_gr1 = len(df1.loc[(df1['pref']==0)&(df1['A']>cutoff_values[0])])
    A_first_choice_admit_gr2 = len(df2.loc[(df2['pref']==0)&(df2['A']>cutoff_values[0])])
    A_second_choice_admit_gr1 = len(df1.loc[(df1['pref']==1)&(df1['A']>cutoff_values[0])&(df1['B']<cutoff_values[1])])
    A_second_choice_admit_gr2 = len(df2.loc[(df2['pref']==1)&(df2['A']>cutoff_values[0])&(df2['B']<cutoff_values[1])])
    B_first_choice_admit_gr1 = len(df1.loc[(df1['pref']==1)&(df1['B']>cutoff_values[1])])
    B_first_choice_admit_gr2 = len(df2.loc[(df2['pref']==1)&(df2['B']>cutoff_values[1])])
    B_second_choice_admit_gr1 = len(df1.loc[(df1['pref']==0)&(df1['A']<cutoff_values[0])&(df1['B']>cutoff_values[1])])
    B_second_choice_admit_gr2 = len(df2.loc[(df2['pref']==0)&(df2['A']<cutoff_values[0])&(df2['B']>cutoff_values[1])])
    print('Proportion of student admitted to A with first choice:', (A_first_choice_admit_gr1 + A_first_choice_admit_gr2)/capacities[0])
    print('Proportion of student admitted to A with second choice:', (A_second_choice_admit_gr1 + A_second_choice_admit_gr2)/capacities[0])
    print('Proportion of student admitted to B with first choice:', (B_first_choice_admit_gr1 + B_first_choice_admit_gr2)/capacities[1])
    print('Proportion of student admitted to B with second choice:', (B_second_choice_admit_gr1 + B_second_choice_admit_gr2)/capacities[1])
    return True   

def cdf(x, sigma):
    return norm.cdf(x, scale = sigma)



def cdfmsi(x, y, sigma, rho):
    return norm.cdf(y, scale = sigma) - multivariate_normal([0, 0], [[sigma**2, sigma**2 * rho], [sigma**2 * rho, sigma**2]]).cdf(np.array([x,y]))



def cdfmis(x, y, sigma, rho):
    return norm.cdf(x, scale = sigma) - multivariate_normal([0, 0], [[sigma**2, sigma**2 * rho], [sigma**2 * rho, sigma**2]]).cdf(np.array([x,y]))


# def market_clear(Pa, Pb, grade_estimated, prop, capA, capB, prefi, prefii, sigmai, sigmaii, cori, corii, chi, sigma,bayes = 'none'):
#     type_bayes = ('right','left','both','none')
#     if bayes not in type_bayes:
#         raise ValueError(f'bayes_update must be one of {type_bayes}')
#     if bayes == 'both':
#         gr1_ecdf_pa,gr2_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = sampling_ecdf(grade_estimated,Pa,Pb,chi,sigma,type=bayes)    
#         f1 = prop*prefi*(1 - gr1_ecdf_pa) + (1 - prop)*prefii*(1 - gr2_ecdf_pa) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
#         f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(gr1_ecdf_pa - multi_ecdf_gr1) + (1 -prop)*prefii*(gr2_ecdf_pa - multi_ecdf_gr2) - capB
#     elif bayes == 'right':
#         gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = sampling_ecdf(grade_estimated,Pa,Pb,chi,sigma,type=bayes)    
#         f1 = prop*prefi*(1 - cdf(Pa, sigmai)) + (1 - prop)*prefii*(1 - cdf(Pa, sigmaii)) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
#         f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(cdf(Pa,sigmai) - multi_ecdf_gr1) + (1 -prop)*prefii*(cdf(Pa,sigmaii) - multi_ecdf_gr2) - capB   
#     return f1, f2



# def market_clear_noise_corr(Pa, Pb, grade_estimated, stud_pref, prop, capA, capB, prefi, prefii, sigmai, sigmaii, cori, corii, chi,  sigma, lambdas = [0,0], bayes = 'none'):
#     type_bayes = ('right_all','left','both','none','right_partial')
#     if bayes not in type_bayes:
#         raise ValueError(f'bayes_update must be one of {type_bayes}')
#     if bayes == 'both':
#         gr1_ecdf_pa,gr2_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = sampling_ecdf(grade_estimated,Pa,Pb,chi,sigma,lambdas, type=bayes)    
#         f1 = prop*prefi*(1 - gr1_ecdf_pa) + (1 - prop)*prefii*(1 - gr2_ecdf_pa) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
#         f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(gr1_ecdf_pa - multi_ecdf_gr1) + (1 -prop)*prefii*(gr2_ecdf_pa - multi_ecdf_gr2) - capB
#     elif (bayes == 'right_all') :
#         gr1_ecdf_pb,gr2_ecdf_pb,multi_ecdf_gr1,multi_ecdf_gr2 = sampling_ecdf(grade_estimated,stud_pref,Pa,Pb,chi,sigma,lambdas,type=bayes)    
#         f1 = prop*prefi*(1 - cdf(Pa, sigmai)) + (1 - prop)*prefii*(1 - cdf(Pa, sigmaii)) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2) - capA
#         f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(cdf(Pa,sigmai) - multi_ecdf_gr1) + (1 -prop)*prefii*(cdf(Pa,sigmaii) - multi_ecdf_gr2) - capB   
#     elif (bayes == 'right_partial'):
#         gr1_ecdf_pa,gr1_ecdf_pb,gr2_ecdf_pa,gr2_ecdf_pb,multi_ecdf_gr1_A,multi_ecdf_gr1_B,multi_ecdf_gr2_A,multi_ecdf_gr2_B = sampling_ecdf(grade_estimated,stud_pref,Pa,Pb,chi,sigma,lambdas,type=bayes)
#         f1 = prop*prefi*(1 - gr1_ecdf_pa) + (1 - prop)*prefii*(1 - gr2_ecdf_pa) + prop*(1 - prefi)*(gr1_ecdf_pb - multi_ecdf_gr1_B) + (1 -prop)*(1 - prefii)*(gr2_ecdf_pb - multi_ecdf_gr2_B) - capA
#         f2 = prop*(1 - prefi)*(1 - gr1_ecdf_pb) + (1 - prop)*(1 - prefii)*(1 - gr2_ecdf_pb) + prop*prefi*(gr1_ecdf_pa - multi_ecdf_gr1_A) + (1 -prop)*prefii*(gr2_ecdf_pa - multi_ecdf_gr2_A) - capB   

#     return f1, f2

# def objective(params):
#     Pa,Pb = params
#     f1,f2 = market_clear(Pa, Pb, grade_estimated, prop_gp[0], capacities_rate[0], capacities_rate[1], prop_all_g_prefer, prop_all_g_prefer, sigma_i, sigma_ii, cor_i, cor_ii,chi,sigma)
#     return f1**2 + f2**2