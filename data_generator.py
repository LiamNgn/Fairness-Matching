#%%
from util import *
import numpy as np
#%%

def data_generator(n_stud,prop_gp,mean_gp,std_gp,n_col,noise_mean,sigma, lambdas):
    students = create_students(n_stud,prop_gp,mean_gp,std_gp) #Create the latent qualities vector
    # grade_estimated = create_col_estim(n_col,students, noise_mean, sigma) #Create the college estimated vectors
    grade_estimated_gr = create_col_estim_corr_nois(n_col,students,noise_mean,sigma,lambdas)
#print(grade_estimated_gr)
#%%
# grade_estimated_gr = grades_col_to_grades_gr(grade_estimated)
### The final result before exporting should have the form 
# [array([[ -3.62656623, -5.26878081], 
# [ 2.18116201, -0.81916207], 
# [-11.56393959, -9.60125782], 
# [ 9.52276646, 7.91264667]]), 
# array([[-2.14095061, -3.97569478], 
# [-3.15453398, -2.6409442 ], 
# [ 0.26865271, 4.08664789], 
# [-0.6574626 , -1.35712839], 
# [-2.25286877, 0.86611285], 
# [ 1.8937661 , 0.95869861]])] 
# with 2 colleges, 2 groups, group 1 has 4 students, 
# group 2 has 6 students. 
    for i in range(len(grade_estimated_gr)):
        np.save(f'grade_estimated_gr{i+1}.npy',grade_estimated_gr[i])
    #%%
    stud_pref = create_stud_pref_2(students,prop_all_g_prefer)
    for i in range(len(stud_pref)):
        np.save(f'stud_pref_gr{i+1}.npy',stud_pref[i])
    print('Data generated')
# %%

# %%
