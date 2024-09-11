from util import *
import numpy as np
from param_def import *

students = create_students(n_stud,prop_gp,mean_gp,std_gp = chi) #Create the latent qualities vector
grade_estimated = create_col_estim(n_col,students, noise_mean, sigma) #Create the college estimated vectors

grade_estimated_gr = grades_col_to_grades_gr(grade_estimated)
for i in range(len(grade_estimated_gr)):
    np.save(f'grade_estimated_gr{i+1}.npy',grade_estimated_gr[i])

stud_pref = create_stud_pref_2(students,prop_all_g_prefer)
for i in range(len(stud_pref)):
    np.save(f'stud_pref_gr{i+1}.npy',stud_pref[i])


