import numpy as np

#Parameter to create student
n_stud = 20000
prop_gp = [0.4,0.6] #vector gamma in the paper
mean_gp = [0,0]
chi = [3,3] #std of latent qualities of groups

#Parameter of college
n_col = 2 #Number of college
noise_mean = [0,0]
sigma = [5,5] #std of noise of each groups
lambdas = [3,0.5] #cov of noies of each groups
prop_all_g_prefer = [0.4,0.4] #This should be the vector beta in the paper

capacities_rate = [0.2,0.2] #This should be the vector alpha in the paper.
capacities = [int(r * n_stud) for r in capacities_rate]#This vector must have the same length as the number of college, i.e. len(capacities) = n_col. The componenets should be integers

sigma_i = np.sqrt(chi[0]**2 + sigma[0]**2)
sigma_ii = np.sqrt(chi[1]**2 + sigma[1]**2)
cor_i = chi[0]**2/sigma_i**2
cor_ii = chi[1]**2/sigma_ii**2
std_estimated = [np.sqrt(i**2 + j**2) for i,j in zip(chi,sigma)]