#Import packages
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import seaborn as sns


#In the case of Gaussian marginals
def joint_dist(w,w_a,p_b,chi,sigma):
    return norm.pdf(w,0,chi)*norm.pdf((w_a - w),0,sigma)*norm.cdf((p_b-w),0,sigma)

def gaussian_denominator(w_a,P_B,chi,sigma):
    mean_cond_dist = chi**2/(chi**2 + sigma**2)*w_a
    var_cond_dist = (sigma**4 + 2*sigma**2*chi**2)/(chi**2 + sigma**2)
    return norm.cdf(P_B,mean_cond_dist,np.sqrt(var_cond_dist))*norm.pdf(w_a,0,np.sqrt(sigma**2 + chi**2))

#Define function inside the integral
def integrand(w,w_a,p_b,chi,sigma):
    return w*norm.pdf(w,0,chi)*norm.pdf((w_a - w),0,sigma)*norm.cdf((p_b-w),0,sigma)#/gaussian_denominator(w_a,p_b,chi,sigma)

#Calculate the integral with respect to w
def exp_val(w_a,p_b,chi,sigma):
    return quad(integrand, -np.inf,np.inf,args=(w_a,p_b,chi,sigma))

def bayesian_est(w_a,p_b,chi,sigma):
    return exp_val(w_a,p_b,chi,sigma)[0]/gaussian_denominator(w_a,p_b,chi,sigma)

