#Import packages
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import erf

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

###Analytically calculate the conditional expectation
def numerator(w_a,p_b,chi,sigma):
    a = chi**2/(chi**2+sigma**2)*w_a
    b = np.sqrt(chi**2*sigma**2/(chi**2+sigma**2))
    return a/2 + a/2*erf((p_b-a)/(np.sqrt(2*(sigma**2+b**2)))) - (b**2)/(np.sqrt(2*np.pi*(b**2 + sigma**2)))*np.exp(-(p_b-a)**2/(2*(b**2+sigma**2)))
    

def denominator(w_a,p_b,chi,sigma):
    mean_cond_dist = chi**2/(chi**2 + sigma**2)*w_a
    var_cond_dist = (sigma**4+2*sigma**2*chi**2)/(chi**2 + sigma**2)
    return norm.cdf(p_b,mean_cond_dist,np.sqrt(var_cond_dist))

def anal_cond_exp(w_a,p_b,chi,sigma):
    return numerator(w_a,p_b,chi,sigma)/denominator(w_a,p_b,chi,sigma)