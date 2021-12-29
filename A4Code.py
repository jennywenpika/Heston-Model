#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:15:59 2021

@author: xiaolux
"""
# Load packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# initialize variables 

# create winer process i.e. standard BM


#Question 2
# t is different set of maturities
# get contingent claim 

# get contingent claim 

def contingent_Analytical(T):
    cont_claim = theta * T + (1 - np.exp(-kappa *T)) * (v0 - theta)/kappa
    return cont_claim


def contingent_Sim(T,dt,var_path_1):
    nSim = len(var_path_1[:,1])
    
    simulated_vals = []
    for ii in range(nSim):
        # for each iteration we get a list of values (250)
        y_value = var_path_1[ii,:]
        num_of_y = len(y_value)

        # the index of values at left
        #x_left = np.linspace(0,num_of_y-1, num_of_y)
        
        # calculate contingent by left riemann sum
        
        left_riemann_sum = np.sum(y_value[0:num_of_y-1] * dt)
        simulated_vals.append(max(left_riemann_sum,0))
    return np.mean(simulated_vals)


def contingent_Analytical_sqr(T):
    part = (2 * kappa * theta + eta **2) / (2 * kappa)
    part1 = part * theta * T
    part2 = (v0**2 - part * (2*v0 - theta)) * (1 - np.exp(-2*kappa*T))/ (2 *kappa)
    part3 = part *2*(v0 - theta) * (1-np.exp(- kappa * T))/ kappa
    cont_claim = part1 + part2 + part3
    return cont_claim

def contingent_Sim_sqr(T,dt,var_path_1):
    nSim = len(var_path_1[:,1])
    
    simulated_vals = []
    for ii in range(nSim):
        # for each iteration we get a list of values (250)
        y_value = var_path_1[ii,:]
        num_of_y = len(y_value)

        # the index of values at left
        #x_left = np.linspace(0,num_of_y-1, num_of_y)
        
        # calculate contingent by left riemann sum
        
        left_riemann_sum = np.sum(y_value[0:num_of_y-1]**2 * dt)
        simulated_vals.append(max(left_riemann_sum,0))
    return np.mean(simulated_vals)
  
#print(contingent_Analytical_sqr(T1))
#print(contingent_Sim_sqr(T1,dt,var_path_1))

def plot_value_sqr():
    T = np.linspace(0.1, 1,10)
    simulated_list = []
    analytical_list = []
    for item in T:
        len_t = int(item/dt)
        mu = np.array([0,0])
        cov = np.array([[1, rho] , [rho , 1]])

        z_sim_v = np.zeros((Nsims,len_t))
        z_sim_s = np.zeros((Nsims,len_t))
        for i in range(0, Nsims):
            W = np.random.multivariate_normal(mu, cov, size = len_t)
            z_sim_v[i, :] = W[:,0]
            z_sim_s[i, :] = W[:,1]
        [var_path, S_path] = Risk_Neutral_Mils_col(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s)
        sim = contingent_Sim_sqr(item,dt,var_path)
        simulated_list.append(sim)
        
        ana = contingent_Analytical_sqr(item)
        analytical_list.append(ana)
    plt.plot(T, simulated_list, label='Simulated Contingent Claim')
    plt.plot(T, analytical_list, label='Analytical Contingent Claim')
    #plt.fill_between(K_call, ci_call_up, ci_call_low, color='b', alpha=.1)
    #plt.fill_between(K_put, ci_put_up, ci_put_low, color='b', alpha=.1)
    plt.xlabel('Maturity T')
    plt.ylabel('Value of Contingent Claim')
    plt.title('Analytical Vs Simulated Contingent Claim Value')
    plt.legend()
    
plot_value_sqr()

def plot_value():
    T = np.linspace(0.1, 1,10)
    simulated_list = []
    analytical_list = []
    for item in T:
        len_t = int(item/dt)
        mu = np.array([0,0])
        cov = np.array([[1, rho] , [rho , 1]])

        z_sim_v = np.zeros((Nsims,len_t))
        z_sim_s = np.zeros((Nsims,len_t))
        for i in range(0, Nsims):
            W = np.random.multivariate_normal(mu, cov, size = len_t)
            z_sim_v[i, :] = W[:,0]
            z_sim_s[i, :] = W[:,1]
        [var_path, S_path] = Risk_Neutral_Mils_col(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s)
        sim = contingent_Sim(item,dt,var_path)
        simulated_list.append(sim)
        
        ana = contingent_Analytical(item)
        analytical_list.append(ana)
    plt.plot(T, simulated_list, label='Simulated Contingent Claim')
    plt.plot(T, analytical_list, label='Analytical Contingent Claim')
    #plt.fill_between(K_call, ci_call_up, ci_call_low, color='b', alpha=.1)
    #plt.fill_between(K_put, ci_put_up, ci_put_low, color='b', alpha=.1)
    plt.xlabel('Maturity T')
    plt.ylabel('Value of Contingent Claim')
    plt.title('Analytical Vs Simulated Contingent Claim Value')
    plt.legend()
    
plot_value() 
print(contingent_Analytical(T1))
print(contingent_Sim(T1,var_path_1))

