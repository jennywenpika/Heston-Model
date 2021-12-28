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

def contingent_Analytical(T):
    cont_claim = theta * T + (1 - np.exp(-kappa *T)) * (v0 - theta)/kappa
    return cont_claim

def contingent_Sim(T,var_path_1):
    sim_Var = var_path_1[:,-1]
    simulated_vals = []
    for i in range(len(sim_Var)):
        #print(i)
        # calculate
        simulated = sim_Var[i]**2/2 - v0**2/2
        simulated_vals.append(max(simulated,0))
    return np.mean(simulated_vals)

# different maturities

def plot_value():
    T = np.linspace(0.1, 1,10)
    simulated_list = []
    analytical_list = []
    for item in T:
        len_t = int(item/dt)
        [var_path, S_path] = Risk_Neutral_Mils_col(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s)
        sim = contingent_Sim(item,var_path)
        simulated_list.append(sim)
        
        ana = contingent_Analytical(item)
        analytical_list.append(ana)
    plt.plot(T, simulated_list, label='Simulated Contingent Claim')
    plt.plot(T, analytical_list, label='Analytical Contingent Claim')
    #plt.fill_between(K_call, ci_call_up, ci_call_low, color='b', alpha=.1)
    #plt.fill_between(K_put, ci_put_up, ci_put_low, color='b', alpha=.1)
    #plt.xlabel('Strike')
    #plt.ylabel('Implied Volatility')
    #plt.title('Volatility Simulation for T = 0.25')
    plt.legend()
    
      
  
print(contingent_Analytical(T1))
print(contingent_Sim(T1,var_path_1))

