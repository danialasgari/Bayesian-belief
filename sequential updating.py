#!/usr/bin/env python
# coding: utf-8
# In[ ]:

import random
import numpy as np
import numba as nb
import pandas as pd
import itertools 
import arviz as az
import rpy2
import pyjags
import math
import multiprocessing
from multiprocessing import Pool
import time


#the length of each side of the lattice
side = 100

#number of Food sources
nFood = 1000

#Food density
#d = nFood / (side_lenght**2)

#degree of the patchiness
p = 1

#use numba to speed up simulations
nb.jit(nopython=True)
def Food_sim(i0,j0):
    #Coordinate of the starting point
    i = i0
    j = j0
    
    #Repeat the loop until nFood is added to the lattice
    while len(np.where(lattice==1)[1])<nFood:
        #Choose a direction randomly (up, down, right or left)
        val = random.randint(1, 4)
        
        #for right:
        if val == 1 and i < side-p:
            i = i + p
            j = j
        #boundary condition for right side of the lattice:
        if val == 1 and i == side-p:
            i = 0
            j = j
        #for left:
        if val == 2 and i >= p:
            i = i - p
            j = j
        #boundary condition for left side of the lattice:
        if val == 2 and i < p:
            i = side-1
            j = j
        #for up:
        if val == 3 and j < side-p:
            i = i
            j = j + p
        #boundary condition for top of the lattice:
        if val == 3 and j == side-p:
            i = i
            j = 0
        #for down:
        if val == 4 and j >= p:
            i = i
            j = j - p
        #boundary condition for bottom of the lattice:
        if val == 4 and j < p:
            i = i
            j = side-1
            
        #place a Food at (i,j) coordinate
        lattice[i,j] = 1

#Model0
Model0 =  '''model {
  for(i in 1:N) {
    y[i,1:2] ~ dmnorm(mu[], prec[ , ])
  }
  # Constructing the covariance matrix and the corresponding precision matrix.
  prec[1:2,1:2] <- inverse(cov[,])
  cov[1,1] <- sigma[1] * sigma[1]
  cov[1,2] <- sigma[1] * sigma[2] * rho
  cov[2,1] <- sigma[1] * sigma[2] * rho
  cov[2,2] <- sigma[2] * sigma[2]
  
  mu[1] ~ dnorm(0,  1.0/1e6)
  mu[2] ~ dnorm(0,  1.0/1e6)}
 '''

#Model1
Model1 =  '''model {
  for(i in 1:N) {
    y[i,1:2] ~ dmnorm(mu[], prec[ , ])
  }
  # Constructing the covariance matrix and the corresponding precision matrix.
  prec[1:2,1:2] <- inverse(cov[,])
  cov[1,1] <- sigma[1] * sigma[1]
  cov[1,2] <- sigma[1] * sigma[2] * rho
  cov[2,1] <- sigma[1] * sigma[2] * rho
  cov[2,2] <- sigma[2] * sigma[2]
  
  mu[1] ~ dnorm(mean1,  1.0/sigma_1)
  mu[2] ~ dnorm(mean2,  1.0/sigma_2)}
 '''
      
#Boundaries:
boundary = side-1
X = np.arange(0,+boundary)
Y = np.arange(0,+boundary)

#Update the belief repeat_n times:
repeat_n = 1000
    
#Sample size
ndata = 10

#Uncertainty in location of food (uncertainty in memory?)
sigma = [1, 1]
rho = 0.01

cov_mat  = [[sigma[0]**2, sigma[0]*sigma[1]*rho], 
            [sigma[0]*sigma[1]*rho, sigma[1]**2]]


#The effect of food unit on the fitness
food_effect = 0.15
#The effect of distance unit on the fitnes
dist_eff = 0.001

#Coordinate of all positions on the lattice
Coord_all = list(itertools.product(np.arange(0,boundary), np.arange(0,boundary)))
    
def simulation():
    
    Total_food = 0
    dist_trav  = 0
    
    
    global lattice
    #Make a lattice of size side x side
    lattice = np.zeros((side,side))
    
    #Find a random starting point for simulation of the Food location (i0,j0)
    i0=round(random.uniform(0, side-1))
    j0=round(random.uniform(0, side-1))
    
    #Simulate the Food location
    Food_sim(i0=i0,j0=j0)

    
    #Data with certain noise
    #Random starting point    
    startingpoint = random.sample(Coord_all,1)[0]
    
    Start_X = startingpoint[0]
    Start_Y = startingpoint[1]
    
    Data = np.random.multivariate_normal([Start_X, Start_Y], cov_mat, ndata)
    
    
    #Run the model
    jags_model \
    = pyjags.Model(Model0, data= dict(y=Data, N = ndata,  sigma = sigma, rho = rho), chains=3,  threads=3, chains_per_thread=1,progress_bar=False)
    

    
    #Sample the stationary dist
    Model_samples = jags_model.sample(iterations=1000, vars=['mu'])
    
    x = [Model_samples['mu'][0, :, 0],Model_samples['mu'][0, :, 1],Model_samples['mu'][0, :, 2]][0]
    y = [Model_samples['mu'][1, :, 0],Model_samples['mu'][1, :, 1],Model_samples['mu'][1, :, 2]][0]
    
    
    #Periodic Boundary Condition for the posterior belief
    x[np.where(x>boundary)]     =  (x[np.where(x>boundary)]-boundary) -1
    x[np.where(x<0)]            = +boundary - (abs(x[np.where(x<0)])) +1
    y[np.where(y>boundary)]     =  (y[np.where(y>boundary)]-boundary) -1
    y[np.where(y<0)]            = +boundary - (abs(y[np.where(y<0)])) +1
    
    #Find the HDI 99%
    CI_x = az.hdi(x, hdi_prob=.99)
    CI_y = az.hdi(y, hdi_prob=.99)
    
    CI_x.astype(int)
    CI_y.astype(int)
    
    CI_x_r = np.arange(CI_x.astype(int)[0], CI_x.astype(int)[1]+1)
    CI_y_r = np.arange(CI_y.astype(int)[0], CI_y.astype(int)[1]+1)
    
    
    #sample size
    sample_size = ndata
    
    
    #If HDI interval is smaller than sample size, reduce the sample size:
    if len(CI_x_r)  <= sample_size or len(CI_y_r) <= sample_size:
        sample_size = min(len(CI_x_r), len(CI_y_r))
    else: 
        sample_size = ndata
        
    
    #if sample is the whole grid:
    if len(list(set(Coord_all) - set(list(itertools.product(CI_x_r, CI_y_r))))) ==0:
        searched_coor0 = random.sample((Coord_all),sample_size)  
    else:
        #take sample with 99% probability from here:
        if random.random() < 0.99:
            searched_coor0 = random.sample(list(itertools.product(CI_x_r, CI_y_r)),sample_size)
        else:
            searched_coor0 = random.sample(list(set(Coord_all) - set(list(itertools.product(CI_x_r, CI_y_r)))),sample_size)

    #eating the foods
    food_taken = np.zeros(len(searched_coor0))
    for i in range(len(searched_coor0)):
        food_taken[i] = lattice[searched_coor0[i]]
                
    #foods are eaten!
    for i in range(len(searched_coor0)):
        lattice[searched_coor0[i]] = 0 
        
    Total_food = np.sum(food_taken)
    
    for zz in range(repeat_n):
        
        #for the next round, update the belief by the sample taken in the previous round:
        if len(np.where(food_taken ==1)[0]) > 0:
            random_food  = np.random.choice(np.where(food_taken ==1)[0])
            new_coord    = searched_coor0[random_food]
            food_loc_X   = new_coord[0]
            food_loc_Y   = new_coord[1]
        else:
            new_coord    = random.sample(searched_coor0,1)[0]
            food_loc_X   = new_coord[0]
            food_loc_Y   = new_coord[1]
        
        #Data with certain noise using the sample from the previous round
        Data = np.random.multivariate_normal([food_loc_X, food_loc_Y], cov_mat, ndata)
        
        #Mean and sd of the previous (new) posterior (prior)
        mean1   = np.mean(x)
        mean2   = np.mean(y)
        sigma_1 = np.std(x)
        sigma_2 = np.std(y)
        
        #run the model
        jags_model \
        = pyjags.Model(code=Model1, data=dict(y=Data, N = ndata,  sigma = sigma, rho = rho, mean1 = mean1, mean2 = mean2 , sigma_1 = sigma_1, sigma_2 = sigma_2), chains=3,  threads=3, chains_per_thread=1,progress_bar=False)
       
 
        Model_samples = jags_model.sample(iterations=1000, vars=['mu'])
        
        #Get the posterior dist
        x = [Model_samples['mu'][0, :, 0],Model_samples['mu'][0, :, 1],Model_samples['mu'][0, :, 2]][0]
        y = [Model_samples['mu'][1, :, 0],Model_samples['mu'][1, :, 1],Model_samples['mu'][1, :, 2]][0]
        
        #Periodic Boundary Condition for the new posterior
        x[np.where(x>boundary)]     =  (x[np.where(x>boundary)]-boundary) -1
        x[np.where(x<0)]            = +boundary - (abs(x[np.where(x<0)])) +1
        y[np.where(y>boundary)]     =  (y[np.where(y>boundary)]-boundary) -1
        y[np.where(y<0)]            = +boundary - (abs(y[np.where(y<0)])) +1
        
        #Find the HDI
        CI_x = az.hdi(x, hdi_prob=.99)
        CI_y = az.hdi(y, hdi_prob=.99)
        
        CI_x.astype(int)
        CI_y.astype(int)
        
        CI_x_r = np.arange(CI_x.astype(int)[0], CI_x.astype(int)[1]+1)
        CI_y_r = np.arange(CI_y.astype(int)[0], CI_y.astype(int)[1]+1)
        
        #Sample size
        sample_size = ndata
                
        #If HDI interval is smaller than sample size, reduce the sample size:
        if len(CI_x_r)  <= sample_size or len(CI_y_r) <= sample_size:
            sample_size = min(len(CI_x_r), len(CI_y_r))
        else: 
            sample_size = ndata
        
        #if sample is the whole grid:
        if len(list(set(Coord_all) - set(list(itertools.product(CI_x_r, CI_y_r))))) ==0:
            searched_coor = random.sample((Coord_all),sample_size)  
        else:
            #take sample with 99% probability from here:
            if random.random() < 0.99:
                searched_coor = random.sample(list(itertools.product(CI_x_r, CI_y_r)),sample_size)
            else:
                searched_coor = random.sample(list(set(Coord_all) - set(list(itertools.product(CI_x_r, CI_y_r)))),sample_size)


        #eating the foods
        food_taken = np.zeros(len(searched_coor))
        for i in range(len(searched_coor)):
            food_taken[i] = lattice[searched_coor[i]]
            
        #foods are eaten!
        for i in range(len(searched_coor)):
            lattice[searched_coor[i]] = 0 
        Total_food = np.sum(food_taken) + Total_food
        
        
        #Calculate the traveled distance
        distance = []
        
        for i in range(len(searched_coor0)):
            i_coor  = searched_coor0[i]
            for j in range(len(searched_coor)):
                j_coor = searched_coor[j]
                distance.append(math.dist(i_coor, j_coor))
        
        dist_trav = int(np.max(distance)) + dist_trav
        
        
        #Save the previous coordinates
        searched_coor0 = searched_coor
        
        
        
        #update mean and sd      
        mean1   = np.mean(x)
        mean2   = np.mean(y)
        sigma_1 = np.std(x)
        sigma_2 = np.std(y)
        
        
        
            
   
    Fitness = 1/(1+np.exp(-(food_effect*Total_food - dist_eff*dist_trav)))
    return [Total_food, dist_trav, Fitness]






dfout = pd.DataFrame({"Total food": [] , "Distance": [],"Fitness":[]})

for i in range(1000):
    dfout.loc[len(dfout)] =  simulation()

df = pd.DataFrame(data=dfout)

df.to_csv('p_1_sd_1.csv', index=False)