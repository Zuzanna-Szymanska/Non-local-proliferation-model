##################################################################################################
##################################################################################################
###
### Postprocessing code for the random walk Metropolis–Hastings algorithm for a non-local proliferation function
###
##################################################################################################
##################################################################################################
###
### This file contains Python3 code for computing the mean trajectory of the growth curve of diameters
### of multicellular spheroids estimated with a non-local proliferation model. The program also finds
### a MAP estimator for proliferation rate, kernel size, measurement error, and initial colony radius.
###
### The code was prepared for computations in the paper:
###
### Bayesian inference of a non-local proliferation model
### Z. Szymańska, J.Skrzeczkowski, B.Miasojedow, P.Gwiazda
### arXiv: 2106.05955
###
##################################################################################################
##################################################################################################

import numpy as np
import os.path

### The function returns a table with colony diameters in time.
### Files "colony_diameter%d.csv" with data on diameters dynamic were generated by program metropolis_hastings.py.
def read_path(path_name,generic_name,min_number,max_number):
    data_out=np.ndarray(max_number-min_number,dtype=object)
    tmp_data=np.zeros(1)
    for i in range(max_number-min_number):
        act_number=min_number+i
        a=os.path.isfile(path_name+"/"+(generic_name % act_number))
        if a: tmp_data=np.loadtxt(path_name+"/"+(generic_name % act_number))
        data_out[i]=tmp_data
        #print((generic_name % act_number),a)
    return(data_out)

### Reading the data
generic_name="colony_diameter%d.csv"
test = read_path("./",generic_name,0,19)
data_0=np.stack(test)

mean=data_0.mean(axis=0)
upper = np.quantile(data_0,q=.975,axis=0)
lower = np.quantile(data_0,q=.025,axis=0)

print("mean=",mean)
print("lower=",lower)
print("upper=",upper)
np.savetxt("mean.csv", mean, delimiter=",")
np.savetxt("lower.csv", lower, delimiter=",")
np.savetxt("upper.csv", upper, delimiter=",")

### Finding the MAP estimator
pom_loglik = np.loadtxt('current_loglik.csv', dtype='float', delimiter=',', unpack=True)
ind_loglik = np.where(pom_loglik==pom_loglik.max())
indeks = ind_loglik[0][0]
print("Indeks=",indeks)
print("Maks=",pom_loglik.max())
pom_mu = np.loadtxt('log_mu.csv', dtype='float', delimiter=',', unpack=True)
pom_sigma_k = np.loadtxt('log_sigma_k.csv', dtype='float', delimiter=',', unpack=True)
pom_sigma_i = np.loadtxt('log_sigma_i.csv', dtype='float', delimiter=',', unpack=True)
pom_sigma_o = np.loadtxt('log_sigma_o.csv', dtype='float', delimiter=',', unpack=True)
print("mu=",np.exp(pom_mu[indeks]))
print("sigma_k=",np.exp(pom_sigma_k[indeks]))
print("sigma_i=",np.exp(pom_sigma_i[indeks]))
print("sigma_o=",np.exp(pom_sigma_o[indeks]))


