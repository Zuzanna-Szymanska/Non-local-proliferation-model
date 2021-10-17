##################################################################################################
##################################################################################################
###
### Auxiliary code for solving a non-local proliferation model.
### It is intended to generate the data needed to compare the mean trajectory obtained with the
### Metropolis-Hastings algorithm with the solution obtained with the MAP estimator.
###
##################################################################################################
##################################################################################################
###
### The code was prepared for computations in the paper:
###
### Bayesian inference of a non-local proliferation model
### Z. Szymańska, J.Skrzeczkowski, B.Miasojedow, P.Gwiazda
### arXiv: 2106.05955
###
##################################################################################################
##################################################################################################

#!/bin/env python
import time
import numpy as np
from scipy.stats import chi2
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pylab
from multiprocessing import Pool
from math import isclose
from math import ceil
from scipy import sparse
from tqdm import tqdm
     

##################################################################################################
##################################################################################################
###
### The "init_function(R,sigma,dr)" returns the vector C with the initial mass of the colony at a given radius.
###
##################################################################################################
##################################################################################################

### The initial mass of the colony for data set a.
def init_function(R,sigma_i,dr):
    C = np.arange(rmin,rmax,dr)
    sigma_i_new  = sigma_i*1.065
    alpha = 13
    for k in range(len(R)):
        if k*dr<=sigma_i_new:
            C[k] = (4*np.pi*R[k]*R[k]*dr)*(max(0,1-(R[k]/sigma_i_new)**alpha))
        else:
            C[k] = 0
    return C

### The initial mass of the colony for data set b.
#def init_function(R,sigma_i,dr):
#    C = np.arange(rmin,rmax,dr)
#    sigma_i_new  = sigma_i*1.065
#    alpha = 13
#    for k in range(len(R)):
#        if k*dr<=sigma_i_new:
#            C[k] = (4*np.pi*R[k]*R[k]*dr)*(max(0,1-(R[k]/sigma_i_new)**alpha))
#        else:
#            C[k] = 0
#    return C

### The initial mass of the colony for data set c.
#def init_function(R,sigma_i,dr):
#    C = np.arange(rmin,rmax,dr)
#    sigma_i_new  = sigma_i*1.06
#    alpha = 13
#    for k in range(len(R)):
#        if k*dr<=sigma_i_new:
#            C[k] = (4*np.pi*R[k]*R[k]*dr)*(max(0,1-(R[k]/sigma_i_new)**alpha))
#        else:
#            C[k] = 0
#    return C

### The function returns the radius that gives 95% of the tumour mass
def get_radius(C,R,dr):
    masa = np.sum(C)*0.95
    pom = 0
    prom = 0
    for r in range(len(R)):
        if (pom + C[r] <= masa):
            pom += C[r]
            prom = r
        else:
            break
    return (prom+2)*dr

##################################################################################################
##################################################################################################
###
### The function "funkcja_L(R,sigma_k)" returns an auxiliary matrix for the calculation of the convolution.
###
##################################################################################################
##################################################################################################

def funkcja_L(R,sigma_k):
    norm = 3/(16*np.pi*(sigma_k**3))
    sigma_kpow = sigma_k**2
    r0 = R[0]
    dr = R[1]-R[0]
    diags = list()
    offsets = list()
    k = 0
    tmp = np.ones_like(R) *(k*dr)**2 - sigma_kpow
    diags.append(tmp)
    offsets.append(k)
    k = 1
    while k <  np.sqrt(sigma_kpow)/dr:
        tmp = np.ones_like(R) *(k*dr)**2 - sigma_kpow
        diags.append(tmp)
        diags.append(tmp)
        offsets.append(k)
        offsets.append(-k)
        k+=1
    L11 = sparse.spdiags(diags, offsets, len(R), len(R))
    subBlock = np.arange(int( np.sqrt(sigma_kpow)/dr-2*r0/dr ) +  1)
    sI, sJ = np.meshgrid(subBlock,subBlock)
    sL = (np.minimum( sI+sJ , np.sqrt(sigma_kpow)/dr-2*r0/dr )+2*r0/dr)**2*dr**2 - sigma_kpow
    L12 = sparse.dia_matrix( sparse.coo_matrix( (sL.ravel(), ( sI.ravel(), sJ.ravel() ) ), shape=(len(R), len(R)) ) )
    L11 = L12 - L11
    diags = list()
    offsets = list()
    N = len(R)
    k = 0
    tmp = list()
    for i in range(N-k):
        j = i + k
        tmp.append( norm / ((i*dr+r0)*(j*dr+r0)) )
    diags.append(tmp)
    offsets.append(k)
    k = 1
    while k <  np.sqrt(sigma_kpow)/dr + 3:
        tmp = list()
        for i in range(N-k):
            j = i + k
            tmp.append( norm / ((i*dr+r0)*(j*dr+r0)) )
        diags.append(tmp)
        offsets.append(k)
        tmp = list()
        for i in range(k,N):
            j = i - k
            tmp.append( norm / ((i*dr+r0)*(j*dr+r0)) )
        diags.append(tmp)
        offsets.append(-k)
        k+=1
    normRij1 = sparse.csr_matrix(sparse.diags(diags, offsets, shape=(N, N )))
    L11 = L11.multiply(normRij1)
    return L11

### The function returns the convolution of the kernel and the vector of masses.
def splot(L,C):
    return L.dot(C)

### 4th order Runge-Kutta IV scheme.
def RK(C,R,L,delta_t,dr,sigma_k,mu):
    kc = splot(L,C)
    R2 = np.asarray([x**2 for x in R])
    k_1 = delta_t * mu*kc*(4*np.pi*dr*R2-C)
    C_1 = C+0.5*k_1
    kc_1 = splot(L,C_1)
    k_2  = delta_t * mu*kc_1*(4*np.pi*dr*R2-C_1)
    C_2 = C_1+0.5*k_2
    kc_2 = splot(L,C_2)
    k_3 = delta_t * mu*kc_2*(4*np.pi*dr*R2-C_2)
    C_3 = C_2+k_3
    kc_3 = splot(L,C_3)
    k_4 = delta_t * mu*kc_3*(4*np.pi*dr*R2-C_3)
    return C + (k_1 + 2*k_2 + 2*k_3 + k_4)/6

### Euler scheme (alternative to 4th order Runge-Kutta IV scheme).
def Euler(C,R,L,delta_t,dr,sigma_k,mu):
    kc = splot(L,C)
    R2 = np.asarray([x**2 for x in R])
    k_1 = delta_t * mu*kc*(4*np.pi*dr*R2-C)
    return C + k_1

### The function calculates the cell colony diameter vector in successive time steps with given the parameters using 4th order Runge-Kutta method.
def simulate_RK(rmin,rmax,tmax,dr,dt,sigma_i,sigma_k,mu):
    czas= int(np.round(tmax/dt))+1
    colony_mass_RK = np.zeros(czas)
    colony_diameter_RK = np.zeros(czas)
    time = np.zeros(czas)
    R = np.arange(rmin,rmax,dr)
    C = init_function(R,sigma_i,dr)
    colony_diameter_RK[0]= 2*get_radius(C,R,dr)
    L=funkcja_L(R,sigma_k)
    for i in tqdm(range(1,czas)):
        C = RK(C,R,L,dt,dr,sigma_k,mu)
        colony_diameter_RK[i] = 2*get_radius(C,R,dr)
        colony_mass_RK[i] = sum(C)
        time[i] = i*dt
    return colony_mass_RK,C,colony_diameter_RK,time

### The function calculates the cell colony diameter vector in successive time steps with given the parameters using Euler method.
def simulate_E(rmin,rmax,tmax,dr,dt,sigma_i,sigma_k,mu):
    czas= int(np.round(tmax/dt))+1
    colony_mass_E = np.zeros(czas)
    colony_diameter_E = np.zeros(czas)
    time = np.zeros(czas)
    R = np.arange(rmin,rmax,dr)
    C = init_function(R,sigma_i,dr)
    colony_diameter_E[0]= 2*get_radius(C,R,dr)
    L=funkcja_L(R,sigma_k)
    for i in tqdm(range(1,czas)):
        C = Euler(C,R,L,dt,dr,sigma_k,mu)
        colony_diameter_E[i] = 2*get_radius(C,R,dr)
        colony_mass_E[i] = sum(C)
        time[i] = i*dt
    return colony_mass_E,C,colony_diameter_E,time

##################################################################################################
##################################################################################################
###
### Parameters for data set a.
###
##################################################################################################
##################################################################################################

dr = 0.005                   # Discretization in space
dt = 0.005                   # Discretization in time
rmin = dr
tau = 8
rmax = dr*ceil(3.2/dr)       # Size of the domain
tmax = 26 - tau              # Maximal time
scale = 0.22                 # Acceptance probability parameter
mu= 1.7264                   # Proliferation rate
sigma_k= 0.0806              # Kernel size
sigma_i= 0.2469              # Initial colony radius
sigma_o= 0.0957              # Measurement error


##################################################################################################
##################################################################################################
###
### Parameters for data set b.
###
##################################################################################################
##################################################################################################

#dr = 0.005                   # Discretization in space
#dt = 0.05                    # Discretization in time
#rmax = dr*ceil(3.0/dr)       # Size of the domain
#tau = 19.3939393939
#tmax = 163 - tau             # Maximal time
#scale = 0.13                 # Acceptance probability parameter
#rmin = dr
#mu= 0.3603                   # Proliferation rate
#sigma_k= 0.0479              # Kernel size
#sigma_i= 0.3744              # Initial colony radius
#sigma_o= 0.0649              # Measurement error

##################################################################################################
##################################################################################################
###
### Parameters for data set c.
###
##################################################################################################
##################################################################################################

#dr = 0.002                   # Discretization in space
#dt = 0.01                    # Discretization in time
#rmin = dr
#rmax = dr*ceil(2.0/dr)
#tau = 13.9597315436
#tmax = 76 - tau
#scale = 0.1                  # Acceptance probability parameter
#rmin = dr
#mu= 0.3616                   # Proliferation rate
#sigma_k= 0.0342              # Kernel size
#sigma_i= 0.7518              # Initial colony radius
#sigma_o= 0.0256              # Measurement error


# Main program - solving the model for given parameters
R = np.arange(rmin,rmax,dr)
L = funkcja_L(R,sigma_k)
# Solution obtained with 4th order Runge-Kutta scheme
colony_mass_RK,C_RK,colony_diameter_RK,time_RK = simulate_RK(rmin,rmax,tmax,dr,dt,sigma_i,sigma_k,mu)
# Solution obtained with Euler scheme
colony_mass_E,C_E,colony_diameter_E,time_E = simulate_E(rmin,rmax,tmax,dr,dt,sigma_i,sigma_k,mu)

np.savetxt("colony_mass_best_RK.csv",colony_mass_RK, delimiter=",")
np.savetxt("colony_diameter_best_RK.csv",colony_diameter_RK, delimiter=",")
np.savetxt("time_best_RK.csv",time_RK, delimiter=",")
np.savetxt("colony_mass_best_E.csv",colony_mass_E, delimiter=",")
np.savetxt("colony_diameter_best_E.csv",colony_diameter_E, delimiter=",")
np.savetxt("time_best_E.csv",time_E, delimiter=",")

