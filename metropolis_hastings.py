##################################################################################################
##################################################################################################
###
###      Random walk Metropolis–Hastings algorithm for a non-local proliferation function
###
##################################################################################################
##################################################################################################
###
### This file contains Python3 code for computing Bayesian inference for a non-local proliferation model.
###
### The code was prepared for computations in the following papers:
###
### Bayesian inference of a non-local proliferation model
### Z. Szymanska, J.Skrzeczkowski, B.Miasojedow, P.Gwiazda
### arXiv: 2106.05955
###
### Convergence of EBT method for nonlocal model of cell proliferation with discontinuous interaction kernel
### P.Gwiazda, B.Miasojedow, J.Skrzeczkowski, Z. Szymanska
### arXiv: 2106.05115
###
### The theoretical background concerning the theory of measure aspects is explained in the book (Chapter 4.2):
### Spaces of Measures and their Applications to Structured Population Models
### C. Duell, P. Gwiazda, A. Marciniak-Czochra, J. Skrzeczkowski
### to be published in October 2021 by Cambridge University Press
### https://www.cambridge.org/pl/academic/subjects/mathematics/differential-and-integral-equations-dynamical-systems-and-co/spaces-measures-and-their-applications-structured-population-models?format=HB
###
### PLEASE CITE IF YOU USE THIS CODE.
###
##################################################################################################
##################################################################################################
###
### Simulations were run for three data sets originating from the paper:
### J. Folkman and M. Hochberg. Self-regulation of growth in three dimensions. J Exp Med., 138(4):745–753, 1973.
###
### Data are redrawn in the paper:
### Bayesian inference of a non-local proliferation model
### Z. Szymanska, J.Skrzeczkowski, B.Miasojedow, P.Gwiazda
### arXiv: 2106.05955
###
### Data concerns three different cell lines, i.e a) L-5178Y murine leukaemia cells, b) V-79 Chinese hamster lung, and c) B-16 mouse melanoma.
###
### The simulations were carried out in tranches.
### The first 50,000 iterations were dismissed as burn-out, and then four tranches of 100,000 iterations were simulated.
### The results from previous tranches were initial data for the subsequent ones.
###
##################################################################################################
##################################################################################################
###
### SETTING:
### We assume that we deal with discrete measures accumulated at points x1, x2, ..., xN
### with masses m1, m2, ..., mN at these points
### In this code we use R to denote vector of points R = [x1, x2, ..., xN]
### and C to denote vector of masses C = [m1, m2, ..., mN]
###
### Estimated parameters:
### mu -- proliferation rate (in the paper denoted by lambda);
### sigma_k -- kernel size;
### sigma_i -- initial colony radius;
### sigma_o -- measurement error;
###
##################################################################################################
##################################################################################################

### SOME PACKAGES:

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
from tqdm import tqdm
from scipy import sparse

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

##################################################################################################
##################################################################################################
###
### The functions F(R,C,s) and invF(R,C,q,s_init) are used to determine the convolution with Laplace distribution to obtain the continuous solution of r(t).
### The test comparison computations we conducted indicate that such a regularisation is not needed in practice.
### The differences between the simulation results are imperceptible and concern distant decimal places.
### The regularisation requires much more computing resources.
### Therefore the main simulations were conducted without this regularisation, however, we provide that code for comparison.
### To run the simulations with the regularisation in function "simulate(rmin,rmax,tmax,dr,dt,sigma,sigma_k,mu)" replace "get_radius(C,R,dr)" with "invF(R,C,q,s_init)".
###
##################################################################################################
##################################################################################################

def F(R,C,s):
    calka = 0
    ee = 0.00001
    for i in range(len(R)):
        if (R[i]<s):
           calka = calka + C[i]*(1-0.5*np.exp((R[i]-s)/ee))
        else:
           calka = calka + 0.5*C[i]*np.exp((s-R[i])/ee)
    return calka

def invF(R,C,q,s_init):
    maxs = np.max(R)
    mins = np.min(R)
    total = np.sum(C)
    tol = 1e-10
    s = s_init
    Fs = F(R,C,s)
    k = 0
    while abs(Fs-q*total)>tol:
        k+=1
        if k==1000:
            break
        if Fs > q*total:
            maxs = s
        else:
            mins = s
        s = (maxs-mins)/2 + mins
        Fs = F(R,C,s)
    return s

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
def Euler(C,R,L,delta_t,dr,sigma2,mu):
    kc = splot(L,C)
    R2 = np.asarray([x**2 for x in R])
    k_1 = delta_t * mu*kc*(4*np.pi*dr*R2-C)
    return C + k_1

### The function calculates the cell colony diameter vector in successive time steps with given the parameters.
def simulate(rmin,rmax,tmax,dr,dt,sigma_i,sigma_k,mu):
    czas= int(np.round(tmax/dt))+1
    colony_mass = np.zeros(czas)
    colony_diameter = np.zeros(czas)
    time = np.zeros(czas)
    R = np.arange(rmin,rmax,dr)
    C = init_function(R,sigma_i,dr)
    colony_diameter[0]= 2*get_radius(C,R,dr)
    #colony_diameter[0]= 2*invF(R,C,0.95,0.5*sigma_i)
    L=funkcja_L(R,sigma_k)
    for i in tqdm(range(1,czas)):
        #for i in (range(1,czas)):
        C = RK(C,R,L,dt,dr,sigma_k,mu)
        #C = Euler(C,R,L,dt,dr,sigma_k,mu)
        colony_diameter[i] = 2*get_radius(C,R,dr)
        #colony_diameter[i]= 2*invF(R,C,0.95,0.5*sigma_i)
        colony_mass[i] = sum(C)
        time[i] = i*dt
    return colony_mass,C,colony_diameter,time

### Logarithm of prior density data set a
def logprior(log_mu,log_sigma_k,log_sigma_o,log_sigma_i):
    out = -(log_mu-np.log(1.38629/2.4))**2/2          #Prior mu
    out -= (log_sigma_k-np.log(0.06))**2/2            #Prior sigma_k
    out -= (log_sigma_o)**2/10                        #Prior sigma_o
    out -= (log_sigma_i-np.log(0.2735966))**2/2	      #Prior sigma_i
    return out

### Logarytm gestosci priori data set b
#def logprior(log_mu,log_sigma2,log_sigma_obs,log_sigma_init):
#    out = -(log_mu-np.log(1.03972/2.4))**2/2            #Prior mu
#    out -= (log_sigma2-np.log(0.06))**2/2               #Prior sigma_k
#    out -= (log_sigma_obs)**2/10                        #Prior sigma_o
#    out -= (log_sigma_init-np.log(0.4030418251))**2/2   #Prior sigma_i
#    return out

### Logarytm gestosci priori data set c
#def logprior(log_mu,log_sigma2,log_sigma_obs,log_sigma_init):
#    out = -(log_mu-np.log(0.92419/2.4))**2/2            #Prior mu
#    out -= (log_sigma2-np.log(0.09))**2/2               #Prior sigma_k
#    out -= (log_sigma_obs)**2/10                        #Prior sigma_o
#    out -= (log_sigma_init-np.log(0.7333333333))**2/2   #Prior sigma_i
#    return out

##################################################################################################
##################################################################################################
###
### Functions "log_likelihood(radius_obs, radius_calc, sigma_o, mu, sigma_k,sigma_i)" and
### "loglik(rmin,rmax,tmax,dr,log_mu,log_sigma_k,log_sigma_o,log_sigma_i)" are to compute the
### probability of receiving the vector of observations given the parameters.
###
##################################################################################################
##################################################################################################

def log_likelihood(radius_obs, radius_calc, sigma_o, mu, sigma_k,sigma_i):
    return -.5/sigma_o**2*np.sum((radius_obs-radius_calc)**2)-(radius_obs.size-1)*np.log(sigma_o)+np.log(mu)+.5*np.log(sigma_k)+.5*np.log(sigma_i)

def loglik(rmin,rmax,tmax,dr,log_mu,log_sigma_k,log_sigma_o,log_sigma_i):
    colony_mass,c,colony_diameter,time = simulate(rmin,rmax,tmax,dr,dt,np.exp(log_sigma_i),np.exp(log_sigma_k),np.exp(log_mu))
    # Table "radius_calc" stores the radius values for the time points for which they are observations.
    radius_calc = np.array([0.5*colony_diameter[i] for i in ind])
    return log_likelihood(np.log(radius_obs),np.log(radius_calc),np.exp(log_sigma_o),np.exp(log_mu),np.exp(log_sigma_k),np.exp(log_sigma_i)),colony_mass,colony_diameter,time


##################################################################################################
##################################################################################################
###
### Metropolis-Hastings algorithm step
###
##################################################################################################
##################################################################################################

def metropolis_step(rmin,rmax,tmax,dr,log_mu,log_sigma_k,log_sigma_o,log_sigma_i,current_loglik,loglik,logprior,scale,niter):
    #  Selection of a candidate for the next state of the Markov chain taking into account the current state theta.
    z= np.random.normal(0,1,4)
    print("theta     ",np.exp(log_mu)," ",np.exp(log_sigma_k)," ",np.exp(log_sigma_o)," ",np.exp(log_sigma_i))
    log_mu_new = log_mu +scale*z[0]
    log_sigma_k_new = log_sigma_k +scale*z[1]
    log_sigma_o_new = log_sigma_o+ scale*z[2]
    log_sigma_i_new = log_sigma_i + scale*z[3]
    print("theta new ",np.exp(log_mu_new)," ",np.exp(log_sigma_k_new)," ",np.exp(log_sigma_o_new)," ",np.exp(log_sigma_i_new))
    u = np.random.random_sample(1)
    if (dr>np.exp(log_sigma_k_new)):
        print("too coarse discretization: dr=",dr, "sigma_k", np.exp(log_sigma_k_new))
    new_loglik,colony_mass,colony_diameter,time= loglik(rmin,rmax,tmax,dr,log_mu_new,log_sigma_k_new,log_sigma_o_new,log_sigma_i_new)
    new_loglik+=logprior(log_mu_new,log_sigma_k_new,log_sigma_o_new,log_sigma_i_new)
    print("\n current_loglik=",current_loglik,"; new_loglik=",new_loglik)
    if new_loglik-current_loglik>np.log(u):
        # Acceptance of a new theta.
        print("Substitution of theta.")
        log_mu = log_mu_new
        log_sigma_o = log_sigma_o_new
        log_sigma_k = log_sigma_k_new
        log_sigma_i = log_sigma_i_new
        current_loglik = new_loglik
        np.savetxt("colony_mass"+str(niter)+".csv",colony_mass, delimiter=",")
        np.savetxt("colony_diameter"+str(niter)+".csv",colony_diameter, delimiter=",")
    return log_mu,log_sigma_k,log_sigma_o,log_sigma_i,current_loglik

##################################################################################################
##################################################################################################
###
####################################### Main program #############################################
###
##################################################################################################
##################################################################################################

### Parameters
M = 100000

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
# Measurement data
time_obs = [8-tau,11.1034482759-tau,13.1724137931-tau,18.0689655172-tau,22-tau,25.1724137931-tau]
radius_obs = np.array([0.2637931034,0.4655172414,0.5948275862,0.9293103448,1.3517241379,1.9155172414])
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
## Measurement data
#time_obs = [19.3939393939-tau,39.3939393939-tau,59.8484848485-tau,80-tau,100.7575757576-tau,120.7575757576-tau,140.9090909091-tau,161.6666666667-tau]
#radius_obs = np.array([0.4030418251,0.4980988593,0.7053231939,0.9657794677,1.2623574144,1.4923954373,1.6254752852,2.0209125475])
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
## Measurement data
#time_obs = [13.9597315436-tau,31.0738255034-tau,41.1409395973-tau,54.0939597315-tau,67.9194630872-tau,74.0268456376-tau]
#radius_obs = np.array([0.7333333333,0.8858333333,0.9341666667,1.0541666667,1.1108333333,1.1741666667])
#mu= 0.3519                   # Proliferation rate
#sigma_k= 0.035               # Kernel size
#sigma_i= 0.7419              # Initial colony radius
#sigma_o= 0.0284              # Measurement error

##################################################################################################
##################################################################################################
###
### Setting up the index table of the time moments for which there are observations.
###
##################################################################################################
##################################################################################################

ind=[]
i=0;
j=0;
time=0
while (time<tmax and j<len(time_obs)):
    if isclose(time, time_obs[j], rel_tol=dt/2):
        ind.append(i)
        j+=1
    i+=1
    time += dt

### Initialization of variables.
log_mu = np.log(mu)
log_sigma_k = np.log(sigma_k)
log_sigma_o = np.log(sigma_o)
log_sigma_i = np.log(sigma_i)
vec_log_mu = []
vec_log_sigma_k = []
vec_log_sigma_o = []
vec_log_sigma_i = []
vec_current_loglik = []

R = np.arange(rmin,rmax,dr)
C = init_function(R,sigma_i,dr)
L = funkcja_L(R,sigma_k)
kc = splot(L,C)
colony_mass,C,colony_diameter,time = simulate(rmin,rmax,tmax,dr,dt,sigma_i,sigma_k,mu)
np.savetxt("colony_mass0.csv",colony_mass, delimiter=",")
np.savetxt("colony_diameter0.csv",colony_diameter, delimiter=",")
current_loglik,colony_mass,colony_diameter,time = loglik(rmin,rmax,tmax,dr,log_mu,log_sigma_k,log_sigma_o,log_sigma_i)
current_loglik += logprior(log_mu,log_sigma_k,log_sigma_o,log_sigma_i)

### Main loop <-- Computes the next step of the Metropolis-Hastings algorithm and saves it to the files.
for zu in range(M):
    vec_log_mu.append(log_mu)
    vec_log_sigma_k.append(log_sigma_k)
    vec_log_sigma_o.append(log_sigma_o)
    vec_log_sigma_i.append(log_sigma_i)
    vec_current_loglik.append(current_loglik)
    
    print("M-H step nr:",zu)
    log_mu,log_sigma_k,log_sigma_o,log_sigma_i,current_loglik = metropolis_step(rmin,rmax,tmax,dr,log_mu,log_sigma_k,log_sigma_o,log_sigma_i,current_loglik,loglik,logprior,scale,zu)
    np.savetxt("log_mu.csv", vec_log_mu, delimiter=",")
    np.savetxt("log_sigma_k.csv", vec_log_sigma_k, delimiter=",")
    np.savetxt("log_sigma_o.csv", vec_log_sigma_o, delimiter=",")
    np.savetxt("log_sigma_i.csv", vec_log_sigma_i, delimiter=",")
    np.savetxt("current_loglik.csv", vec_current_loglik, delimiter=",")


