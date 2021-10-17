##################################################################################################
##################################################################################################
###
###      Order of convergence of the EBT method for a non-local 2D model of cell proliferation
###
##################################################################################################
##################################################################################################
###
### This file contains Python3 code for computing the order of convergence of the EBT method for 
### a non-local 2D model of cell proliferation with a discontinuous interaction kernel.
### 
### The code was prepared for computations in the following paper:
###
### Convergence of the EBT method for a non-local model of cell proliferation with discontinuous interaction kernel
### P.Gwiazda, B.Miasojedow, J.Skrzeczkowski, Z. Szymańska
### arXiv: 2106.05115
###
### The theoretical background concerning the theory of measure aspects is explained in the book (Chapter 4.2):
### Spaces of Measures and their Applications to Structured Population Models
### C. Duell, P. Gwiazda, A. Marciniak-Czochra, J. Skrzeczkowski
### to be published in October 2021 by Cambridge University Press
### https://www.cambridge.org/pl/academic/subjects/mathematics/differential-and-integral-equations-dynamical-systems-and-co/spaces-measures-and-their-applications-structured-population-models?format=HB
###
##################################################################################################
##################################################################################################

import numpy as np
from scipy.integrate import quad
from math import isclose
from math import ceil,inf
from tqdm import tqdm
from scipy import sparse

### The function returns the vector C with the initial mass of the colony at a given radius. 
def init_function(rmin,rmax,R,sigma,dr):
    C = np.arange(rmin,rmax,dr)
    sigma_new  = sigma*1.07
    alpha = 13
    for k in range(len(R)):
        if k*dr<=sigma_new:
            C[k] = (2*np.pi*R[k]*dr)*(max(0,1-(R[k]/sigma_new)**alpha))
        else:
            C[k] = 0
    return C

### The function returns the weighted masses (mass divided by the radius and by the total mass) needed to calculate the auxiliary metric rho.
def get_weighted_mass_rho(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    pierw = np.sqrt(R)
    for k in range(len(R)-1):
        D[k] = C[k]/pierw[k]
    return D/np.sum(D)

### The function returns the weighted masses (mass divided by the radius) needed to calculate the auxiliary metric rho.
def get_weighted_mass(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    pierw = np.sqrt(R)
    for k in range(len(R)-1):
        D[k] = C[k]/pierw[k]
    return D

### The function returns an auxiliary matrix for the calculation of the convolution. 
def funkcja_L(R,sigma2):
    norm = 1/(np.pi*np.pi*(sigma2**2))
    dr = R[1]-R[0]
    N = int((2*(sigma2/dr)+1) * len(R))
    L11_data = np.zeros([N])
    L11_I = np.zeros_like(L11_data)
    L11_J = np.zeros_like(L11_data)
    p = 0
    for i in range(len(R)):
        k = 0
        while k <=  sigma2/dr :
            j = i + k
            if j < len(R) and  (np.abs(R[i]-R[j]) <= sigma2):
                pom = np.maximum( (R[i]**2+R[j]**2-sigma2**2)/(2*R[i]*R[j]),-1 )
                if (pom>1):
                    pom = 1
                L11_data[p] = norm*((np.pi/2)-np.arcsin(pom))
                L11_I[p] = i
                L11_J[p] = j
                p += 1
            j = i - k
            if j >= 0 and  (np.abs(R[i]-R[j]) <= sigma2) and k > 0:
                pom = np.maximum( (R[i]**2+R[j]**2-sigma2**2)/(2*R[i]*R[j]),-1 )
                if (pom>1):
                    pom = 1
                L11_data[p] = norm*((np.pi/2)-np.arcsin(pom))
                L11_I[p] = i
                L11_J[p] = j
                p += 1
            k+=1
    L1 = sparse.coo_matrix((L11_data[:p], (L11_I[:p], L11_J[:p])), shape = (len(R), len(R)))
    return L1

### The function returns the convolution of the kernel and the vector of masses.
def splot(L,C):
    return L.dot(C)

### 4th order Runge-Kutta IV scheme.
def RK(C,R,L,delta_t,dr,sigma2,mu):
    schemat = 2
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

### Euler scheme.
def Euler(C,R,L,delta_t,dr,sigma2,mu):
    kc = splot(L,C)
    k_1 = delta_t * mu*kc*(2*np.pi*dr*R-C)
    return C + k_1

### The function calculates the cell colony mass vector in successive time steps with the given mu and sigma2 parameters.
def simulate(rmin,rmax,tmax,dr,dt,sigma_init,sigma2,mu,schemat):
    czas= int(np.round(tmax/dt))+1
    R = np.arange(rmin,rmax,dr)
    C = init_function(rmin,rmax,R,sigma_init,dr)
    L=funkcja_L(R,sigma2)
    L = sparse.csr_matrix(L)
    if (schemat ==2):
        #print("Runge - Kutta")
        for i in tqdm(range(1,czas)):
            C = RK(C,R,L,dt,dr,sigma2,mu)
    else:
        #print("Euler")
        for i in tqdm(range(1,czas)):
            C = Euler(C,R,L,dt,dr,sigma2,mu)
    return C

######################################################################################################################
### Counting distances between measures. 
### It consists of two stages: i) diff function - calculates the difference of two measures, and ii) function rho calculates the proper auxiliary metric rho.
### The result vectors of the diff function are the arguments for the function calculating the rho metrics.
######################################################################################################################

### The function returns the order of convergence. Note metryka_new is the distance for 2 * dr = dt.
def Error(metryka_new,metryka_old):
    return np.log(metryka_new/metryka_old)/np.log(2)

### The function returns the distance between two measures. 
def diff(R1,R2,C1,C2):
    # R1 - support of the first measure; C1 - masses for the first measure.
    # R2 - support of the second measure; C2 - masses for the second measure.
    i = 0
    j = 0
    koniec_1=0 # Marker becomes 1 if the end of the first vector is reached.
    koniec_2=0 # Marker becomes 1 if the end of the second vector is reached.
    supp = []
    diff = []

    while (koniec_1 == 0 or koniec_2 == 0):
        if (koniec_1 == 0 and koniec_2 == 1):
            supp.append(R1[i])
            diff.append(C1[i])
            i+=1
            if (i == len(R1)):
                koniec_1 = 1
        else:
            if (koniec_1 == 1 and koniec_2 == 0):
                supp.append(R2[j])
                diff.append((-1)*round(C2[j],13))
                j+=1
                if (j == len(R2)):
                    koniec_2 = 1
            else:
                if (isclose(R1[i],R2[j])):
                    supp.append(R1[i])
                    diff.append(round(C1[i]-C2[j],13))
                    i+=1
                    j+=1
                    if (i == len(R1)):
                        koniec_1 = 1
                    if (j == len(R2)):
                        koniec_2 = 1
                else:
                    if (R1[i]<R2[j]):
                        supp.append(R1[i])
                        diff.append(round(C1[i],13))
                        i+=1
                        if (i == len(R1)):
                            koniec_1 = 1
                    else:
                        if (R1[i]>R2[j]):
                            supp.append(R2[j])
                            diff.append((-1)*round(C2[j],13))
                            j+=1
                            if (j == len(R2)):
                                koniec_2 = 1
    return supp,diff

### The function returns the Wasserstain distance.
def W1(R,C):
    partial_sum = 0
    distance = 0
    for i in range(0,len(R)-1):
        partial_sum+=C[i]
        distance = distance + np.abs(partial_sum)*np.sqrt((R[i+1]-R[i]))
    return distance

### The function returns the auxiliary metric rho. 
def rho(R_diff,M_diff,Masa1,Masa2):
    wasser = W1(R_diff,M_diff)
    return min(Masa1,Masa2)*wasser+np.abs(Masa1-Masa2)


######################################################
################# Main program #####################
######################################################

# Parameters
Len = 2 
tmax = 10
mu= 0.5
sigma2= 0.04
sigma_init= 0.74
#schemat = 1 <-- Euler, schemat == 2 <-- RK
schemat = 1
dr_pop = 0.000125 # The smallest discretization.
dt_pop = dr_pop
rmin_pop = dr_pop
rmax_pop = rmin_pop + Len
dr = dr_pop

# I chose the RK scheme or Euler scheme.
if (schemat == 1):
    print("\n Schemat = Euler dr_ref = ",dr_pop," tmax = ",tmax)
else:
    if (schemat == 2):
        print("\n Schemat = RK dr_ref = ",dr_pop," tmax = ",tmax)

# Calculation of the solutions for the smallest discretization considered <=== reference solution.
R_pop = np.arange(rmin_pop,rmax_pop,dr_pop)
C_pop = simulate(rmin_pop,rmax_pop,tmax,dr_pop,dt_pop,sigma_init,sigma2,mu,schemat)
#C_pop = init_function(rmin_pop,rmax_pop,R_pop,sigma_init,dr_pop)

# Weighted by radius.
D_pop = get_weighted_mass(rmin_pop,rmax_pop,R_pop,C_pop,dr_pop)
# Weighted by mass.
M_pop = get_weighted_mass_rho(rmin_pop,rmax_pop,R_pop,C_pop,dr_pop)

R_new = []
D_new = []
flat_old = 0
rho_old = 0

# In the loop, Increase of the discretization 2 times, calculation of the new solution and calculation of the distance to the previous one.
while (4*dr<sigma2):
    dr = dr_pop*2
    rmin = dr
    rmax = rmin + Len
    dt = dr
    R_new = np.arange(rmin,rmax,dr)
    C_new = simulate(rmin,rmax,tmax,dr,dt,sigma_init,sigma2,mu,schemat)
    #C_new = init_function(rmin,rmax,R_new,sigma_init,dr)

    # Weighted by radius.
    D_new = get_weighted_mass(rmin,rmax,R_new,C_new,dr)
    # Weighted by mass.
    M_new = get_weighted_mass_rho(rmin,rmax,R_new,C_new,dr)

    print("\n dr=",dr)

    # Calculation of the differences of 2 measures for the rho metric.
    R_new_diff,M_new_diff = diff(R_new,R_pop,M_new,M_pop)
    rho_new = rho(R_new_diff,M_new_diff,np.sum(D_pop),np.sum(D_new))
    print("Rho  = ",rho_new)
    if (rho_old!=0):
        print("Error rho  =",Error(rho_new,rho_old))
    dr_pop = dr
    R_pop = R_new
    D_pop = D_new
    M_pop = M_new
    rho_old = rho_new
