### Convergence of EBT method for nonlocal model of cell proliferation with discontinuous interaction kernel I
### P.Gwiazda, B.Miasojedow, J.Skrzeczkowski, Z. Szymanska

import numpy as np
from scipy.integrate import quad
from math import isclose
from math import ceil,inf
from copy import deepcopy
from tqdm import tqdm


### The function returns the vector C with the initial mass of the colony at a given radius.
def init_function(rmin,rmax,R,sigma,dr):
    C = np.arange(rmin,rmax,dr)
    sigma_new  = sigma*1.07
    alpha = 13
    for k in range(len(R)):
        if k*dr<=sigma_new:
            # Funkcja charakterystyczna kuli zmiekczona na brzegu
            C[k] = (4*np.pi*R[k]*R[k]*dr)*(max(0,1-(R[k]/sigma_new)**alpha))
        else:
            C[k] = 0
    return C

### The function returns the density vector (mass divided by volume).
def get_density(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    for k in range(len(R)):
        D[k] = C[k]/(4*np.pi*R[k]*R[k]*dr)
    return D

### The function returns the weighted masses (mass divided by the radius and by the total mass) needed to calculate the auxiliary metric rho.
def get_weighted_mass_rho(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    for k in range(len(R)-1):
        D[k] = C[k]/R[k]
    return D/np.sum(D)

### The function returns the weighted masses (mass divided by the radius) needed to calculate the auxiliary metric rho.
def get_weighted_mass(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    for k in range(len(R)-1):
        D[k] = C[k]/R[k]
    return D

### The function returns an auxiliary matrix for the calculation of the convolution.
def funkcja_L(R,sigma2):
    norm = 3/(16*np.pi*(sigma2**3))
    sigma2pow = sigma2**2
    r0 = R[0]
    dr = R[1]-R[0]
    diags = list()
    offsets = list()
    k = 0
    tmp = np.ones_like(R) *(k*dr)**2 - sigma2pow
    diags.append(tmp)
    offsets.append(k)
    k = 1
    while k <  np.sqrt(sigma2pow)/dr:
        tmp = np.ones_like(R) *(k*dr)**2 - sigma2pow
        diags.append(tmp)
        diags.append(tmp)
        offsets.append(k)
        offsets.append(-k)
        k+=1
    L11 = sparse.spdiags(diags, offsets, len(R), len(R))
    subBlock = np.arange(int( np.sqrt(sigma2pow)/dr-2*r0/dr ) +  1)
    sI, sJ = np.meshgrid(subBlock,subBlock)
    sL = (np.minimum( sI+sJ , np.sqrt(sigma2pow)/dr-2*r0/dr )+2*r0/dr)**2*dr**2 - sigma2pow
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
    while k <  np.sqrt(sigma2pow)/dr + 3:
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
    schemat = 1
    kc = splot(L,C)
    R2 = np.asarray([x**2 for x in R])
    k_1 = delta_t * mu*kc*(4*np.pi*dr*R2-C)
    return C + k_1

from scipy import sparse

### The function calculates the cell colony mass vector in successive time steps with the given mu and sigma2 parameters.
def simulate(rmin,rmax,tmax,dr,dt,sigma_init,sigma2,mu,schemat):
    czas= int(np.round(tmax/dt))+1
    R = np.arange(rmin,rmax,dr)
    C = init_function(rmin,rmax,R,sigma_init,dr)
    L=funkcja_L(R,sigma2)
    #for i in tqdm(range(1,czas)):
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
### It consists of two stages: i) diff function - calculates the difference of two measures, and ii) function rho calculates the proper auxiliary metric rho and flat metric.
### The result vectors of the diff function are the arguments for the functions calculating the rho and flat metrics.
######################################################################################################################

### The function returns the order of convergence. Note metryka_new is the distance for 2 * dr = dt.
def Error(metryka_new,metryka_old):
    return np.log(metryka_new/metryka_old)/np.log(2)

### The function returns the distance between two measures.
def diff(R1,R2,C1,C2):
    i = 0
    j = 0
    koniec_1=0
    koniec_2=0
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
        distance = distance + np.abs(partial_sum)*(R[i+1]-R[i])
    return distance

### The function returns the auxiliary metric rho.
def rho(R_diff,M_diff,Masa1,Masa2):
    wasser = W1(R_diff,M_diff)
    return min(Masa1,Masa2)*wasser+np.abs(Masa1-Masa2)

### The function returns the flat metric.
def flat(R,C):
    left_value = -C[0]
    V=set([(-1,C[0]),(1,-inf)])
    for i in range(1,len(R)):
        d = round(R[i]-R[i-1],13)
        f_left = set([(v-d,p) for (v,p) in V if round(p,13) >= 0])
        f_right = set([(v+d,p) for (v,p) in V if round(p,13) < 0])
        vmin = np.min([v for (v,p) in f_right])
        V = f_left.union(f_right.union(set([(round(vmin-2*d,13),0)])))
        W = list(V)
        W = sorted(W, key=lambda tup: tup[0])
        pom_list = []
        for k in range(len(W)-1):
            pom_list.append((W[k][0],W[k][1],W[k+1][0]))
        left_value += np.sum([round(p*(min(w,-1)-v),13) for (v,p,w) in pom_list if round(v,13)<-1])
        for k in range(len(W)):
            if round(W[k][0],13)<=-1:
                v_min,p_min = W[k]
            else:
                break
        V = set([(v,p) for (v,p) in V if round(v,13)>-1 and round(v,13)<1])
        V=V.union(set([(-1,p_min),(1,-inf)]))
        V = set([(v,round(p+C[i],13)) for (v,p) in V])
        m = C[i]
        left_value = round(left_value-C[i],13)
    W = list(V)
    W = sorted(W, key=lambda tup: tup[0])
    pom_list = []
    for k in range(len(W)-1):
        pom_list.append((W[k][0],W[k][1],W[k+1][0]))
    left_value += np.sum([round(p*(w-v),13) for (v,p,w) in pom_list if round(p,13) > 0 and round(v,13) >= -1])
    return left_value

######################################################
################# Main program #####################
######################################################

# Paramers
Len = 2
tmax = 10
mu= 0.5
sigma2= 0.04
sigma_init= 0.74
#schemat = 1 <-- Euler, schemat == 2 <-- RK
schemat = 1
dr_ref = 8*0.000125 # The smallest discretization.
dt_ref = dr_ref
rmin_ref = dr_ref
rmax_ref = rmin_ref + Len
dr = dr_ref

# RK scheme or Euler scheme.
if (schemat == 1):
    print("\n Schemat = Euler dr_ref = ",dr_ref," tmax = ",tmax)
else:
    if (schemat == 2):
        print("\n Schemat = RK dr_ref = ",dr_ref," tmax = ",tmax)

# Calculation of the solutions for the smallest discretization considered <=== reference solution.
R_ref = np.arange(rmin_ref,rmax_ref,dr_ref)
C_ref = simulate(rmin_ref,rmax_ref,tmax,dr_ref,dt_ref,sigma_init,sigma2,mu,schemat)
#C_ref = init_function(rmin_ref,rmax_ref,R_ref,sigma_init,dr_ref)

D_ref = get_weighted_mass(rmin_ref,rmax_ref,R_ref,C_ref,dr_ref)
M_ref = get_weighted_mass_rho(rmin_ref,rmax_ref,R_ref,C_ref,dr_ref)
R_new = []
D_new = []
flat_old = 0
rho_old = 0
dr_pop=dr_ref

# In the loop, Increase of the discretization 2 times, calculation of the new solution and calculation of the distance to the previous one.
while (4*dr<sigma2):
    
    dr = dr_pop*2
    rmin = dr
    rmax = rmin + Len
    dt = dr
    R_new = np.arange(rmin,rmax,dr)
    C_new = simulate(rmin,rmax,tmax,dr,dt,sigma_init,sigma2,mu,schemat)
    #C_new = init_function(rmin,rmax,R_new,sigma_init,dr)a
    
    D_new = get_weighted_mass(rmin,rmax,R_new,C_new,dr)
    M_new = get_weighted_mass_rho(rmin,rmax,R_new,C_new,dr)
    
    print("\n dr=",dr)
    
    # Calculation of the flat metric.
    R_new_diff,D_new_diff = diff(R_new,R_ref,D_new,D_ref)
    flat_new = flat(R_new_diff,D_new_diff)
    print("Flat = ",flat_new)
    if (flat_old!=0):
        print("Error flat =",Error(flat_new,flat_old))
 
    # Calculation of the rho metric.
    R_new_diff,M_new_diff = diff(R_new,R_ref,M_new,M_ref)
    rho_new = rho(R_new_diff,M_new_diff,np.sum(D_ref),np.sum(D_new))
    print("Rho  = ",rho_new)
    if (rho_old!=0):
        print("Error rho  =",Error(rho_new,rho_old))
    
    dr_pop = dr
    flat_old = flat_new
    rho_old = rho_new
