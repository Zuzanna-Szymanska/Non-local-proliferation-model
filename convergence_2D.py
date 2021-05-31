### Convergence of EBT method for nonlocal model of cell proliferation with discontinuous interaction kernel II
### P.Gwiazda, B.Miasojedow, J.Skrzeczkowski, Z. Szymanska
import numpy as np
from scipy.integrate import quad
from math import isclose
from math import ceil,inf
from tqdm import tqdm
from scipy import sparse


### Masa poczatkowa kolonii komorkowej.
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

### Funkcja zwraca zwagowane masy (masa podzielona przez promień i przez calkowita mase) potrzebne do liczenia metryki rho.
def get_weighted_mass_rho(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    pierw = np.sqrt(R)
    for k in range(len(R)-1):
        D[k] = C[k]/pierw[k]
    return D/np.sum(D)

### Funkcja zwraca zwagowane masy (masa podzielona przez promień) potrzebne do liczenia metryki flat i metryki rho
def get_weighted_mass(rmin,rmax,R,C,dr):
    D = np.arange(rmin,rmax,dr)
    pierw = np.sqrt(R)
    for k in range(len(R)-1):
        D[k] = C[k]/pierw[k]
    return D

### Liczenie macierzy pomocniczej do splotu (wzor A.18)
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

### Liczenie splotu
def splot(L,C):
    return L.dot(C)

### Runge-Kutta IV rzedu
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

### Euler
def Euler(C,R,L,delta_t,dr,sigma2,mu):
    kc = splot(L,C)
    k_1 = delta_t * mu*kc*(2*np.pi*dr*R-C)
    return C + k_1

### Funkcja zwraca wektory gestosci i masy kolonii komorkowej w kolejnych krokach czasowych przy zadanych parametrach mu i sigma2
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
### Liczenie odległości między miarami. 
### Składa się z dwóch etapów: funkcji diff - liczenie różnicy dwóch miar i liczenia właściwej metryki - flat lub rho.
### Danymi dla funkcji liczących odpowiednio metryki flat i rho są wektory wynikowe funkcji diff.
######################################################################################################################

### Funkcja zwraca rząd zbiezności. Uwaga metryka_new to odległośc dla 2*dr=dt
def Error(metryka_new,metryka_old):
    return np.log(metryka_new/metryka_old)/np.log(2)

### Funkcja liczy odległośćmiędzy dwoma miarami
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

### Liczenie odległości Wasserstaina
def W1(R,C):
    partial_sum = 0
    distance = 0
    for i in range(0,len(R)-1):
        partial_sum+=C[i]
        distance = distance + np.abs(partial_sum)*np.sqrt((R[i+1]-R[i]))
    return distance

### Liczenie metryki rho 
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
dr_pop = 0.000125
dt_pop = dr_pop
rmin_pop = dr_pop
rmax_pop = rmin_pop + Len
dr = dr_pop

if (schemat == 1):
    print("\n Schemat = Euler dr_ref = ",dr_pop," tmax = ",tmax)
else:
    if (schemat == 2):
        print("\n Schemat = RK dr_ref = ",dr_pop," tmax = ",tmax)

#Liczę rozwiązania dla najdrobniejszej rozważanej dyskretyzacji <=== rozwiazanie referencyjne
R_pop = np.arange(rmin_pop,rmax_pop,dr_pop)
C_pop = simulate(rmin_pop,rmax_pop,tmax,dr_pop,dt_pop,sigma_init,sigma2,mu,schemat)
#C_pop = init_function(rmin_pop,rmax_pop,R_pop,sigma_init,dr_pop)

#Zwagowane przez promien
D_pop = get_weighted_mass(rmin_pop,rmax_pop,R_pop,C_pop,dr_pop)

#Zwagowane przez mase
M_pop = get_weighted_mass_rho(rmin_pop,rmax_pop,R_pop,C_pop,dr_pop)

R_new = []
D_new = []
flat_old = 0
rho_old = 0

#W pętli zwiększam dyskretyzację 2 razy, liczę nowe rozwiązania i liczę odległość od poprzedniego
while (4*dr<sigma2):
    dr = dr_pop*2
    rmin = dr
    rmax = rmin + Len
    dt = dr
    R_new = np.arange(rmin,rmax,dr)
    C_new = simulate(rmin,rmax,tmax,dr,dt,sigma_init,sigma2,mu,schemat)
    #C_new = init_function(rmin,rmax,R_new,sigma_init,dr)

    #Licze funkcje odpowiednio zwagowane
    D_new = get_weighted_mass(rmin,rmax,R_new,C_new,dr)
    M_new = get_weighted_mass_rho(rmin,rmax,R_new,C_new,dr)

    print("\n dr=",dr)

    #Liczę różnice 2 miar dla metryki rho
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
