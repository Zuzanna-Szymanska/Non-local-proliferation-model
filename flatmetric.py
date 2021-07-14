#####################################################################
#####################################################################
### DISTANCES FOR MEASURES ##########################################
#####################################################################
#####################################################################

### This file contains Python3 code for computing Wasserstein and flat metric

### The code was prepared for computations in the following papers:

### Convergence of EBT method for nonlocal model of cell proliferation with discontinuous interaction kernel
### P.Gwiazda, B.Miasojedow, J.Skrzeczkowski, Z. Szymanska
### arXiv: 2106.05115

### Bayesian inference of a non-local proliferation model
### Z. Szymanska, J.Skrzeczkowski, B.Miasojedow, P.Gwiazda
### arXiv: 2106.05955

### The theoretical background of these algorithms is explained in the book (Chapter 4.2):

### Spaces of Measures and their Applications to Structured Population Models
### C. Duell, P. Gwiazda, A. Marciniak-Czochra, J. Skrzeczkowski
### to be published in October 2021 by Cambridge University Press
### https://www.cambridge.org/pl/academic/subjects/mathematics/differential-and-integral-equations-dynamical-systems-and-co/spaces-measures-and-their-applications-structured-population-models?format=HB

### PLEASE CITE IF YOU USE THIS CODE.

#####################################################################
#####################################################################

### SETTING:
### We assume that we deal with discrete measures accumulated at points x1, x2, ..., xN 
### with masses m1, m2, ..., mN at these points
### Usually in this code we use R (or R1 or R2) to denote vector of points R = [x1, x2, ..., xN]
### and C (or C1 or C2) to denote vector of masses C = [m1, m2, ..., mN]

### SOME PACKAGES:

import numpy as np
from math import isclose
from math import inf

#####################################################################
### AUXILLARY DIFFERENCE FUNCTION ###################################
#####################################################################

### This function writes difference of two discrete measures as one measures
### Measures do not need to have the same support
### Arguments:
### R1, R2 - vectors of positions where the mass is accumulated
### C1, C2 - vectors containing values of masses 


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

#####################################################################
### WASSERSTEIN DISTANCE ############################################
#####################################################################

### This function computes Wasserstein distance of two measures
### Measures do not need to have the same support!
### Measures are represented with:
### R1, R2 - vectors of positions where the mass is accumulated
### C1, C2 - vectors containing values of masses
### IMPORTANT: The algorithm assumes that both masses are the same, i.e. sum(C1) = sum(C2)!!!
### (otherwise, the Wasserstein distance is infinity)

def W1(R1,R2,C1,C2):
    R, C = diff(R1,R2,C1,C2)
    partial_sum = 0
    distance = 0
    for i in range(0,len(R)-1):
        partial_sum+=C[i]
        distance = distance + np.abs(partial_sum)*(R[i+1]-R[i])
    return distance


#####################################################################
### FLAT METRIC #####################################################
#####################################################################

### This function computes flat metric of two measures
### Measures do not need to have the same support!
### Measures are represented with:
### R1, R2 - vectors of positions where the mass is accumulated
### C1, C2 - vectors containing values of masses

### Example:
### Suppose we compute distance between two measures
### -> u concentrated at points 0, 1, 2 with massess 0.3, 0.7, 1.5
### -> v concentrated at points 1, 3 with massess 2.1, 0.6
### We call flat([0,1,2], [1,3], [0.3,0.7,1.5], [2.1, 0.6])

def flat(R1,R2,C1,C2):
    R, C = diff(R1,R2,C1,C2)    
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
