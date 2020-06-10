# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:13:54 2020

@author: Værksted Matilde
"""

#PR_A1, TPR_C1 = Tpr_afri, Tpr_cau
import matplotlib.pyplot as plt
import numpy as np
def equal_opportunity(TPR_A1, TPR_C1, FPR_A, FPR_C, sigma, T, CLVar, plot = False):
    """
    T: list of thresholds
    TPR_A: List of TPR of African- American
    TPR_C: List of TPR of Caucasian
    sigma: Accepted difference between  TPR_A and TPR_C
    CLVar: Name of equal class variable 
    FPR_A: List of FPR, African-American 
    TPR_A: List of FPR, caucasian
    """

    TPR_A = np.asarray(TPR_A1)
    TPR_C = np.asarray(TPR_C1)
    #Define 2 thresholds
    T_A = np.asarray(T)
    T_C = np.asarray(T)
    
    #remove i's from TPR of both races, and remove coherent thresholds
    for i in range(2):
        
        idxa = TPR_A != i
        idxc = TPR_C != i
        
        TPR_A = TPR_A[idxa]
        TPR_C = TPR_C[idxc]
    
        T_A = T_A[idxa]
        T_C = T_C[idxc]
    
    TtotalC = []
    TtotalA = []
    T_Anew, T_Cnew = [], []
    for t_idx, A in enumerate(TPR_A): 
        
        #Create list with all entries = A and the same length as TPR_C
        A =[A]
        A = len(TPR_C)*A
        
        #Find difference between A and all TPR of C
        Diff = np.asarray(A) - np.asarray(TPR_C)
        
        #Define idx of TPR_C that are identical to A and store them in list 
        idx = abs(Diff) < sigma
        if sum(idx) > 0:    
            t = T_C[idx]
            for i in range(len(t)):
                TtotalC.append(t[i])
                TtotalA.append(T_A[t_idx])
                T_Anew.append(T_A[t_idx])
                T_Cnew.append(T_C[t_idx])
            
    
    #compute (TPR, FPR) and accuracy 
    pairsA, pairsC = [], []
    accA, accC = [], []
    for a,c in zip(TtotalA, TtotalC):
        
        confa = CLVar.conf_models(a,0) #conf mtrx with threshold i, for african-american
        confc = CLVar.conf_models(c,1) #conf mtrx with threshold i for caucasian
        
        pairsA.append(CLVar.FP_TP_rate(confa)) #collect (FPR, TPR) with threshold i, for african-american
        pairsC.append(CLVar.FP_TP_rate(confc))  #collect (FPR, TPR) with threshold i for caucasian
        
        accA.append(CLVar.acc_with_conf(confa)) #collect accuracy with threshold i, for african-american
        accC.append(CLVar.acc_with_conf(confc)) #collect accuracy with threshold i for caucasian
    
    
    #compute highest accuracy 
    N_A = CLVar.Freq[0] #number of 
    N_C = CLVar.Freq[1]
    
    weighted_acc = []
    for a, c in zip(accA, accC):
        weighted_acc.append((a*N_A+c*N_C))
    
        
    #maximum accuracy of pairs
    maxw = np.argmax(weighted_acc)
    max_acc = weighted_acc[maxw]
    maxtA = T_Anew[maxw]
    maxtC = T_Cnew[maxw]
    rateA = pairsA[maxw]
    rateC = pairsC[maxw]
    
    #plot with max accu point
    plt.plot(FPR_C, TPR_C1,'g', label = 'Caucasian')
    plt.plot(FPR_A, TPR_A1,'b', label = 'African-american')
    plt.plot(rateA[0],rateA[1] ,'b*', label = "Equal opportunity")
    plt.plot(rateC[0],rateC[1], 'g*', label = "Equal opportunity")
    plt.plot([rateA[0],rateC[0]],[rateA[1],rateC[1]] ,'r')
    plt.legend()
    plt.show()   
    
    return max_acc, maxtA, maxtC, rateA, rateC

#TPR_A1, TPR_C1, FPR_A, FPR_C, sigma, T, CLVar = Tpr_afri, Tpr_cau, Fpr_afri, Fpr_cau, Sigma, T, Equal_rf
