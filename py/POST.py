# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:05:59 2020

@author: Bruger
"""

#from Process_data import *
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib as plt
import matplotlib.pyplot as plt
import collections
ConfusionMatrix = collections.namedtuple('conf', ['tp','fp','tn','fn']) 
from Process_data import A, ytrue, yhat


class equal:
    def __init__(self,A, yhat, ytrue, N=400):
        """ data_path: path to csv file
            yhatName: Name of attribute containing yhat
            ytrueName: Name of attribute containing ytrue
            AName: Name of protected attribute
            N: Read describtion of "Groups" below
        """
        
        #self.data = data
        #Y = self.data[[yhatName, ytrueName]].to_numpy(dtype=float)
        self.A = A.to_numpy()
        self.Yhat = yhat
        self.Ytrue = ytrue
        
        #Groups: List with dictionaries. Each group of A, where number of observations > N has a dictionary
        #Each dictionary has 3 keys: ytrue, yhat and A
        Groups = []
        Race, Freq = np.unique(self.A,return_counts=True)
        self.Race = Race[Freq>N]
        self.Freq = Freq[Freq>N]
        #self.Race = Race
        for i in range(len(self.Race)):
            Groups.append({})
            Groups[i]['groupname'] = self.Race[i] 
            Groups[i]['ytrue'] = self.Ytrue[self.A == self.Race[i]]
            Groups[i]['yhat'] = self.Yhat[self.A == self.Race[i]]
            
        self.Groups = Groups

    
    def calc_ConfusionMatrix(self,t1, t2, g, p2, OnlyOne = False, positive_label = 1):
        """
        Computes confusionsmatrix for randomised predictor
        
        Computestp, fp, tn, fn for all groups by default.
        If OnlyOne is True it only computes for one group
        
        p2: probability of t2, which is the distance from t1 to the "correct point"
        "t1 and t2": thresholds
        
        """
        tp=fp=tn=fn=0
        bool_actuals = [act==positive_label for act in self.Groups[g]['ytrue']]
        for truth, score in zip(bool_actuals,self.Groups[g]['yhat']):  
            if score > t2:
                if truth:                              # actually positive 
                    tp += 1
                else:                                  # actually negative              
                    fp += 1
                
            elif score < t1:
                if not truth:                          # actually negative 
                    tn += 1                          
                else:                                  # actually positive 
                    fn += 1
                
            elif score>=t1 and score<=t2: 
                mid = np.random.choice([0,1], p = [1-p2,p2])
                if mid: 
                    if truth: 
                        tp+=1
                    else: fp+=1
                else: 
                    if truth: 
                        fn+=1
                    else: 
                        tn+=1
        return ConfusionMatrix(tp, fp, tn, fn)
    
    
                    
    def conf_(self,t, g, positive_label=1):
        bool_actuals = [act==positive_label for act in self.Groups[g]['ytrue']]
        tp=fp=tn=fn=0
        for truth, score in zip(bool_actuals,self.Groups[g]['yhat']):
            if score < t:
                if truth:                              # actually positive 
                    tp += 1
                else:                                  # actually negative              
                    fp += 1
                
            else:
                if not truth:                          # actually negative 
                    tn += 1                          
                else:                                  # actually positive 
                    fn += 1
    
        return ConfusionMatrix(tp, fp, tn, fn)

    def conf_models(self,t, g, positive_label=1):
        bool_actuals = [act==positive_label for act in self.Groups[g]['ytrue']]
        tp=fp=tn=fn=0
        for truth, score in zip(bool_actuals,self.Groups[g]['yhat']):
            if score > t:
                if truth:                              # actually positive 
                    tp += 1
                else:                                  # actually negative              
                    fp += 1
                
            else:
                if not truth:                          # actually negative 
                    tn += 1                          
                else:                                  # actually positive 
                    fn += 1
    
        return ConfusionMatrix(tp, fp, tn, fn)


    
    def FP_TP_rate(self,conf_mtrx):
        RFPR = conf_mtrx.fp / (conf_mtrx.fp + conf_mtrx.tn) if (conf_mtrx.fp + conf_mtrx.tn)!=0 else 0
        RTPR = conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn)!=0 else 0
        return RFPR, RTPR
    
    
    def ROC_(self,T, models, makeplot=True, GetAllOutput=False):
        """
        Allthresholds: list of all thresholds of ROC curve. 
        For A=a , allthresholds[i] is the thresholds used to compute (allfpr[i],alltpr[i])
        alltpr and allfpr: for a given ROC curve they define points in the FP, TP plane 
        allauc: AUC of all ROC curves
        """
        ALLfpr, ALLtpr = {}, {}
        
        
        for idx,R in enumerate(self.Race): 
            fprl, tprl = [], []
            for thres in T: 
                if models:    
                    conf_mtrx = self.conf_models(thres, idx)
                    
                else: 
                    conf_mtrx = self.conf_(thres, idx)
                RFPR, RTPR = self.FP_TP_rate(conf_mtrx)
                RFPR = [RFPR]
                RTPR = [RTPR]
                fprl = fprl + RFPR
                tprl = tprl + RTPR 
            ALLfpr[R] = fprl
            ALLtpr[R] = tprl
        
        return ALLfpr, ALLtpr


    def acc_(self,T,makeplot=True, GetAllOutput=False):
        """
        T: list of thresholds 
        Computes accuracies given lidt of thresholds
        accs: accuracies given list of thresholds
        """

        accs = []

        for idx,R in enumerate(self.Race, models):
            k = []
            
            
            if models: 
                for thres in T:
                    conf_mtrx = self.conf_models(thres, idx)
                    tp = conf_mtrx[0]
                    fp = conf_mtrx[1]
                    tn = conf_mtrx[2]
                    fn = conf_mtrx[3]   
    
                    acc = (tn+tp)/(tn+fp+fn+tp)
    
                    k.append(acc)
                accs.append(k)
                 
            else:
                for thres in T:
                    conf_mtrx = self.conf_models(thres, idx)
                    tp = conf_mtrx[0]
                    fp = conf_mtrx[1]
                    tn = conf_mtrx[2]
                    fn = conf_mtrx[3]   
    
                    acc = (tn+tp)/(tn+fp+fn+tp)
    
                    k.append(acc)
                accs.append(k)
                

            
        return accs

#pandas.core.series.Series

#Name: decile_score.1, Length: 6907, dtype: int64
       
DATA = equal(A, yhat, ytrue)
#allthresholds, allfpr, alltpr, allauc, thresholds = DATA.ROC(False, True)
#Compute fpr anf tpr on lowest ROC with threshold t: 


#Compute confusion matrix values with the two thresholds: t1, t2, with probability p1 for group g
t1, t2, g, p1 = 2, 7, 1, 0.9
conf = DATA.calc_ConfusionMatrix(t1, t2, g, p1)

t = 5
#p1, p2 = DATA.THEPOINT(t, t1, t2, g, p1, PlotAllt = True)
#Compute fpr and tpr for the predictor for group g, with the above mentioned thresholds and prob
#gfpr, gtpr = DATA.Point_with_randomised_thresholds(conf)


##Kode fra https://www.daniweb.com/programming/computer-science/tutorials/520084/understanding-roc-curves-from-scratch


TP1, FP1 = DATA.ROC_([0,1,2,3,4,5,6,7,8,9,10], models = False)

Ax = TP1['African-American']
Ay = FP1['African-American']

Cx = TP1['Caucasian']
Cy = FP1['Caucasian']

accs = DATA.acc_(np.arange(0,11))

max_accs = [np.argmax(accs[0]), np.argmax(accs[1])]


# Test fra stack overflow, vi skal have skrevet vores egen, hvis det er worth

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def percentile(p1,p2,p3):
    l1 = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    l2 = np.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)

    return l1/l2

l1 = line([Cx[4],Cx[4]],[Cx[3],Cy[3]])
l2 = line([0,0],[Ax[5],Ay[5]])

inter = intersection(l1,l2)
print(inter)

perc1 = percentile([Cx[4],Cx[4]],[Cx[3],Cy[3]],[inter[0],inter[1]])
perc2 = percentile([0,0],[Ax[5],Ay[5]],[inter[0],inter[1]])

print("1. interpolation:", perc1)
print("2. interpolation:", perc2)

"""
print(Ax[5],Ay[5])
print(Cx[0],Cy[0])
print(Cx[3],Cy[3])
print(Cx[4],Cy[4])
"""

plt.plot(Ax,Ay, "o")
plt.plot(Cx,Cy, "o")

#plt.plot([Cx[9],Cx[3]],[Cy[9],Cy[3]])
#plt.plot([Cx[4],Cx[3]],[Cy[4],Cy[3]])

#plt.plot(Cx[5],Cy[5], "o")
plt.plot(Ax[5],Ay[5], "o")
plt.plot(inter[0],inter[1],"o")

c1 = DATA.calc_ConfusionMatrix(4,3, 1, perc1)
c2 = DATA.calc_ConfusionMatrix(10,5, 0, perc2)

DATA.FP_TP_rate(c1)
DATA.FP_TP_rate(c2)
plt.show()
