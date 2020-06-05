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


class equal:
    def __init__(self,data_path, yhatName, ytrueName, AName, N):
        """ data_path: path to csv file
            yhatName: Name of attribute containing yhat
            ytrueName: Name of attribute containing ytrue
            AName: Name of protected attribute
            N: Read describtion of "Groups" below
        """
        data = pd.read_csv(data_path)
        self.data = data
        Y = self.data[[yhatName, ytrueName]].to_numpy(dtype=float)
        self.A = self.data[AName].to_numpy()
        self.Yhat = Y[:,0]
        self.Ytrue = Y[:,1]
        
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

    def ROC(self ,makeplot=True, GetAllOutput=False):
        """
        Allthresholds: list of all thresholds of ROC curve. 
        For A=a , allthresholds[i] is the thresholds used to compute (allfpr[i],alltpr[i])
        alltpr and allfpr: for a given ROC curve they define points in the FP, TP plane 
        allauc: AUC of all ROC curves
        """
        
        allthresholds, allfpr, alltpr, allauc = [], [], [], []
        if makeplot: 
            plt.figure(figsize=(8,6))
        for g in self.Groups:  
            fpr, tpr, thresholds = metrics.roc_curve(g['ytrue'], g['yhat'],pos_label=1)
            fpr = fpr.tolist()
            tpr = tpr.tolist()
            thresholds = thresholds.tolist()
            allfpr = allfpr + fpr
            alltpr = alltpr + tpr
            allthresholds = allthresholds+thresholds
          
            roc_auc  = metrics.auc(fpr, tpr)
            allauc.append(roc_auc)
            if makeplot:  
                plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (g['groupname'], roc_auc))
        
        if makeplot: 
            plt.plot([0, 1], [0, 1], 'k--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc=0, fontsize='small')
            plt.show()
        
        if GetAllOutput:             
            return allthresholds, allfpr, alltpr, allauc, thresholds
        else: 
            return allthresholds, allfpr, alltpr, allauc
    
    def calc_ConfusionMatrix(self,t1, t2, g, p1, OnlyOne = False, positive_label = 1):
        """
        Computestp, fp, tn, fn for all groups by default.
        If OnlyOne is True it only computes for one group
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
                mid = np.random.choice([0,1], p = [1-p1,p1])
                if mid: 
                    if truth: 
                        tp+=1
                    else: fp+=1
                else: 
                    if not truth: 
                        tn+=1
                    else: 
                        fn+=1
        return ConfusionMatrix(tp, fp, tn, fn)
                    
    def conf_(self,t, g, positive_label=1):
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
    
    
    def ROC_(self,T,makeplot=True, GetAllOutput=False):
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
                conf_mtrx = self.conf_(thres, idx)
                RFPR, RTPR = self.FP_TP_rate(conf_mtrx)
                RFPR = [RFPR]
                RTPR = [RTPR]
                fprl = fprl + RFPR
                tprl = tprl + RTPR 
            ALLfpr[R] = fprl
            ALLtpr[R] = tprl
        



        #plt.figure(figsize=(8,6))   
        #for idx,R in enumerate(self.Race):

            #roc_auc  = metrics.auc(fpr, tpr)
            #allauc.append(roc_auc)
            #plt.plot(ALLfpr[R], ALLtpr[R])#, label='%s ROC (area = %0.2f)' % (g['groupname'], roc_auc))
        
        #plt.plot([0, 1], [0, 1], 'k--')
        
       # plt.xlim([0.0, 1.0])
       # plt.ylim([0.0, 1.0])
       ## plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.legend(loc=0, fontsize='small')
        #plt.show()
        return ALLfpr, ALLtpr


    def acc_(self,T,makeplot=True, GetAllOutput=False):
        """
        Allthresholds: list of all thresholds of ROC curve. 
        For A=a , allthresholds[i] is the thresholds used to compute (allfpr[i],alltpr[i])
        alltpr and allfpr: for a given ROC curve they define points in the FP, TP plane 
        allauc: AUC of all ROC curves
        """

        accs = []

        for idx,R in enumerate(self.Race):
            k = []
            for thres in T:
                conf_mtrx = self.conf_(thres, idx)
                tp = conf_mtrx[0]
                fp = conf_mtrx[1]
                tn = conf_mtrx[2]
                fn = conf_mtrx[3]   

                acc = (tn+tp)/(tn+fp+fn+tp)

                k.append(acc)

            accs.append(k)
        return accs


    def THEPOINT(self,t, t1, t2, g, p1, PlotAllt= False, MakePlot = True):
        
            """
            This function always computes all ROC curves. 
            
            
            If PlotAllt=True: Computes FPR, TPR corresponding to threshold=t for all groups 
            and plots it if MakePlot is True
            
            If PlotAllt = False (default) following happens: 
            We wish to choose a point in the (FP, TP) plane, on the ROC curve
            that defines the upper-left edge of the Convex Hull. This is done as follows: 
            Define the wantsed ROC curve as the ROC curve with the smallest AUC. 
            (minROC) and then we compute FPR and TPR for minROC corresponding to t (input)
            For group g, it computes a randomised predictor with t1, t2 and p1. 
            Ig MakePlot = True the results are plotted
            
            """
            allthresholds, allfpr, alltpr, allauc, thresholds = self.ROC(False,True)
            
            #Define number of points in one ROC curve
            M = len(thresholds)
            
            #Rates pr group (ROC curve values)
            pltfpr = (np.asarray(allfpr)).reshape(len(self.Groups),M)
            plttpr= (np.asarray(alltpr)).reshape(len(self.Groups),M)
            
            curveidx= np.argmin(allauc)
                #index of chosen threshold (assuning the same thresholds are used for all ROC curves)
            thresholdidx = thresholds.index(t)
            #Define FPR, TPR according to t, for each group: 
            if PlotAllt: 
                FPRall = pltfpr[:,thresholdidx]
                TPRall = plttpr[:,thresholdidx]
                
                if MakePlot:
                    #Plot with point and coherent threshold
                    for i,gr in enumerate(self.Groups): 
                                   plt.plot(pltfpr[i,:], plttpr[i,:], label='%s ROC (area = %0.2f)' % (gr['groupname'], allauc[i]))
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.plot(FPRall, TPRall, 'go', label = "threshold = %s" %t)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend(loc=0, fontsize='small')
                    plt.show()
                return FPRall, TPRall
                    
                 
            else: 

                #rate with t on lowest ROC 
                FPR = pltfpr[curveidx,thresholdidx]
                TPR = plttpr[curveidx,thresholdidx]
                #Points in randomised threshold 
                conf = self.calc_ConfusionMatrix(t1, t2, g, p1)
                RandomFPR, RandomTPR = self.Point_with_randomised_thresholds(conf)
                
                if MakePlot: 
                    for i,gr in enumerate(self.Groups): 
                                   plt.plot(pltfpr[i,:], plttpr[i,:], label='%s ROC (area = %0.2f)' % (gr['groupname'], allauc[i]))
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.plot(FPR,TPR,'o', label = "Threshold = %s" %t)
                    plt.plot(RFPR, RTPR, 'bo', label = "$t_1$=%s, $t_1$=%s,$p_1$=%s group = %s" %(t1, t2, p1, g))
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend(loc=0, fontsize='small')
                    plt.show()

                
                return FPR, TPR, RandomFPR, RandomTPR

            
DATA = equal("./data/compas-scores-two-years.csv", 'decile_score.1','two_year_recid', 'race',400)
allthresholds, allfpr, alltpr, allauc, thresholds = DATA.ROC(False, True)
#Compute fpr anf tpr on lowest ROC with threshold t: 


#Compute confusion matrix values with the two thresholds: t1, t2, with probability p1 for group g
t1, t2, g, p1 = 2, 7, 1, 0.9
conf = DATA.calc_ConfusionMatrix(t1, t2, g, p1)

t = 5
#p1, p2 = DATA.THEPOINT(t, t1, t2, g, p1, PlotAllt = True)
#Compute fpr and tpr for the predictor for group g, with the above mentioned thresholds and prob
#gfpr, gtpr = DATA.Point_with_randomised_thresholds(conf)

T = np.arange(0,10,0.1)
hej, hej1 = DATA.ROC_(T)

##Kode fra https://www.daniweb.com/programming/computer-science/tutorials/520084/understanding-roc-curves-from-scratch


TP1, FP1 = DATA.ROC_([0,1,2,3,4,5,6,7,8,9,10])

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

plt.show()
