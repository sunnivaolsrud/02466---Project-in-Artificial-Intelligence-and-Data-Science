# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:32:30 2020

@author: Værksted Matilde
"""
import numpy as np
import seaborn as sns
import pandas as pd
def plot_conf(conf_mtrx):
    """
    input: confusion matrix of type conf(tp, fp, tn, fn)
    """
    conf = np.empty([2,2])
    conf[0,0] = conf_mtrx[0]
    
    conf[0,1] = conf_mtrx[1]
    conf[1,0] = conf_mtrx[3]
    conf[1,1] = conf_mtrx[2]
    
    df = pd.DataFrame(conf, index = ["Predictive (1)", "Predictive (0)"], columns = ["Actual (1)","Actual (0)"])


    sns.heatmap(df/np.sum(df), annot=True, 
            fmt='.2%', cmap='Blues')
    return conf 
    

def rates(conf_mtrx):
    """input: confusion matrix of type conf(tp, fp, tn, fn)
    """
    