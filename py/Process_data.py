# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:47:11 2020

@author: mat05
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
date_format = "%Y-%m-%d"
#from POST import *

class dataprocess:
    
    def __init__(self,data_path):
        data = pd.read_csv(data_path)
        self.data = data
        
    def days_len(self,aval,bval,name):
        """ 
        input: 
            aval = start date
            bval = end date
            name = column name of new columns
        
        return: 
            Array with days between aval and bval, or nans if aval/bval = nan
        """
        d = []
        try:
            for i in range(len(self.data[aval])): 
                if type(self.data[aval][i])!=float:
                    a = (datetime.strptime(self.data[aval][i], date_format))
                    b = (datetime.strptime(self.data[bval][i], date_format))
                    d.append((b-a).days)
                else: 
                    d.append(np.nan)
                    
        except ValueError:
            for i in range(len(self.data[aval])): 
                if type(self.data[aval][i])!=float:
                    a = (datetime.strptime(self.data[aval][i].split()[0], date_format))
                    b = (datetime.strptime(self.data[bval][i].split()[0], date_format))
                    d.append((b-a).days)
                else: 
                    d.append(np.nan)
        
        self.data[name]=d
        return d
    
    def hotK(self, featurename, remove): 
        """
        featurename: Name of the feature for onehotK
        num: False by default. define as True if data is nummerical. 
        (classes needs to be sorted for nummerical data)
        remove: features to be removed after performing one-hot coding
        """
        hot =[]
        for i in range(len(featurename)): 
            hot.append(pd.get_dummies(self.data[featurename[i]]))
        
        self.data = self.data.drop(remove, axis=1)
        
        for i in range(len(hot)):
            self.data = pd.concat([self.data,hot[i]], axis=1, sort=False)
        
        return
    
    def newlabels(self):
        
        self.data["decile_score.1"] = self.data["decile_score.1"] >5
        return  
    

    
    

##define class variabel
twoyears = dataprocess("./data/compas-scores-two-years.csv")

## Compute legth of stay of compas-score prison time
length  = twoyears.days_len("c_jail_in","c_jail_out","c_len_of_stay")

##Drop some columns
keeplist = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 
       'juv_misd_count', 'juv_other_count', 'priors_count', 'c_days_from_compas',
       'c_charge_degree', 'r_charge_degree', #'vr_charge_degree',
       'decile_score.1', 'score_text','two_year_recid', 'c_len_of_stay']

twoyears.data = twoyears.data[keeplist]

## One hot K. One hot k encoded features are removed except "race". 
klist = ['sex', 'age_cat', 'race', 'c_charge_degree', 'r_charge_degree',  'score_text'] #'vr_charge_degree'
remove = ['sex', 'age_cat', 'c_charge_degree', 'r_charge_degree',  'score_text']#'vr_charge_degree'
twoyears.hotK(klist, remove)


#remove nans
twoyears.data = twoyears.data.dropna(axis = 0)



#%% Prepair data
#Prepair data for POST step on compas data 
A = twoyears.data["race"]
ytrue = twoyears.data['two_year_recid']
yhat = twoyears.data['decile_score.1']

#Prepair data for models
#Redefine labels (1: 6-10, 0: 1-5)
twoyears.newlabels()

#twoyears.data now has all the attributes needed to run model
twoyears.data = twoyears.data.drop(['race', 'two_year_recid'], axis = 1)


    