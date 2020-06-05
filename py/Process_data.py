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
        
        self.data.join(d)
        return d
    
    def hotK(self, featurename): 
        """
        featurename: Name of the feature for onehotK
        num: False by default. define as True if data is nummerical. 
        (classes needs to be sorted for nummerical data)
        """
        hot =[]
        for i in range(len(featurename)): 
            hot.append(pd.get_dummies(self.data[featurename[i]]))
            
        self.data = self.data.drop(featurename, axis=1)
        self.data = self.data.join(hot)
        
        return 
    
  #  def as_numbers(self,attri):
   #     """
    #    input: Categorical attribute consisting of strings 
     #   function: changes attribute to nummerical data in dataset
      ## """

        #group = self.data[attri].unique()
     #   idx = [type(group[i]) for i in range(len(group))]
      #  if [float in idx]:
       #     group.remove(group[idx==float])
        #dic = {}
        #for i,j in enumerate(group):
        #    dic[j] = i
                
       # self.data.replace(dic, inplace=True)
           
        #return dic
    
    

##define class variabel
twoyears = dataprocess("./data/compas-scores-two-years.csv")

## Compute legth of stay of compas-score prison time
length  = twoyears.days_len("c_jail_in","c_jail_out",'c_len_of_stay')

##Drop some columns
keeplist = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 
       'juv_misd_count', 'juv_other_count', 'priors_count', 'c_days_from_compas',
       'c_charge_degree', 'r_charge_degree', 'vr_charge_degree',
       'decile_score.1', 'score_text','two_year_recid', 'c_len_of_stay']

twoyears.data = twoyears.data[keeplist]

## One hot K 
klist = ['sex', 'age_cat', 'race', 'c_charge_degree', 'r_charge_degree', 'vr_charge_degree', 'score_text']
twoyears.hotK(klist)


#remove nans
twoyears.data = twoyears.data.dropna(axis = 0)

