# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:47:11 2020

@author: mat05
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
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
        
        self.data[name] = d
        return d
    
    
    def as_numbers(self,attri):
        """
        input: Categorical attribute consisting of strings 
        function: changes attribute to nummerical data in dataset
        returns: dictionary with strings and corresponding nummerical value
        """

        group = self.data[attri].unique()
        idx = [type(group[i]) for i in range(len(group))]
        if [float in idx]:
            group.remove(group[idx==float])
        dic = {}
        for i,j in enumerate(group):
            dic[j] = i
                
        self.data.replace(dic, inplace=True)
           
        return dic
    
    



##define class variabel
twoyears = dataprocess("./data/compas-scores-two-years.csv")

## Compute legth of stay of compas-score prison time
length  = twoyears.days_len("c_jail_in","c_jail_out",'c_len_of_stay')

##compute nummerical categorical attributes
d_sex = twoyears.as_numbers('sex')
d_agecat = twoyears.as_numbers('age_cat')
d_race = twoyears.as_numbers('race')
d_c_charge_desc = twoyears.as_numbers('c_charge_desc')

#.....

#Der mangler stadig en funktion med one-out-of-K (tænker egentlig det er bedre en min as_numbers lol)
#Og det ville være fedt med en function der tager n attributes, og fjerner observationer hvor disse har nans 
#hvis vi altså vil det