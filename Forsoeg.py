# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:16:17 2020

@author: mat05
"""

import pandas as pd 
#Comments on loading dataset are obtained from https://arxiv.org/pdf/1906.04711.pdf


#dataset created by ProPublica with purpose of studying two-year general recidivism
#The data set compas-scores-two-years-violent.csv is a subset of violent recidivism
data_2years = pd.read_csv("./compas-scores-two-years.csv")
data_2years.head()
data_2years_head = data_2years.columns.values

#full dataset of pretrial defendants that ProPublica obtained from the Broward County Sheriff’s Office
data_pretrial = pd.read_csv("./compas-scores.csv")
data_pretrial.head()
data_pretrial_head = data_pretrial.columns.values



