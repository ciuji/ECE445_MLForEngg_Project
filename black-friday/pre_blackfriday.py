#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:19:09 2018

@author: ciuji
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df_allData=pd.read_csv('BlackFriday.csv')

groupByUserData=df_allData.groupby(['User_ID'])
#kmeans = KMeans(n_clusters=2, random_state=0).fit(df_allData)
