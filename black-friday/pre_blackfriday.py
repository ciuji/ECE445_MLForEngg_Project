#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:19:09 2018

@author: ciuji
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from sklearn import metrics

df_allData=pd.read_csv('BlackFriday.csv')
i=0
def oneUserData(User):
    global i
    if(i==0):
        print (User.indices)
        #print(User.loc[0]['User_ID'])
        a=User.loc[0]['User_ID']
        mean=User.Purchase.mean()
        newSerie=pd.Series({'User_ID':User.loc[0]['User_ID'],'Mean':mean})
        #User['mean_purchase']=User.Purchase.mean()
        print (newSerie)
        #print (User)
    i+=1
groupByUserData=df_allData.groupby(['User_ID'])
#groupByUserData.apply(lambda x:oneUserData(x))
#kmeans = KMeans(n_clusters=2, random_state=0).fit(df_allData)

meanData=groupByUserData.mean()
modeData=groupByUserData.agg(lambda x: stats.mode(x)[0][0])

mean_mode_data={'Gender':modeData['Gender'],'Age':modeData['Age'],'City_Category':modeData['City_Category'],'Marital_Status':modeData['Marital_Status'],'Product_CateGory_1':modeData['Product_Category_1'],'Purchase':meanData['Purchase']}
mean_mode_data=pd.DataFrame(mean_mode_data)


mean_mode_data['City_A']=pd.get_dummies(mean_mode_data['City_Category'])['A']
mean_mode_data['City_B']=pd.get_dummies(mean_mode_data['City_Category'])['B']
mean_mode_data['City_C']=pd.get_dummies(mean_mode_data['City_Category'])['C']
mean_mode_data['Gender_M']=pd.get_dummies(mean_mode_data['Gender'])['M']
inputData=mean_mode_data.drop(['Gender','Age','City_Category'],axis=1)
X=inputData
kmeans=KMeans(n_clusters=4,random_state=0).fit(X)
y=kmeans.labels_
metrics.calinski_harabaz_score(X,y)
'''
aggCluster=AgglomerativeClustering(n_clusters=4,).fit(X)
y_agg=aggCluster.labels_
metrics.calinski_harabaz_score(X,y_agg)
'''