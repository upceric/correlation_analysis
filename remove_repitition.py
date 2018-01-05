#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:39:42 2017

@author: lww
"""
import numpy as np
a=[[1,2],[2,4],[3,6],[4,8],[5,10],[6,12],[7,14],[8,16]]
b=np.zeros(len(a),dtype=np.int)+1
#print(b)
for i,j in enumerate(a):
    for k in range(i+1,len(a)):
        if len(set(a[i])-(set(a[i])-set(a[k])))>0:
            b[i]=0
            continue
from itertools import compress

c=list(compress(a,b))
print(c)
v_i=[i[0] for i in c]
v_j=[i[1] for i in c]
m_v_i=np.mean(v_i)
m_v_j=np.mean(v_j)
sigma_i_s=np.std(v_i)**2#sigma of v_i array
sigma_j_s=np.std(v_j)**2#sigma of v_j array

#correlation=sum([(i-m_v_i)*(j-m_v_j) for i,j in zip(v_i,v_j)])\
#                /np.sqrt((sigma_i_s*sigma_j_s))
correlation=sum([(i-m_v_i)*(j-m_v_j) for i,j in zip(v_i,v_j)])\
                /(sigma_i_s+sigma_j_s)*2/len(v_i)
print(correlation)
