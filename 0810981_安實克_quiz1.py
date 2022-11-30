#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:45:19 2022

@author: 安實克 0810981
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

dataset = load_iris()

X = dataset.data
y = dataset.target

def dist(a, b):
	d = 0
	for i in range(4):
		d = d + (a[i] - b[i])**2
	return np.sqrt(d)

def k_nearest_pred(data):# data includes 4 parameters, without output
    distances = [dist(data, x) for x in X]

    distances.sort()
    nei = []
    for i in range(len(X) - 1):
        if dist(data, X[i]) in distances[0:3]:
            nei.append(np.append(X[i], y[i]))
    
    nei = np.array(nei)
    return sum(nei[:,4])/3

print(k_nearest_pred(X[55]))