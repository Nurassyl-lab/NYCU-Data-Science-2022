#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:20:10 2022

@author: raksa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
def MSE(act, pred):
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   return mean_diff

def problem1():
    df = pd.read_csv('CarPrice_Assignment.csv')
    print('Column labels:')
    for col in df.columns:
        print(col, end = ', ')
    print('')
    print('First twenty rows')
    print(df.loc[0:19].values)
    x = pd.Series(df.loc[:, 'horsepower'])
    y = pd.Series(df.loc[:, 'price'])
    plt.figure()
    plt.scatter(x, y)
    return

def problem2():
    bo = -3000
    b1 = np.arange(0,251)#251m so that 250 is also in a set
    df = pd.read_csv('CarPrice_Assignment.csv')
    x = pd.Series(df.loc[:, 'horsepower'])
    y = np.array(pd.Series(df.loc[:, 'price']))
    y = np.array(y)
    mse_list = []
    for beta in b1:
        tmp = []
        for i in x:
            tmp.append( bo+ beta*i)
        y_pred = np.mean(tmp)
        mse_list.append(MSE(y, y_pred))
    plt.figure()
    plt.plot(b1, mse_list)
    return

def problem3():
    df = pd.read_csv('CarPrice_Assignment.csv')
    x = np.array(pd.Series(df.loc[:, 'horsepower']))
    y = np.array(pd.Series(df.loc[:, 'price']))
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(MSE(y_test, pred))
    plt.figure()
    plt.scatter(X_test, y_test)
    plt.plot(X_test, pred, color = 'r')
    return

def problem4():
    df = pd.read_csv('CarPrice_Assignment.csv')
    
    
    x = df.loc[:, ['horsepower', 'peakrpm', 'citympg', 'highwaympg']]
    y = df.loc[:, 'price']
    
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)
    model = LinearRegression()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(mean_squared_error(y_test, pred))
    return 