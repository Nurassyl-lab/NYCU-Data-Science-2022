#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:22:39 2022

@author: raksa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def replace_nan(x):
    if type(x)!=str:
        return  0 if np.isnan(x) else x
    else:
        return x

def problem_1():
    n = np.arange(2011, 2021)
    length = np.array([103, 101, 99, 100, 100, 95, 95, 96, 93, 90])
    
    return {'years':n, 'durations':length}

def problem_2(problem_1):
    df = pd.DataFrame(problem_1)
    return df

def problem_3(problem_2, flag = False):
    if flag == False:
        matplotlib.use('Agg')
    else:
        matplotlib.use('qt5Agg')
    a = pd.Series(problem_2['years'])
    b = pd.Series(problem_2['durations'])
    plt.figure(figsize = (12, 9))
    plt.title('Movie Durations 2011-2020')
    plt.plot(a, b)
    if flag == False:
        plt.close()
    return a,b

def problem_4(file = 'movies.csv'):
    return pd.read_csv(file)

def problem_5(problem_4):
    tmp = problem_4.loc[problem_4.type == 'Movie']
    return tmp.loc[:,['title', 'country', 'genre', 'release_year', 'duration']]

def problem_6(problem_5, flag = False):
    problem_5 = problem_5.applymap(replace_nan)
    if flag == False:
        matplotlib.use('Agg')
    else:
        matplotlib.use('qt5Agg')
        
    a = pd.Series(problem_5['release_year'])
    b = pd.Series(problem_5['duration'])
    c = []
    for i in b:#converting string to int
        c.append(int(''.join(x for x in str(i) if x.isdigit())))
    plt.figure(figsize=(12,8))
    plt.title("Movie Duration by Year of Release")
    plt.scatter(a, c)
    if flag == False:
        plt.close()
    return a,pd.Series(c)

def problem_7(problem_5):
    tmp = []
    index = np.array(problem_5.index)
    problem_5 = problem_5.applymap(replace_nan)
    
    for i in index:
        a = problem_5.loc[i,:]
        if int(''.join(x for x in str(a['duration']) if x.isdigit())) < 60:
            tmp.append(i)
    return problem_5.loc[tmp].loc[tmp[0:20]]
    
def problem_8(problem_5):
    colors = []
    index = np.array(problem_5.index)
    
    for i in index:
        if problem_5.loc[i, 'genre'] == 'Documentaries':
            colors.append('blue')
        elif problem_5.loc[i, 'genre'] == 'Children & Family Movies':
            colors.append('red')
        elif problem_5.loc[i, 'genre'] == 'Stand-Up Comedy':
            colors.append('green')
        else:
            colors.append('black')
    return colors
        
def problem_9(problem_5, problem_8, flag = False):
    problem_5 = problem_5.applymap(replace_nan)
    if flag == False:
        matplotlib.use('Agg')
    else:
        matplotlib.use('qt5Agg')
    a = pd.Series(problem_5['release_year'])
    b = pd.Series(problem_5['duration'])
    c = []
    for i in b:
        c.append(int(''.join(x for x in str(i) if x.isdigit())))
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,8))
    plt.title("Movie Duration by year of release")
    plt.scatter(a, c, color = problem_8)
    plt.ylabel('Duration(min)')
    plt.xlabel('Release year')
    if flag == False:
        plt.close()
    return a, pd.Series(c), problem_8

def problem_10():
    # c = problem_5(problem_4())
    # c = c.applymap(replace_nan)
    
    # a = pd.Series(c['release_year'])
    # b = pd.Series(c['duration'])
    # ind =np.array(a.index)
    # dic = {}
    # for i in ind:
    #     if a[i] in dic:
    #         dic[a[i]] += int(''.join(x for x in str(b[i]) if x.isdigit()))
    #     else:
    #         dic[a[i]] = int(''.join(x for x in str(b[i]) if x.isdigit()))
    # for key in dic:
    #     dic[key] /= len(np.where(np.array(a) == key)[0])
    # plt.figure(figsize=(12,8))
    # x = list(dic.keys())
    # y = list(dic.values())
    # x.sort()
    # y.sort()
    # plt.plot(x, y)
    return 'No'