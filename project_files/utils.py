import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, find_peaks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import IPython
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from os import listdir
from os.path import isfile, join
import pickle

def get_song(val_data, song_names):
    'randomly select a song from a {song_names}'
    row = np.random.randint(low = 1, high = 858)
    
    'get name of one song and get corresponding validtion data'
    name = song_names.iloc[row][0]
    val = val_data.iloc[row:row+1]
    
    'assign and load models'
    dt_classifier = DecisionTreeClassifier(random_state=0)
    kn_classifier = KNeighborsClassifier(n_neighbors=3)
    rf_classifier = RandomForestClassifier(random_state=0)
    
    rf_classifier = pickle.load(open('random_forest.pkl', 'rb'))
    kn_classifier = pickle.load(open('k_neighbours.pkl', 'rb'))
    dt_classifier = pickle.load(open('decision_tree.pkl', 'rb'))
    
    'predic the scale based on the input validation'
    rd_pred = rf_classifier.predict(val)[0]
    kn_pred = kn_classifier.predict(val)[0]
    dt_pred = dt_classifier.predict(val)[0]

    if sum([rd_pred, kn_pred, dt_pred]) >= 2:
        scale = 1
    else:
        scale = 0
    
    'return row in the csv files (list.csv and val_data.csv)'
    'return name of the track in .wav format'
    'return scale (major/minor)'
    return row, name, scale