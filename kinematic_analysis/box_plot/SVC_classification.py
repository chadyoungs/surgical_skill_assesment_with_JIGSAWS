#! /usr/bin/env python3

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

DATA_PATH = r"C:\Users\yangcheng\Desktop\box_plot_new"

def load_data(data_path = DATA_PATH):
    csv_path = os.path.join(data_path, "data_for_SVM.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] 

if __name__ == "__main__":
    data = load_data()
    # test the data
    #data.head()
    #data.info()

    # data visulization
    #data.hist(bins=50, figsize=(20,15))
    #plt.show()

    train_set, test_set = train_test_split(data, test_size=0.5, random_state=42)
    #print(test_set)

    train_set_X = train_set.drop("class", axis=1) 
    train_set_y = train_set["class"].copy()

    test_set_X = test_set.drop("class", axis=1) 
    test_set_y = test_set["class"].copy()

    #print(train_set_X)
    #print(train_set_y)
    

    # preparing the data
    scaler = MinMaxScaler()
    scaler.fit(train_set_X)
    scaler.fit(test_set_X)
    train_set_X_scaled = scaler.transform(train_set_X)
    test_set_X_scaled = scaler.transform(test_set_X)

    svm = SVC(C=100)
    svm.fit(train_set_X_scaled, train_set_y)

    svm.score(train_set_X_scaled, train_set_y)
    svm.score(test_set_X_scaled, test_set_y)
    print(svm.score(train_set_X_scaled, train_set_y))
    print(svm.score(test_set_X_scaled, test_set_y))
    
    '''
    rbf_kernel_svm_clf = Pipeline((  ("Scaler", StandardScalar()), 
                        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)) ))
    rbf_kernel_svm_clf.fit(train_set_X, train_set_y)
    rbf_kernel_svm_clf.score(train_set_X, train_set_y)
    print(rbf_kernel_svm_clf.score(train_set_X, train_set_y))
    '''
