# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:22:57 2020
@author: xiaoxiaoyang

This code was inspired by <<Towards automatic skill evaluation: Detection and segmentation
                     of robot-assisted surgical motions>>
!!!
!!!Notice this method is not done in real time but in post analysis
!!!
Step 1 (Clustering): BoF, we obtain features for each of 5 train sets
Step 2 (Feature Engineering): Feature Extraction, Though STIP contains temporal information,
                              we still add STIPs of sevral frames pre- or process- in current 
                              frame, for the reason of surgical gesture last for seconds in 
                              surgery;
                              Feature Normalization(L2, for now);
Step 3 (Classification):SVM and kNN and Bayes
Step 4 (Sliding)

This script mainly complete Step 3
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
import joblib

from data_extract_stip import DataExtract
import Global_Var

import matplotlib.pyplot as plt

# 0 for Suturing
TASK_SYMBOL = Global_Var.TASK_SYMBOL_Global
task_list = ["Suturing", "Knot_Tying", "Needle_Passing"]

N_CLUSTERS = Global_Var.CLUSTERS

test = DataExtract()
test.get_category()
test.get_txt_index()  
test.get_score()

train_list = test.train_sum()
test_list = test.test_sum()

# saving paths
time_series_hog = [[] for _ in range(len(train_list))]
time_series_hof = [[] for _ in range(len(train_list))]
#extract_labels = [[] for _ in range(len(train_list))]
for i in range(len(train_list)):
    time_series_hog[i] = joblib.load(r'.\time_series_data\{}\time_series_hog_set{}_{}.pkl'.format(task_list[TASK_SYMBOL], i+1, N_CLUSTERS))
    time_series_hof[i] = joblib.load(r'.\time_series_data\{}\time_series_hof_set{}_{}.pkl'.format(task_list[TASK_SYMBOL], i+1, N_CLUSTERS))
    #extract_labels[i] = joblib.load(r'.\time_series_data\{}\time_series_surgery_name_set{}_{}.pkl'.format(task_list[TASK_SYMBOL], i+1, N_CLUSTERS))
    
train_txt_No_sum = []
# extract the index for every classifier
for i in train_list:
    train_txt_No = []
    for j in i:
        train_txt_No_item = test.metaData_index[j]
        train_txt_No.append(train_txt_No_item)
    train_txt_No_sorted = sorted(train_txt_No)
    train_txt_No_sum.append(train_txt_No_sorted)

test_txt_No_sum = []
# extract the index for every classifier
for i in test_list:
    test_txt_No = []
    for j in i:
        test_txt_No_item = test.metaData_index[j]
        test_txt_No.append(test_txt_No_item)
    test_txt_No_sorted = sorted(test_txt_No)
    test_txt_No_sum.append(test_txt_No_sorted)

# get the train set's data
train_hist_vectors_hog = [[] for _ in range(5)]
train_hist_vectors_hof = [[] for _ in range(5)]
train_hist_vectors = [[] for _ in range(5)]

# get the train set's data
test_hist_vectors_hog = [[] for _ in range(5)]
test_hist_vectors_hof = [[] for _ in range(5)]
test_hist_vectors = [[] for _ in range(5)]

for i in range(len(train_hist_vectors)):
    train_hist_vectors_hog[i] = np.zeros((1, N_CLUSTERS))
    train_hist_vectors_hof[i] = np.zeros((1, N_CLUSTERS))
    for count, (j, k) in enumerate(zip(time_series_hog[i], time_series_hof[i])):
        if count in train_txt_No_sum[i]:
            j = np.reshape(j, (1, N_CLUSTERS))
            k = np.reshape(k, (1, N_CLUSTERS))
            train_hist_vectors_hog[i] = np.vstack((train_hist_vectors_hog[i], j))
            train_hist_vectors_hof[i] = np.vstack((train_hist_vectors_hof[i], k))
    train_hist_vectors_hog[i] = np.delete(train_hist_vectors_hog[i], 0, 0)
    train_hist_vectors_hof[i] = np.delete(train_hist_vectors_hof[i], 0, 0)
    
    train_hist_vectors[i] = np.hstack((train_hist_vectors_hog[i], train_hist_vectors_hof[i]))
    train_hist_vectors[i] = np.reshape(train_hist_vectors[i], (-1, 2*N_CLUSTERS))
    #print(train_hist_vectors[i])
    
for i in range(len(test_hist_vectors)):
    test_hist_vectors_hog[i] = np.zeros(N_CLUSTERS)
    test_hist_vectors_hof[i] = np.zeros(N_CLUSTERS)
    for count, (j, k) in enumerate(zip(time_series_hog[i], time_series_hof[i])):
        if count in test_txt_No_sum[i]:
            j = np.reshape(j, (1, N_CLUSTERS))
            k = np.reshape(k, (1, N_CLUSTERS))
            test_hist_vectors_hog[i] = np.vstack((test_hist_vectors_hog[i], j))
            test_hist_vectors_hof[i] = np.vstack((test_hist_vectors_hof[i], k))
    test_hist_vectors_hog[i] = np.delete(test_hist_vectors_hog[i], 0, 0)
    test_hist_vectors_hof[i] = np.delete(test_hist_vectors_hof[i], 0, 0)
    test_hist_vectors[i] = np.hstack((test_hist_vectors_hog[i], test_hist_vectors_hof[i]))
    test_hist_vectors[i] = np.reshape(test_hist_vectors[i], (-1, 2*N_CLUSTERS))
    #print(test_hist_vectors[i])
    
# get the train set's target   
train_targets = [[] for _ in range(5)]

# get the train set's target        
test_targets = [[] for _ in range(5)]

for i in range(len(train_targets)):
    for j in train_txt_No_sum[i]:
        if j in test.metaData_index.values():
            name_no_property = list(test.metaData_index.keys())[list(test.metaData_index.values()).index(j)]
            label = test.metaData_score[name_no_property][1]
            train_targets[i].append(label)

for i in range(len(test_targets)):
    for j in test_txt_No_sum[i]:
        if j in test.metaData_index.values():
            name_no_property = list(test.metaData_index.keys())[list(test.metaData_index.values()).index(j)]
            label = test.metaData_score[name_no_property][1]
            test_targets[i].append(label)

print("Start to classify the histogram")

# SVM method
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        sum_score = 0
        for i in range(len(train_hist_vectors)):
            svm = SVC(gamma=gamma, C=C)
            svm.fit(train_hist_vectors[i], train_targets[i])
            score = svm.score(test_hist_vectors[i], test_targets[i])
            sum_score += score
        print("gamma: {} and C: {}".format(gamma, C))
        print("average score for current parameters: {:.4f}".format(sum_score/len(train_hist_vectors)))
        if sum_score > best_score:
            best_score = sum_score
            best_parameters = {'C':C, 'gamma':gamma}
            
print("best score: {:.4f}".format(best_score))
print("best parameters: {}".format(best_parameters))
print("Test set average score for 5 sets: {:.4f}".format(best_score/len(train_hist_vectors))) 

