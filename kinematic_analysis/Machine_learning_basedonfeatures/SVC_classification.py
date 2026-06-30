#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:46:39 2020
@author: xiaoxiaoyang
"""
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from data_extract import DataExtract 

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import TASKS, get_task_index

script_dir = Path(__file__).resolve().parent

def load_data():
    task_lists = list(TASKS)
    task_symbol = get_task_index()
    
    feature = np.load(str(script_dir / "{}_feature.npy".format(task_lists[task_symbol])))
    label = np.load(str(script_dir / "{}_grs.npy".format(task_lists[task_symbol])))
    
    # selct vel and smooth features
    feature_re=np.reshape(feature,(feature.shape[0],feature.shape[1]*feature.shape[2]))
    
    # t
    temp_00 = feature_re[:, 0:1]
    temp_01 = feature_re[:, 10:11]
    temp_02 = feature_re[:, 20:21]
    temp_03 = feature_re[:, 30:31]
    
    # v
    temp_10 = feature_re[:, 2:4]
    temp_11 = feature_re[:, 12:14]
    temp_12 = feature_re[:, 22:24]
    temp_13 = feature_re[:, 32:34]
    
    # s
    temp_20 = feature_re[:, 6:8]
    temp_21 = feature_re[:, 16:18]
    temp_22 = feature_re[:, 26:28]
    temp_23 = feature_re[:, 36:38]
    
    # curvity
    temp_30 = feature_re[:, 4:6]
    temp_31 = feature_re[:, 14:16]
    temp_32 = feature_re[:, 24:26]
    temp_33 = feature_re[:, 34:36]
    
    # ta
    temp_40 = feature_re[:, 8:10]
    temp_41 = feature_re[:, 18:20]
    temp_42 = feature_re[:, 28:30]
    temp_43 = feature_re[:, 38:]
    
    # d
    temp_50 = feature_re[:, 1:2]
    temp_51 = feature_re[:, 11:12]
    temp_52 = feature_re[:, 21:22]
    temp_53 = feature_re[:, 31:32]
    
    
    feature_select_SU = np.hstack((temp_00, temp_01, temp_02, temp_03,
                                   temp_10, temp_11, temp_12, temp_13, 
                                   temp_20, temp_21, temp_22, temp_23,
                                   temp_40, temp_41, temp_42, temp_43))
    feature_select_KT = np.hstack((temp_00, temp_01, temp_02, temp_03,
                                   temp_10, temp_11, temp_12, temp_13, 
                                   temp_20, temp_21, temp_22, temp_23))
    feature_select_NP = np.hstack((temp_10, temp_11, temp_12, temp_13, 
                                   temp_30, temp_31, temp_32, temp_33,
                                   temp_40, temp_41, temp_42, temp_43))
    
    return feature_select_SU, feature_select_KT, feature_select_NP, feature_re, label

if __name__ == "__main__":
    # ST 28
    # KT 20
    # NP 24
    FEATURES = 28
    
    data_su, data_kt, data_np, data, label = load_data()
    
    # st 0 kt 1 np 2
    SELECT_SYMBOL = get_task_index()
    if SELECT_SYMBOL == 0:
        data_f = data_su
    elif SELECT_SYMBOL == 1:
        data_f = data_kt
    elif SELECT_SYMBOL == 2:
        data_f = data_np
        
    test = DataExtract()
    test.get_category()
    test.get_txt_index()  
    test.get_score()

    train_list = test.train_sum()
    test_list = test.test_sum()
    
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
    train_hist_vectors = [[] for _ in range(5)]

    # get the train set's data
    test_hist_vectors = [[] for _ in range(5)]

    for i in range(len(train_hist_vectors)):
        for count, j in enumerate(train_txt_No_sum[i]):
            if count == 0:
                train_hist_vectors[i] = np.zeros((1, FEATURES))
            train_hist_vectors[i] = np.vstack((train_hist_vectors[i],data_f[j]))
        train_hist_vectors[i] = np.delete(train_hist_vectors[i], 0, 0)

    for i in range(len(test_hist_vectors)):
        for count, j in enumerate(test_txt_No_sum[i]):
            if count == 0:
                test_hist_vectors[i] = np.zeros((1, FEATURES))
            test_hist_vectors[i] = np.vstack((test_hist_vectors[i],data_f[j]))
        test_hist_vectors[i] = np.delete(test_hist_vectors[i], 0, 0)
    
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
   
    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            sum_score = 0
            for i in range(len(train_hist_vectors)):
                svm = SVC(gamma=gamma, C=C)
                svm.fit(train_hist_vectors[i], train_targets[i])
                score = svm.score(test_hist_vectors[i], test_targets[i])
                sum_score += score
            #print("gamma: {} and C: {}".format(gamma, C))
            #print("average score for current parameters: {:.4f}".format(sum_score/len(train_hist_vectors)))
            if sum_score > best_score:
                best_score = sum_score
                best_parameters = {'C':C, 'gamma':gamma}
            
    print("best score: {:.4f}".format(best_score))
    print("best parameters: {}".format(best_parameters))
    print("Test set average score for 5 sets: {:.4f}".format(best_score/len(train_hist_vectors))) 
    
