#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 01:06:45 2020
@author: xiaoxiaoyang
"""

import scipy.signal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import numpy as np

import glob 
import joblib

from sklearn.cluster import KMeans
import Global_Var

def load_data(path, surgeme):
    txt_files = glob.glob(path + '\*.txt')
    X_MTM_L_sum = np.zeros(shape = (1, 19))
    X_MTM_R_sum = np.zeros(shape = (1, 19))
    X_PSM_1_sum = np.zeros(shape = (1, 19))
    X_PSM_2_sum = np.zeros(shape = (1, 19))
    
    for count, i in enumerate(txt_files):
        temp = i.split("\\")[-1]
        surgery_name = temp.split(".")[0]
        if surgery_name.__contains__(surgeme):
            #print(surgery_name)
            data = np.loadtxt(i, dtype=np.float64)
            X_MTM_L = data[:, 0:19]
            X_MTM_R = data[:, 19:38]
            X_PSM_1 = data[:, 38:57]
            X_PSM_2 = data[:, 57:]
            #y = test.metaData_score[surgery_name][1]
      
            X_MTM_L_sum = np.vstack((X_MTM_L_sum, X_MTM_L))
            X_MTM_R_sum = np.vstack((X_MTM_R_sum, X_MTM_R))
            X_PSM_1_sum = np.vstack((X_PSM_1_sum, X_PSM_1))
            X_PSM_2_sum = np.vstack((X_PSM_2_sum, X_PSM_2))
            #grade = np.array([[y],])
            #np.vstack((y_MTM_L_sum, grade))
           
    X_MTM_L_sum = np.delete(X_MTM_L_sum, 0, 0)
    X_MTM_R_sum = np.delete(X_MTM_R_sum, 0, 0)
    X_PSM_1_sum = np.delete(X_PSM_1_sum, 0, 0)
    X_PSM_2_sum = np.delete(X_PSM_2_sum, 0, 0)
    
    return (X_MTM_L_sum, X_MTM_R_sum, X_PSM_1_sum, X_PSM_2_sum)

def PCA_trans(data):
    #stft
    #result = scipy.signal.stft(X_MTM_L_sum,fs=30.0,window='hann',nperseg=76,noverlap=None,nfft=None,
                            #detrend=False,return_onesided=True,boundary='zeros',padded=True,axis=-1)
    #LDA 
    #lda = LDA(n_components=2)
    #X_r2 = lda.fit(X_MTM_L_sum).transform(X_MTM_L_sum)  

    #PCA
    pca = PCA(n_components=4, whiten=True, random_state=0).fit(data)
    data_trans = pca.transform(data)  
    
    return data_trans
    

if __name__ == '__main__':
    N_CLUSTERS = Global_Var.N_CLUSTERS
    
    # 0 set as Suturing
    TASK_SYMBOL = Global_Var.TASK_SYMBOL
    task_list = ["Suturing", "Knot_Tying", "Needle_Passing"]
    
    split_files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Data_clips\{}_clips\kinetic'.format(task_list[TASK_SYMBOL])
    
    if TASK_SYMBOL == 0 or TASK_SYMBOL == 2:
        # For suturing and needle passing
        surgeme_list = ["_G1_", "_G2_", "_G3_", "_G4_", "_G5_", "_G6_", "_G8_", "_G9_", "_G10_", "_G11_"]
    
        # for suturing and needle passing
        save_path_list = [".\observation_clusters\{}\G1.pkl".format(task_list[TASK_SYMBOL]), 
                      ".\observation_clusters\{}\G2.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G3.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G4.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G5.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G6.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G8.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G9.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G10.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G11.pkl".format(task_list[TASK_SYMBOL])]
    else:
        # For Knot_Passing
        surgeme_list = ["_G1_", "_G11_", "_G12_", "_G13_", "_G14_", "_G15_"]
        
    
        #for knot tying
        save_path_list = [".\observation_clusters\{}\G1.pkl".format(task_list[TASK_SYMBOL]), 
                      ".\observation_clusters\{}\G11.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G12.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G13.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G14.pkl".format(task_list[TASK_SYMBOL]),
                      ".\observation_clusters\{}\G15.pkl".format(task_list[TASK_SYMBOL])]           
    
    for i in range(len(surgeme_list)):
        data_raw = load_data(split_files_path, surgeme_list[i])
        data_MTM_L = PCA_trans(data_raw[0])
        data_MTM_R = PCA_trans(data_raw[1])
        data_PSM_1 = PCA_trans(data_raw[2])
        data_PSM_2 = PCA_trans(data_raw[3])
    
        data_result = np.hstack((data_MTM_L, data_MTM_R, data_PSM_1, data_PSM_2))
        
        data_result = np.float64(data_result)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(data_result)
        joblib.dump(kmeans, save_path_list[i])
        
        print("Done. Dumping to", save_path_list[i].split("\\")[-1])
        print("####################\n")
    




