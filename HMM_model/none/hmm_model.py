#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 23:50:59 2020
@author: xiaoxiaoyang
"""
import scipy.signal

import numpy as np
from hmmlearn import hmm
import glob 
import joblib

from data_extract import DataExtract

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def load_data(path):
   temp = path.split("\\")[-1]
   surgery_name = temp.split(".")[0] 
   data = np.loadtxt(path, dtype=np.float64)

   X_MTM_L = data[:, 0:19]
   X_MTM_R = data[:, 19:38]
   X_PSM_1 = data[:, 38:57]
   X_PSM_2 = data[:, 57:] 
   #y = test.metaData_score[surgery_name][1] 
   #grade = np.array([[y],])
  
   return X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2, surgery_name

def PCA_trans(data):
    #stft
    #result = scipy.signal.stft(X_MTM_L_sum,fs=30.0,window='hann',nperseg=76,noverlap=None,nfft=None,
                            #detrend=False,return_onesided=True,boundary='zeros',padded=True,axis=-1)
    #LDA
    #lda = LDA(n_components=2)
    #X_r2 = lda.fit(X_MTM_L_sum).transform(X_MTM_L_sum)  

    pca = PCA(n_components=4, whiten=True, random_state=0).fit(data)
    data_trans = pca.transform(data)  
    
    return data_trans

def kmeans_selected(surgeme):
    #  for suturing task
    kmeans_G1 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G1.pkl')
    kmeans_G2 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G2.pkl')
    kmeans_G3 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G3.pkl')
    kmeans_G4 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G4.pkl')
    kmeans_G5 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G5.pkl')
    kmeans_G6 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G6.pkl')
    kmeans_G8 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G8.pkl')
    kmeans_G9 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G9.pkl')
    kmeans_G10 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G10.pkl')
    kmeans_G11 = joblib.load(r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observation_clusters\Suturing\G11.pkl')
    
    surgemes_list = ['G1', 'G2', 'G3', 'G4',
                     'G5', 'G6', 'G8', 'G9',
                     'G10', 'G11'] 
    kmeans_list = [kmeans_G1, kmeans_G2, kmeans_G3, kmeans_G4, kmeans_G5, kmeans_G1,
                   kmeans_G6, kmeans_G8, kmeans_G9, kmeans_G10, kmeans_G11]
    
    return kmeans_list[surgemes_list.index(surgeme)]

def hMM(data, surgery_name):
     # for suturing
     states = ['G1', 'G2', 'G3', 'G4',
               'G5', 'G6', 'G8', 'G9',
               'G10', 'G11']   
     # states transition
     A = np.zeros((10, 10))
     # observation, we have obtain 64 clusters in training process
     B = np.zeros((10, 10))   
     # states initial distribution
     pi = np.zeros(10)
     
     # pi calculation
     # there are only 'G1'/'G5'/'G8' in JHU dataset suturing usually
     for i in range(10):
         if test.metaData_surgeme[surgery_name][2][0] == states[i]:  
             pi[i] += 1
            
     pi /= np.sum(pi)   
     
     observation = []
     # A and B calculation
     list_ = test.metaData_surgeme[surgery_name][2]
     for count, i in enumerate(list_):
         surgeme = test.metaData_surgeme[surgery_name][2][count]
         
         start = int(test.metaData_surgeme[surgery_name][0][count])
         end = int(test.metaData_surgeme[surgery_name][1][count])  # +1?
         kmeans = kmeans_selected(surgeme)
         
         result = kmeans.predict(np.float64(data[start:end, :]))
         temp = np.bincount(result)
         
         
         
         # B
         for i in result:
             B[states.index(surgeme), i] += 1
             
         #result = int(sum(result)/len(result))
         result = np.argmax(temp)
         
         # observation series
         observation.append(result)
         
         # A
         if count != (len(list_) - 1):
             surgeme_ = test.metaData_surgeme[surgery_name][2][count+1]
             
             A[states.index(surgeme), states.index(surgeme_)] += 1
    
     for i in range(10):
         if np.sum(A[i]) != 0:
             A[i] = A[i] / np.sum(A[i])

     for i in range(10):
         if np.sum(B[i]) != 0:
             B[i] /= np.sum(B[i])

     model = hmm.MultinomialHMM(n_components=len(states))
     model.startprob_ = pi
     model.emissionprob_ = B
     model.transmat_ = A
    
     return model, observation


if __name__ == '__main__':
    test = DataExtract()
    test.get_category()
    test.get_txt_index()  
    test.get_score()
    test.get_frame_surgeme()
    test.get_frame_No()
    test.get_hmm_states()

    train_list = test.train_sum()
    test_list = test.test_sum()
    
    # Suturing
    files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Suturing\kinematics\AllGestures'
    split_files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Data_clips\Suturing_clips\kinetic'
    save_model_path = r'F:\Projects\surgical_project\Kinematic_code\HMM_model\models\suturing.pkl'
    save_obser_path = r'F:\Projects\surgical_project\Kinematic_code\HMM_model\observations\suturing_obser.pkl'
    # Knot_Tying
    #files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Knot_Tying\kinematics\AllGestures'
    #split_files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Data_clips\Knot_Tying_clips\kinetic'

    # Needle_Passing
    #files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Needle_Passing\kinematics\AllGestures'
    #split_files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Data_clips\Needle_Passing_clips\kinetic'
    
    txt_files = glob.glob(files_path + '\*.txt')
    
    suturing_models = []
    suturing_novice_models = []
    suturing_inter_models = []
    suturing_expert_models = []
    
    suturing_obser = []
    suturing_novice_obser = []
    suturing_inter_obser = []
    suturing_expert_obser = []
    
    for count, txt_file in enumerate(txt_files):
        data_raw = load_data(txt_file)
        
        data_surgery_name = data_raw[4]
        data_MTM_L_trans = PCA_trans(data_raw[0])
        data_MTM_R_trans = PCA_trans(data_raw[1])
        data_PSM_1_trans = PCA_trans(data_raw[2])
        data_PSM_2_trans = PCA_trans(data_raw[3])
        data_trans = np.hstack((data_MTM_L_trans, data_MTM_R_trans, data_PSM_1_trans, data_PSM_2_trans))

        hmm_model = hMM(data_trans, data_surgery_name)
        
        suturing_models.append(hmm_model[0])
        suturing_obser.append(hmm_model[1])
        '''
        # for test
        if count == 1:
            break
        '''
    
    # previous code
    joblib.dump(suturing_models, save_model_path)
    joblib.dump(suturing_obser, save_obser_path)

    print("Done. Dumping models to", save_model_path.split("\\")[-1])
    print("Done. Dumping observations to", save_obser_path.split("\\")[-1])
    print("####################\n")
    
    '''    
    #for test       
    x = hmm_forward(suturing_models[1].transmat_, suturing_models[1].emissionprob_, 
                suturing_models[1].startprob_, suturing_obser[0])
    y = hmm_forward(suturing_models[1].transmat_, suturing_models[1].emissionprob_, 
                suturing_models[1].startprob_, suturing_obser[1])
    print(x)
    print(x[0])
    print(np.log(np.sum(x[0], axis=0)[0]))
    print(np.log(np.sum(y[0], axis=0)[0]))
    #print(suturing_models[2].score(suturing_obser[1]))
    '''
    


