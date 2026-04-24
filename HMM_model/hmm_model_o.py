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

import Global_Var

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

    pca = PCA(n_components=Global_Var.PCA_COMPONENTS, whiten=True, random_state=0).fit(data)
    data_trans = pca.transform(data)  
    
    return data_trans

def kmeans_selected(surgeme):
    if Global_Var.TASK_SYMBOL == 0 or Global_Var.TASK_SYMBOL == 2:
        #  for suturing task
        kmeans_G1 = joblib.load(r'.\observation_clusters\{}\G1.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G2 = joblib.load(r'.\observation_clusters\{}\G2.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G3 = joblib.load(r'.\observation_clusters\{}\G3.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G4 = joblib.load(r'.\observation_clusters\{}\G4.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G5 = joblib.load(r'.\observation_clusters\{}\G5.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G6 = joblib.load(r'.\observation_clusters\{}\G6.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G8 = joblib.load(r'.\observation_clusters\{}\G8.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G9 = joblib.load(r'.\observation_clusters\{}\G9.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G10 = joblib.load(r'.\observation_clusters\{}\G10.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G11 = joblib.load(r'.\observation_clusters\{}\G11.pkl'.format(surgical_task_list[TASK_SYMBOL]))
    
        surgemes_list = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'] 
        kmeans_list = [kmeans_G1, kmeans_G2, kmeans_G3, kmeans_G4, kmeans_G5, kmeans_G1,
                   kmeans_G6, kmeans_G8, kmeans_G9, kmeans_G10, kmeans_G11]
    else:
        # for knot_tying
        kmeans_G1 = joblib.load(r'.\observation_clusters\{}\G1.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G11 = joblib.load(r'.\observation_clusters\{}\G11.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G12 = joblib.load(r'.\observation_clusters\{}\G12.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G13 = joblib.load(r'.\observation_clusters\{}\G13.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G14 = joblib.load(r'.\observation_clusters\{}\G14.pkl'.format(surgical_task_list[TASK_SYMBOL]))
        kmeans_G15 = joblib.load(r'.\observation_clusters\{}\G15.pkl'.format(surgical_task_list[TASK_SYMBOL]))
    
        surgemes_list = ['G1', 'G11', 'G12', 'G13', 'G14', 'G15'] 
        kmeans_list = [kmeans_G1, kmeans_G11, kmeans_G12, kmeans_G13, kmeans_G14, kmeans_G15]
    
    return kmeans_list[surgemes_list.index(surgeme)]

# training the novice and expert model
def hMM(files_path):
     
     CLUSTERS = Global_Var.CLUSTERS
     
     if Global_Var.TASK_SYMBOL == 0 or Global_Var.TASK_SYMBOL == 2:
         # for suturing and needle passing
         states = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'] 
         A_ = 10
     else:
         # for knot_tying
         states = ['G1', 'G11', 'G12', 'G13', 'G14', 'G15']
         A_ = 6
     
     # states transition
     # suturing  10  knot_tying  6   needle_passing
     A_expert = np.zeros((A_, A_))
     A_novice = np.zeros((A_, A_))
     # observation, we have obtain 64 clusters in training process
     B_expert = np.zeros((A_, CLUSTERS))   
     B_novice = np.zeros((A_, CLUSTERS))
     # states initial distribution
     pi_expert = np.zeros(A_)
     pi_novice = np.zeros(A_)
     
     observations = []
     
     txt_files = glob.glob(files_path + '\*.txt')
     for count, txt_file in enumerate(txt_files):
         data_raw = load_data(txt_file)
         surgery_name = data_raw[4]
         data_MTM_L_trans = PCA_trans(data_raw[0])
         data_MTM_R_trans = PCA_trans(data_raw[1])
         data_PSM_1_trans = PCA_trans(data_raw[2])
         data_PSM_2_trans = PCA_trans(data_raw[3])
         data_trans = np.hstack((data_MTM_L_trans, data_MTM_R_trans, data_PSM_1_trans, data_PSM_2_trans))
    
         # pi calculation
         # there are only 'G1'/'G5'/'G8' in JHU dataset suturing usually
         # suturing, needle passing  10  knot_tying  6
         for i in range(A_):
             if test.metaData_surgeme[surgery_name][2][0] == states[i]: 
                 # novice
                 if test.metaData_score[surgery_name][1] == 0:
                     pi_novice[i] += 1
                 # expert
                 else:
                     pi_expert[i] += 1
         
         observation = []
         # A and B calculation
         list_ = test.metaData_surgeme[surgery_name][2]
         for count, i in enumerate(list_):
             surgeme = test.metaData_surgeme[surgery_name][2][count]
         
             start = int(test.metaData_surgeme[surgery_name][0][count])
             end = int(test.metaData_surgeme[surgery_name][1][count])  # +1?
             kmeans = kmeans_selected(surgeme)
         
             result = kmeans.predict(np.float64(data_trans[start:end, :]))
             # choices of result calculation
             #temp = np.bincount(result)
             #result = int(sum(result)/len(result))
             #result = np.argmax(temp)
             
             # B
             for i in result:
                 if test.metaData_score[surgery_name][1] == 0:
                     B_novice[states.index(surgeme), i] += 1
                 else:
                     B_expert[states.index(surgeme), i] += 1
         
             # observation series
             observation.append(result.tolist())
             #print(observation)
         
             # A
             if count != (len(list_) - 1):
                 surgeme_ = test.metaData_surgeme[surgery_name][2][count+1]
                 if test.metaData_score[surgery_name][1] == 0:
                     A_novice[states.index(surgeme), states.index(surgeme_)] += 1
                 else:
                     A_expert[states.index(surgeme), states.index(surgeme_)] += 1
                     
             
         '''    
         # previous code which the ovservations series is as long as frames, and 
         # and each frame mapping to one state, the result is disappoint 
         for i, line in enumerate(data_trans):
             frame_No = i + 1
         
             # as the start frame No and end frame No of txt may have no suegeme.
             # we ignore these frames here
             if frame_No >= test.metaData[surgery_name][0] and frame_No <= test.metaData[surgery_name][1]:
                 surgeme = test.metaData_hmmstates[frame_No]
                 # for suturing
                 kmeans = kmeans_selected(surgeme)
                 result = kmeans.predict(np.float64(line.reshape(1, -1)))
                 
                 result = int(result)
                 observation.append(result)
                 # novice
                 if test.metaData_score[surgery_name][1] == 0:
                     B_novice[states.index(surgeme), result] += 1
                 # expert
                 else:
                     B_expert[states.index(surgeme), result] += 1
         
             if frame_No != test.metaData[surgery_name][1]:
                 surgeme = test.metaData_hmmstates[frame_No]
                 next_frame = frame_No + 1
                 surgeme_ = test.metaData_hmmstates[next_frame]
                 # novice
                 if test.metaData_score[surgery_name][1] == 0:
                     A_novice[states.index(surgeme), states.index(surgeme_)] += 1
                 # expert
                 else:
                     A_expert[states.index(surgeme), states.index(surgeme_)] += 1
         '''           
         # observation series
         observations.append(observation)
     
     # pi calculation
     pi_novice /= np.sum(pi_novice)
     pi_expert /= np.sum(pi_expert)
     
     # A calculation
     # suturing, needle passing 10 knot tying 6
     for i in range(A_):
         if np.sum(A_novice[i]) != 0:
             A_novice[i] = A_novice[i] / np.sum(A_novice[i])
         if np.sum(A_expert[i]) != 0:
             A_expert[i] = A_expert[i] / np.sum(A_expert[i])
     
     # B calculation
     # suturing, needle passing 10 knot tying 6
     for i in range(A_):
         if np.sum(B_novice[i]) != 0:
             B_novice[i] /= np.sum(B_novice[i])
         if np.sum(B_expert[i]) != 0:
             B_expert[i] = B_expert[i] / np.sum(B_expert[i])

     model_expert = hmm.MultinomialHMM(n_components=len(states))
     model_expert.startprob_ = pi_expert
     model_expert.emissionprob_ = B_expert
     model_expert.transmat_ = A_expert
     
     model_novice = hmm.MultinomialHMM(n_components=len(states))
     model_novice.startprob_ = pi_novice
     model_novice.emissionprob_ = B_novice
     model_novice.transmat_ = A_novice
    
     return model_expert, model_novice, observations

if __name__ == '__main__':
    # choose task
    TASK_SYMBOL = Global_Var.TASK_SYMBOL
    
    surgical_task_list = ["Suturing", "Knot_Tying", "Needle_Passing"]
    
    test = DataExtract()
    test.get_category()
    test.get_txt_index()  
    test.get_score()
    test.get_frame_surgeme()
    test.get_frame_No()
    test.get_hmm_states()

    train_list = test.train_sum()
    test_list = test.test_sum()
    
    # remind the split_files_path
    files_path = r'.\da_vici_data_with_iDT_features\{}\kinematics\AllGestures'.format(surgical_task_list[TASK_SYMBOL])
    split_files_path = r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Data_clips\{}_clips\kinetic'.format(surgical_task_list[TASK_SYMBOL])
    save_expert_model_path = r'.\models\{}_expert_model.pkl'.format(surgical_task_list[TASK_SYMBOL])
    save_novice_model_path = r'.\models\{}_novice_model.pkl'.format(surgical_task_list[TASK_SYMBOL])
    save_obser_path = r'.\observations\{}_observations.pkl'.format(surgical_task_list[TASK_SYMBOL])
    
    train_model = hMM(files_path)
    
    suturing_expert_model = train_model[0]
    suturing_novice_model = train_model[1]
    suturing_observations = train_model[2]
    
    # previous code
    joblib.dump(suturing_expert_model, save_expert_model_path)
    joblib.dump(suturing_novice_model, save_novice_model_path)
    joblib.dump(suturing_observations, save_obser_path)
    print("Done. Dumping expert models to", save_expert_model_path.split("\\")[-1])
    print("Done. Dumping novice models to", save_novice_model_path.split("\\")[-1])
    print("Done. Dumping observations to", save_obser_path.split("\\")[-1])
    print("####################\n")

    


