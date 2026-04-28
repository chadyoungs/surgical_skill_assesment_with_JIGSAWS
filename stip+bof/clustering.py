# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:55:25 2020
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
Step 3 (Classification):SVM， kNN and Bayes
Step 4 (Sliding)

This script mainly complete Step 1
"""

import Global_Var

import joblib
from glob import glob
from data_extract_stip import DataExtract

from sklearn.cluster import KMeans

import numpy as np

import cv2

def find_last_frame(videos_features_location):
    videos_features = glob(videos_features_location + '\\*.txt')
    for i in videos_features:
        data = np.loadtxt(i, dtype=float)
        lastframe = max(data[:, 6])
        with open(i, 'r+') as f:
            for count, line in enumerate(f):
                if count == 1:
                    line.strip()
                    surgery_name_temp = line[2:].replace('\n', '')
                    
                    temp_0 = surgery_name_temp.split('_')
                    temp_0.reverse()
                    temp_1 = temp_0[1:]
                    temp_1.reverse()
                    surgery_name = '_'.join(temp_1)

                    lastline_surgeryname[surgery_name] = lastframe
    
def kmeans_predict(video_feature_name):
    surgery_name = video_feature_name.split('\\')[-1].split('.')[0]
    #lastframe_No = int(lastline_surgeryname[surgery_name])
    #lastframe_No += 1
    
    result_hog = np.zeros(N_CLUSTERS)
    result_hof = np.zeros(N_CLUSTERS)
    raw_data = np.loadtxt(video_feature_name, dtype=float)
    for i in raw_data:
        #frame_No = int(i[6])
        hog_data = i[9:81]
        hof_data = i[81:]
        hog_data = np.resize(hog_data, (1, 72))
        hof_data = np.resize(hof_data, (1, 90))
        
        pred_hog = hog_kmeans.predict(hog_data)
        pred_hof = hof_kmeans.predict(hof_data)
        result_hog[pred_hog[0]] += 1
        result_hof[pred_hof[0]] += 1
    
    cv2.normalize(result_hog, result_hog, norm_type=cv2.NORM_L2)
    cv2.normalize(result_hof, result_hof, norm_type=cv2.NORM_L2)
    
    return surgery_name, result_hog, result_hof 
  
def kmeans_data(video_feature_name):
    videos_features = glob(videos_features_location + '\\*.txt')
    if not len(videos_features):
        raise AssertionError
        
    for i in videos_features:
        temp = i.split("\\")[-1].split('.')[0]
        if temp == video_feature_name:
            print(temp)
            raw_data = np.loadtxt(i, dtype=float)
            # we only extract the HOG-HOF data, 72 + 90 = 162
            data_HOG = raw_data[:, 9:81]
            data_HOF = raw_data[:, 81:]
            data_HOG = np.reshape(data_HOG, (-1, 72))
            data_HOF = np.reshape(data_HOF, (-1, 90))
            
    return data_HOG, data_HOF

def data_extraction(data, No):
    rand_arr = np.arange(data.shape[0])
    np.random.seed(0)
    np.random.shuffle(rand_arr)
    
    return data[rand_arr[:No]]


if __name__ == '__main__':
    
    task_list = ['Suturing', 'Knot_Tying', 'Needle_Passing']
    # 0 for suturing
    TASK_SYMBOL = Global_Var.TASK_SYMBOL_Global
    # we set num of clusters as 4, tips and some other two places, maybe some other cut edges
    N_CLUSTERS = Global_Var.CLUSTERS
    SELECTED_FEATURES_No = Global_Var.SELECTED_FEATURES_No
    
    # call the data_extract_stip to obtain the train set 
    test = DataExtract()
    test.get_category()
    test.get_txt_index()  
    test.get_score()

    train_list = test.train_sum()
    test_list = test.test_sum()
    
    videos_features_location = r'..\stip+dct_code\raw_data\{}'.format(task_list[TASK_SYMBOL])   
    
#training process
    hog_repositories = [[] for _ in range(len(train_list))]
    hof_repositories = [[] for _ in range(len(train_list))]
    for i in range(len(train_list)):
        hog_repositories[i] = r'.\\clusters\\{}\\clusters_{}_set{}_hog.pkl'.format(task_list[TASK_SYMBOL], N_CLUSTERS, i+1)
        hof_repositories[i] = r'.\\clusters\\{}\\clusters_{}_set{}_hof.pkl'.format(task_list[TASK_SYMBOL], N_CLUSTERS, i+1)    
     
    for i in range(len(train_list)):
        HOG_vector = np.zeros(shape = (1, 72))
        HOF_vector = np.zeros(shape = (1, 90))
        #print(train_list[i])
        for j in train_list[i]:
            video_feature_name = j
            hog_extract_data,  hof_extract_data = kmeans_data(video_feature_name)
            HOG_vector = np.vstack((HOG_vector, hog_extract_data))
            HOF_vector = np.vstack((HOF_vector, hof_extract_data))
    
        HOG_vector = np.delete(HOG_vector, 0, 0)
        HOF_vector = np.delete(HOF_vector, 0, 0)
    
        HOG_vector = np.float64(HOG_vector)
        HOF_vector = np.float64(HOF_vector)
        
        HOG_vector = data_extraction(HOG_vector, SELECTED_FEATURES_No)
        HOG_kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(HOG_vector)
        joblib.dump(HOG_kmeans, hog_repositories[i])
        
        HOF_vector = data_extraction(HOF_vector, SELECTED_FEATURES_No)
        HOF_kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(HOF_vector)
        joblib.dump(HOF_kmeans, hof_repositories[i])
        print("\n####################")
        print("Done. Dumping HOG and HOF clusters of train set {} to".format(i+1), hog_repositories[i].split("\\")[-1], hof_repositories[i].split("\\")[-1])
    print("\n####################")
    print("Done. HOG and HOF clusters of all train sets have been dumped!")
    
#predicting process
    hog_clusters_locations = [[] for _ in range(len(train_list))]
    hof_clusters_locations = [[] for _ in range(len(train_list))]
    
    repositories_0 = [[] for _ in range(len(train_list))]
    repositories_1 = [[] for _ in range(len(train_list))]
    repositories_2 = [[] for _ in range(len(train_list))]
    for i in range(len(train_list)):
        hog_clusters_locations[i] = r'.\clusters\{}\clusters_{}_set{}_hog.pkl'.format(task_list[TASK_SYMBOL], N_CLUSTERS, i+1)
        hof_clusters_locations[i] = r'.\clusters\{}\clusters_{}_set{}_hof.pkl'.format(task_list[TASK_SYMBOL], N_CLUSTERS, i+1)
        
        repositories_0[i] = r'.\time_series_data\{}\time_series_surgery_name_set{}_{}.pkl'.format(task_list[TASK_SYMBOL], i+1, N_CLUSTERS)
        repositories_1[i] = r'.\time_series_data\{}\time_series_hog_set{}_{}.pkl'.format(task_list[TASK_SYMBOL], i+1, N_CLUSTERS)
        repositories_2[i] = r'.\time_series_data\{}\time_series_hof_set{}_{}.pkl'.format(task_list[TASK_SYMBOL], i+1, N_CLUSTERS)
    
    # obtain the max frame No. in data
    # the last line's frame No. is not always the maximum frame No.
    lastline_surgeryname = {}
    lastline_txt_file_name = {}
    find_last_frame(videos_features_location)
   
    video_feature_names = [[] for _ in range(len(train_list))]
    time_series_hog = [[] for _ in range(len(train_list))]
    time_series_hof = [[] for _ in range(len(train_list))]
    
    videos_features = glob(videos_features_location + '\\*.txt')
    for i in range(len(train_list)):
        for j in videos_features:
            #if i.split("\\")[-1] not in expert_video_features:
            hog_kmeans = joblib.load(hog_clusters_locations[i])
            hof_kmeans = joblib.load(hof_clusters_locations[i])
            surgery_name, hog_predict_result, hof_predict_result = kmeans_predict(j)
            
            video_feature_names[i].append(surgery_name)
            time_series_hog[i].append(hog_predict_result)
            time_series_hof[i].append(hof_predict_result)
        
        joblib.dump(video_feature_names[i], repositories_0[i])
        joblib.dump(time_series_hog[i], repositories_1[i])
        joblib.dump(time_series_hof[i], repositories_2[i])
        print("\n####################")
        print("Done. Dumping time series of set{} to".format(i+1), repositories_0[i].split("\\")[-2:])
    print("\n####################")
    print("Done. time series of all train sets have been dumped!")
