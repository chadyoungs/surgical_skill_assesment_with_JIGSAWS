#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 23:50:59 2020
@author: xiaoxiaoyang
"""
import os
import numpy as np
from hmmlearn import hmm
import glob
import joblib

from data_extract import DataExtract

from sklearn.decomposition import PCA

import Global_Var


def load_data(path):
    surgery_name = os.path.splitext(os.path.basename(path))[0]
    data = np.loadtxt(path, dtype=np.float64)

    X_MTM_L = data[:, 0:19]
    X_MTM_R = data[:, 19:38]
    X_PSM_1 = data[:, 38:57]
    X_PSM_2 = data[:, 57:]

    return X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2, surgery_name


def PCA_trans(data):
    pca = PCA(n_components=Global_Var.PCA_COMPONENTS, whiten=True, random_state=0).fit(data)
    return pca.transform(data)


def kmeans_selected(surgeme):
    task_dir = os.path.join('.', 'observation_clusters', surgical_task_list[TASK_SYMBOL])
    if Global_Var.TASK_SYMBOL == 0 or Global_Var.TASK_SYMBOL == 2:
        # for suturing and needle passing
        surgemes_list = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
    else:
        # for knot_tying
        surgemes_list = ['G1', 'G11', 'G12', 'G13', 'G14', 'G15']

    kmeans_list = [joblib.load(os.path.join(task_dir, g + '.pkl')) for g in surgemes_list]
    return kmeans_list[surgemes_list.index(surgeme)]


# training the novice and expert model
def build_hmm(files_path):

    CLUSTERS = Global_Var.CLUSTERS

    if Global_Var.TASK_SYMBOL == 0 or Global_Var.TASK_SYMBOL == 2:
        # for suturing and needle passing
        states = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
    else:
        # for knot_tying
        states = ['G1', 'G11', 'G12', 'G13', 'G14', 'G15']

    n_states = len(states)

    # states transition
    A_expert = np.zeros((n_states, n_states))
    A_novice = np.zeros((n_states, n_states))
    # observation, we have obtained 64 clusters in training process
    B_expert = np.zeros((n_states, CLUSTERS))
    B_novice = np.zeros((n_states, CLUSTERS))
    # states initial distribution
    pi_expert = np.zeros(n_states)
    pi_novice = np.zeros(n_states)

    observations = []

    txt_files = glob.glob(os.path.join(files_path, '*.txt'))
    for txt_file in txt_files:
        X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2, surgery_name = load_data(txt_file)
        data_trans = np.hstack((PCA_trans(X_MTM_L), PCA_trans(X_MTM_R),
                                PCA_trans(X_PSM_1), PCA_trans(X_PSM_2)))

        # pi calculation
        # there are only 'G1'/'G5'/'G8' in JHU dataset suturing usually
        first_surgeme = test.metaData_surgeme[surgery_name][2][0]
        for i in range(n_states):
            if first_surgeme == states[i]:
                if test.metaData_score[surgery_name][1] == 0:
                    pi_novice[i] += 1
                else:
                    pi_expert[i] += 1

        observation = []
        # A and B calculation
        surgeme_seq = test.metaData_surgeme[surgery_name][2]
        for seg_idx, surgeme in enumerate(surgeme_seq):
            start = int(test.metaData_surgeme[surgery_name][0][seg_idx])
            end = int(test.metaData_surgeme[surgery_name][1][seg_idx])
            kmeans = kmeans_selected(surgeme)

            result = kmeans.predict(np.float64(data_trans[start:end, :]))

            # B
            state_idx = states.index(surgeme)
            is_novice = test.metaData_score[surgery_name][1] == 0
            for obs in result:
                if is_novice:
                    B_novice[state_idx, obs] += 1
                else:
                    B_expert[state_idx, obs] += 1

            # observation series
            observation.append(result.tolist())

            # A
            if seg_idx != (len(surgeme_seq) - 1):
                next_surgeme = surgeme_seq[seg_idx + 1]
                if is_novice:
                    A_novice[state_idx, states.index(next_surgeme)] += 1
                else:
                    A_expert[state_idx, states.index(next_surgeme)] += 1

        observations.append(observation)

    # pi calculation
    pi_novice /= np.sum(pi_novice)
    pi_expert /= np.sum(pi_expert)

    # A calculation
    for i in range(n_states):
        if np.sum(A_novice[i]) != 0:
            A_novice[i] /= np.sum(A_novice[i])
        if np.sum(A_expert[i]) != 0:
            A_expert[i] /= np.sum(A_expert[i])

    # B calculation
    for i in range(n_states):
        if np.sum(B_novice[i]) != 0:
            B_novice[i] /= np.sum(B_novice[i])
        if np.sum(B_expert[i]) != 0:
            B_expert[i] /= np.sum(B_expert[i])

    model_expert = hmm.MultinomialHMM(n_components=n_states)
    model_expert.startprob_ = pi_expert
    model_expert.emissionprob_ = B_expert
    model_expert.transmat_ = A_expert

    model_novice = hmm.MultinomialHMM(n_components=n_states)
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

    files_path = os.path.join('.', 'da_vici_data_with_iDT_features',
                              surgical_task_list[TASK_SYMBOL], 'kinematics', 'AllGestures')
    save_expert_model_path = os.path.join('.', 'models',
                                          surgical_task_list[TASK_SYMBOL] + '_expert_model.pkl')
    save_novice_model_path = os.path.join('.', 'models',
                                          surgical_task_list[TASK_SYMBOL] + '_novice_model.pkl')
    save_obser_path = os.path.join('.', 'observations',
                                   surgical_task_list[TASK_SYMBOL] + '_observations.pkl')

    model_expert, model_novice, observations = build_hmm(files_path)

    joblib.dump(model_expert, save_expert_model_path)
    joblib.dump(model_novice, save_novice_model_path)
    joblib.dump(observations, save_obser_path)
    print("Done. Dumping expert models to", os.path.basename(save_expert_model_path))
    print("Done. Dumping novice models to", os.path.basename(save_novice_model_path))
    print("Done. Dumping observations to", os.path.basename(save_obser_path))
    print("####################\n")
