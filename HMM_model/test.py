# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:00:29 2020
@author: xiaoxiaoyang
"""
import os
import numpy as np
import joblib

from data_extract import DataExtract

import Global_Var


def hmm_forward(A, B, pi, O):
    T = len(O)
    N = len(A[0])

    # initialize
    alpha = [[0] * T for _ in range(N)]

    for i in range(N):
        temp_sum = sum(pi[i] * B[i][O[0][j]] for j in range(len(O[0])))
        alpha[i][0] = temp_sum / len(O[0])

    # calculate alpha(t)
    for t in range(1, T):
        for i in range(N):
            temp = sum(alpha[j][t - 1] * A[j][i] for j in range(N))
            temp_sum = sum(temp * B[i][O[t][k]] for k in range(len(O[t])))
            alpha[i][t] = temp_sum / len(O[t])

    # sum final column
    proba = sum(alpha[i][-1] for i in range(N))
    return proba, alpha


if __name__ == '__main__':

    task_list = ['Suturing', 'Knot_Tying', 'Needle_Passing']
    # 0 for suturing
    TASK_SYMBOL = Global_Var.TASK_SYMBOL

    hmm_expert_model = joblib.load(os.path.join('.', 'models',
                                                 task_list[TASK_SYMBOL] + '_expert_model.pkl'))
    hmm_novice_model = joblib.load(os.path.join('.', 'models',
                                                 task_list[TASK_SYMBOL] + '_novice_model.pkl'))
    obser = joblib.load(os.path.join('.', 'observations',
                                     task_list[TASK_SYMBOL] + '_observations.pkl'))

    test = DataExtract()
    test.get_category()
    test.get_txt_index()
    test.get_score()
    test.get_index()

    novice_values = []
    expert_values = []
    # calculate each observation's probability against both HMM models
    for obs in obser:
        prob_expert = hmm_forward(hmm_expert_model.transmat_, hmm_expert_model.emissionprob_,
                                  hmm_expert_model.startprob_, obs)
        prob_novice = hmm_forward(hmm_novice_model.transmat_, hmm_novice_model.emissionprob_,
                                  hmm_novice_model.startprob_, obs)
        novice_values.append(prob_novice[0])
        expert_values.append(prob_expert[0])

    # novice and expert sets
    novice_set = test.metaData_hmmindex["novice"]
    expert_set = test.metaData_hmmindex["expert"]

    n_n = sum(1 for i in novice_set if novice_values[int(i)] > expert_values[int(i)])
    n_e = len(novice_set) - n_n

    e_e = sum(1 for i in expert_set if expert_values[int(i)] > novice_values[int(i)])
    e_n = len(expert_set) - e_e

    sum_test = e_n + e_e + n_e + n_n
    print("accuracy = {:.4f}".format((e_e + n_n) / sum_test))
    print("expert_accuracy = {:.4f}, novice_accuracy = {:.4f}".format(
        e_e / len(expert_set), n_n / len(novice_set)))
