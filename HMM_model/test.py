# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:00:29 2020
@author: xiaoxiaoyang
"""
import numpy as np
import joblib

from data_extract import DataExtract

import Global_Var


def hmm_forward(A, B, pi, O):
    T = len(O)
    N = len(A[0])
    
    #initialize
    alpha = [[0]*T for _ in range(N)]
    
    for i in range(N):
        temp_sum = 0
        for j in range(len(O[0])):
            temp_sum += pi[i]*B[i][O[0][j]]
        alpha[i][0] = temp_sum / len(O[0])
        #alpha[i][0] = pi[i]*B[i][O[0]]
        #print(alpha[i][0])

    #calculate alpha(t)
    for t in range(1,T):
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += alpha[j][t-1]*A[j][i]
                #print(temp)
            temp_sum = 0
            for k in range(len(O[t])):
                temp_sum += temp*B[i][O[t][k]]   
            #alpha[i][t] = temp*B[i][O[t]]
            alpha[i][t] = temp_sum / len(O[t])
            
    #step3
    proba = 0
    for i in range(N):
        proba += alpha[i][-1]
        #print(proba)
    return proba, alpha

def hmm_backward(A, B, pi, O):
    T = len(O)
    N = len(A[0])
    
    #initialize
    beta = [[0]*T for _ in range(N)]
    for i in range(N):
        beta[i][-1] = 1
        
    #step2
    for t in reversed(range(T-1)):
        for i in range(N):
            for j in range(N):
                beta[i][t]  += A[i][j]*B[j][O[t+1]]*beta[j][t+1]
            
    #step3
    proba = 0
    for i in range(N):
        proba += pi[i]*B[i][O[0]]*beta[i][0]
    return proba,beta


if __name__ == '__main__':
    
    task_list = ['Suturing', 'Knot_Tying', 'Needle_Passing']
    # 0 for suturing
    TASK_SYMBOL = Global_Var.TASK_SYMBOL
    
    hmm_expert_model = joblib.load(r'.\models\{}_expert_model.pkl'.format(task_list[TASK_SYMBOL]))
    hmm_novice_model = joblib.load(r'.\models\{}_novice_model.pkl'.format(task_list[TASK_SYMBOL]))
    obser = joblib.load(r'.\observations\{}_observations.pkl'.format(task_list[TASK_SYMBOL]))
    
    test = DataExtract()
    test.get_category()
    test.get_txt_index()  
    test.get_score()
    test.get_index()
    
    novice_values = []
    expert_values = []
    # calulate each observation's prop to every hmm models include itself
    for i in range(len(obser)):
        prob_expert = hmm_forward(hmm_expert_model.transmat_, hmm_expert_model.emissionprob_, 
                hmm_expert_model.startprob_, obser[i])
        prob_novice = hmm_forward(hmm_novice_model.transmat_, hmm_novice_model.emissionprob_, 
                hmm_novice_model.startprob_, obser[i])
        novice_values.append(prob_novice[0])
        expert_values.append(prob_expert[0])
        
    # novice and expert set   
    novice_set = test.metaData_hmmindex["novice"]
    expert_set = test.metaData_hmmindex["expert"]
        
    n_n = 0
    n_e = 0
    for i in novice_set:
        if novice_values[int(i)] > expert_values[int(i)]:
            n_n += 1
        else:
            n_e += 1
    
    e_n = 0
    e_e = 0
    for i in expert_set:
        if expert_values[int(i)] > novice_values[int(i)]:
            e_e += 1
        else:
            e_n += 1
    
    sum_test = e_n + e_e + n_e + n_n
    print("accuracy = {:.4f}".format((e_e + n_n)/sum_test))
    print("expert_accuracy = {:.4f}, novice_accuracy = {:.4f}".format(e_e/len(expert_set), n_n/len(novice_set)))      
           
        
               
       
    
    
    
    
