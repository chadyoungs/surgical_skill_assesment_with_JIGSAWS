#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 01:06:45 2020
@author: xiaoxiaoyang
"""
import os
from pathlib import Path
import sys
import numpy as np
import glob
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import Global_Var

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import get_data_root


def load_data(path, surgeme):
    txt_files = glob.glob(os.path.join(path, '*.txt'))
    X_MTM_L_list = []
    X_MTM_R_list = []
    X_PSM_1_list = []
    X_PSM_2_list = []

    for i in txt_files:
        surgery_name = os.path.splitext(os.path.basename(i))[0]
        if surgeme in surgery_name:
            data = np.loadtxt(i, dtype=np.float64)
            X_MTM_L_list.append(data[:, 0:19])
            X_MTM_R_list.append(data[:, 19:38])
            X_PSM_1_list.append(data[:, 38:57])
            X_PSM_2_list.append(data[:, 57:])

    if not X_MTM_L_list:
        raise ValueError("No data files found matching surgeme '{}' in {}".format(surgeme, path))
    return (np.vstack(X_MTM_L_list), np.vstack(X_MTM_R_list),
            np.vstack(X_PSM_1_list), np.vstack(X_PSM_2_list))


def PCA_trans(data):
    pca = PCA(n_components=Global_Var.PCA_COMPONENTS, whiten=True, random_state=0).fit(data)
    return pca.transform(data)


if __name__ == '__main__':
    N_CLUSTERS = Global_Var.N_CLUSTERS

    # 0 set as Suturing
    TASK_SYMBOL = Global_Var.TASK_SYMBOL
    task_list = ["Suturing", "Knot_Tying", "Needle_Passing"]

    split_files_path = os.path.join(
        get_data_root(),
        'Data_clips',
        task_list[TASK_SYMBOL] + '_clips',
        'kinetic',
    )

    if TASK_SYMBOL == 0 or TASK_SYMBOL == 2:
        # For suturing and needle passing
        surgeme_list = ["_G1_", "_G2_", "_G3_", "_G4_", "_G5_",
                        "_G6_", "_G8_", "_G9_", "_G10_", "_G11_"]
        gestures = ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"]
    else:
        # For Knot_Tying
        surgeme_list = ["_G1_", "_G11_", "_G12_", "_G13_", "_G14_", "_G15_"]
        gestures = ["G1", "G11", "G12", "G13", "G14", "G15"]

    save_path_list = [
        os.path.join('.', 'observation_clusters', task_list[TASK_SYMBOL], g + '.pkl')
        for g in gestures
    ]

    for surgeme, save_path in zip(surgeme_list, save_path_list):
        X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2 = load_data(split_files_path, surgeme)
        data_result = np.float64(np.hstack((PCA_trans(X_MTM_L), PCA_trans(X_MTM_R),
                                            PCA_trans(X_PSM_1), PCA_trans(X_PSM_2))))
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(data_result)
        joblib.dump(kmeans, save_path)

        print("Done. Dumping to", os.path.basename(save_path))
        print("####################\n")
