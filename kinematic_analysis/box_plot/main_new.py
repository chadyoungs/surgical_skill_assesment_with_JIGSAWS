# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:22:56 2020

@author: yumin
"""
from load_and_import_new import *
from analysis_new import *

if __name__ == '__main__':
    #for singal file data

    timeseriesdata = TimeSeriesData ()
    timeseriesdata.choose_file(r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\Suturing')
    timeseriesdata.getMetaData ()
    [feature, label] = timeseriesdata.getKinematicData(r'F:\Projects\surgical_project\da_vici_data_with_iDT_features\AllGestures')

    feature_novice=feature[np.where(label == 0)[0]]
    feature_expert=feature[np.where(label == 2)[0]]
    
    #对每一侧器械轨迹特征画箱式图对比
    analysisexpert_mtfL = TotalData()
    analysisexpert_mtfL.getFile(feature_expert, 'MTF_L')
    analysisexpert_mtfL.total_analysis()
    
    analysisnovice_mtfL = TotalData()
    analysisnovice_mtfL.getFile(feature_novice, 'MTF_L')
    analysisnovice_mtfL.total_analysis()
    
    analysisexpert_mtfR = TotalData()
    analysisexpert_mtfR.getFile(feature_expert, 'MTF_R')
    analysisexpert_mtfR.total_analysis()
    
    analysisnovice_mtfR = TotalData()
    analysisnovice_mtfR.getFile(feature_novice, 'MTF_R')
    analysisnovice_mtfR.total_analysis()
    
    analysisnovice_mtfL.visual_comparison(analysisexpert_mtfL,analysisexpert_mtfR,analysisnovice_mtfR )
    
    analysisexpert_psm1 = TotalData()
    analysisexpert_psm1.getFile(feature_expert, 'PSM_1')
    analysisexpert_psm1.total_analysis()
    
    analysisnovice_psm1 = TotalData()
    analysisnovice_psm1.getFile(feature_novice, 'PSM_1')
    analysisnovice_psm1.total_analysis()
    
    analysisexpert_psm2 = TotalData()
    analysisexpert_psm2.getFile(feature_expert, 'PSM_2')
    analysisexpert_psm2.total_analysis()
    
    analysisnovice_psm2 = TotalData()
    analysisnovice_psm2.getFile(feature_novice, 'PSM_2')
    analysisnovice_psm2.total_analysis()
    
    analysisnovice_psm1.visual_comparison(analysisexpert_psm1,analysisexpert_psm2,analysisnovice_psm2)

