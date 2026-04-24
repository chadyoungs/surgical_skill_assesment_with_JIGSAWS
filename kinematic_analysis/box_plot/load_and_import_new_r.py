

import os
import glob
import numpy as np

import csv

from calculation_new import *


class TimeSeriesData(object):
    def __init__(self):
        self.classmode = ['GRS'] # ['SELFPROCLAIMED']
        self.metaData = {}
        
    def choose_file(self,file_name):
        self.file_name = file_name
        print("loading data name is:",self.file_name)

    def getMetaData(self):        
        count = 0
        for file in glob.glob (self.file_name + '\meta_file_' + '*' + '.txt'):
            for line in open (file, 'r'):
                line = line.strip ()
                if len (line) == 0:
                    break
                b = line.split ()
                surgery_name = b[0]
                skill_level = b[1]
                b = b[2:]
                scores = [int (e) for e in b]
                self.metaData[surgery_name] = (skill_level, scores)

    def getSkillLevel(self, surgery_name):
        if self.metaData.__contains__ (surgery_name):
            if self.classmode[0] == 'GRS':
                score_grs = self.metaData[surgery_name][1][0]

                if surgery_name.__contains__ ('Knot_Tying'):
                    if score_grs <= 15:
                        y = 0   #novice
                    else:
                        y = 2   #expert
                if surgery_name.__contains__ ('Suturing'):
                    if score_grs <= 19:
                        y = 0   #novice
                    else:
                        y = 2   #expert
                elif surgery_name.__contains__ ('Needle_Passing'):
                    if score_grs <= 15:
                        y = 0   #novice
                    elif score_grs > 15 and score_grs < 20:
                        y = 1
                    else:
                        y = 2   #expert
                return y
        return None

    # import raw data and get window slides
    def getKinematicData(self, url):
        dataX = np.zeros ((0, 4, 10))
        dataX = dataX.tolist()
        dataY = np.zeros ((0, 1))

        print ("loading data from url:\t", str (url))
        filelist = glob.glob (url + "\*.txt")  # return a list of all txt files in the directory

        f = open('data_for_SVM.csv','w',encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["time_sum", "displacement_sum_mtfl", "velocity_ave_mtfl", \
                             "velocity_var_mtfl", "curvity_ave_mtfl", "curvity_var_mtfl", \
                             "smooth_ave_mtfl", "smooth_var_mtfl", "turning_ave_mtfl", "turning_var_mtfl",  \
                             "displacement_sum_mtfr", "velocity_ave_mtfr", \
                             "velocity_var_mtfr", "curvity_ave_mtfr", "curvity_var_mtfr", \
                             "smooth_ave_mtfr", "smooth_var_mtfr", "turning_ave_mtfr", "turning_var_mtfr",  \
                             "displacement_sum_psm1", "velocity_ave_psm1", \
                             "velocity_var_psm1", "curvity_ave_psm1", "curvity_var_psm1", \
                             "smooth_ave_psm1", "smooth_var_psm1", "turning_ave_psm1", "turning_var_psm1",  \
                             "displacement_sum_psm2", "velocity_ave_psm2", \
                             "velocity_var_psm2", "curvity_ave_psm2", "curvity_var_psm2", \
                             "smooth_ave_psm2", "smooth_var_psm2", "turning_ave_psm2", "turning_var_psm2", \
                             "class"
                             ])
        
        for file in filelist:
            file_name = os.path.basename(file)  # 'Needle_Passing_H004.txt'
            surgery_name = os.path.splitext(file_name)[0]  # 'Needle_Passing_H004'
            
            y = self.getSkillLevel(surgery_name)
            if y is None:
                continue
            
            print(y)
            # reading kinematic data from a file
            x = np.genfromtxt (file, delimiter='', dtype=np.float32)

           # caculating global movement features
           
            x_calculate = DataCal()
        
            x_calculate.getFile(x,'MTF_L')
            feature_MTF_L= x_calculate.cal_processing()
            x_calculate.getFile(x,'MTF_R')
            feature_MTF_R= x_calculate.cal_processing()
            x_calculate.getFile(x,'PSM_1')
            feature_PSM_1= x_calculate.cal_processing()
            x_calculate.getFile(x,'PSM_2')
            feature_PSM_2= x_calculate.cal_processing()
            
            csv_writer.writerow([str(feature_MTF_L[0]),  str(feature_MTF_L[1]), str(feature_MTF_L[2]), \
                                 str(feature_MTF_L[3]),  str(feature_MTF_L[4]), str(feature_MTF_L[5]), \
                                 str(feature_MTF_L[6]),  str(feature_MTF_L[7]), str(feature_MTF_L[8]), str(feature_MTF_L[9]), \
                                 str(feature_MTF_R[1]),  str(feature_MTF_R[2]), \
                                 str(feature_MTF_R[3]),  str(feature_MTF_R[4]), str(feature_MTF_R[5]), \
                                 str(feature_MTF_R[6]),  str(feature_MTF_R[7]), str(feature_MTF_R[8]), str(feature_MTF_R[9]), \
                                 str(feature_PSM_1[1]),  str(feature_PSM_1[2]), \
                                 str(feature_PSM_1[3]),  str(feature_PSM_1[4]), str(feature_PSM_1[5]), \
                                 str(feature_PSM_1[6]),  str(feature_PSM_1[7]), str(feature_PSM_1[8]), str(feature_PSM_1[9]), \
                                 str(feature_PSM_2[1]),  str(feature_PSM_2[2]), \
                                 str(feature_PSM_2[3]),  str(feature_PSM_2[4]), str(feature_PSM_2[5]), \
                                 str(feature_PSM_2[6]),  str(feature_PSM_2[7]), str(feature_PSM_2[8]), str(feature_PSM_2[9]), \
                                 str(y)
                                 ])
            
            feature=np.vstack ((feature_MTF_L,feature_MTF_R,feature_PSM_1,feature_PSM_2))
            
            feature=feature.tolist()
            dataX.append(feature)
            dataY = np.vstack ((dataY, y))
            
        dataX=np.array(dataX)
        return dataX, dataY
