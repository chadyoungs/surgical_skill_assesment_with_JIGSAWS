
import os
import glob
import numpy as np

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
        metadata_files = [
            file
            for file in glob.glob(os.path.join(self.file_name, "meta_file_*"))
            if file.endswith(".txt")
        ]
        for file in metadata_files:
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
                    elif score_grs > 15 and score_grs < 19:
                        y = 1
                    else:
                        y = 2   #expert
                if surgery_name.__contains__ ('Suturing'):
                    if score_grs <= 15:
                        y = 0   #novice
                    elif score_grs > 15 and score_grs < 19:
                        y = 1
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
        filelist = glob.glob(os.path.join(url, "*.txt"))  # return a list of all txt files in the directory
        for file in filelist:
            file_name = os.path.basename(file)  # 'Needle_Passing_H004.txt'
            surgery_name = os.path.splitext(file_name)[0]  # 'Needle_Passing_H004'
            y = self.getSkillLevel(surgery_name)
            if y is None:
                continue

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

            feature=np.vstack ((feature_MTF_L,feature_MTF_R,feature_PSM_1,feature_PSM_2))
            
            feature=feature.tolist()
            dataX.append(feature)
            dataY = np.vstack ((dataY, y))
            
        dataX=np.array(dataX)
        return dataX, dataY
