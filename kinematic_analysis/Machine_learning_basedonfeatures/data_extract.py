#! /usr/bin/env python3
'''
Created on Mon Jul 20 17:29:20 2020
@author: xiaoxiaoyang

This program aiming at extract the start and end frame
for each video's iDT features.
'''
import os
from pathlib import Path
import sys
from glob import glob

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import (
    TASKS,
    get_experimental_setup_root,
    get_task_index,
    get_task_root,
)

TASK_SYMBOL = get_task_index()
task_lists = list(TASKS)
root_path = str(get_experimental_setup_root() / "GestureClassification" / "SuperTrialOut")
root_path_score = str(get_task_root())
root_path_trans = str(get_task_root() / "transcriptions")

class DataExtract(object):
    def __init__(self):
        self.metaData = {}
        self.metaData_surgeme = {}
        self.metaData_score = {}
        self.metaData_index = {}
    
    def get_category(self):
         #retrieves directories inside root path
        self.category = glob(root_path + "/" + "*")
        # eliminate absolute path
        self.category_abs = [os.path.basename(i) for i in self.category]
    
    def get_frame_surgeme(self):
        txt_files = glob(os.path.join(root_path_trans, "*.txt"))
        for file in txt_files:
            list_surgeme = []
            list_frameStartNo = []
            list_frameEndNo = []
            
            temp_0 = os.path.basename(file)
            surgery_name = temp_0.split('.')[0]
            for line in open(file, 'r'):
                line = line.strip()
                if len(line) == 0:
                    break
                
                start_frame_No = line.split()[0]
                list_frameStartNo.append(start_frame_No)
                end_frame_No = line.split()[1]
                list_frameEndNo.append(end_frame_No)
                surgeme = line.split()[2]
                list_surgeme.append(surgeme)
                
            self.metaData_surgeme[surgery_name] = (list_frameStartNo, list_frameEndNo, list_surgeme)
        
    # each file's starting and endding frame No.
    def get_frame_No(self):
        for count, c in enumerate(self.category):
            # actually there is only one file under the category, we applied "for" for reusing
            txt_files = glob(os.path.join(c, "itr_1", "*.txt"))
            
            if count >= 1:
                break
            
            for file in txt_files:
                for line in open(file, 'r'):
                    line = line.strip()
                    if len(line) == 0:
                        break
                    # b = line.split()
                    # surgery_name_and_No = b[0]
                    # surgery_score_sum = int(b[1])
                    
                    c = line.split('.')
                
                    start_frame_No = int(c[0].split('_')[-2])
                    end_frame_No = int(c[0].split('_')[-1])
                    
                    c_temp = c[0].split('_')
                    list.reverse(c_temp)
                    c_temp_1 = [x+"_"  for x in c_temp[2:]]
                    list.reverse(c_temp_1)
                    
                    surgery_name = "".join(c_temp_1)[:-1] 
                    #scores_grade = self.skill_level(surgery_name, surgery_score_sum)
                
                    self.metaData[surgery_name] = (start_frame_No, end_frame_No)
    
    # each file's score
    def get_score(self):
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, "meta_file_" + category + ".txt")
        
        for line in open(txt_file, 'r'):
            line = line.strip()
            if len(line) == 0:
                break
            b = line.split()
            surgery_name = b[0]
            surgery_score_sum = int(b[2])
                    
            scores_grade = self.skill_level(surgery_name, surgery_score_sum)
                
            self.metaData_score[surgery_name] = (surgery_score_sum, scores_grade)
        
    def skill_level(self, surgery_name, score):
        if surgery_name.__contains__('Knot_Tying'):
            if score <= 15:
                y = 0 # novice
            else:
                y = 2 # expert
                    
        elif surgery_name.__contains__('Suturing'):
            if score <= 19:
                y = 0 # novice
            else:
                y = 2 # expert
                
        elif surgery_name.__contains__('Needle_Passing'):
            if score <= 15:
                y = 0 # novice
            else:
                y = 2 # expert
        
        return y
    
    # only for ML train and test process
    def get_txt_index(self):
        count = 0
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, "meta_file_" + category + ".txt")
        
        for line in open(txt_file, 'r'):
            line = line.strip()
            if len(line) == 0:
                break
            b = line.split()
            surgery_name = b[0]
                
            self.metaData_index[surgery_name] = count
            count += 1
    
    # ML train list depends on LOSO        
    def train_sum(self):
        txt_files_train_names = []
        
        for c in self.category:
            # actually there is only one file under the category, we applied "for" for reusing
            txt_files_train = os.path.join(c, "itr_1", "Train.txt")
            txt_files_train_name = []
            
            with open(txt_files_train, 'r') as f:
                for line in f:
                    data = line.split()
                    
                    temp = data[0].split('_')
                    list.reverse(temp)
                    temp_1 = [x+"_"  for x in temp[2:]]
                    list.reverse(temp_1)
                    train_file_name = "".join(temp_1)[:-1]
                    
                    txt_files_train_name.append(train_file_name)
                
            txt_files_train_names.append(txt_files_train_name)
            
        return txt_files_train_names    
    
    # ML test list depends on LOSO 
    def test_sum(self):
        txt_files_test_names = []
        
        for c in self.category:
            # actually there is only one file under the category, we applied "for" for reusing
            txt_files_test = os.path.join(c, "itr_1", "Test.txt")
            txt_files_test_name = []
            
            with open(txt_files_test, 'r') as f:
                for line in f:
                    data = line.split()
                    
                    temp = data[0].split('_')
                    list.reverse(temp)
                    temp_1 = [x+"_"  for x in temp[2:]]
                    list.reverse(temp_1)
                    test_file_name = "".join(temp_1)[:-1]
                    
                    txt_files_test_name.append(test_file_name)
                
            txt_files_test_names.append(txt_files_test_name)
        return txt_files_test_names
    
    def print_test(self):
        '''for test'''
        print(self.metaData["Suturing_E004"][0])
    
'''
# for test
if __name__ == "__main__":
    test = DataExtract()
    #test.get_category()
    #test.get_frame_No()
    #test.get_score()
    #test.print_test()
    #test.test_sum()
    test.get_frame_surgeme()
'''