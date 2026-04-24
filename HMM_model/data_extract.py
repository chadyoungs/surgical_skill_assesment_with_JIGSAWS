#! /usr/bin/env python3
'''
Created on Mon Jul 20 17:29:20 2020
@author: xiaoxiaoyang

This program aiming at extract the start and end frame
for each video's iDT features.
'''
import os
from glob import glob
import Global_Var

task_list = ['Suturing', 'Knot_Tying', 'Needle_Passing']
# 0 for suturing
TASK_SYMBOL = Global_Var.TASK_SYMBOL

# remind that the root_path must point to SuperTrialOut folder!!!
root_path = os.path.join('.', 'da_vici_data_with_iDT_features', 'Experimental_setup',
                         task_list[TASK_SYMBOL], 'unBalanced', 'SkillDetection', 'SuperTrialOut')
root_path_score = os.path.join('.', 'da_vici_data_with_iDT_features', task_list[TASK_SYMBOL])
root_path_trans = os.path.join('.', 'da_vici_data_with_iDT_features', task_list[TASK_SYMBOL], 'transcriptions')


def _surgery_name_from_stem(stem):
    """Extract surgery name by dropping the last two '_'-separated tokens (frame numbers)."""
    return '_'.join(stem.split('_')[:-2])


class DataExtract(object):
    def __init__(self):
        self.metaData = {}
        self.metaData_surgeme = {}
        self.metaData_score = {}
        self.metaData_index = {}
        self.metaData_hmmstates = {}
        self.metaData_hmmindex = {}

    def get_category(self):
        # retrieves directories inside root path
        self.category = glob(root_path + "/" + "*")
        # eliminate absolute path
        self.category_abs = [os.path.basename(i) for i in self.category]

    def get_frame_surgeme(self):
        txt_files = glob(os.path.join(root_path_trans, '*.txt'))
        for file in txt_files:
            list_surgeme = []
            list_frameStartNo = []
            list_frameEndNo = []

            surgery_name = os.path.splitext(os.path.basename(file))[0]
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        break
                    parts = line.split()
                    list_frameStartNo.append(parts[0])
                    list_frameEndNo.append(parts[1])
                    list_surgeme.append(parts[2])

            self.metaData_surgeme[surgery_name] = (list_frameStartNo, list_frameEndNo, list_surgeme)

    def get_hmm_states(self):
        txt_files = glob(os.path.join(root_path_trans, '*.txt'))
        for file in txt_files:
            surgery_name = os.path.splitext(os.path.basename(file))[0]
            starts = self.metaData_surgeme[surgery_name][0]
            ends = self.metaData_surgeme[surgery_name][1]
            surgemes = self.metaData_surgeme[surgery_name][2]

            for i in range(int(ends[-1])):
                frame_No = i + 1
                for j in range(len(starts)):
                    if int(starts[j]) <= frame_No <= int(ends[j]):
                        self.metaData_hmmstates[frame_No] = surgemes[j]

    # each file's starting and ending frame No.
    def get_frame_No(self):
        for count, c in enumerate(self.category):
            # actually there is only one file under the category, we applied "for" for reusing
            txt_files = glob(os.path.join(c, 'itr_1', '*.txt'))

            if count >= 1:
                break

            for file in txt_files:
                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0:
                            break

                        stem = line.split('.')[0]
                        parts = stem.split('_')
                        start_frame_No = int(parts[-2])
                        end_frame_No = int(parts[-1])
                        surgery_name = _surgery_name_from_stem(stem)

                        self.metaData[surgery_name] = (start_frame_No, end_frame_No)

    # each file's score
    def get_score(self):
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, 'meta_file_' + category + '.txt')

        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    break
                b = line.split()
                surgery_name = b[0]
                surgery_score_sum = int(b[2])

                scores_grade = self.skill_level(surgery_name, surgery_score_sum)
                self.metaData_score[surgery_name] = (surgery_score_sum, scores_grade)

    def skill_level(self, surgery_name, score):
        if 'Knot_Tying' in surgery_name:
            return 0 if score <= 15 else 2  # novice / expert
        elif 'Suturing' in surgery_name:
            return 0 if score <= 19 else 2
        elif 'Needle_Passing' in surgery_name:
            return 0 if score <= 15 else 2
        else:
            raise ValueError("Unrecognised task in surgery name: {}".format(surgery_name))

    # only for HMM test process
    def get_index(self):
        '''get the index of train/test set'''
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, 'meta_file_' + category + '.txt')

        novice_list = []
        expert_list = []
        with open(txt_file, 'r') as f:
            for count, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    break
                surgery_name = line.split()[0]

                if self.metaData_score[surgery_name][1] == 0:
                    novice_list.append(count)
                else:
                    expert_list.append(count)

        self.metaData_hmmindex["novice"] = novice_list
        self.metaData_hmmindex["expert"] = expert_list

    # only for ML train and test process
    def get_txt_index(self):
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, 'meta_file_' + category + '.txt')

        with open(txt_file, 'r') as f:
            for count, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    break
                surgery_name = line.split()[0]
                self.metaData_index[surgery_name] = count

    def _read_split_file(self, split_file):
        """Parse a Train.txt or Test.txt split file and return surgery names."""
        names = []
        with open(split_file, 'r') as f:
            for line in f:
                stem = line.split()[0]
                names.append(_surgery_name_from_stem(stem))
        return names

    # ML train list depends on LOSO
    def train_sum(self):
        return [self._read_split_file(os.path.join(c, 'itr_1', 'Train.txt'))
                for c in self.category]

    # ML test list depends on LOSO
    def test_sum(self):
        return [self._read_split_file(os.path.join(c, 'itr_1', 'Test.txt'))
                for c in self.category]

    def print_test(self):
        '''for test'''
        print(self.metaData["Suturing_E004"][0])


'''
# for test
if __name__ == "__main__":
    test = DataExtract()
    test.get_category()
    test.get_frame_No()
    test.get_score()
    #test.print_test()
    test.test_sum()
    test.get_frame_surgeme()
    test.get_hmm_states()
    test.get_index()
'''
