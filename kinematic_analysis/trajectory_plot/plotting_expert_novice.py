from mpl_toolkits import mplot3d
#matplotlib inline
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import numpy as np
from data_extract_stip import DataExtract

test = DataExtract()
#test.get_score()

data_expert_name, data_novice_name = test.get_score()

task_list = ["Suturing", "Knot_Tying", "Needle_Passing"]

# 0 for suturing
TASK_SYMBOL = 0

data_expert = np.loadtxt(".\da_vici_data_with_iDT_features\{}\kinematics\AllGestures\{}.txt".format(task_list[TASK_SYMBOL], data_expert_name))
data_novice = np.loadtxt(".\da_vici_data_with_iDT_features\{}\kinematics\AllGestures\{}.txt".format(task_list[TASK_SYMBOL], data_novice_name))

position_dict = {"left_MTM": (0, 1, 2),
                 "right_MTM": (19, 20, 21),
                 "PSM1": (38, 39, 40),
                 "PSM2": (57, 58, 59)}
tip_list = ["left_MTM", "right_MTM", "PSM1", "PSM2"]

# j = 0 or j = 2
j = 0
x_expert_left = data_expert[:, position_dict[tip_list[j]][0]]
y_expert_left = data_expert[:, position_dict[tip_list[j]][1]]
z_expert_left = data_expert[:, position_dict[tip_list[j]][2]]
x_novice_left  = data_novice[:, position_dict[tip_list[j]][0]]
y_novice_left = data_novice[:, position_dict[tip_list[j]][1]]
z_novice_left = data_novice[:, position_dict[tip_list[j]][2]]

# i = 1 or i = 3
i = 1
x_expert_right = data_expert[:, position_dict[tip_list[i]][0]]
y_expert_right = data_expert[:, position_dict[tip_list[i]][1]]
z_expert_right = data_expert[:, position_dict[tip_list[i]][2]]
x_novice_right = data_novice[:, position_dict[tip_list[i]][0]]
y_novice_right = data_novice[:, position_dict[tip_list[i]][1]]
z_novice_right = data_novice[:, position_dict[tip_list[i]][2]]

ax = plt.axes(projection='3d')
ax.set_title("3D_trajectories")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

mark = 'l'
if mark == 'l':
    figure_expert_left = ax.plot(x_expert_left,y_expert_left,z_expert_left,c='b',label='expert')
    figure_novice_left = ax.plot(x_novice_left,y_novice_left,z_novice_left,c='r',label='novice')
    
    plt.legend(loc="best")
    #plt.show()

    plt.savefig('3D trajectories of left hand of expert and novice surgeon',dpi=300)
else:
    figure_expert_right = ax.plot(x_expert_right,y_expert_right,z_expert_right,c='b',label='expert')#PSM2
    figure_novice_right = ax.plot(x_novice_right,y_novice_right,z_novice_right,c='r',label='novice')#PSM2

    plt.legend(loc="best")
    #plt.show()

    plt.savefig('3D trajectories of right hand of expert and novice surgeon',dpi=300)
    
