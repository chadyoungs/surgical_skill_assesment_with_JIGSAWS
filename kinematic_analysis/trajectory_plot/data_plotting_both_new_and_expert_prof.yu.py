from mpl_toolkits import mplot3d
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


data_expert = np.loadtxt("F:\Projects\surgical_project\da_vici_data_with_iDT_features\Suturing\kinematics\AllGestures\Suturing_D001.txt")
data_novice = np.loadtxt("F:\Projects\surgical_project\da_vici_data_with_iDT_features\Suturing\kinematics\AllGestures\Suturing_B001.txt")

x_expert_left = data_expert[:,38]
y_expert_left = data_expert[:,39]
z_expert_left = data_expert[:,40]

x_expert_right = data_expert[:,57]
y_expert_right = data_expert[:,58]
z_expert_right = data_expert[:,59]

x_novice_left  = data_novice[:,38]
y_novice_left = data_novice[:,39]
z_novice_left = data_novice[:,40]

x_novice_right = data_novice[:,57]
y_novice_right= data_novice[:,58]
z_novice_right = data_novice[:,59]

ax = plt.axes(projection='3d')
ax.set_title("3D_Position_Curve_Suturing")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

figure_expert_left = ax.plot(x_expert_left,y_expert_left,z_expert_left,c='b',label='expert_PSM1')
#figure_expert_right = ax.plot(x_expert_right,y_expert_right,z_expert_right,c='r',label='expert_PSM2')

figure_novice_left = ax.plot(x_novice_left,y_novice_left,z_novice_left,c='m',label='newhand_PSM1')
#figure_novice_right = ax.plot(x_novice_right,y_novice_right,z_novice_right,c='y',label='newhand_PSM2')

plt.legend()

#plt.show()

plt.savefig('fig_expert_and_novice',dpi=300)
    
