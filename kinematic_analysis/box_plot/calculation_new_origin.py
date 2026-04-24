#! usr/bin/env python3

import numpy as np
import math

import os

import matplotlib.pyplot as plt


class DataCal:

    def getFile(self,x,choose):
        self.data = x
        self.description_choose=choose
    
    def mov_determine(self):
   
        #data frequecy and timestamp
        self.frequency = 30
        self.timestamp = 1/self.frequency

        #move_start_judgement
        #mov_start_judge = False
        self.move_start_moment = 0
        
        #move_stop_judgement
        #move_stop_judge = False
        self.move_stop_moment = 0
    
        #extract position data

        mtf_l_x = self.data[:,0]
        mtf_l_y = self.data[:,1]
        mtf_l_z = self.data[:,2]
        
        mtf_r_x = self.data[:,19]
        mtf_r_y = self.data[:,20]
        mtf_r_z = self.data[:,21]

        psm1_x = self.data[:,38]
        psm1_y = self.data[:,39]
        psm1_z = self.data[:,40]
        
        psm2_x = self.data[:,57]
        psm2_y = self.data[:,58]
        psm2_z = self.data[:,59]
        
        if self.description_choose=='MTF_L':
            self.des_x=mtf_l_x
            self.des_y=mtf_l_y
            self.des_z=mtf_l_z
        elif self.description_choose=='MTF_R':
            self.des_x=mtf_r_x
            self.des_y=mtf_r_y
            self.des_z=mtf_r_z
        elif self.description_choose=='PSM_1':
            self.des_x=psm1_x
            self.des_y=psm1_y
            self.des_z=psm1_z
        elif self.description_choose=='PSM_2':
            self.des_x=psm2_x
            self.des_y=psm2_y
            self.des_z=psm2_z

        #bulid the displacement list
        self.p_var_x = []
        self.p_var_y = []
        self.p_var_z = []
        self.p_var = []

        #rows number of data
        data_lines = self.data.shape[0]- 1
        #reverse
        data_lines_reverse = - (data_lines + 1)
 
        #displacement_calulation
        for i in range(data_lines):
            #current moment
            p_var_x_num = self.des_x[i+1] - self.des_x[i]
            p_var_y_num = self.des_y[i+1] - self.des_y[i]
            p_var_z_num = self.des_z[i+1] - self.des_z[i]
            p_var_num = math.sqrt(p_var_x_num**2 +p_var_y_num**2 +p_var_z_num**2)
        
            i += 1
            #axis_x.append(i)
            self.p_var_x.append(p_var_x_num)
            self.p_var_y.append(p_var_y_num)
            self.p_var_z.append(p_var_z_num)
            self.p_var.append(p_var_num)
    
        #move_start_judgement
        for i in range(data_lines):
            if abs(self.p_var_x[i]) > 1e-05 and abs(self.p_var_y[i]) > 1e-05 and abs(self.p_var_z[i]) > 1e-05:
                i += 1
                # move_start_recording
                self.move_start_moment = i
                self.move_start_time = self.move_start_moment * self.timestamp
                break

        #move_stop_judgement
        for i in range(-1,data_lines_reverse,-1):
            if abs(self.p_var_x[i]) > 1e-05 and abs(self.p_var_y[i]) > 1e-05 and abs(self.p_var_z[i]) > 1e-05:
               i -= 1
               # move_stop_recording
               self.move_stop_moment = (data_lines + 1) + i
               self.move_stop_time = self.move_stop_moment * self.timestamp
               break

        #move_data_sum
        #move_stop_moment += 1
        self.move_data_x_sum = []
        self.move_data_y_sum = [] 
        self.move_data_z_sum = []
        self.move_data_sum = []

        for i in range(self.move_start_moment,self.move_stop_moment,1):
            self.move_data_x_sum.append(self.p_var_x[i])
            self.move_data_y_sum.append(self.p_var_y[i])
            self.move_data_z_sum.append(self.p_var_z[i])
            self.move_data_sum.append(self.p_var[i])

    
    def time_cal(self):
        self.time_moments_sum = self.move_stop_moment - self.move_start_moment
        self.time_sum = self.time_moments_sum*self.timestamp



    def p_displacement_sum_cal(self):
        #sum of displacement calculation
        p_sum_x = sum(self.move_data_x_sum)
        p_sum_y = sum(self.move_data_y_sum)
        p_sum_z = sum(self.move_data_z_sum)

        self.p_sum = sum(self.move_data_sum)


    def vel_average_cal(self):
        #velocity calculation between every two timesteps
        self.v_var_x = [x/self.timestamp for x in self.move_data_x_sum]
        self.v_var_y = [y/self.timestamp for y in self.move_data_y_sum]
        self.v_var_z = [z/self.timestamp for z in self.move_data_z_sum]
        self.v_var = [s/self.timestamp for s in self.move_data_sum]

        #velocity calculation for whole process
        v_average_x = np.mean(self.v_var_x)
        v_average_y = np.mean(self.v_var_y)
        v_average_z = np.mean(self.v_var_z)
        self.v_average = np.mean(self.v_var)

        #velocity variance calculation for whole process
        v_variance_x = np.var(self.v_var_x)
        v_variance_y = np.var(self.v_var_y)
        v_variance_z = np.var(self.v_var_z)
        self.v_variance = np.var(self.v_var)
    

    def acc_cal(self):
        self.acc_data_x = np.gradient(self.v_var_x, self.timestamp)
        self.acc_data_y = np.gradient(self.v_var_y, self.timestamp)
        self.acc_data_z = np.gradient(self.v_var_z, self.timestamp)
        
        self.acc_data = [(self.acc_data_x[s]**2 + self.acc_data_y[s]**2 + self.acc_data_z[s]**2)**0.5  for s in range(self.time_moments_sum)]
    
    def curvity_cal(self):
        #add curvity list
        curvity = []

        for i in range(self.time_moments_sum):
            self.v_vector = []
            self.a_vector = []

            #denfine the velocity vector
            self.v_vector.append(self.v_var_x[i])
            self.v_vector.append(self.v_var_y[i])
            self.v_vector.append(self.v_var_z[i])

            #define the accelerate vector
            self.a_vector.append(self.acc_data_x[i])
            self.a_vector.append(self.acc_data_y[i])
            self.a_vector.append(self.acc_data_z[i])

            #calculation the upper and lower part of curvity calculate equation          

            up_cross = np.cross(self.v_vector,self.a_vector)
            up_nomal = np.linalg.norm(up_cross)
            down_nomal = np.linalg.norm(self.v_vector)
            down = down_nomal**3    
            
            if down ==0:
                cur=0
            else:
                cur=up_nomal/down      
            
            curvity.append(cur)

        self.curvity_average = np.mean(curvity)
        self.curvity_variance = np.var(curvity)

        
    def smoothness_cal(self):
        smoothness_x = np.gradient(self.acc_data_x, self.timestamp) 
        smoothness_y = np.gradient(self.acc_data_y, self.timestamp)
        smoothness_z = np.gradient(self.acc_data_z, self.timestamp)
        
        self.smoothness = [(smoothness_x[s]**2 + smoothness_y[s]**2 + smoothness_z[s]**2)**0.5  for s in range(self.time_moments_sum)]

        # smoothness average value calculation for whole process
        smoothness_average_x = np.mean(smoothness_x)
        smoothness_average_y = np.mean(smoothness_y)
        smoothness_average_z = np.mean(smoothness_z)
        
        self.smoothness_average = np.mean(self.smoothness)
        self.smoothness_variance = np.var(self.smoothness)

    def Turning_angle_cal(self):
        Turning_angle = []
        self.u_vector = []

        for i in range(self.time_moments_sum):

            self.u_vector.append((self.move_data_x_sum[i],self.move_data_y_sum[i],self.move_data_z_sum[i]))

            #calculation the upper and lower part of curvity calculate equation          
        for i in range(self.time_moments_sum-1):
            up_dot = np.dot(self.u_vector[i],self.u_vector[i+1])
            down1_nomal = np.linalg.norm(self.u_vector[i])
            down2_nomal = np.linalg.norm(self.u_vector[i+1])
            if down1_nomal*down2_nomal==0:
                Tur=1
            else: 
                Tur = up_dot/(down1_nomal*down2_nomal)

            if Tur>1:
                Tur = 1
            elif Tur<-1:
                Tur = -1
                
            Tur_angle = math.acos(Tur)
            Turning_angle.append(Tur_angle)

        self.Turning_angle_average = np.mean(Turning_angle)
        self.Turning_angle_variance = np.var(Turning_angle)

    def cal_processing(self):

        self.mov_determine()
                
        self.time_cal()
        self.p_displacement_sum_cal()

        self.vel_average_cal()

        self.acc_cal()

        self.curvity_cal()
        self.smoothness_cal()
        self.Turning_angle_cal()
        
        return self.time_sum, self.p_sum, \
        self.v_average, self.v_variance, \
        self.curvity_average, self.curvity_variance,\
        self.smoothness_average, self.smoothness_variance,\
        self.Turning_angle_average, self.Turning_angle_variance
