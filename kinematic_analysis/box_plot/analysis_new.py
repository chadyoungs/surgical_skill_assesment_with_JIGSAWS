# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:51:02 2020

@author: yumin
"""
import numpy as np
import math

import os

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


class TotalData():

    def getFile(self,x,choose):
        self.description_choose=choose
        if self.description_choose=='MTF_L':
            self.feature = x[:,0,:]
        elif self.description_choose=='MTF_R':
            self.feature = x[:,1,:]
        elif self.description_choose=='PSM_1':
            self.feature = x[:,2,:]
        elif self.description_choose=='PSM_2':
            self.feature = x[:,3,:]

    def total_analysis(self):
        '''average_calculation'''
        self.total_time_sum=self.feature[:,0]
        self.total_displacement_sum=self.feature[:,1]
        self.total_v_average=self.feature[:,2]
        self.total_v_variance=self.feature[:,3]
        self.total_curvity_average=self.feature[:,4]
        self.total_curvity_variance=self.feature[:,5]
        self.total_smoothness_average=self.feature[:,6]
        self.total_smoothness_variance=self.feature[:,7]
        self.total_Turning_angle_average=self.feature[:,8]
        self.total_Turning_angle_variance=self.feature[:,9]       

    def visual_comparison(self,other_data, data, data_other_data):
        
        data_t = [other_data.total_time_sum, self.total_time_sum]
        data_p = [other_data.total_displacement_sum, self.total_displacement_sum, \
                  data.total_displacement_sum, data_other_data.total_displacement_sum ]
        data_v = [other_data.total_v_average, self.total_v_average,\
                  data.total_v_average, data_other_data.total_v_average ]
        data_v_var = [other_data.total_v_variance, self.total_v_variance, \
                      data.total_v_variance, data_other_data.total_v_variance]
        data_cur_ave = [other_data.total_curvity_average, self.total_curvity_average, \
                        data.total_curvity_average, data_other_data.total_curvity_average ]
        data_cur_var = [other_data.total_curvity_variance, self.total_curvity_variance, \
                        data.total_curvity_variance, data_other_data.total_curvity_variance ]
        data_smooth_ave = [other_data.total_smoothness_average, self.total_smoothness_average, \
                           data.total_smoothness_average, data_other_data.total_smoothness_average ]
        data_smooth_var = [other_data.total_smoothness_variance, self.total_smoothness_variance, \
                           data.total_smoothness_variance, data_other_data.total_smoothness_variance ]
        data_Turning_angle_ave = [other_data.total_Turning_angle_average, self.total_Turning_angle_average, \
                                  data.total_Turning_angle_average, data_other_data.total_Turning_angle_average ]
        data_Turning_angle_var = [other_data.total_Turning_angle_variance, self.total_Turning_angle_variance, \
                                  data.total_Turning_angle_variance, data_other_data.total_Turning_angle_variance ]

        plot_data = [data_t, data_p, data_v, data_v_var, data_cur_ave, \
                     data_cur_var, data_smooth_ave, data_smooth_var, \
                     data_Turning_angle_ave,data_Turning_angle_var]

        plot_title = ['Total time', 'Total dispalcement', 'Velocity_mean', 'Velocity_variance',\
                      'Curvity_mean', 'Curvity_variance', 'Smoothness_mean','Smoothness_variance',\
                      'Turning_angle_mean', 'Turning_angle_variance']
        
        fig,axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        #ax = axs.ravel()
        #figure_decorate
        font_dict={'fontsize':10,\
                   'fontweight':20
                  }
        pad_setting = 11.5
        
        labels = ['Exp', 'Nov', 'Exp','Nov']
        labels_time_plot = ['Exp', 'Nov']
        
        # plotting
        boxplot_count = 0
        # rectangular box plot
        for i in range(2):
            for j in range(5):
                if i == 0 and j == 0:
                    axs[i, j].boxplot(plot_data[boxplot_count],vert=True,labels=labels_time_plot)  # vertical box alignment
                    axs[i, j].set_title(plot_title[boxplot_count],fontdict=font_dict, pad=pad_setting)
                else:    
                    axs[i, j].boxplot(plot_data[boxplot_count],vert=True,labels=labels)  # vertical box alignment
                    axs[i, j].set_title(plot_title[boxplot_count],fontdict=font_dict, pad=pad_setting)
                    axs[i, j].set_xlabel('L               R')
                # adding horizontal grid lines
                axs[i, j].yaxis.grid(True)
                axs[i, j].yaxis.get_major_formatter().set_powerlimits((0,2)) 
                axs[i, j].set_ylabel('values')
                
                boxplot_count += 1
        '''
        # fill with colors
        colors = ['pink', 'lightblue']

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
                 xticklabels=['Exp', 'Nov'])
        '''
        fig.suptitle(self.description_choose.split('_')[0])
        plt.show()
