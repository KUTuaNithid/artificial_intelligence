# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:23:04 2021

@author: Nithid Mahattanasin
"""
from my_lib import *

#############   ENERGY   #############
######################################
#######         TASK 3         #######
############### START! ###############
x_arr = np.array([1, 1, 1, 1])

w_arr = np.array([[0, 1, 1, 1], # 0 
                  [0, 0, -2, -2], # 1
                  [0, -2, 0, -2], # 2 
                  [0, -2, -2, 0],
                  ])# 3
test = my_rnn(x_arr, w_arr, "test", 1000, 3)
# ans_arr = test.train_to_converg()
ans_arr, energy = test.train_to_converg("deter")
test.report_energy_short(energy, test.n)
print("A Converged state is", ans_arr[1:])

# x_arr = np.array([1, 1, 1, 1, 1, 1])

# w_arr = np.array([[0, 1, 1, 1, 1, 1], # 0 
#                   [0, 0, -2, -2, -2, -2], # 1
#                   [0, -2, 0, -2, -2, -2], # 2 
#                   [0, -2, -2, 0, -2, -2],
#                   [0, -2, -2, -2, 0, -2],
#                   [0, -2, -2, -2, -2, 0]
#                   ])# 3
# test = my_rnn(x_arr, w_arr, "test", 1000, 5)
# # ans_arr = test.train_to_converg()
# ans_arr, energy = test.train_to_converg("deter")
# test.report_energy_short(energy, test.n)
# print("A Converged state is", ans_arr[1:])

# x_arr = np.array([1, 1, 1, 1, 1, 1])

# w_arr = np.array([[0, 3, 3, 4, 5, 2], # 0 
#                   [0,  0,  2, -5, -2, -2], # 1
#                   [0,  2,  0, -1, -2, -2], # 2 
#                   [0, -5, -1,  0, -4, -2],
#                   [0, -2, -2, -4,  0, -2],
#                   [0, -2, -2, -2, -2,  0]
#                   ])# 3
# test = my_rnn(x_arr, w_arr, "test", 1000, 5)
# # ans_arr = test.train_to_converg()
# ans_arr, energy = test.train_to_converg("deter")
# test.report_energy_short(energy, test.n)
# print("A Converged state is", ans_arr[1:])
######################################
#######         TASK 3         #######
################ END! ################
