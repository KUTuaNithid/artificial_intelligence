# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:19:34 2021

@author: Nithid Mahattanasin
"""

from my_lib import *

w_arr = np.array([[0, 1, 1, 1], # 0 
                  [0, 0, -2, -2], # 1
                  [0, -2, 0, -2], # 2 
                  [0, -2, -2, 0]])# 3
x_arr = np.array([1, 0, 0, 0])
test = my_rnn(x_arr, w_arr, "test", 1000, 3)
ans_arr, energy = test.train_to_converg()
# ans_arr = test.train_to_converg("deter")
print("A Converged state is", ans_arr[1:])

x_arr = np.array([1, 1, 1, 1])
test = my_rnn(x_arr, w_arr, "test", 1000, 3)
ans_arr, energy = test.train_to_converg()
# ans_arr = test.train_to_converg("deter")
print("A Converged state is", ans_arr[1:])

x_arr = np.array([1, 0, 0, 0])
test = my_rnn(x_arr, w_arr, "test", 1000, 3)
ans_arr, energy = test.train_to_converg()
# ans_arr = test.train_to_converg("deter")
print("A Converged state is", ans_arr[1:])

x_arr = np.array([1, 0, 0, 0])
freq_tab = test.run_trail(0.2, 100)
test.report_freq(freq_tab, test.n)

freq_tab = test.run_trail(0.5, 100)
test.report_freq(freq_tab, test.n)

freq_tab = test.run_trail(1.0, 100)
test.report_freq(freq_tab, test.n)
