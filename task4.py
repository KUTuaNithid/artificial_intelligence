# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:58:04 2021

@author: Nithid Mahattanasin
"""

from my_lib import *

######  !ENERGY + PROB MODEL!  #######
######################################
#######         TASK 4         #######
############### START! ###############
trials = 1000
copies = 1000
a = 0.5
x_arr = np.array([1, 1, 1, 1, 1, 1])

w_arr = np.array([[0, 0.5, 0.5, 0.5, 0.5, 0.5], # 0 
                  [0.5, 0, -2, -2, -2, -2], # 1
                  [0.5, -2, 0, -2, -2, -2], # 2 
                  [0.5, -2, -2, 0, -2, -2],
                  [0.5, -2, -2, -2, 0, -2],
                  [0.5, -2, -2, -2, -2, 0]
                  ])# 3
# x_arr = np.array([1, 1, 1, 1])

# w_arr = np.array([[0, 1, 1, 1], # 0 
#                   [0, 0, -2, -2], # 1
#                   [0, -2, 0, -2], # 2 
#                   [0, -2, -2, 0]
#                   ])# 3

test_list = []
for i in range(copies):
    test_list.append(my_rnn(x_arr, w_arr, "Copy{}".format(i), a, (x_arr.size-1)))

target = 1
energy_tab = np.zeros(pow(2,test_list[0].n)) # Therotical
freq_tab = np.zeros(pow(2,test_list[0].n))   # Experimental

# Calculate energy of all possible state
for i in range(2**(test_list[0].n)):
    # Size of array is n + 1(dummy)
    # i is [current state] + [1 in LSB for dummy]. Dummy always 1
    # Use np.base_repr to create string of binary to represent n neuron in each state
    # in_arr[0] dummy
    # in_arr[1] neuron#1
    # ..
    # in_arr[n] neuron#n
    in_arr = np.base_repr(i+pow(2,test_list[0].n), base=2) # Create array of base 2 format
                                                           # pow(2,test_list[0].n) to make LSB always 1
    energy_tab[i] = test_list[0].get_energy(in_arr, test_list[0].w_arr, test_list[0].n)

mornitor_tab = [] # To check equilibrium
target = 1
# Train for 100 times
for i in range(trials):
    # State selection
    if target > test_list[0].n:
        target = 1
    add = np.zeros(pow(2,test_list[0].n))
    # Train all copies following selected state
    for test in test_list:
        # Get current state of each neuron and count up before state changing
        add[test.get_result(test.x_arr,test.n)] += 1
        test.update_state(target)
    mornitor_tab.append(add)
    target += 1

# Check freq of each state after trainning 100 times
for test in test_list:
    freq_tab[test.get_result(test.x_arr,test.n)] += 1

# for test in test_list:
#     for i in range(trials):
#         if target > test.n:
#             target = 1
#         test.update_state(target)
#         # Correct energy to table
#         # Record result to freq_tab every times
#         target += 1
#     freq_tab[test.get_result(test.x_arr,test.n)] += 1

df = test_list[0].report_energy(energy_tab, freq_tab, test_list[0].n, a, copies)

Th_vs_Ex = pd.concat([df['Theoretical'], df['Experimental']], axis = 1)
plt.figure()
Th_vs_Ex.plot()

morn_df = test_list[0].report_mornitor(mornitor_tab, test_list[0].n)

    
######################################
#######         TASK 4         #######
################ END! ################