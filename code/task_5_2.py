# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:54:53 2021

@author: Nithid Mahattanasin
"""

from my_lib import *

######   APPLICATION EQUATION  #######
######################################
#######       TASK 5 _2        #######
############### START! ###############

trials = 100
copies = 1000
a = 5

minus = -1

x_arr = np.array([1, 1, 1, 1, 1, 1])
# Equation
# 11x^2+2nx-10mx-10xy-6xz-6x+5m^2+2mz+11n^2+10n+4ny+11y^2+10y+2yz+5z^2+7-2m-2mn-2my-10nz-6z
                  # 0      1       2       3       4       5
w_arr = np.array([[0,      2.5*minus,     10.5*minus,     -0.5*minus,      1.5*minus,     10.5*minus],      # 0
                  [2.5*minus,      0,     -5*minus,     -3*minus,     -5*minus,      1*minus],      # 1
                  [10.5*minus,     -5*minus,      0,      1,     -1*minus,      2*minus],      # 2
                  [-0.5*minus,     -3*minus,      1*minus,      0,      1,     -5*minus],      # 3
                  [1.5*minus,     -5*minus,     -1*minus,      1*minus,      0,     -1*minus],      # 4
                  [10.5*minus,      1*minus,      2*minus,     -5*minus,     -1*minus,      0]])    # 5

test_list = []
# Create set of gibbs following copies
# 1 gibb has size(x_arr) neurons (include dummy)
for i in range(copies):
    test_list.append(my_rnn(x_arr, w_arr, "Copy{}".format(i), a, (x_arr.size-1)))

# For get result
target = 1 # State selector
energy_tab = np.zeros(pow(2,test_list[0].n)) # Therotical
freq_tab = np.zeros(pow(2,test_list[0].n))   # Experimental
mornitor_tab = [] # To check equilibrium

# Therotical
# Calculate energy of all possible state
for i in range(2**(test_list[0].n)):
    # Size of array is n + 1(dummy)
    # i is current state + 1 in LSB for dummy. Dummy always 1
    # Use np.base_repr to create string of binary to represent n neuron in each state
    # in_arr[0] dummy
    # in_arr[1] neuron#1
    # ..
    # in_arr[n] neuron#n
    in_arr = np.base_repr(i+pow(2,test_list[0].n), base=2)
    energy_tab[i] = test_list[0].get_energy(in_arr, test_list[0].w_arr, test_list[0].n)

# Experimental
# Train for trials times
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

# Ergodicity
ergo_rnn = my_rnn(x_arr, w_arr, "Ergodicity_neuron", a, (x_arr.size-1))
target = 1
ergo_freq_tab = np.zeros(pow(2,ergo_rnn.n))   # Experimental
for i in range(copies):
    if target > ergo_rnn.n:
        target = 1
    ergo_rnn.update_state(target)
    target += 1
    ergo_freq_tab[ergo_rnn.get_result(ergo_rnn.x_arr,ergo_rnn.n)] += 1

df = ergo_rnn.report_gibb_vs_ergo(energy_tab, freq_tab, ergo_freq_tab, ergo_rnn.n, a, copies)

Th_vs_Ex = pd.concat([df['Ergodicity'], df['Experimental']], axis = 1)
plt.figure()
Th_vs_Ex.plot()
