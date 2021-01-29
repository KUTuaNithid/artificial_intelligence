# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:05:33 2021

@author: Nithid Mahattanasin
"""

from my_lib import *

# Input
x_arr1 = np.array([1, 0, 1, -1, 0.5, -0.5])
w_arr1 = np.array([1.0, -1, 2, -1, 2, -1])
x_arr2 = np.array([1, -1, 1, 1, -0.5, 1])
w_arr2 = np.array([1, -1, 2, -1, 2, -1])

# Test
test1 = my_neurons(x_arr1, w_arr1, "TEST 1 a=0.2", 0.2)
u1 = test1.run()

test2 = my_neurons(x_arr1, w_arr1, "TEST 1 a=0.5", 0.5)
u2 = test2.run()

test3 = my_neurons(x_arr1, w_arr1, "TEST 1 a=1", 1)
u3 = test3.run()

test4 = my_neurons(x_arr1, w_arr1, "TEST 1 a=5", 5)
u4 = test4.run()

test5 = my_neurons(x_arr1, w_arr1, "TEST 1 a=10", 10)
u5 = test5.run()

fig = plt.figure()
fig.suptitle("Input 1: Compare percent error for different gain(a)", fontsize=10)
ax = fig.add_axes([0,0,0.5,1])
cate = ['a = 0.2', 'a = 0.5', 'a = 1', 'a = 5', 'a = 10']
results = [u1,u2,u3,u4,u5]
ax.bar(cate,results)
plt.show()

test1 = my_neurons(x_arr2, w_arr2, "TEST 2 a=0.2", 0.2)
u1 = test1.run()

test2 = my_neurons(x_arr2, w_arr2, "TEST 2 a=0.5", 0.5)
u2 = test2.run()

test3 = my_neurons(x_arr2, w_arr2, "TEST 2 a=1", 1)
u3 = test3.run()

test4 = my_neurons(x_arr2, w_arr2, "TEST 2 a=5", 5)
u4 = test4.run()

test5 = my_neurons(x_arr2, w_arr2, "TEST 2 a=10", 10)
u5 = test5.run()

fig = plt.figure()
fig.suptitle("Input 2: Compare percent error for different gain(a)", fontsize=10)
ax = fig.add_axes([0,0,0.5,1])
cate = ['a = 0.2', 'a = 0.5', 'a = 1', 'a = 5', 'a = 10']
results = [u1,u2,u3,u4,u5]
ax.bar(cate,results)
plt.show()