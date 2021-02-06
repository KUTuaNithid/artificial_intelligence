# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:29:14 2021

@author: Nithid Mahattanasin
"""
import matplotlib.pyplot as plt 
import numpy as np 
import random
import statistics
import pandas as pd
import math
import statistics

# Setup
RAND_MAX = 10000
SEED = 0
random.seed(SEED)
def boltzMan(A, gain, energy):
    return A*np.exp(-1*gain*energy)
def my_sigmoid(s, a):
    # Sigmoid function, to nomalize result to 0, 1
    # P = 1 / (1 + e^-ax)
    return 1.0/(1 + np.exp(-1*a*s))
def output(s, a, RAND_MAX):
    # Get output follow probabilistic model
    # RAND_MAX is maximum of random number
    
    rand = random.randint(0, RAND_MAX)
    if (rand <= (my_sigmoid(s, a) * RAND_MAX)):
        return 1
    elif (rand > (my_sigmoid(s, a) * RAND_MAX)):
        return 0

def output_deter(s, th):
    # Get output according to deteministic binary model
    if s >= -th:
        return 1
    else:
        return 0

class my_rnn:
    def __init__(self, x_arr, w_arr, test_name, a, N):
        # N: Size of input nit include dummy
        self.x_arr = np.copy(x_arr)
        self.w_arr = np.copy(w_arr)
        self.n = N
        self.a = a
        # print("Name:", test_name)
        # print("Input:", self.x_arr)
        # print("Weight:", self.w_arr)
        # print("Size is", self.n)

    def compute_state(self, target):
        # For probabilistic model
        # target : Target state to compute
        #       N
        # Sn = sum Win * Xi
        #      i=0
        s = 0
        for i in range(0, self.n+1):
            s = s + (self.w_arr[i][target] * self.x_arr[i])
        return output(s, self.a, RAND_MAX)

    def compute_state_deter(self, target):
        # For determistic model
        # target : Target state to compute
        #       N
        # Sn = sum Win * Xi
        #      i=1
        # Then compare s with threshold (W0target)
        s = 0
        for i in range(1, self.n+1):
            s = s + (self.w_arr[i][target] * self.x_arr[i])
        return output_deter(s, self.w_arr[0][target])
    
    def update_state(self, target, model="prob"):
        # Update target state of model
        if model == "deter":
            self.x_arr[target] = self.compute_state_deter(target)
        else:
            self.x_arr[target] = self.compute_state(target)
        
    def train_to_converg(self, model="prob"):
        # To find converge. If the neuron have same state more than 3 times
        # I will judge it is converged
        # This network should use with a -> inf to assume that is deterministic model
        count = 0
        self.energy_tab = []
        while count < 3:
            x_arr_temp = np.copy(self.x_arr)
            for i in range(1, self.n + 1):
                # Collect energy of current state
                self.energy_tab.append(np.concatenate((np.copy(self.x_arr[1:]), [self.get_energy(self.x_arr, self.w_arr, self.n)]), axis = 0))

                if model == "deter":
                    self.x_arr[i] = self.compute_state_deter(i)
                else:
                    self.x_arr[i] = self.compute_state(i)
            # Compare that this time is same as previous or not
            count = (count + 1) if np.array_equal(x_arr_temp, self.x_arr) else 0
        return self.x_arr, self.energy_tab

    def run_trail(self, a, trail):
        # Run RNN for many times as specied in trail
        # Return freq_tab that implies about how often of each state
        # The index is implies state, values is frequency
        # Ex freq_tab[0] = 20 -> 0 0 0 occurs 20 times
        #    freq_tab[1] = 20 -> 0 0 1 occurs 20 times
        self.a = a
        print("Gain: ", a)
        self.freq_tab = np.zeros(pow(2,self.n))
        for t in range(1, trail+1):
            res = 0
            # Compute all state, 1 state per iteration
            # Lowest index is computed first
            for i in range(1, self.n + 1):
                # Compute each state per time
                self.x_arr[i] = self.compute_state(i)
                # Check all state per iteration
                res = self.get_result(self.x_arr,self.n)
                # Record result to freq_tab every times
                self.freq_tab[res] += 1
        return self.freq_tab
    
    def get_energy(self, x_arr, w_arr, n):
        #            n   n
        # E = -0.5 * E   E (Wij*Xi*Xj)
        #           i=0 j=0
        energy = 0
        for i in range(0, n+1):
            for j in range(0, n+1):
                energy = energy + (float(w_arr[i][j])*float(x_arr[i])*float(x_arr[j]))
        return energy * -0.5
      
    def get_result(self, x_arr,n):
        # Compute the result of all state
        # X1 is MSB
        # Ex. 0 -> 0 0 0 -> X1 = 0 X2 = 0 X3 = 0
        #     1 -> 0 0 1 -> X1 = 0 X2 = 0 X3 = 1
        #     4 -> 1 0 0 -> X1 = 1 X2 = 0 X3 = 0
        res = 0
        for i in range(1,n+1):
            res = res + (x_arr[i]<<(n-i))

        return res
    def get_BoltzTheo(self, energy_tab, gain, copies):
        # Get A
        # A = copies/ sum(e^(gain*all_energy))
        s = 0
        for energy in energy_tab:
            s += np.exp(-1*gain*energy)
        A = copies/s
        thero_tab = []
        for energy in energy_tab:
            thero_tab.append(boltzMan(A, gain, energy))

        return thero_tab, A

    def report_mornitor(self, morn_tab, n):
        header = ["{0:b}".format(i).zfill(n) for i in range(pow(2,n))]
        df = pd.DataFrame(morn_tab, columns = header)
        print(df)
        return df
    
    def report_freq(self, freq_tab, n):
        # Print table that show how often of each state result
        res = {'Neurons': ["{0:b}".format(i).zfill(n) for i in range(pow(2,n))],
        'How_often': freq_tab
        }
        df = pd.DataFrame(res, columns = ['Neurons', 'How_often'])
        print(df)
        print("Max is", df.loc[df.How_often == df['How_often'].max(), 'Neurons'].values[0], "Occured", df['How_often'].max(), "times")
        
        cate = ["{0:b}".format(i).zfill(n) for i in range(pow(2,n))]
        results = freq_tab
        plt.xlabel('[x1, x2, x3]: State of neurons')
        plt.ylabel('Frequency')
        plt.suptitle('State frequency table when gain(a) is {}'.format(self.a))
        plt.bar(cate, results)
        plt.show()

        return df
    
    def report_energy_short(self, energy_tab, n):
        header = ['X{}'.format(i) for i in range(1,n+1)] + ['Energy']
        df = pd.DataFrame(energy_tab, columns = header)
        print(df)
        return df

    def report_energy(self, energy_tab, freq_tab, n, gain, copies):
        # Compare theritical(Boltzman) vs Experimental
        thero_tab, A = self.get_BoltzTheo(energy_tab, gain, copies)
        res = {'Neurons': ["{0:b}".format(i).zfill(n) for i in range(pow(2,n))],
        'Energy': energy_tab, 'Theoretical': thero_tab, 'Experimental': freq_tab
        }
        df = pd.DataFrame(res, columns = ['Neurons', 'Energy', 'Theoretical', 'Experimental'])
        # pd.options.display.float_format = "{:,.2f}".format
        print(df)
        return df, A
        # df = pd.DataFrame(energy_tab, columns = header)
    
    def report_gibb_vs_ergo(self, energy_tab, freq_tab, ergo_freq_tab, n, gain, copies):
        # theritical(Boltzman) vs Gibbs vs Ergodicity
        thero_tab = self.get_BoltzTheo(energy_tab, gain, copies)
        res = {'Neurons': ["{0:b}".format(i).zfill(n) for i in range(pow(2,n))],
        'Theoretical': thero_tab, 'Experimental': freq_tab, 'Ergodicity': ergo_freq_tab
        }
        df = pd.DataFrame(res, columns = ['Neurons', 'Theoretical', 'Experimental', 'Ergodicity'])
        print(df)
        return df

class my_neurons:
    def __init__(self, x_arr, w_arr, test_name, a):
        self.x_arr = x_arr
        self.w_arr = w_arr
        self.test_name = test_name
        self.a = a
        print("-----------", test_name, "-----------")
        # print("Input:")
        # print("x", x_arr)
        # print("w", w_arr)
        # print("a", a)
    
    def compute_s (self, x_arr, w_arr):
        # Compute the Result
        #     N
        # S = E WnXn : N is # of input
        #    n=1
        
        s = 0
        for x, w in zip(x_arr, w_arr):
            s = s + w*x
        return s

    def run_trial(self, trails):
        # Compute the node many times as specified in trails
        # Return
        #   Experimental (count) : How often when output is 1
        #   Therotical (my_sigmoid) : Sigmoid normalize our result to 0-1, 
        #                             so sigmoid * result will tell how strong of our result
        #                             Or it implies kind of probability detail.

        # Start operation
        count = 0
        s = self.compute_s(self.x_arr, self.w_arr)
        for x in range(1, trails):
            # Get output by probabilistic model
            out = output(s, self.a, RAND_MAX)
            if out == 1:
                count = count + 1
        return count,  my_sigmoid(self.compute_s(self.x_arr, self.w_arr), self.a) * trails
    
    def percent_error(self, ex, th):
        # Calculate percent error between experimental and therotical
        return ((th-ex)/th)*100
    
    def run(self):
        # Execute neuron by 100, 1000, 10000 times
        # Return average of percent error of all trails
        
        Experimantal_100, Theoretical_100 = self.run_trial(100)
        Experimantal_1000, Theoretical_1000 = self.run_trial(1000)
        Experimantal_10000, Theoretical_10000 = self.run_trial(10000)
        # fig = plt.figure()
        # fig.suptitle(self.test_name, fontsize=16)
        # ax = fig.add_axes([0,0,1,1])
        # cate = ['Experimantal', 'Theoretical']
        # results = [Experimantal_100,Theoretical_100]
        # ax.bar(cate,results)
        # plt.show()
        
        # Compute percent error of each trails
        pe_100 = (self.percent_error(Theoretical_100, Experimantal_100))
        pe_1000 = (self.percent_error(Theoretical_1000, Experimantal_1000))
        pe_10000 = (self.percent_error(Theoretical_10000, Experimantal_10000))
        print("Percent error")
        print("Trial 100:", "%.2f" % pe_100, "%")
        print("Trial 1000:", "%.2f" % pe_1000, "%")
        print("Trial 10000:", "%.2f" % pe_10000, "%")
        mean = statistics.mean([abs(pe_100), abs(pe_1000), abs(pe_10000)])
        print("Average of percent error:", "%.2f" % mean, "%")
        # fig = plt.figure()
        # fig.suptitle("Percent error"+self.test_name, fontsize=10)
        # ax = fig.add_axes([0,0,0.5,1])
        # cate = ['Trial 100', 'Trial 1000', 'Trial 10000']
        # results = [pe_100,pe_1000, pe_10000]
        # ax.bar(cate,results)
        # plt.show()
        return mean