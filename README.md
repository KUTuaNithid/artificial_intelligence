# Artificial Intelligence Report

## **Task 1 - Simulation of Probabilistic Binary Model**


![task1_detail](task1_detail.jpg) 
[Source code](code/task1.py)  
Procedure  
1. Create 1 neuron which has 5 inputs and use a probabilistic Binary Model (Using random generator to create probabilistic model).
1. Execute the neuron with parameters defined in table.
1. Repeatdly execute and count when `y = 1` for the experimental result
1. Calculate the theoritical result by ` Trials * P`
1. Calculate a ⭐percent error(pe) between the experimental result and the theritical result
1. Calculate a ⭐average of percent error(mean) from each trail  
```py
# ⭐How to get percent error and average
# Percent error can be negative to show more or less of result
pe = ((theoritical-experimental)/th)*100 
# Mean is calculated from absolute values to show only a margin of result
mean = statistics.mean([abs(pe_100), abs(pe_1000), abs(pe_10000)]) 
```
<center>Parameter table

| Name | Description | value |
| ---- | ----------- | ----- |
| a | Gain of the sigmoid function | 0.2, 0.5, 1, 5 10 |
| Trials | Repeatly execute the neuron | 100, 1000, 10000ฅ
| RAND_MAX | Maximum range of a random generator | RAND_MAX = 10000|
| SEED | Seed of a random generatot | SEED = 0 |
| Input | Input of a neuron | x1 = [1, 0, 1, -1, 0.5, -0.5] |  
| | |x2 = [1, -1, 1, 1, -0.5, 1] |
| weight | Weight of a neuron | w = [1, -1, 2, -1, 2, -1] |
</center>
*x[0], w[0]  is dummy  

&nbsp;  

### **1.1. Input x1 = [1, 0, 1, -1, 0.5, -0.5]**
<center>

| Gain(a) | Trials | Percent error | Average of 3 Trials |
|--|--|--|--|
|**0.2**| 100  | -8.73 % | **4.04 %** |
| ↑  | 1000 | 2.69 %  |   ↑      |
| ↑  | 10000|  <span style="color:blue">0.71 %</span> |   ↑      |
|**0.5**| 100  | -3.29 % |**1.51 %**  |
| ↑  | 1000 | 1.17 %  |   ↑      | 
| ↑  | 10000| <span style="color:blue">0.07 %</span>  |   ↑      | 
|**1**  | 100  | -0.60 % | **0.37 %** |
| ↑  | 1000 | -0.50 % |   ↑      |
| ↑  | 10000| <span style="color:blue">-0.02 %</span> |   ↑      |
|**5**  | 100  | -1.01 % |**0.38 %**  |
| ↑  | 1000 | -0.10 % |   ↑      |
| ↑  | 10000| <span style="color:blue">-0.02 %</span> |   ↑      |
|**10** | 100  | -1.01 % |**0.37 %**  |
| ↑  | 1000 | -0.10 % |   ↑      |
| ↑  | 10000| <span style="color:blue">-0.01 %</span> |   ↑      |

&nbsp; 
![Input1_summary](task1_input1.png) 
</center>

### **1.2. Input x2 = [1, -1, 1, 1, -0.5, 1]**
<center>

| Gain(a) | Trials | Percent error | Average of 3 Trials |
|--|--|--|--|
|0.2| 100  | -7.82 %  | 3.66 % |
| ↑  | 1000 | 2.51 %  |    ↑     |
| ↑  | 10000|  <span style="color:blue">0.66 %</span> |    ↑     |
|0.5| 100  | -2.04 % |1.44 %  |
| ↑  | 10000| 1.20 %  |    ↑     | 
| ↑  | 1000 | <span style="color:blue">1.07 %</span>  |    ↑     | 
|1  | 100  | -2.97 % | 1.57 % |
| ↑  | 1000 | 1.74 % |     ↑    |
| ↑  | 10000| <span style="color:blue">0.01 %</span> |     ↑    |
|5  | 100  | -2.40 % |0.90 %  |
| ↑  | 1000 | -0.23 % |    ↑     |
| ↑  | 10000| <span style="color:blue">0.07 %</span> |     ↑    |
|10 | 100  | -1.01 % |0.37 %  |
| ↑  | 1000 | -0.10 % |    ↑     |
| ↑  | 10000| <span style="color:blue">-0.02 %</span> |    ↑     |

![Input2_summary](task1_input2.png)  
</center>

### **Summary**
Trials: As the result, the percent error will be minimum when `Trial = 10000`, so we can conclude that **<mark>the more trial, the less error</mark>**.

Gain(a): As the result, the percent error will be minimum when `a = 10`, so we can conclude that **<mark>the more gain, the less error</mark>**.  

For my opinion, this conclusion could not apply for all situation. It may suitable for these set of input and weigth only.

## **Task 2 - RNN**
![task2_detail](task2_detail.jpg) 
[Source code](code/task2.py)  
Procedure
1. Make a general program for the winner takes all RNN.
1. Use the probabilistic binary model for its neurons.
1. Execute the neuron with parameters defined in table.

### 

### **2.1. Compute the convergent state of neurons**
Repeatdly update each neuron until reach the convergent state.
<center>

| Input(x) | weight(x) |gain(a) | Convergent state(x) |
| :----: | :-------:|:----: | :-----: |
|<pre>[1, 0, 0, 0]</pre>|<pre>[0,  1,  1,  1], #0<br/>[0,  0, -2, -2], #1<br/>[0, -2,  0, -2], #2<br/>[0, -2, -2,  0]  #3</pre>| 1000 | <pre>[1, 1, 0, 0]</pre> |
|<pre>[1, 1, 1, 1]</pre>|↑| ↑ | <pre>[1, 0, 0, 1]</pre> |
|<pre>[1, 1, 0, 1]</pre>|↑| ↑ | <pre>[1, 0, 0, 1]</pre> |
</center>
⭐x[0], w[0] is dummy

### **2.2. Try with different gain**
Change the gain and see what happen
| Input(x) | weight(x) |
| :----: | :-------:|
|<pre>[1, 0, 0, 0]</pre>|<pre>[0,  1,  1,  1], #0<br/>[0,  0, -2, -2], #1<br/>[0, -2,  0, -2], #2<br/>[0, -2, -2,  0]  #3</pre>|

| gain(a) | Most existing state (x) | Graph |
|:----: | :-----: | :----: |
| 0.2 | <pre>[1, 1, 0, 0]</pre> | ![task2_a02](task2_a02.png)|
| 0.5 | <pre>[1, 0, 1, 0]</pre> | ![task2_a05](task2_a05.png)|
| 1.0 | <pre>[1, 1, 0, 0]</pre> |![task2_a10](task2_a10.png)|

### **Summary**
Convergent: When we set gain to large number, the model will be deterministic and we can find the convergent state. In the other hand, if we set gain to small number, the model will more stochastic and we can not find the convergent state.  
Gain(a): As the result, the distribution decreases when gain increases. We can conclude that if we want the output to be more deterministic we should increase gain, but if we want the output to be more stochastic we should decrease gain.

## **Task 3 - The Decreasing of Energy**
![task3_detail](task3_detail.jpg)  
[Source code](code/task3.py)
Procedure
1. Make a general program for the winner takes all RNN.
1. Use the deterministic binary model for its neurons.
1. Execute the neuron with parameters defined in table.
1. Updating each neuron and record energy.
1. Change weight and check the changing of energy.


| Input(x) | weight(x) | Convergent state(x) |
| :----: | :-------|:----: | :-----: |
|<pre>[1, 1, 1, 1, 1, 1]</pre>|<font size="2"><pre>[0, 1, 1, 1, 1, 1],#0<br/>[1, 0,-2,-2,-2,-2],#1<br/>[1,-2, 0,-2,-2,-2],#2<br/>[1,-2,-2, 0,-2,-2],#3<br/>[1,-2,-2,-2, 0,-2],#4<br/>[1,-2,-2,-2,-2, 0] #5<br/></pre></font>| <pre>[1, 0, 0, 0, 0, 1]</pre> | 
⭐x[0], w[0] is dummy
<p align="center">
Energy Table
</p>
<p align="center">
  <img src="task3_1enetab.PNG" />
</p>

| Input(x) | weight(w) | Convergent state(x) |
| :----: | :------- |:----: | :-----: |
|<font size="1"><pre>[1, 1, 1, 1, 1, 1, 1, 1]</pre></font> |<font size="1"><pre>[0, 1, 1, 1, 1, 1, 1, 1],#0<br/>[1, 0,-2,-3,-4,-2,-3,-4],#1<br/>[1,-2, 0,-4,-5,-6,-1,-2],#2<br/>[1,-3,-4, 0,-1,-2,-3,-4],#3<br/>[1,-4,-5,-1, 0,-1,-2,-3],#4<br/>[1,-2,-6,-2,-1, 0,-1,-2],#5<br/>[1,-3,-1, 3,-2,-1, 0,-1],#6<br/>[1,-4,-2, 4,-3,-2,-1, 0] #7</pre></font> | <font size="1"><pre>[1, 0, 0, 1, 1, 0, 0, 0]</pre></font> |
⭐x[0], w[0] is dummy
<p align="center">
Energy Table
</p>
<p align="center">
  <img src="task3_2enetab.PNG" />
</p>

### **Summary**
As the result, the energy of RNN always decreases whtn the neuron update its state. The convergence state will has a minimum energy. 

## **Task 4 - RNNs with Boltzmann's Distribution**
![task4_detail](task4_detail.jpg)  
[Source code](code/task4.py)

### **4.1. Boltzman's distribution**
Procedure
1. Make a copy of RNN
1. Update the state of neuron 1 by 1 for each copy
1. The experimental result: Collect the number of Gibbs copy in each state
1. The theoritical result: Caculate follows Boltzman's distribution for all possible state.

| Input(x) | weight(w) |
| :----: | :-------|
|<font size="2"><pre>[1, 1, 1, 1, 1, 1]</pre></font> |<font size="2"><pre>[0,  1,  1,  1,  1,  1],#0<br/>[1,  0, -2, -2, -2, -2],#1<br/>[1, -2,  0, -2, -2, -2],#2<br/>[1, -2, -2,  0, -2, -2],#3<br/>[1, -2, -2, -2,  0, -2],#4<br/>[1, -2, -2, -2, -2,  0] #5</pre></font> |

| # of Trials | # of Copies | Gain(a) | Distribution graph
| :---- | :------- | :------- | :----: |
| 100 | 100 | 0.5 | ![task4_100x100](task4_100x100.png) 
| 100 | 1000 | 0.5 |![task4_100x1000](task4_100x1000.png) 
| 1000 | 1000 | 0.5 |![task4_1000x1000](task4_1000x1000.png) 
| 1000 | 1000 | 0.2 |![task4_1000x1000_a0.2](task4_1000x1000_a0.2.png) 
| 1000 | 1000 | 1.0 |![task4_1000x1000_a1](task4_1000x1000_a1.png) 
| 1000 | 1000 | 1.5 |![task4_1000x1000_a15](task4_1000x1000_a15.png)

<p align="center">
Energy Table: This table shows the energy of each state.
</p>

| # of Trials | # of Copies | Gain(a) | Energy table
| :---- | :------- | :------- | :----: |
| 1000 | 1000 | 0.5 |![task4_1000x1000_table](task4_1000x1000_table.jpg) 

### **Summary**
As the result, the experimental result and the theroetical result is going to the same way. When we increase the number of copies, the result is more closer. The increasing of the number of trials also effects the result, but it not that much.  
The Changing of gain also effects to the result. The more gain makes the result closer, but the distribution of result will be decreased.  
For my opinion, the most suitable parameters is ``` Trials = 1000, Copies = 1000, Gain = 0.5```  
As the result of energy, the most of experimental result will have the lowest energy. Thus, we predict the equilibrium state from an energy. If the energy is low, it is more likely to be equilibrium state.
### **4.2. Equilibrium**
Procedure
1. Make a copy of RNN
1. Update the state of neuron 1 by 1 for each copy
1. Collect the number of copy in each state
1. Check the changing of number of copy after updated many times

| Input(x) | weight(w) | # of Trials | # of Copies |
| :----: | :-------| :------- | :----: |
|<font size="2"><pre>[1, 1, 1, 1]</pre></font> |<font size="2"><pre>[0,  1,  1,  1],#0<br/>[1,  0, -2, -2],#1<br/>[1, -2,  0, -2],#2<br/>[1, -2, -2,  0],#3<br/></pre></font> | 100 | 1000|

| Gain(a) | # of copy in each state | ⭐The changing state of each copy in every trails |
| :---- | :-------: | :-------: |
| 0.1 | ![task4_equi_01](task4_equi_01_TAB.png) | ![task4_equi_01](task4_equi_01.png) |
| 0.5 | ![task4_equi_01](task4_equi_05_TAB.png) | ![task4_equi_01](task4_equi_05.png) |
| 1.0 | ![task4_equi_01](task4_equi_10_TAB.png) | ![task4_equi_01](task4_equi_10.png) |
| 5.0 | ![task4_equi_01](task4_equi_50_TAB.png) | ![task4_equi_01](task4_equi_50.png) |

⭐ This value shows the number of copy which changes to new state. 

### **Summary**
As the result, many copies change their state in the starting time. After updating around 10-20 times, the number of changing decreases a lot. After that, this number just change a little bit. Thus, we can conclude that when this number is not changed too much comparing to previos times, then this system reaches the equilibrium.  
The increasing of gain makes copies less state change.

## **Task 5.1 - Ergodicity**
Due to the Gibbs copies method consumes a lot of memory, Ergodicity has been publish. We can use a time series of states generated by a single RNN, instead of the states of Gibbs copies.  
Procedure
1. Ergodicity result: Make 1 RNN and repeatdly update and collect the state of RNN in each time
1. Experimental result(Gibbs copies): Use Gibbs copies RNN from task 4
1. Compare result: the theoritical result vs the experimental result vs the Ergodicity result
[Source code](code/task5.py)

Adjust Gain  

| Input(x) | weight(w) | # of Trials | # of copies
| :----: | :-------| :---: | :---: |
|<font size="2"><pre>[1, 1, 1, 1, 1, 1]</pre></font> |<font size="2"><pre>[0,  1,  1,  1,  1,  1],#0<br/>[1,  0, -2, -2, -2, -2],#1<br/>[1, -2,  0, -2, -2, -2],#2<br/>[1, -2, -2,  0, -2, -2],#3<br/>[1, -2, -2, -2,  0, -2],#4<br/>[1, -2, -2, -2, -2,  0] #5</pre></font> | 1000 | 1000

| Gain(a) | the theoritical result vs the experimental result vs the Ergodicity result |
| :---- | :-------: |
| 0.1 | ![task5_ergo_0.1](task5_ergo_0.1.png)
| 0.5 | ![task5_ergo_0.5](task5_ergo_0.5.png)
| 1.0 | ![task5_ergo_1.0](task5_ergo_1.0.png)
| 5.0 | ![task5_ergo_5.0](task5_ergo_5.0.png)

Adjust the number of trials and copies

| Input(x) | weight(w) | Gain
| :----: | :-------| :---: | :---: |
|<font size="2"><pre>[1, 1, 1, 1, 1, 1]</pre></font> |<font size="2"><pre>[0,  1,  1,  1,  1,  1],#0<br/>[1,  0, -2, -2, -2, -2],#1<br/>[1, -2,  0, -2, -2, -2],#2<br/>[1, -2, -2,  0, -2, -2],#3<br/>[1, -2, -2, -2,  0, -2],#4<br/>[1, -2, -2, -2, -2,  0] #5</pre></font> | 0.5

| # of Trials | # of copies | the theoritical result vs the experimental result vs the Ergodicity result |
| :---- | :-------: | :---: |
| 100 | 100 |![task5_ergo_trial100](task5_ergo_trial100.png)
| 1000 | 1000 |![task5_ergo_trial1000](task5_ergo_trial1000.png)
| 10000 | 10000 |![task5_ergo_trial10000](task5_ergo_trial10000.png)

### **Summary**
As the result, the ergodicity can be replaced the Gibbs copies if we have a good parameter tuning.  
For Gain tuning, the lower gain makes the oscillated result, so 3 results are going to different ways. The more gain makes the result too determined. As you can see, when the gain is 5 only the ergodicity result is not compatible. The suitable gain should be 0.5  
For trials and copies tuning, the higher number makes the result better.

## **Task 5.1 - Application**
An application of RNN solves the simultaneous equation.
![task5_equation](task5_equation.png)  

Procedure
1. Construct an energy function with the standard for, so that its minimum is the solution of the simultaneous equation.  
<img src="task5_construct.png" width="350">
1. Expand this equation nad get the standard form.
1. Get all weight and theta values from standard equation. Don't forget Wnm = Wmn, so the weight array need to be adjusted.  
<img src="task5_w.png" width="300">
1. Create the RNN with this weight.
1. Repeatdly update. The most frequently appeared state is the solution

[Source code](code/task5.py)

| Input(x) | weight(w) | # of Trials | # of copies |
| :----: | :-------| :-------: | :---: |
|<font size="2"><pre>[1, 1, 1, 1, 1, 1]</pre></font> |<font size="2"><pre>[  0. ,  -2.5, -10.5,   0.5,  -1.5, -10.5],#0<br/>[ -2.5,   0. ,   5. ,   3. ,   5. ,  -1. ],#1<br/>[-10.5,   5. ,   0. ,   1. ,   1. ,  -2. ],#2<br/>[  0.5,   3. ,  -1. ,   0. ,   1. ,   5. ],#3<br/>[ -1.5,   5. ,   1. ,  -1. ,   0. ,   1. ],#4<br/>[-10.5,  -1. ,  -2. ,   5. ,   1. ,   0. ] #5</pre></font> | 1000 | 1000

| Gain (a) | Solution[X1, X2, X3, X4, X5] | The frequency table of each result | Graph |
| :----: | :-------: | :-------: | :---: |
| 0.1 | [1, 0, 1, 1, 0] | <img src="task5_equa_tab_01.jpg" width="300"> | <img src="task5_equation_a0.1.png" width="300">
| 0.5 | ↑ | <img src="task5_equa_tab_05.jpg" width="300"> | <img src="task5_equation_a0.5.png" width="300">
| 1.0 | ↑ | <img src="task5_equa_tab_10.jpg" width="300"> | <img src="task5_equation_a1.0.png" width="300">

### **Summary**
As the result, we can use the RNN method to find the solution of the simultaneius equation. All of method whether a theoritical, Gibbs copies, Ergodicity can get the correct result. Gain makes the confident of result changes, but we can get the correct result from all gain.