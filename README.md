# Artificial Intelligence Report

## **Task 1 - Simulation of Probabilistic Binary Model**


![task1_detail](task1_detail.jpg) 

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

### **1. Input x1 = [1, 0, 1, -1, 0.5, -0.5]**
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

### **2. Input x2 = [1, -1, 1, 1, -0.5, 1]**
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

### **Summary***
Trials: As the result, the percent error will be minimum when `Trial = 10000`, so we can conclude that **<mark>the more trial, the less error</mark>**.

Gain(a): As the result, the percent error will be minimum when `a = 10`, so we can conclude that **<mark>the more gain, the less error</mark>**.  

For my opinion, this conclusion could not apply for all situation. It may suitable for these set of input and weigth only.

