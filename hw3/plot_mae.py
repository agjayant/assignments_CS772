
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np


# In[4]:

mae_train = []
mae_test = []
with open("mae.txt", "r") as ins:
    for line in ins:
        line_split = line.split(' ')
        mae_train.append(line_split[1])
        mae_test.append(line_split[2])


# In[7]:

mae_avg_train = []
mae_avg_test = []
with open("mae_avg.txt", "r") as ins:
    for line in ins:
        line_split = line.split(' ')
        mae_avg_train.append(line_split[1])
        mae_avg_test.append(line_split[2])


# In[26]:

x = range(1,1001)
x2 = range(501,1001)


# In[72]:

fig, ax1 = plt.subplots()
ax1.plot(x, mae_train, 'blue', label="train")
ax1.plot(x, mae_test, 'green', label="test")
ax1.plot(x2, mae_avg_train, 'red', label="train_avg")
ax1.plot(x2, mae_avg_test, 'black', label="train_avg")
ax1.set_xlabel('iterations', color="white")
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('mae', color='white')
ax1.tick_params('y', colors='white')
ax1.tick_params('x', colors='white')
l1 = ax1.legend(bbox_to_anchor=(1.1 ,0.5), loc=2, borderaxespad=0.)
ax1.set_ylim([1.105,1.125])

plt.show()

