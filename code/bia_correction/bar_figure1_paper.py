import pandas as pd
import scipy.stats

'''
import pylab
import scipy.stats as stats

measurements = np.random.normal(loc = 20, scale = 5, size=100)
print(np.shape(measurements))
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
'''
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression

import statistics

from matplotlib import rcParams

TINY_SIZE = 42
SMALL_SIZE = 42
MEDIUM_SIZE = 48
BIGGER_SIZE = 48

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams["legend.frameon"] = False
#plt.rc('legend',**{'fontsize':16})


rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False

rcParams["axes.linewidth"] = 2

img_width = 11
img_height = 7


sample_num = 40
sample_list = range(1, sample_num+1)
label_list = [str(int(x+1)) for x in sample_list]
real_age_list = np.array(list(range(0, sample_num)))/3 + 10 + np.random.rand(sample_num) * 4 - 2
age_diff_list = np.random.rand(sample_num) * 10 - 5
predict_age_list = real_age_list + age_diff_list

plt.figure(figsize=(img_width, img_height))  # width:20, height:3
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xticklabels([])
#ax.set_yticklabels([])
plt.bar(sample_list, real_age_list, color=(128/255, 0/255, 32/255), align='center', width=0.9, alpha=1, edgecolor='dimgray') # 0，47， 167
#plt.xlabel("Sample index", labelpad=-5)
plt.ylabel("Chronological age", labelpad=0)
plt.subplots_adjust(left = 0.15)
plt.savefig('../../imgs_for_figs/real_age.svg', format='svg', transparent=True)

plt.figure(figsize=(img_width, img_height))  # width:20, height:3
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xticklabels([])
#ax.set_yticklabels([])

ax.set_ylim(-8, 8)
plt.bar(sample_list, age_diff_list, color=(0/255,149/255,182/255), align='center', width=0.9, alpha=1, edgecolor='dimgray')
#plt.xlabel("Sample Index", labelpad=-5)
plt.ylabel("PAD", labelpad=0)
plt.subplots_adjust(left = 0.15)
plt.savefig('../../imgs_for_figs/age_diff.svg', format='svg', transparent=True)

plt.figure(figsize=(img_width, img_height))  # width:20, height:3
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xticklabels([])
#ax.set_yticklabels([])

plt.bar(sample_list, predict_age_list, color=(0/255,47/255,167/255), align='center', width=0.9, alpha=1, edgecolor='dimgray')
#plt.xlabel("Sample Index", labelpad=-5)
plt.ylabel("Predicted age", labelpad=0)

plt.subplots_adjust(left = 0.15)

plt.savefig('../../imgs_for_figs/predict_age.svg', format='svg', transparent=True)