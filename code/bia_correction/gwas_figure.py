import pandas as pd


from bioinfokit import analys, visuz
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
img_height = 8


# load dataset as pandas dataframe
df = analys.get_data('mhat').data
df.head(2)

# create Manhattan plot with default parameters
# visuz.marker.mhat(df=df, chr='chr',pv='pvalue')
color=("#a7414a", "#696464", "#00743f", "#563838", "#6a8a82", "#a37c27", "#5edfff", "#282726", "#c0334d", "#c9753d")

visuz.marker.mhat(df=df, chr='chr',pv='pvalue', dotsize = 40, color=color, gwas_sign_line=True, gwasp=5E-06, figtype='svg', axlabelfontsize=50, dim=(img_width,img_height))

# set parameter show=True, if you want view the image instead of saving