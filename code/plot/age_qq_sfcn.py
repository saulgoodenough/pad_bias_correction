
import numpy as np
import pandas as pd
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


def age_scatter(left_predict_age, right_predict_age):
    plt.scatter(left_predict_age, right_predict_age, alpha=0.5, s=5)
    plt.plot(left_predict_age, left_predict_age)
    plt.title("left and right predicted age")
    plt.xlabel("left hemisphere")
    plt.ylabel("right hemisphere")
    plt.show()

left_pd = pd.read_csv('../../sfcn_predict/left/2021-06-21_age_predict.csv')
left_predict_age = left_pd['predict age'].values
#print(left_predict_age.shape)

right_pd = pd.read_csv('../../sfcn_predict/right/2021-06-21_age_predict.csv')
right_predict_age = right_pd['predict age'].values

real_age = right_pd['real age'].values

# Fixing random state for reproducibility
np.random.seed(19680801)

len_vec = range(1, len(left_predict_age)+1)


plt.scatter(len_vec, right_predict_age, c='b', alpha=0.5, label='right hemisphere',s=4)
plt.scatter(len_vec, left_predict_age, c='r', alpha=0.5, label='left hemisphere',s=4)
plt.scatter(len_vec, real_age, c='g', alpha=0.5, label='chronological',s=4)
plt.title("left and right predicted age")
plt.xlabel("subjects")
plt.ylabel("age")
plt.legend(loc='upper left')
plt.show()

age_scatter(left_predict_age, right_predict_age)




