
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


def age_scatter(left_predict_age, right_predict_age, save_path):
    plt.scatter(left_predict_age, right_predict_age, alpha=0.5, s=5)
    plt.plot(left_predict_age, left_predict_age)
    plt.title("left and right predicted age")
    plt.xlabel("left hemisphere")
    plt.ylabel("right hemisphere")
    plt.savefig(save_path)
    plt.show()



def plot_whole_brain(whole_path, plot_title):
    whole_pd = pd.read_csv(whole_path)
    whole_predict_age = whole_pd['predict age'].values

    real_age = whole_pd['real age'].values

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    len_vec = range(1, len(whole_predict_age)+1)


    plt.scatter(len_vec, whole_predict_age, c='b', alpha=0.5, label='right hemisphere',s=4)
    #plt.scatter(len_vec, left_predict_age, c='r', alpha=0.5, label='left hemisphere',s=4)
    plt.scatter(len_vec, real_age, c='g', alpha=0.5, label='chronological',s=4)
    plt.title(plot_title+'whole brain age prediction')
    plt.xlabel("subjects")
    plt.ylabel("age")
    plt.legend(loc='upper left')
    plt.savefig(whole_path + '_whole.png')
    plt.show()



def plot_left_right(left_path, right_path, plot_title):
    left_pd = pd.read_csv(left_path)
    left_predict_age = left_pd['predict age'].values
    #print(left_predict_age.shape)

    right_pd = pd.read_csv(right_path)
    right_predict_age = right_pd['predict age'].values

    real_age = right_pd['real age'].values

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    len_vec = range(1, len(left_predict_age)+1)


    plt.scatter(len_vec, right_predict_age, c='b', alpha=0.5, label='right hemisphere',s=4)
    plt.scatter(len_vec, left_predict_age, c='r', alpha=0.5, label='left hemisphere',s=4)
    plt.scatter(len_vec, real_age, c='g', alpha=0.5, label='chronological',s=4)
    plt.title(plot_title+"left and right predicted age")
    plt.xlabel("subjects")
    plt.ylabel("age")
    plt.legend(loc='upper left')
    plt.savefig(left_path + '_left_right_plot.png')
    plt.show()


    age_scatter(left_predict_age, right_predict_age, save_path=left_path+'_left_right_scatter.png')
#age_scatter(left_predict_age, right_predict_age)

'''
# resnet range
whole_path = '../../resnet3d_predict/range/whole/2021-06-26_age_predict.csv'
left_path = '../../resnet3d_predict/range/left/2021-06-26_age_predict.csv'
right_path = '../../resnet3d_predict/range/right/2021-06-26_age_predict.csv'

plot_whole_brain(whole_path, 'resnet3d range 30-90')
plot_left_right(left_path, right_path, 'resnet3d range 30-90')
'''

'''
# sfcn
whole_path = '../../sfcn_predict/whole/2021-06-29_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

plot_whole_brain(whole_path, 'sfcn range 42-82')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')

# resnet3d
whole_path = '../../resnet3d_predict/whole/2021-06-29_age_predict.csv'
#left_path = '../../resnet3d_predict/range/left/2021-06-26_age_predict.csv'
#right_path = '../../resnet3d_predict/range/right/2021-06-26_age_predict.csv'

plot_whole_brain(whole_path, 'resnet3d range 42-82')
#plot_left_right(left_path, right_path, 'resnet3d range 30-90')
'''

# sfcn
whole_path = '../../sfcn_predict/whole/2021-07-11_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

plot_whole_brain(whole_path, 'sfcn range 42-82')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')

# sfcn range
whole_path = '../../sfcn_predict/range/whole/2021-07-05_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

plot_whole_brain(whole_path, 'sfcn range 38-86')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')


# sfcn sample
whole_path = '../../sfcn_predict/sampler/whole/2021-07-14_age_predict.csv'
#left_path = '../../sfcn_predict/left/2021-06-26_age_predict.csv'
#right_path = '../../sfcn_predict/right/2021-06-26_age_predict.csv'

plot_whole_brain(whole_path, 'sampled sfcn range 42-82')
#plot_left_right(left_path, right_path, 'resnet3d range 42-82')

