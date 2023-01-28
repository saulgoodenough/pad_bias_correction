from code.utils.plot_utils import age_statistics_real as age_statistics_real

from code.utils.bias_correction import correct_age as correct_age
import numpy as np
import matplotlib.pyplot as plt
path = '../../data/image_list_train.npy'
train_age_path_list = np.load(path)
temp_path = train_age_path_list[0]

train_age_list_key = [x.split('/')[-3] for x in train_age_path_list]


#print(train_age.shape)
path_age_dict = np.load('../../data/age_dict.npy', allow_pickle=True).item()
#print(path_age_dict)
#train_age_list = path_age_dict.values()
#print(len(train_age_list))

train_age_list = [path_age_dict[k] for k in train_age_list_key]

mean_age = np.mean(np.array(train_age_list))
print(f'The mean age is {mean_age}') # 62.024529649916644

save_path = '../../data/'
plt.figure()
plt.hist(train_age_list, bins = range(38,86), color='orange')
plt.title("real age distribution for all data")
plt.xlabel("age")
plt.ylabel("number")
plt.savefig(save_path + 'train_real_age.png')