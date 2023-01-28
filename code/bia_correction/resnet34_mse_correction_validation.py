import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt


fig_save_path = '../../resnet3d34_predict/range/mse_adam_correction/'
path = '../../resnet3d34_predict/range/mse_adam_correction/validation/'

path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('_')[0]))

Epoch_list = [x for x in range(1, 21)]
corrected_age_list_1 = []
corrected_pad_list_1 = []
predict_age_list_1 = []
real_age_1 = []
pad_list_1 = []

corrected_age_list_2 = []
corrected_pad_list_2 = []
predict_age_list_2 = []
real_age_2 = []
pad_list_2 = []

corrected_age_list_3 = []
predict_age_list_3 = []
real_age_3 = []

corrected_age_list_4 = []
predict_age_list_4 = []
real_age_4 = []

k = 1
for file in path_list:
    if k <= 20:
        k += 1
        whole_path = path+file
        info_pd = pd.read_csv(whole_path)
        info_pd = info_pd.sort_values('user id')

        print(info_pd['user id'].values[0])
        print(info_pd['user id'].values[1])

        real_age_1.append(info_pd['real age'].values[0])
        real_age_2.append(info_pd['real age'].values[1])
        real_age_3.append(info_pd['real age'].values[2])
        real_age_4.append(info_pd['real age'].values[3])

        corrected_age_list_1.append(info_pd['corrected predict age'].values[0])
        corrected_age_list_2.append(info_pd['corrected predict age'].values[1])
        corrected_age_list_3.append(info_pd['corrected predict age'].values[3])
        corrected_age_list_4.append(info_pd['corrected predict age'].values[4])

        corrected_pad_list_1.append(info_pd['corrected age difference'].values[0])
        corrected_pad_list_2.append(info_pd['corrected age difference'].values[1])

        predict_age_list_1.append(info_pd['predict age'].values[0])
        predict_age_list_2.append(info_pd['predict age'].values[1])
        predict_age_list_3.append(info_pd['predict age'].values[3])
        predict_age_list_4.append(info_pd['predict age'].values[4])

        pad_list_1.append(info_pd['age difference'].values[0])
        pad_list_2.append(info_pd['age difference'].values[1])



labels = [str(x) for x in Epoch_list]
plt.figure(figsize=(18, 6))  # width:20, height:3
plt.plot(Epoch_list, real_age_1, 'k', label='Sample 1 chronological age', linewidth=2, alpha = 1)
plt.plot(Epoch_list, real_age_2, 'ks--', label='Sample 2 chronological age', linewidth=2, alpha = 0.6)


plt.plot(Epoch_list, predict_age_list_1, 'b-', label='Sample 1 predict age', linewidth=2, alpha = 1)
plt.plot(Epoch_list, predict_age_list_2, 'bs--', label='Sample 2 predict age', linewidth=2, alpha = 0.6)

plt.plot(Epoch_list, corrected_age_list_1, 'r-', label='Sample 1 corrected predict age', linewidth=2, alpha = 1)
plt.plot(Epoch_list, corrected_age_list_2, 'rs--', label='Sample 2 corrected predict age', linewidth=2, alpha = 0.6)

plt.plot(Epoch_list, pad_list_1, 'm-', label='Sample 1 PAD', linewidth=2, alpha = 1)
plt.plot(Epoch_list, pad_list_2, 'ms--', label='Sample 2 PAD', linewidth=2, alpha = 0.6)

plt.plot(Epoch_list, corrected_pad_list_1, 'g-', label='Sample 1 corrected PAD', linewidth=2, alpha = 1)
plt.plot(Epoch_list, corrected_pad_list_2, 'gs--', label='Sample 2 corrected PAD', linewidth=2, alpha = 0.6)

plt.xlabel("Epoch")
plt.ylabel("Age or PAD")
plt.title("Various age prediction as training epoch increases")
plt.xticks(Epoch_list, labels)
plt.legend()
plt.savefig(fig_save_path + 'statistics_corrected_along_epoch.png')




plt.figure(figsize=(18, 6))  # width:20, height:3
plt.plot(Epoch_list, real_age_1, 'k', label='Sample 1 chronological age', linewidth=2, alpha = 0.6)


plt.plot(Epoch_list, predict_age_list_1, 'b-', label='Sample 1 predict age', linewidth=2, alpha = 0.6)

plt.plot(Epoch_list, corrected_age_list_1, 'r-', label='Sample 1 corrected predict age', linewidth=2, alpha = 0.6)

plt.plot(Epoch_list, pad_list_1, 'm-', label='Sample 1 PAD', linewidth=2, alpha = 0.6)

plt.plot(Epoch_list, corrected_pad_list_1, 'g-', label='Sample 1 corrected PAD', linewidth=2, alpha = 0.6)

plt.xlabel("Epoch")
plt.ylabel("Age or PAD")
plt.title("Various age prediction as training epoch increases")
plt.xticks(Epoch_list, labels)
plt.legend()
plt.savefig(fig_save_path + 'statistics_corrected_along_epoch_onesample.png')



labels = [str(x) for x in Epoch_list]
plt.figure(figsize=(18, 8))  # width:20, height:3
plt.plot(Epoch_list, real_age_1, 'k-', label='Sample 1 chronological age', linewidth=2, alpha = 1)
plt.plot(Epoch_list, real_age_2, 'b-', label='Sample 2 chronological age', linewidth=2, alpha = 1)
plt.plot(Epoch_list, real_age_3, 'g-', label='Sample 3 chronological age', linewidth=2, alpha = 1)
plt.plot(Epoch_list, real_age_4, 'r-', label='Sample 4 chronological age', linewidth=2, alpha = 1)


plt.plot(Epoch_list, predict_age_list_1, 'k*--', label='Sample 1 predict age', linewidth=1, alpha = 0.7)
plt.plot(Epoch_list, predict_age_list_2, 'b*--', label='Sample 2 predict age', linewidth=1, alpha = 0.7)
plt.plot(Epoch_list, predict_age_list_3, 'g*--', label='Sample 3 predict age', linewidth=1, alpha = 0.7)
plt.plot(Epoch_list, predict_age_list_4, 'r*--', label='Sample 4 predict age', linewidth=1, alpha = 0.7)


plt.plot(Epoch_list, corrected_age_list_1, 'ks-.', label='Sample 1 corrected predict age', linewidth=1, alpha = 0.4)
plt.plot(Epoch_list, corrected_age_list_2, 'bs-.', label='Sample 2 corrected predict age', linewidth=1, alpha = 0.4)
plt.plot(Epoch_list, corrected_age_list_3, 'gs-.', label='Sample 3 corrected predict age', linewidth=1, alpha = 0.4)
plt.plot(Epoch_list, corrected_age_list_4, 'rs-.', label='Sample 4 corrected predict age', linewidth=1, alpha = 0.4)


plt.xlabel("Epoch")
plt.ylabel("Age or PAD")
plt.title("Various age prediction as training epoch increases")
plt.xticks(Epoch_list, labels)
plt.legend()
plt.savefig(fig_save_path + 'statistics_corrected_along_epoch_4samples.png')