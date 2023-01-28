import torch
import numpy as np

def age_diff(x, target_age, bc):
    x = torch.squeeze(x.detach())
    prob_x = torch.exp(x)
    age_x = torch.matmul(prob_x, bc)
    return age_x,  torch.mean(torch.abs(age_x-target_age))

def age_diff_mse(x, target_age):
    x = torch.squeeze(x.detach())
    age_x = x
    return age_x,  torch.mean(torch.abs(age_x-target_age))

def age_diff_crossentropy(x, target_age, bin_range):
    x = x.detach().float()
    #print('x shape:', x.shape)
    #print(torch.argmax(x, axis=1))
    #print(x.data)
    #print(torch.argmax(x, axis=1))
    age_x = bin_range[0] + torch.argmax(x, axis=1)
    return age_x, torch.mean(torch.abs(torch.argmax(x, axis=1) - target_age.float()))