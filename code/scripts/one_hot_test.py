import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

bin_range = [38, 86]
x = torch.tensor([38, 76, 63, 45, 85])
print(F.one_hot(x - bin_range[0], num_classes=48))

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
output = loss(input, target)