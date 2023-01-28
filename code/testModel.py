
import torch
from torchviz import make_dot
import msd3d
input_size=(162, 226, 209)

x = torch.randn(2, 1, 162, 226, 209)

y = msd3d.MSD9(input_size=input_size)(x)
vise=make_dot(y, params=dict(msd3d.MSD9(input_size=input_size).named_parameters()))
vise.view()

from torchsummary import summary

model = msd3d.MSD9(input_size=input_size, output_dim=32)
summary(model, (1, 162, 226, 209))
	#log.write(stat_print)



