import torch.nn as nn
import torch.optim as optim


def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    #print(loss)
    return loss


def TripleMSE(y_sum, y_f, y_real):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    #print(loss)
    lambda_para = 0.4
    loss = nn.MSELoss()(y_sum, y_real) + lambda_para * nn.MSELoss()(y_f, y_real)
    return loss

def makeCritirion(lossname):
    if lossname == 'KL':
        return my_KLDivLoss
    elif lossname == 'MSE':
        return nn.MSELoss()
    elif lossname == 'TripleMSE':
        return TripleMSE
    elif lossname == 'NLLLoss':
        return nn.NLLLoss()



def makeSolver(cfg, Net):
    if cfg.SOLVER.NAME == 'SGD':
        optimizer = optim.SGD(Net.parameters(), lr = cfg.SOLVER.SGD_LR, momentum = cfg.SOLVER.SGD_MOMENTUM)
    elif cfg.SOLVER.NAME == 'Adam':
        optimizer = optim.Adam(Net.parameters(), lr=cfg.SOLVER.ADAM_LR, betas=cfg.SOLVER.ADAM_BETAS,
                               eps=cfg.SOLVER.ADAM_EPS, weight_decay=cfg.SOLVER.ADAM_WEIGHT_DECAY)
    return optimizer


