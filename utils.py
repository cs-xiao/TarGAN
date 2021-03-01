import os
import torch
import random
import numpy as np

def loss_filter(mask,device="cuda"):
    list = []
    for i, m in enumerate(mask):
        if torch.any(m == 1):
            list.append(i)
    index = torch.tensor(list, dtype=torch.long).to(device)
    return index

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res

def renorm(x):
    res = (x - 0.5) / 0.5
    res.clamp_(-1, 1)
    return res


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    for i in range(batch_size):
        out[i, labels[i].long()] = 1
    return out

def getLabel(imgs, device, index, c_dim=2):
    syn_labels = torch.zeros((imgs.size(0), c_dim)).to(device)
    syn_labels[:, index] = 1.
    return syn_labels

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def save_state_net(net, parameters, index, optim=None, parents_root='../checkpoints/MICCAI2021'):
    save_path = os.path.join(parents_root, parameters.save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path,parameters.net_name)
    torch.save(net.state_dict(), save_file+'_'+str(index)+'.pkl')
    if optim is not None:
        torch.save(optim.state_dict(), save_file+'_optim_'+str(index)+'.pkl')
    if not os.path.isfile(save_path+'/outputs.txt'):
        with open(save_path+'/outputs.txt', mode='w') as f:
            argsDict = parameters.__dict__;
            f.writelines(parameters.note + '\n')
            for i in argsDict.keys():
                f.writelines(str(i) + ' : ' + str(argsDict[i]) + '\n')

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
