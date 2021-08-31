import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacker.linf_sgd import Linf_SGD
from torch.optim import SGD, Adam
from torch.autograd import Variable
from sota.cnn.model import Network as NetworkCIFAR
import optimizers.darts.utils as utils
import csv
import math, operator

def gaussian(d,bandwidth):
    return (-1/(bandwidth*math.sqrt(2*math.pi)))*torch.exp(-0.5*(d))

########################
def Mean_Shift_alpha(model, X, y, epsilon, bandwidth, T, N):
    training = model.training
    if training:
        model.eval()
    alpha = [p.clone() for p in model.arch_parameters()]
    
    for iteration in range(T):
        saved_params = [p.clone() for p in model.arch_parameters()]

        with torch.no_grad():
            mean_shift_params = [torch.zeros_like(p) for p in model.arch_parameters()]

            cell_num = len(model.arch_parameters()) # 2
            omega_list = torch.zeros(N) # [3] [3points]
            kernel_list = torch.zeros(N,cell_num) # [3,2] [3points,2alpha]

            perturb_list = [] # [3,2,7,14]
            for i in range(N):
                perturb_list.append([torch.zeros_like(p) for p in model.arch_parameters()]) #[2,7,14]

            for n in range(N):
                # get perturb alpha_nor/alpha_red (Xi)
                for i, p in enumerate(model.arch_parameters()):
                    p.data.copy_(saved_params[i])            
                    p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
                    perturb_list[n][i].data.copy_(p)

                model.clip()
                # get valloss from perturbed alpha (Xi)
                omega = model._loss(X, y, updateType='weight')
                omega_list[n] = omega

                # calculate G(Xi-alpha)
                for i in range(len(model.arch_parameters())):
                    A_Xi = (saved_params[i] - model.arch_parameters()[i]).view(-1)
                    dist = torch.sqrt((A_Xi**2).sum(0))/bandwidth
                    dist = dist**2
                    kernel = (-1)*gaussian(dist,bandwidth)
                    kernel_list[n][i] = kernel

            #print('Before softmax omega:',omega_list)
            omega_list = F.softmax(omega_list, dim=-1)
            #print('After softmax omega:',omega_list)

            total_down = omega_list@kernel_list            
            total_up = [torch.zeros_like(p) for p in model.arch_parameters()]

            for i in range(len(model.arch_parameters())):
                for n in range(N):
                    total_up[i] +=  perturb_list[n][i]*omega_list[n]*kernel_list[n][i]

            for i, p in enumerate(model.arch_parameters()):
                mean_shift_params[i] = total_up[i] / total_down[i]

        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(mean_shift_params[i])

    if training:
        model.train()
        
##################################################################################
##################################################################################

def Mean_Shift_alpha_RNN(model, X, y, hidden, epsilon, bandwidth, T, N):
    training = model.training
    if training:
        model.eval()
    alpha = [p.clone() for p in model.arch_parameters()]
    
    for iteration in range(T):
        saved_params = [p.clone() for p in model.arch_parameters()]

        with torch.no_grad():
            mean_shift_params = [torch.zeros_like(p) for p in model.arch_parameters()]

            cell_num = len(model.arch_parameters()) # 2
            omega_list = torch.zeros(N) # [3] [3points]
            kernel_list = torch.zeros(N,cell_num) # [3,2] [3points,2alpha]

            perturb_list = [] # [3,2,7,14]
            for i in range(N):
                perturb_list.append([torch.zeros_like(p) for p in model.arch_parameters()]) #[2,7,14]

            for n in range(N):
                # get perturb alpha_nor/alpha_red (Xi)
                for i, p in enumerate(model.arch_parameters()):
                    p.data.copy_(saved_params[i])            
                    p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
                    perturb_list[n][i].data.copy_(p)

                model.clip()
                # get valloss from perturbed alpha (Xi)
                omega = model._loss(hidden, X, y, updateType='weight')
                omega_list[n] = omega

                # calculate G(Xi-alpha)
                for i in range(len(model.arch_parameters())):
                    A_Xi = (saved_params[i] - model.arch_parameters()[i]).view(-1)
                    dist = torch.sqrt((A_Xi**2).sum(0))/bandwidth
                    dist = dist**2
                    kernel = (-1)*gaussian(dist,bandwidth)
                    kernel_list[n][i] = kernel

            #print('Before softmax omega:',omega_list)
            omega_list = F.softmax(omega_list, dim=-1)
            #print('After softmax omega:',omega_list)

            total_down = omega_list@kernel_list            
            total_up = [torch.zeros_like(p) for p in model.arch_parameters()]

            for i in range(len(model.arch_parameters())):
                for n in range(N):
                    total_up[i] +=  perturb_list[n][i]*omega_list[n]*kernel_list[n][i]

            for i, p in enumerate(model.arch_parameters()):
                mean_shift_params[i] = total_up[i] / total_down[i]

        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(mean_shift_params[i])

    if training:
        model.train()
    