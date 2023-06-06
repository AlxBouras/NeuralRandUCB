import numpy as np
import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad


# for RandUCB
def randomize_confidence(N=20, decoupled_arms=None, std=None, is_optimistic=True):
    lower_bound = 0 if is_optimistic else -1
    upper_bound = 1
    delta = upper_bound - lower_bound

    x = np.linspace(lower_bound, upper_bound, N)

    if std:
        pdf = scipy.stats.norm.pdf(x, loc=0, scale=std)
    else:
        pdf = scipy.stats.uniform.pdf(x, loc=-lower_bound, scale=delta)

    pdf = pdf / np.sum(pdf)

    n = np.random.choice(N, size=decoupled_arms, p=pdf)

    if decoupled_arms:
        n = torch.from_numpy(n).float().cuda()    

    return lower_bound + n * delta / (N - 1)

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
        
class NeuralTSDiag:
    def __init__(self, dim, lamdba=1, nu=1e-2, hidden=100, style='ts', delay=1, **kwargs):
        self.func = extend(Network(dim, hidden_size=hidden).cuda())
        self.context_list = None
        self.len = 0
        self.reward = None
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.style = style
        self.delay = delay
        self.loss_func = nn.MSELoss()
        self.kwargs = kwargs

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        mu = self.func(tensor)
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()
        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
        if self.style == 'rand_ucb':
            sigma = torch.sqrt(torch.sum(g_list * g_list / self.U, dim=1))
            z = randomize_confidence(**self.kwargs)
            sample_r = mu.view(-1) + z * sigma.view(-1)
        else:    
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
            if self.style == 'ts':
                sample_r = torch.normal(mu.view(-1), sigma.view(-1))
            elif self.style == 'ucb':
                sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), 0, 0
    
    def train(self, context, reward):
        self.len += 1
        optimizer = optim.Adam(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba / self.len)
        if self.context_list is None:
            self.context_list = torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.reward = torch.tensor([reward], device='cuda', dtype=torch.float32)
        else:
            self.context_list = torch.cat((self.context_list, torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.reward = torch.cat((self.reward, torch.tensor([reward], device='cuda', dtype=torch.float32)))
        if self.len % self.delay != 0:
            return 0
        for _ in range(100):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(self.context_list).view(-1)
            loss = self.loss_func(pred, self.reward)
            loss.backward()
            optimizer.step()
        return 0