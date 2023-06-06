import numpy as np
import dataset
import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from sklearn.metrics.pairwise import rbf_kernel
from torch.distributions.multivariate_normal import MultivariateNormal


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, alpha, context="user"):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """
        self.n_features = len(dataset.features[0])
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        self.A = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.b = np.zeros((dataset.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        self.name = "LinUCB"

    def select(self, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, dataset.features[pool_idx]))

        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        return np.argmax(p)

    def train(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]  # displayed article's index
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, dataset.features[a]))

        x = x.reshape((self.n_features, 1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])


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
    def __init__(self, dim, n_hidden=2, hidden_size=100):
        super(Network, self).__init__()
        layers = []
        layers = [torch.nn.Linear(dim, hidden_size), torch.nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_size, 1))

        self.fa = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fa(x)
        
class NeuralTSDiag:
    def __init__(self, dim, lamdba=1, nu=1e-2, n_hidden=2, hidden_size=100, style='ts', delay=1, **kwargs):
        self.name = 'Neural_' + style
        self.func = extend(Network(dim, n_hidden=n_hidden, hidden_size=hidden_size).cuda())
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
        self.n_features = len(dataset.features[0])
        self.n_arms = dataset.n_arms

    def select(self, context, pool_idx):
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
        return arm
    
    def train(self, displayed, reward, context, pool_idx):
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