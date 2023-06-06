from datasets import Bandit_multi, SyntheticDataset
from neural_bandits import NeuralTSDiag
from train import train
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle 
import os
import time
import torch
from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':    
    seed = 42
    T = 2000
    dataset = 'synthetic'
    n_runs = 8

    # Neural TS
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'ts')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, lamdba=1, nu=1e-2, hidden=100, style='ts')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralTS_{seed+i}.csv'), cumul_regret, delimiter=',')        

    # Neural RandUCB TS
    kwargs = {'N': 20, 'decoupled_arms': 4, 'std': 0.1, 'is_optimistic': False}
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'randucb_ts')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)
        learner = NeuralTSDiag(bandit.dim, hidden=100, style='rand_ucb', **kwargs)
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralRandUCB_{seed+i}.csv'), cumul_regret, delimiter=',')

    # Neural RandUCB UCB
    kwargs = {'N': 20, 'decoupled_arms': None, 'std': None, 'is_optimistic': True}
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'randucb_ucb')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, hidden=100, style='rand_ucb', **kwargs)
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralRandUCB_{seed+i}.csv'), cumul_regret, delimiter=',')


    # Neural RandUCB best
    kwargs = {'N': 20, 'decoupled_arms': None, 'std': 0.0625, 'is_optimistic': False}
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'randucb_best')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, hidden=100, style='rand_ucb', **kwargs)
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralRandUCB_{seed+i}.csv'), cumul_regret, delimiter=',')


