from datasets import Bandit_multi, SyntheticDataset
from neural_bandits import NeuralTSDiag
from linear_bandits import LinearTS
from kernel_bandits import KernelTS
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
    T = 6000
    dataset = 'synthetic'
    n_runs = 5

    # LinearUCB
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'LinearUCB')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)        
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)
        learner = LinearTS(bandit.dim, lamdba=1, nu=1, style='ucb')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'linUCB_{i}.csv'), cumul_regret, delimiter=',')

    # LinearTS
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'LinearTS')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = LinearTS(bandit.dim, lamdba=1, nu=1e-2, style='ts')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'linTS_{i}.csv'), cumul_regret, delimiter=',')    
    
    # KernelUCB
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'KernelUCB')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = KernelTS(bandit.dim, lamdba=1, nu=1, style='ucb')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'kernelUCB_{i}.csv'), cumul_regret, delimiter=',')

    # KernelTS
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'KernelTS')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)        
        learner = KernelTS(bandit.dim, lamdba=1, nu=1e-2, style='ts')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'kernelTS_{i}.csv'), cumul_regret, delimiter=',')

    # Neural RandUCB
    kwargs = {'N': 20, 'decoupled_arms': None, 'std': 0.0625, 'is_optimistic': False}
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'NeuralRandUCB')
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
        np.savetxt(Path(filepath, f'neuralRandUCB_{i}.csv'), cumul_regret, delimiter=',')

    # Neural UCB
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'NeuralUCB')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, lamdba=1, nu=1e-1, hidden=100, style='ucb')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralUCB_{i}.csv'), cumul_regret, delimiter=',')

    # Neural TS
    filepath = Path(Path(__file__).parent, 'results_performance', dataset, 'NeuralTS')
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
        np.savetxt(Path(filepath, f'neuralTS_{i}.csv'), cumul_regret, delimiter=',')