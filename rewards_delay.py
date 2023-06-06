from datasets import Bandit_multi, SyntheticDataset
from neural_bandits import NeuralTSDiag
from linear_bandits import LinearTS
from kernel_bandits import KernelTS
from train import train
import numpy as np
import os
import torch
from pathlib import Path


if __name__ == '__main__':
    seed = 42
    T = 6000
    delays = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    dataset = 'synthetic'
    n_runs = 5

    # Neural RandUCB
    kwargs = {'N': 20, 'decoupled_arms': None, 'std': 0.0625, 'is_optimistic': False}
    for delay in delays:
        filepath = Path(Path(__file__).parent, 'results_delay', dataset, 'NeuralRandUCB', str(delay))
        os.makedirs(filepath, exist_ok=True)
        for i in range(n_runs):
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            if dataset == 'synthetic':
                bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
            else:
                bandit = Bandit_multi(dataset, seed=seed + i)    
            learner = NeuralTSDiag(bandit.dim, hidden=100, style='rand_ucb', delay=delay, **kwargs)
            cumul_regret = train(bandit, learner, T=T)
            f = open(Path(filepath, f'{i}.csv'), 'w')
            f.write(str(cumul_regret[-1]))
            f.close()

    # Neural UCB
    for delay in delays:
        filepath = Path(Path(__file__).parent, 'results_delay', dataset, 'NeuralUCB', str(delay))
        os.makedirs(filepath, exist_ok=True)
        for i in range(n_runs):
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            if dataset == 'synthetic':
                bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
            else:
                bandit = Bandit_multi(dataset, seed=seed + i)    
            learner = NeuralTSDiag(bandit.dim, lamdba=1, nu=1e-1, hidden=100, style='ucb', delay=delay)
            cumul_regret = train(bandit, learner, T=T)
            f = open(Path(filepath, f'{i}.csv'), 'w')
            f.write(str(cumul_regret[-1]))
            f.close()

    # Neural TS
    for delay in delays:
        filepath = Path(Path(__file__).parent, 'results_delay', dataset, 'NeuralTS', str(delay))
        os.makedirs(filepath, exist_ok=True)
        for i in range(n_runs):
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            if dataset == 'synthetic':
                bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=6000, seed=seed + i)
            else:
                bandit = Bandit_multi(dataset, seed=seed + i)    
            learner = NeuralTSDiag(bandit.dim, lamdba=1, nu=1e-2, hidden=100, style='ts', delay=delay)
            cumul_regret = train(bandit, learner, T=T)
            f = open(Path(filepath, f'{i}.csv'), 'w')
            f.write(str(cumul_regret[-1]))
            f.close()