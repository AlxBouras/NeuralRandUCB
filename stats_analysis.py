from datasets import Bandit_multi, SyntheticDataset
from neural_bandits import NeuralTSDiag
from train import train
import numpy as np
import torch
import os
from pathlib import Path


if __name__ == '__main__':
    seed = 42
    T = 2000
    dataset = 'synthetic'
    n_runs = 10

    # Neural RandUCB
    kwargs = {'N': 20, 'decoupled_arms': None, 'std': 0.0625, 'is_optimistic': False}
    filepath = Path(Path(__file__).parent, 'results_stat_analysis', 'NeuralRandUCB')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=T, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, hidden=100, style='rand_ucb', **kwargs)
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralRandUCB_{seed + i}.csv'), cumul_regret, delimiter=',')

    # Neural UCB
    filepath = Path(Path(__file__).parent, 'results_stat_analysis', 'NeuralUCB')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=T, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, lamdba=1, nu=1e-1, hidden=100, style='ucb')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralUCB_{seed + i}.csv'), cumul_regret, delimiter=',')

    # Neural TS
    filepath = Path(Path(__file__).parent, 'results_stat_analysis', 'NeuralTS')
    os.makedirs(filepath, exist_ok=True)
    for i in range(n_runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        if dataset == 'synthetic':
            bandit = SyntheticDataset(n_features=5, n_arms=4, order=5, noise_std=0.1, T=T, seed=seed + i)
        else:
            bandit = Bandit_multi(dataset, seed=seed + i)    
        learner = NeuralTSDiag(bandit.dim, lamdba=1, nu=1e-2, hidden=100, style='ts')
        cumul_regret = train(bandit, learner, T=T)
        np.savetxt(Path(filepath, f'neuralTS_{seed + i}.csv'), cumul_regret, delimiter=',')