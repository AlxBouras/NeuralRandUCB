
import dataset
from bandits import *
from evaluator import evaluate

import os
from matplotlib import pyplot as plt
from pathlib import Path


if __name__ == '__main__':
    files = ("./dataset/R6/ydata-fp-td-clicks-v1_0.20090501")
    dataset.get_yahoo_events(files)
    dim = dataset.n_context_features * dataset.n_arms

    seed = 42

    kwargs = {'N': 20, 'decoupled_arms': None, 'std': 0.0625, 'is_optimistic': False}

    algos = [NeuralTSDiag(dim, n_hidden=1, hidden_size=100, style='rand_ucb', **kwargs),
             NeuralTSDiag(dim, lamdba=1, nu=1e-1, n_hidden=1, hidden_size=100, style='ucb'),
             NeuralTSDiag(dim, lamdba=1, nu=1e-2, n_hidden=1, hidden_size=100, style='ts'),
             LinUCB(0.3, context="user")]
    
    for algo in algos:
        torch.manual_seed(seed)
        np.random.seed(seed)
        _, deploy = evaluate(algo, size=100)
        filepath = Path(Path(__file__).parent, 'results', algo.name + '.csv')
        np.savetxt(filepath, deploy, delimiter=',')
