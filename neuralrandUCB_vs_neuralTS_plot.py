import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


if __name__ == '__main__':
    dataset = 'synthetic'

    # TS    
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'ts')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='NeuralTS ($\lambda$ = 1, $\\nu = 10^{-2}$)')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()

    # RandUCB  
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'randucb_ucb')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='NeuralRandUCB - Configuration 1')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()


    # RandUCB  
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'randucb_ts')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='NeuralRandUCB - Configuration 2')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()    

    # RandUCB  
    filepath = Path(Path(__file__).parent, 'results_randUCBvsTS', dataset, 'randucb_best')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='NeuralRandUCB - Configuration 3')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()
    plt.xlabel('# of rounds')
    plt.ylabel('Cumulative regret')
    plt.legend()
    plt.savefig("randUCBvsTS.png")