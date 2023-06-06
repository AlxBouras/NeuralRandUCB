import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

if __name__ == '__main__':
    dataset = 'synthetic'
    results_folder = 'results_performance'

    # LinearUCB    
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'LinearUCB')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='LinearUCB ($\lambda$ = 1, $\\nu = 1$)')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()

    # LinearTS    
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'LinearTS')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='LinearTS ($\lambda$ = 1, $\\nu = 10^{-2}$)')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()

    # KernelUCB   
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'KernelUCB')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='KernelUCB ($\lambda$ = 1, $\\nu = 1$)')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()

    # KernelTS   
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'KernelTS')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='KernelTS ($\lambda$ = 1, $\\nu = 10^{-2}$)')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()

    # NeuralUCB   
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'NeuralUCB')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='NeuralUCB ($\lambda$ = 1, $\\nu = 10^{-1}$)')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.legend()

    # NeuralTS   
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'NeuralTS')
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

    # NeuralRandUCB   
    filepath = Path(Path(__file__).parent, results_folder, dataset, 'NeuralRandUCB')
    cumul_regrets = []
    for csv in os.listdir(filepath):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        cumul_regrets.append(cumul_regret)
    
    cumul_regrets = np.array(cumul_regrets)
    cumul_regrets_mean = np.mean(cumul_regrets, axis=0)
    cumul_regrets_mean_with_std = cumul_regrets_mean + np.std(cumul_regrets, axis=0)

    plt.plot(cumul_regrets_mean, label='NeuralRandUCB')
    plt.fill_between(np.arange(len(cumul_regrets_mean)), cumul_regrets_mean, cumul_regrets_mean_with_std, alpha=0.4)
    plt.xlabel('# of rounds')
    plt.ylabel('Cumulative regret')
    #plt.title(f'Performance on {dataset}')
    plt.legend()

    plt.savefig(f"performance_{dataset}.png")