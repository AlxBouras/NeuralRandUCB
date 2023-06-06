import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

from pathlib import Path


if __name__ == '__main__':
    dataset = 'mushroom'

    # Neural RandUCB
    filepath = Path(Path(__file__).parent, 'results_delay', dataset, 'NeuralRandUCB')
    mean_regrets = []
    std_regrets = []
    delays = []
    for delay in natsorted(os.listdir(filepath)):
        delay_path = Path(filepath, delay)
        cumul_regrets = []
        for run in os.listdir(delay_path):
            csv_path = Path(delay_path, run)
            cumul_regret = np.genfromtxt(csv_path, delimiter=',')
            cumul_regrets.append(cumul_regret)
        mean = np.mean(cumul_regrets)
        std = np.std(cumul_regrets)
        mean_regrets.append(mean)
        std_regrets.append(std)
        delays.append(int(delay))

    plt.errorbar(delays, mean_regrets, yerr=std_regrets, capsize=6, uplims=False, lolims=False, label='NeuralRandUCB')

    # Neural UCB
    filepath = Path(Path(__file__).parent, 'results_delay', dataset, 'NeuralUCB')
    mean_regrets = []
    std_regrets = []
    delays = []
    for delay in natsorted(os.listdir(filepath)):
        delay_path = Path(filepath, delay)
        cumul_regrets = []
        for run in os.listdir(delay_path):
            csv_path = Path(delay_path, run)
            cumul_regret = np.genfromtxt(csv_path, delimiter=',')
            cumul_regrets.append(cumul_regret)
        mean = np.mean(cumul_regrets)
        std = np.std(cumul_regrets)
        mean_regrets.append(mean)
        std_regrets.append(std)
        delays.append(int(delay))

    plt.errorbar(delays, mean_regrets, yerr=std_regrets, capsize=6, uplims=False, lolims=False, label='NeuralUCB ($\lambda$ = 1, $\\nu = 10^{-1}$)')


    # Neural TS
    filepath = Path(Path(__file__).parent, 'results_delay', dataset, 'NeuralTS')
    mean_regrets = []
    std_regrets = []
    delays = []
    for delay in natsorted(os.listdir(filepath)):
        delay_path = Path(filepath, delay)
        cumul_regrets = []
        for run in os.listdir(delay_path):
            csv_path = Path(delay_path, run)
            cumul_regret = np.genfromtxt(csv_path, delimiter=',')
            cumul_regrets.append(cumul_regret)
        mean = np.mean(cumul_regrets)
        std = np.std(cumul_regrets)
        mean_regrets.append(mean)
        std_regrets.append(std)
        delays.append(int(delay))

    plt.errorbar(delays, mean_regrets, yerr=std_regrets, capsize=6, uplims=False, lolims=False, label='NeuralTS ($\lambda$ = 1, $\\nu = 10^{-2})$')
    plt.legend()
    plt.xlabel('Reward Delay')
    plt.ylabel('Total Regret')

    plt.savefig(f"reward_delay_{dataset}.png")

