import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

if __name__ == '__main__':
    results_folder = 'results_stat_analysis'

    # NeuralRandUCB   
    filepath = Path(Path(__file__).parent, results_folder, 'NeuralRandUCB')
    total_cumul_regrets_randucb = []
    for i, csv in enumerate(os.listdir(filepath)):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        total_cumul_regrets_randucb.append(cumul_regret[-1])
        if i == 0:
            plt.plot(cumul_regret, color='blue', label='NeuralRandUCB', alpha=0.5)
        else:
            plt.plot(cumul_regret, color='blue', alpha=0.5)  

    # NeuralUCB
    filepath = Path(Path(__file__).parent, results_folder, 'NeuralUCB')
    total_cumul_regrets_ucb = []
    for i, csv in enumerate(os.listdir(filepath)):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        total_cumul_regrets_ucb.append(cumul_regret[-1])
        if i == 0:
            plt.plot(cumul_regret, color='red', label='NeuralUCB ($\lambda$ = 1, $\\nu = 10^{-1}$)', alpha=0.5)
        else:
            plt.plot(cumul_regret, color='red', alpha=0.5) 

    # NeuralTS   
    filepath = Path(Path(__file__).parent, results_folder, 'NeuralTS')
    total_cumul_regrets_ts = []
    for i, csv in enumerate(os.listdir(filepath)):
        csv_path = Path(filepath, csv)
        cumul_regret = np.genfromtxt(csv_path, delimiter=',')
        total_cumul_regrets_ts.append(cumul_regret[-1])
        if i == 0:
            plt.plot(cumul_regret, color='green', label='NeuralTS ($\lambda$ = 1, $\\nu = 10^{-2}$)', alpha=0.5)
        else:
            plt.plot(cumul_regret, color='green', alpha=0.5) 

    plt.xlabel('# of rounds')
    plt.ylabel('Cumulative regret')
    plt.legend()
    plt.savefig(f"stat_analysis.png")

    # check how many times RandUCB worked better
    total_cumul_regrets_randucb = np.array(total_cumul_regrets_randucb)
    total_cumul_regrets_ucb = np.array(total_cumul_regrets_ucb)
    total_cumul_regrets_ts = np.array(total_cumul_regrets_ts)

    randucb_vs_ucb = total_cumul_regrets_randucb - total_cumul_regrets_ucb
    randucb_vs_ts = total_cumul_regrets_randucb - total_cumul_regrets_ts

    randucb_vs_ucb_count = 0
    randucb_vs_ts_count = 0

    for i in range(len(randucb_vs_ucb)):
        if randucb_vs_ucb[i] < 0:
            randucb_vs_ucb_count += 1
        if randucb_vs_ts[i] < 0:
            randucb_vs_ts_count += 1

    print(f'RandUCB was better {randucb_vs_ucb_count} times out of {len(randucb_vs_ucb)} against UCB')
    print(f'RandUCB was better {randucb_vs_ts_count} times out of {len(randucb_vs_ts)} against TS')

