from jsonHandler import save_json, load_json
import matplotlib.pyplot as plt
import numpy as np


FILE_1 = 'pso-metrics-ord2.json'
FILE_2 = 'tribes-metrics-ord2.json'

def plot_metrics(metrics):
    plt.figure(1)
    # plt.figure(figsize=(10, 8))
    # plt.subplot(121)
    x = np.arange(1, len(metrics['error']) + 1)
    plt.plot(x, metrics['error'],'ro', markersize=4)
    plt.grid(which='both', axis='both')
    plt.title('Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    # plt.subplot(122)
    # plt.plot(metrics['it_duration'])
    # plt.title('Iteration duration')
    # plt.xlabel('Iteration')
    # plt.ylabel('Duration (s)')
    plt.tight_layout()
    plt.show()
    # plt.close()


def compare_metrics(file_1, file_2):
    metrics_1 = load_json(file_1)
    metrics_2 = load_json(file_2)
    # plot_metrics(metrics_1)
    # plot_metrics(metrics_2)

    plt.figure(1, figsize=(14, 5))
    plt.subplot(121)
    x = np.arange(1, len(metrics_1['error']) + 1)
    plt.plot(x, metrics_1['error'],'ro', markersize=4, label='PSO')
    plt.plot(x, metrics_1['error'], 'r')
    
    x2 = np.arange(1, len(metrics_2['error']) + 1)
    plt.plot(x2, metrics_2['error'], 'bx', label='Tribes')
    plt.plot(x2, metrics_2['error'], 'b')

    plt.grid(which='both', axis='both')
    plt.axis('tight')

    plt.legend()
    plt.title('Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')

    plt.subplot(122)

    plt.plot(x, metrics_1['it_duration'], 'r', label='PSO')
    plt.plot(x2, metrics_2['it_duration'], 'b--', label='Tribes')

    plt.grid(which='both', axis='both')

    plt.legend()
    plt.title('Iteration duration')
    plt.xlabel('Iteration')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_metrics(FILE_1, FILE_2)
