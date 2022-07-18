from scipy.signal import freqz
from jsonHandler import load_json
import matplotlib.pyplot as plt
import numpy as np


ORDER = 8

FILE_1 = 'results/pso-ord{}-metrics0.json'.format(ORDER)
FILE_2 = 'results/tribes-ord{}-metrics0.json'.format(ORDER)


def compare_metrics(file_1, file_2):
    metrics_1 = load_json(file_1)
    metrics_2 = load_json(file_2)
    # plot_metrics(metrics_1)
    # plot_metrics(metrics_2)

    plt.figure('PSO and TRIBES comparison', figsize=(14, 10))
    plt.suptitle('PSO and TRIBES comparison, filter order {}'.format(ORDER))

    # Error plot
    ax_error = plt.subplot(221)
    x = np.arange(1, len(metrics_1['error']) + 1)
    ax_error.plot(x, metrics_1['error'],'ro', markersize=4, label='PSO')
    ax_error.plot(x, metrics_1['error'], 'r')
    
    x2 = np.arange(1, len(metrics_2['error']) + 1)
    ax_error.plot(x2, metrics_2['error'], 'bo', markersize=2, label='Tribes')
    ax_error.plot(x2, metrics_2['error'], 'b')

    plt.grid(which='both', axis='both')
    plt.axis('tight')

    plt.legend()
    plt.title('Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')

    # Duration plot
    ax_duration = plt.subplot(222)
    ax_duration.plot(x, metrics_1['it_duration'], 'r', label='PSO')
    ax_duration.plot(x2, metrics_2['it_duration'], 'b', label='Tribes')

    plt.grid(which='both', axis='both')
    plt.legend()
    plt.title('Iteration duration')
    plt.xlabel('Iteration')
    plt.ylabel('Seconds')

    # Frequency response plot
    ax_response = plt.subplot(223)
    w_goal, h_goal = freqz(metrics_1['parameters']['desired_filter_denominator'])
    w_pso, h_pso = freqz(metrics_1['final_denominator'])
    w_tribes, h_tribes = freqz(metrics_2['final_denominator'])
    ax_response.plot(w_goal/np.pi, 20 * np.log10(abs(h_goal)), 'g', linewidth=8, label='Desired')
    ax_response.plot(w_pso/np.pi, 20 * np.log10(abs(h_pso)), 'r--', linewidth=4, label='PSO')
    ax_response.plot(w_tribes/np.pi, 20 * np.log10(abs(h_tribes)), 'b-.',linewidth=2, label='Tribes')
    plt.grid(which='both', axis='both')
    plt.axis('tight')
    plt.legend()
    plt.title('Frequency response')
    plt.xlabel('Frequency (normalized)')
    plt.ylabel('Amplitude response [dB]')

    # table plot
    
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_metrics(FILE_1, FILE_2)
