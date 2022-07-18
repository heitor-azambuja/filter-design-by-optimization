from scipy.signal import freqz
from jsonHandler import load_json
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--order', type=int, default=2,
                    help='Order of the filter. Defaults to 2.')
parser.add_argument('-fo', '--file_offset', type=int, 
                    help='Offset of the files to be compared.')

args = parser.parse_args()

if args.file_offset is None:
    FILE_OFFSET = ''
else:
    FILE_OFFSET = '{}'.format(args.file_offset)

ORDER = args.order

FILE_1 = 'results/pso-ord{}-metrics{}.json'.format(ORDER, FILE_OFFSET)
FILE_2 = 'results/tribes-ord{}-metrics{}.json'.format(ORDER, FILE_OFFSET)


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
    # ref: https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
    table_data = [
        ['PSO', 'TRIBES'],
        ['Converged', 'yes' if metrics_1['converged'] else 'no', 'yes' if metrics_2['converged'] else 'no'],
        ['Iterations', len(metrics_1['error']), len(metrics_2['error'])],
        ['Final Error', '{:1.6f}'.format(metrics_1['error'][-1]), '{:1.6f}'.format(metrics_2['error'][-1])],
        ['Evaluations', metrics_1['evaluations'], metrics_2['evaluations']],
        ['Execution Time', '{:1.4f}'.format(sum(metrics_1['it_duration'])), '{:1.4f}'.format(sum(metrics_2['it_duration']))],
    ]
    
    column_headers = table_data.pop(0)
    row_headers = [x.pop(0) for x in table_data]
    ax_table = plt.subplot(224)

    # Table data needs to be non-numeric text. Format the data
    # while I'm at it.
    cell_text = []
    for row in table_data:
        cell_text.append([f'{x}' for x in row])

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    # Add a table at the bottom of the axes
    table = ax_table.table(cellText=cell_text,
        rowLabels=row_headers,
        rowColours=rcolors,
        colLabels=column_headers,
        rowLoc='left',
        colColours=ccolors,
        loc='center',
        bbox=[0.25 ,0 ,0.7 ,1]
    )

    ax_table.axis('tight')
    ax_table.axis('off')
    ax_table.set_title('Metrics')

    # table.scale(0.7, 1.5)
    
    plt.draw()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_metrics(FILE_1, FILE_2)
