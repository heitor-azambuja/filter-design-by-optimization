from jsonHandler import save_json, load_json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--order', type=int, default=2,
                    help='Order of the filter. Defaults to 2.')
parser.add_argument('-n', '--number_of_files', type=int, default=4, 
                    help='Number of files to be averaged. How many offsets of the same file name. Defaults to 4')
parser.add_argument('-s', '--save', action='store_true', 
                    help='Save metrics to json file.')
parser.add_argument('-sf', '--save_figure', action='store_true',
                    help='Save figure as png. It will not be shown.')

args = parser.parse_args()

FILES_NUM = args.number_of_files
ORDER = args.order

SAVE_AVERAGED_METRICS = args.save
SAVE_FIGURE = args.save_figure
PATH = 'results/img/average/'
FIGURE_NAME = 'ord{}-metrics-averaged'.format(ORDER)

first_pso_file = 'results/pso-ord{}-metrics.json'.format(ORDER)
first_tribes_file = 'results/tribes-ord{}-metrics.json'.format(ORDER)

pso_files = [first_pso_file]
tribes_files = [first_tribes_file]

for i in range(FILES_NUM - 1):
    pso_files.append('results/pso-ord{}-metrics{}.json'.format(ORDER, i))
    tribes_files.append('results/tribes-ord{}-metrics{}.json'.format(ORDER, i))


# from: https://stackoverflow.com/questions/10058227/calculating-mean-of-arrays-with-different-lengths
def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def average_metrics(files):
    errors = []
    it_intervals = []
    avg_iterations = 0
    avg_evaluations = 0
    for file in files:
        metrics_file = load_json(file)
        errors.append(metrics_file['error'])
        it_intervals.append(metrics_file['it_duration'])
        avg_iterations += len(metrics_file['error'])
        avg_evaluations += metrics_file['evaluations']
    
    metrics = {}
    metrics['avg_error'], metrics['error_std_deviation'] = tolerant_mean(errors)
    metrics['avg_it_duration'], metrics['it_duration_std_deviation'] = tolerant_mean(it_intervals)
    metrics['avg_iterations'] = avg_iterations / FILES_NUM
    metrics['avg_evaluations'] = avg_evaluations / FILES_NUM

    return metrics


if __name__ == '__main__':
    pso_avg_metrics = average_metrics(pso_files)
    tribes_avg_metrics = average_metrics(tribes_files)

    if SAVE_AVERAGED_METRICS:
        save_json(pso_avg_metrics, 'results/pso-ord{}-metrics-averaged.json'.format(ORDER))
        save_json(tribes_avg_metrics, 'results/tribes-ord{}-metrics-averaged.json'.format(ORDER))

    # y, error = tolerant_mean(it_intervals)
    # plt.figure(1)
    # plt.plot(np.arange(len(y))+1, y, color='green', label='PSO')
    # plt.fill_between(np.arange(len(y))+1, y-error, y+error, color='green', alpha=0.2)
    # plt.grid()
    # plt.show()

    plt.figure('Averaged PSO and TRIBES comparison', figsize=(16, 5))
    plt.suptitle('Comparsion of PSO and TRIBES metrics, {} executions average, filter order {}'.format(FILES_NUM, ORDER))

    # Error plot
    ax_error = plt.subplot(131)
    x = np.arange(1, len(pso_avg_metrics['avg_error']) + 1)
    ax_error.plot(x, pso_avg_metrics['avg_error'],'ro', markersize=4, label='PSO')
    ax_error.plot(x, pso_avg_metrics['avg_error'], 'r')
    
    x2 = np.arange(1, len(tribes_avg_metrics['avg_error']) + 1)
    ax_error.plot(x2, tribes_avg_metrics['avg_error'], 'bo', markersize=2, label='Tribes')
    ax_error.plot(x2, tribes_avg_metrics['avg_error'], 'b')

    plt.grid(which='both', axis='both')
    plt.axis('tight')

    plt.legend()
    plt.title('Averaged Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')

    # Duration plot
    ax_duration = plt.subplot(132)
    ax_duration.plot(x, pso_avg_metrics['avg_it_duration'], 'r', label='PSO')
    ax_duration.fill_between(x, np.array(pso_avg_metrics['avg_it_duration']) - np.array(pso_avg_metrics['it_duration_std_deviation']), 
                    np.array(pso_avg_metrics['avg_it_duration']) + np.array(pso_avg_metrics['it_duration_std_deviation']), 
                    color='r', alpha=0.2)
    ax_duration.plot(x2, tribes_avg_metrics['avg_it_duration'], 'b', label='Tribes')
    ax_duration.fill_between(x2, np.array(tribes_avg_metrics['avg_it_duration']) - np.array(tribes_avg_metrics['it_duration_std_deviation']),
                    np.array(tribes_avg_metrics['avg_it_duration']) + np.array(tribes_avg_metrics['it_duration_std_deviation']),
                    color='b', alpha=0.2)

    plt.grid(which='both', axis='both')
    plt.legend()
    plt.title('Averaged Iteration duration with Deviation')
    plt.xlabel('Iteration')
    plt.ylabel('Seconds')

    # table plot
    # ref: https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
    table_data = [
        ['PSO', 'TRIBES'],
        ['Iterations', pso_avg_metrics['avg_iterations'], tribes_avg_metrics['avg_iterations']],
        ['Final Error', '{:1.6f}'.format(pso_avg_metrics['avg_error'][-1]), '{:1.6f}'.format(tribes_avg_metrics['avg_error'][-1])],
        ['Evaluations', pso_avg_metrics['avg_evaluations'], tribes_avg_metrics['avg_evaluations']],
        ['Execution Time', '{:1.4f}'.format(sum(pso_avg_metrics['avg_it_duration'])), '{:1.4f}'.format(sum(tribes_avg_metrics['avg_it_duration']))],
    ]
    
    column_headers = table_data.pop(0)
    row_headers = [x.pop(0) for x in table_data]
    ax_table = plt.subplot(133)

    cell_text = []
    for row in table_data:
        cell_text.append([f'{x}' for x in row])

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    ax_table.table(cellText=cell_text,
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
    ax_table.set_title('Averaged Metrics')

    plt.tight_layout()
    
    if SAVE_FIGURE:
        counter = -1
        for i in os.listdir(PATH):
            if i.startswith(FIGURE_NAME):
                counter += 1
        
        suffix = '.png'
        if counter >= 0:
            suffix = str(counter) + suffix

        plt.savefig(PATH + FIGURE_NAME + suffix)
    else:
        plt.show()
