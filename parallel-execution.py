import multiprocessing as mp
import os
import argparse

# Create an individial process for every script execution

# WARNING: Running this script with a high number of processes 
# will cause your cpu to go 100% for a while. Run at your own risk.

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--processes', type=int, default=8, 
					help='Number of processes to run. Defaults to 8.\
						WARNING: Running this script with a high number of processes\
						will cause your cpu to go 100%% for a while. Run at your own risk.'
)

args = parser.parse_args()

FILE_1 = 'pso-digital-filter.py -s -o '
FILE_2 = 'TRIBES-digital-filter.py -s -o '
EXECUTIONS = 4
ORDERS = [2, 6, 8, 10, 15, 20]
# PROC_NUM = 2 * EXECUTIONS * len(ORDERS)
PROC_NUM = args.processes

all_processes = []

for order in ORDERS:
	all_processes += [FILE_1 + str(order) for i in range(EXECUTIONS)]
	all_processes += [FILE_2 + str(order) for i in range(EXECUTIONS)]


def execute(process):                                                             
	os.system(f'python3 {process}')   


if __name__ == '__main__':
	process_pool = mp.Pool(processes = PROC_NUM)
	process_pool.map(execute, all_processes)