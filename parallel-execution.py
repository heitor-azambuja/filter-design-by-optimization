import multiprocessing as mp
import os

# Create an individial process for everry script execution
# Warning: This makes all my cpu corers go 100% for a while.
# Run at your own risk.

FILE_1 = 'pso-digital-filter.py -s -o '
FILE_2 = 'TRIBES-digital-filter.py -s -o '
EXECUTIONS = 4
ORDERS = [2, 6, 8, 10, 15, 20]
# PROC_NUM = 2 * EXECUTIONS * len(ORDERS)
PROC_NUM = 8

all_processes = []

for order in ORDERS:
	all_processes += [FILE_1 + str(order) for i in range(EXECUTIONS)]
	all_processes += [FILE_2 + str(order) for i in range(EXECUTIONS)]


def execute(process):                                                             
	os.system(f'python3 {process}')   

if __name__ == '__main__':

	process_pool = mp.Pool(processes = PROC_NUM)
	process_pool.map(execute, all_processes)