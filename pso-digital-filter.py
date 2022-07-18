from scipy.signal import firwin, freqz
from time import time
import matplotlib.pyplot as plt
import numpy as np
from jsonHandler import save_json
import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-bp', '--bandpass', action='store_true', help='Use bandpass filter (default)')
group.add_argument('-lp', '--lowpass', action='store_true', help='Use lowpass filter')

parser.add_argument('-o', '--order', type=int, default=2,
                    help='Order of the filter. Defaults to 2.'
)
parser.add_argument('-i', '--iterations', type=int, default=2000,
                    help='Number of iterations. Defaults to 2000.'
)
parser.add_argument('-po', '--population', type=int, default=100, help='Swarm size. Defaults to 100.')
parser.add_argument('-p', '--plot', action='store_true', help='Plot error and filter response.')
parser.add_argument('-s', '--save', action='store_true', help='Save metrics to json file.')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')

args = parser.parse_args()

# PSO config
ITERATIONS = args.iterations
POPULATION = args.population
PERSONAL_WEIGHT = 2
SOCIAL_WEIGHT = 2
ERROR_THRESHOLD = 0.01
INERTIA = 1
MAX_VEL = 2
MIN_VEL = -2
MAX_POSITION = 1
MIN_POSITION = -1
DAMPINIG = 0.999
# DAMPINIG = 1

# filter config
GOAL_ORDER = args.order
ORDER = GOAL_ORDER
N_COEFICIENTS = ORDER

PLOT_GRAPHS = args.plot
SAVE_METRICS = args.save
FILE_NAME = 'results/pso-ord{}-metrics.json'.format(ORDER)
VERBOSE = args.verbose

metrics = {
    'converged': False,
    'reachedMaxIt': False,
    'error': [],
    'it_duration': [],
    'evaluations': 0,
    'final_denominator': [],
    'parameters': {
        'order': ORDER,
        'goal_order': GOAL_ORDER,
        'dampening': DAMPINIG,
        'max_vel': MAX_VEL,
        'min_vel': MIN_VEL,
        'max_position': MAX_POSITION,
        'min_position': MIN_POSITION,
        'desired_filter_type': None,
        'desired_filter_denominator': None,
        'population': POPULATION,
        'personal_weight': PERSONAL_WEIGHT,
        'social_weight': SOCIAL_WEIGHT,
        'error_threshold': ERROR_THRESHOLD,
        'inertia': INERTIA,
        'iterations': ITERATIONS,
    }
}

if args.lowpass: # lowpass filter
    metrics['parameters']['desired_filter_type'] = 'lowpass'
    goal_h = firwin(GOAL_ORDER, [0.4], pass_zero='lowpass')
else: # bandpass filter [DEFAULT]
    metrics['parameters']['desired_filter_type'] = 'bandpass'
    goal_h = firwin(GOAL_ORDER, [0.4, 0.7], pass_zero='bandpass')

if VERBOSE:
    print('Filter type: {}'.format(metrics['parameters']['desired_filter_type']))
    print('Desired filter denominator: {}'.format(goal_h))

metrics['parameters']['desired_filter_denominator'] = goal_h.tolist()

desired_w, desired_h = freqz(goal_h)

if PLOT_GRAPHS:
    plt.figure(1, figsize=(12, 5))
    ax_response = plt.subplot(122, label='response')
    ax_response.plot(desired_w/np.pi, 20 * np.log10(abs(desired_h)), linewidth=6, label='Desired')
    plt.axis('tight')
    plt.xlabel('Frequency (normalized)')
    plt.ylabel('Amplitude response [dB]')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.title('Frequency Response')

    ax_error = plt.subplot(121, label='error')
    plt.title('Convergency')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.grid(which='both', axis='both')

    plt.tight_layout()

position = []
velocity = []
particle_best_pos = []
particle_best_error = []
global_best_pos = np.ndarray([])
global_best_error = np.inf

def calculateError(denominator):
    metrics['evaluations'] += 1
    error = 0
    _, h = freqz(denominator)
    for i in range(len(h)):
        e = (abs(h[i]) - abs(desired_h[i])) ** 2
        # e = (abs(desired_h[i]) - abs(h[i])) ** 2
        error += e
    return error**0.5


if __name__ == '__main__':
    # Initialize Particles
    for i in range(POPULATION):
        position.append(np.random.uniform(MIN_POSITION, MAX_POSITION, N_COEFICIENTS))
        
        velocity.append(np.random.uniform(MIN_VEL, MAX_VEL, N_COEFICIENTS))
        
        particle_best_pos.append(position[i])

        currentError = calculateError(position[i])
        particle_best_error.append(currentError)
        if currentError < global_best_error:
            global_best_pos = position[i]
            global_best_error = currentError


    # Run PSO
    for i in range(ITERATIONS):
        # measure iteration duration
        start = time()

        if ((i + 1) % 50 == 0) and VERBOSE:
            print('{} Iterations.'.format(i + 1))

        for j in range(POPULATION):
            # Update Velocity

            velocity[j] = INERTIA * velocity[j] + PERSONAL_WEIGHT * np.random.rand() * (particle_best_pos[j] - position[j]) + SOCIAL_WEIGHT * np.random.rand() * (global_best_pos - position[j])

            # Update Position
            position[j] = position[j] + velocity[j]

            # Clamp position
            position[j] = np.clip(position[j], MIN_POSITION, MAX_POSITION)

            # Update Particle Best
            currentError = calculateError(position[j])
            if currentError < particle_best_error[j]:
                particle_best_pos[j] = position[j]
                particle_best_error[j] = currentError

                # Update Global Best
                if currentError < global_best_error:
                    if VERBOSE:
                        print('New Global Best! Error: {}'.format(currentError))
                    global_best_pos = position[j]
                    global_best_error = currentError

        metrics['error'].append(global_best_error)

        if PLOT_GRAPHS:
            plt.ion()
            plt.show()
            ax_error.plot(i + 1, global_best_error, 'ro', markersize=4)
            plt.pause(0.001)
        
        if global_best_error <= ERROR_THRESHOLD:
            if VERBOSE:
                print('Reached error threshold on iteration {}.'.format(i + 1))
            metrics['it_duration'].append(time() - start)
            metrics['converged'] = True
            break
        
        if (i + 1) == ITERATIONS:
            if VERBOSE:
                print('Reached maximum number of iterations')
            metrics['it_duration'].append(time() - start)
            metrics['reachedMaxIt'] = True
            break

        # Damping Inertia
        INERTIA *= DAMPINIG

        metrics['it_duration'].append(time() - start)

    metrics['final_denominator'] = global_best_pos
    
    if VERBOSE:
        print('Final denominator: {}'.format(global_best_pos))
    
    final_w, final_h = freqz(global_best_pos)

    if PLOT_GRAPHS:
        plt.ioff()
        ax_response.plot(final_w / np.pi, 20 * np.log10(abs(final_h)), 'r--', linewidth=3, label='Result')
        ax_response.legend()
        plt.show()

    if SAVE_METRICS:
        save_json(metrics, FILE_NAME)