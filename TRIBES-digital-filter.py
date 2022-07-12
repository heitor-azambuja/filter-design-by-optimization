from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt
import numpy as np
from jsonHandler import save_json

ITERATIONS = 2000
ERROR_THRESHOLD = 0.01
MAX_POSITION = 1
MIN_POSITION = -1

# filter config
DESIRED_FILTER = 2  # 1 = lowpass; any = bandpass
GOAL_ORDER = 2
ORDER = GOAL_ORDER
N_COEFICIENTS = ORDER

PLOT_GRAPHS = False
SAVE_METRICS = True
FILE_NAME = 'tribes-metrics-ord{}.json'.format(ORDER)

metrics = {
    'iterations': ITERATIONS,
    'evaluations': 0,
    'adaptations': 0,
    'pivot': 0,
    'disturbed_pivot': 0,
    'gaussian': 0,
}


if DESIRED_FILTER == 1:
    print('lowpass')
    goal_h = firwin(GOAL_ORDER, [0.4], pass_zero='lowpass')
else:
    print('bandpass')
    goal_h = firwin(GOAL_ORDER, [0.4, 0.7], pass_zero='bandpass')
print('Desired filter response: {}'.format(goal_h))

desired_w, desired_h = freqz(goal_h)

if PLOT_GRAPHS:
    plt.figure(1)
    plt.plot(desired_w/np.pi, 20 * np.log10(abs(desired_h)))
    plt.axis('tight')
    plt.xlabel('Frequency (normalized)')
    plt.ylabel('Amplitude response [dB]')
    plt.grid(which='both', axis='both')
    # plt.show()

    plt.figure(2)

    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.grid(which='both', axis='both')


def calculateError(denominator):
    metrics['evaluations'] += 1
    error = 0
    _, h = freqz(denominator)
    for i in range(len(h)):
        e = (abs(h[i]) - abs(desired_h[i])) ** 2
        # e = (abs(desired_h[i]) - abs(h[i])) ** 2
        error += e
    return error**0.5


dimensions = ORDER

# tribeNb = 1

tribe = {}
currentIteration = 0
nextAdaptation = 0
NL = 1
g = np.inf
g_pos = []

def swarm_adaptation():
    # print('Swarm adaptation')
    metrics['adaptations'] += 1
    global g, g_pos, tribe
    
    tribes_to_remove = []
    created_particles = False
    new_particles = {
        'position': [],
        'last_error': [],
        'current_error': [],
        'particle_best_pos': [],
        'particle_best_error': [],
        'particle_status': [],
        'tribe_best_idx': 0,
        'tribe_best_error': np.inf,
        'tribe_worst_idx': 0,
        'tribe_worst_error': -np.inf,
        'quality': ''
    }
    
    # print('\n')
    for i in tribe:
        # print(i)
        if tribe[i]['quality'] == 'bad':
            var = round((9.5 + 0.124 * (dimensions - 1)) / len(tribe))
            newParticlesNb = max(2, var)
            for j in range(newParticlesNb):
                #  Generate new particles
                rand_chance = np.random.uniform(0, 1)
                if rand_chance <= 0.5:  # create free particle
                    new_particles['position'].append(np.random.uniform(MIN_POSITION, MAX_POSITION, dimensions))
                else:  # create confined particle
                    tribe_best_idx = tribe[i]['tribe_best_idx']
                    radius = np.subtract(g_pos, tribe[i]['particle_best_pos'][tribe_best_idx])

                    low1 = np.subtract(tribe[i]['particle_best_pos'][tribe_best_idx], radius)
                    high1 = np.add(tribe[i]['particle_best_pos'][tribe_best_idx], radius)
                    
                    new_particles['position'].append(np.random.uniform(low1, high1))
                
                error = calculateError(new_particles['position'][-1])
                new_particles['last_error'].append(error)
                new_particles['current_error'].append(error)
                new_particles['particle_best_pos'].append(new_particles['position'][-1])
                new_particles['particle_best_error'].append(error)
                new_particles['particle_status'].append('==')
                
                if error < new_particles['tribe_best_error']:
                    new_particles['tribe_best_idx'] = len(new_particles['position']) - 1
                    new_particles['tribe_best_error'] = error
                    if error < g:
                        print("Generated particle on the new global best: {}".format(error))
                        g = error
                        g_pos = new_particles['position'][-1]
                
                if error > new_particles['tribe_worst_error']:
                    new_particles['tribe_worst_idx'] = len(new_particles['position']) - 1
                    new_particles['tribe_worst_error'] = error
            created_particles = True
        
        elif tribe[i]['quality'] == 'good':
            if len(tribe[i]['position']) == 1:
                #  Remove tribe if not the best particle
                if tribe[i]['tribe_best_error'] != g:
                    tribes_to_remove.append(i)
            else:
                # remove worst particle of tribe i
                worst_idx = tribe[i]['tribe_worst_idx']
                tribe[i]['position'].pop(worst_idx)
                tribe[i]['last_error'].pop(worst_idx)
                tribe[i]['current_error'].pop(worst_idx)
                tribe[i]['particle_best_pos'].pop(worst_idx)
                tribe[i]['particle_best_error'].pop(worst_idx)
                tribe[i]['particle_status'].pop(worst_idx)
                tribe[i]['tribe_worst_error'] = -np.inf
                # Update tibe best particle index
                tribe[i]['tribe_best_error'] = np.inf
                for j in range(len(tribe[i]['particle_best_error'])):
                    if tribe[i]['particle_best_error'][j] <= tribe[i]['tribe_best_error']:
                        tribe[i]['tribe_best_idx'] = j
                        tribe[i]['tribe_best_error'] = tribe[i]['particle_best_error'][j]
                

    if created_particles:
        # Aggregate all the generated particles to the new tribe
        new_tribe_idx = max(tribe.keys()) + 1
        tribe[new_tribe_idx] = new_particles
    
    # Delete tribes marked for removal
    for i in tribes_to_remove:
        del tribe[i]
    # Compute NL
    # NL is the number of informations links at thhe time of the last adaptation
    # The next adaptation will occur after NL/2 iterations.
    # NL is estimated using the following equation:
    NL = 0
    tribeNb = len(tribe)
    for i in tribe:
        NL += (len(tribe[i]['position']) ** 2) + (tribeNb * (tribeNb - 1))
            

def pivot(t, p_idx):
    metrics['pivot'] += 1
    # print('Pivot')
    # Determine particle best informant
    # check if this is the best particle of the tribe
    if tribe[t]['tribe_best_idx'] == p_idx:
        informant_best = g
    else:
        informant_best = tribe[t]['tribe_best_error']
    particle_best_err = tribe[t]['particle_best_error'][p_idx]
    
    c1 = particle_best_err / (particle_best_err + informant_best)
    c2 = informant_best / (particle_best_err + informant_best)
    
    # particle best position
    p_pos = tribe[t]['particle_best_pos'][p_idx]
    # radius of the hypersphere
    radius = np.subtract(p_pos, g_pos)

    low1 = np.subtract(p_pos, radius)
    high1 = np.add(p_pos, radius)

    low2 = np.subtract(g_pos, radius)
    high2 = np.add(g_pos, radius)

    new_position = np.add(np.multiply(c1, np.random.uniform(low1, high1)), np.multiply(c2, np.random.uniform(low2, high2)))
    return new_position


def disturbed_pivot(t, p_idx):
    metrics['disturbed_pivot'] += 1
    # print('Disturbed pivot')
    new_position = pivot(t, p_idx)
    # Determine particle best informant
    # check if this is the best particle of the tribe
    if tribe[t]['tribe_best_idx'] == p_idx:
        informant_best = g
    else:
        informant_best = tribe[t]['tribe_best_error']
    particle_best_err = tribe[t]['particle_best_error'][p_idx]
    
    b_range = np.absolute((particle_best_err - informant_best) / (particle_best_err + informant_best))
    b = np.random.uniform(0, b_range)
    new_position = np.multiply((1 + b), new_position)
    
    return new_position
    

def gaussian(t, p_idx):
    metrics['gaussian'] += 1
    # print('Gaussian')
    # Determine particle best informant
    # check if this is the best particle of the tribe
    if tribe[t]['tribe_best_idx'] == p_idx:
        informant_best = g
    else:
        informant_best = tribe[t]['tribe_best_error']
    # particle_best_err = tribe[t]['particle_best_error'][p_idx]82

    # particle best position
    p_pos = tribe[t]['particle_best_pos'][p_idx]

    center = np.subtract(informant_best, p_pos)
    mean = np.absolute(center)
    new_position = np.random.normal(center,mean)

    return new_position


# In the beginning, the swarm is composed of only one particle that represents a single tribe.
# If, during the first iteration, this particle does not improve its location, new ones are created,
# forming a second tribe. During the second iteration, the same process is repeated
# and this process continues in subsequent iterations.
if __name__ == '__main__':
    # Initialize a population of particles with random positions
    # For each individual i, pi is initialized to Xi
    tribe = {
        0: {
            'position': [],
            'last_error': [],
            'current_error': [],
            'particle_best_pos': [],
            'particle_best_error': [],
            'particle_status': [],
            'tribe_best_idx': 0,
            'tribe_best_error': 0,
            'tribe_worst_idx': 0,
            'tribe_worst_error': 0,
            'quality': ''
        }
    }
    tribe[0]['position'].append(np.random.uniform(MIN_POSITION, MAX_POSITION, N_COEFICIENTS))

    tribe[0]['particle_best_pos'].append(tribe[0]['position'][0])
    
    tribe[0]['particle_status'].append('==')
    #  randomize first particle status
    # tribe[0]['particle_status'].append(np.random.choice(['+','-', '=']))
    # tribe[0]['particle_status'][0] += np.random.choice(['+','-', '='])
    
    # Evaluate the objective function for each particle and compute g
    g = calculateError(tribe[0]['position'][0])
    g_pos = tribe[0]['position'][0]
    tribe[0]['particle_best_error'].append(g)
    tribe[0]['last_error'].append(g)
    tribe[0]['current_error'].append(g)
    tribe[0]['tribe_best_error'] = g
    
    # Sanity check
    print('\n\n')
    print(tribe)
    print('\n\n')
    print(g)
    print('\n\n')

    NL = 1
    n = 0
    
    for i in range(ITERATIONS):
        if (i + 1) % 50 == 0:
            print('{} Iterations.'.format(i + 1))
        
        n += 1
        # for t in range(len(tribe)):
        for t in tribe:
            for p in range(len(tribe[t]['position'])):
                # Determine statuses of all particles
                if tribe[t]['current_error'][p] > tribe[t]['last_error'][p]:
                    tribe[t]['particle_status'][p] += '+'
                elif tribe[t]['current_error'][p] < tribe[t]['last_error'][p]:
                    tribe[t]['particle_status'][p] += '-'
                else:
                    tribe[t]['particle_status'][p] += '='
                if len(tribe[t]['particle_status'][p]) > 2:
                    tribe[t]['particle_status'][p] = tribe[t]['particle_status'][p][1:]
                    
                # Choose the displacement strategies
                # Update the positions of the particles
                p_status = tribe[t]['particle_status'][p]
                if p_status in ['=+', '++']:
                    tribe[t]['position'][p] = gaussian(t, p)
                elif p_status in ['+=', '-+']:
                    tribe[t]['position'][p] = disturbed_pivot(t, p)
                else:  # p_status in ['--', '=-', '+=', '-=', '==']:
                    tribe[t]['position'][p] = pivot(t, p)
        
                # Evaluate the objective function for each particle
                tribe[t]['last_error'][p] = tribe[t]['current_error'][p]
                tribe[t]['current_error'][p] = calculateError(tribe[t]['position'][p])

                # Compute new pi (particle_best) and g (tribe or global best)
                if tribe[t]['current_error'][p] < tribe[t]['particle_best_error'][p]:
                    tribe[t]['particle_best_error'][p] = tribe[t]['current_error'][p]
                    tribe[t]['particle_best_pos'][p] = tribe[t]['position'][p]
                    if tribe[t]['current_error'][p] < tribe[t]['tribe_best_error']:
                        tribe[t]['tribe_best_error'] = tribe[t]['current_error'][p]
                        tribe[t]['tribe_best_idx'] = p
                        if tribe[t]['current_error'][p] < g:
                            g = tribe[t]['current_error'][p]
                            g_pos = tribe[t]['position'][p]
                            
                            print('New Global Best: {}'.format(tribe[t]['current_error'][p]))
                
                # Update the tribe worst
                if tribe[t]['current_error'][p] > tribe[t]['tribe_worst_error']:
                    tribe[t]['tribe_worst_error'] = tribe[t]['current_error'][p]
                    tribe[t]['tribe_worst_idx'] = p

        if PLOT_GRAPHS:
            plt.figure(2)
            plt.ion()
            plt.show()
            plt.plot(i + 1, g, 'ro', markersize=4)
            plt.pause(0.001)

        # Stop if the swarm converged
        if g <= ERROR_THRESHOLD:
            print('Converged on iteration {}'.format(i + 1))
            break
        
        if (i + 1) == ITERATIONS:
            print('Reached maximum number of iterations')
            break
        
        if n > round(NL / 2):
            # Determine the qualities of the tribes
            for t in tribe:
                got_better = False
                for p in range(len(tribe[t]['position'])):
                    if tribe[t]['particle_status'][p][1] == '+':
                        got_better = True
                        break
                if got_better:
                    tribe[t]['quality'] = 'good'
                else:
                    rand = np.random.uniform(0, 1)
                    if rand < 0.5:
                        tribe[t]['quality'] = 'bad'
                    else:
                        tribe[t]['quality'] = 'good'
            # Adapt swarm
            # Compute NL
            swarm_adaptation()
            n = 0

    print('Final denominator: {}'.format(g_pos))
    final_w, final_h = freqz(g_pos)

    if PLOT_GRAPHS:
        plt.ioff()
        plt.figure(1)
        plt.plot(final_w / np.pi, 20 * np.log10(abs(final_h)))
        plt.show()