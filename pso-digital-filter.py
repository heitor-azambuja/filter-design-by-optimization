from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt
import numpy as np

# PSO config
ITERATIONS = 2000
POPULATION = 100
PERSONAL_WEIGHT = 2
SOCIAL_WEIGHT = 2
ERROR_THRESHOLD = 0.001
INERTIA = 1

MAX_VEL = 2
MIN_VEL = -2
MAX_POSITION = 1
MIN_POSITION = -1

DAMPINIG = 0.9995
# DAMPINIG = 1

# filter config
DESIRED_FILTER = 2  # 1 = lowpass; any = bandpass
GOAL_ORDER = 15
ORDER = GOAL_ORDER
N_COEFICIENTS = ORDER


if DESIRED_FILTER == 1:
    print('lowpass')
    goal_h = firwin(GOAL_ORDER, [0.4], pass_zero='lowpass')
else:
    print('bandpass')
    goal_h = firwin(GOAL_ORDER, [0.4, 0.7], pass_zero='bandpass')
print('Desired filter response: {}'.format(goal_h))


desired_w, desired_h = freqz(goal_h)


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

position = []
velocity = []
particle_best = []
global_best = []


def calculateError(denominator):
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
        
        particle_best.append(position[i])

        if i == 0:
            global_best = position[0]
        elif calculateError(position[i]) < calculateError(global_best):
            global_best = position[i]

    # Run PSO
    for i in range(ITERATIONS):
        if (i + 1) % 50 == 0:
            print('{} Iterations.'.format(i + 1))

        for j in range(POPULATION):
            # Update Velocity
            velocity[j] = INERTIA * velocity[j] + PERSONAL_WEIGHT * np.random.rand() * (particle_best[j] - position[j]) + SOCIAL_WEIGHT * np.random.rand() * (global_best - position[j])

            # Update Position
            position[j] = position[j] + velocity[j]

            # Clamp position
            position[j] = np.clip(position[j], MIN_POSITION, MAX_POSITION)

            # Update Particle Best
            currentError = calculateError(position[j])
            if currentError < calculateError(particle_best[j]):
                particle_best[j] = position[j]

                # Update Global Best
                if currentError < calculateError(global_best):
                    print('New Global Best! Error: {}'.format(currentError))
                    global_best = position[j]

        globalError = calculateError(global_best)
        plt.figure(2)
        plt.ion()
        plt.show()
        plt.plot(i + 1, globalError, 'ro', markersize=4)
        plt.pause(0.001)
        
        # print(globalError)
        
        if globalError <= ERROR_THRESHOLD:
            print('Reached error threshold on iteration {}.'.format(i + 1))
            break
        
        # Damping Inertia
        INERTIA *= DAMPINIG

    print('Final denominator: {}'.format(global_best))
    final_w, final_h = freqz(global_best)


    plt.ioff()
    plt.figure(1)
    plt.plot(final_w / np.pi, 20 * np.log10(abs(final_h)))
    plt.show()
    # plt.xlabel('Frequency (normalized)')
    # plt.ylabel('Amplitude response [dB]')
    # plt.grid(which='both', axis='both')