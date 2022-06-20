from scipy.signal import freqs, iirfilter
import matplotlib.pyplot as plt
import numpy as np

# PSO config
ITERATIONS = 500
POPULATION = 20
PERSONAL_WEIGHT = 2
SOCIAL_WEIGHT = 2
ERROR_THRESHOLD = 0.1
INERTIA = 1

MAX_VEL = 1000
MIN_VEL = -1000
# MAX_POSITION = 1e12
# MIN_POSITION = -1e12
# DAMPINIG = 0.999

# MAX_VEL = 2000
# MIN_VEL = -2000
MAX_POSITION = 1e10
MIN_POSITION = 0
# MAX_POSITION = 1e99
# MIN_POSITION = -1e99
DAMPINIG = 1

# filter config
DESIRED_FILTER = 2  # 1 = lowpass; any = bandpass
GOAL_ORDER = 2
ORDER = 2
# N_COEFICIENTS = ORDER + 1
INIT_FREQ = 0.1
FINAL_FREQ = 10000
FREQ_STEPS = 1000

if DESIRED_FILTER == 1:
    print('lowpass')
    goal_num, goal_den = iirfilter(GOAL_ORDER, [100], analog=True, btype='lowpass')
else:
    print('bandpass')
    goal_num, goal_den = iirfilter(GOAL_ORDER, [100, 1000], analog=True, btype='bandpass')
print('Desired filter Numerator: {}'.format(goal_num))
print('Desired filter Denumerator: {}'.format(goal_den))

N_COEFICIENTS = len(goal_den)

desired_w, desired_h = freqs(goal_num, goal_den, worN=np.logspace(-1, 4, FREQ_STEPS))


plt.figure(1)
plt.semilogx(desired_w, 20 * np.log10(abs(desired_h)))
plt.axis((0.1, 10000, -100, 10))
plt.xlabel('Frequency')
plt.ylabel('Amplitude response [dB]')
plt.grid(which='both', axis='both')
# plt.show()

plt.figure(2)
# plt.axis((0.1, 10000, -100, 20))
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.grid(which='both', axis='both')


# position_num = []
position_den = []
# velocity_num = []
velocity_den = []
# particle_best_num = []
particle_best_den = []
# global_best_num = []
global_best_den = []

def calculateError(denominator, numerator=goal_num):
    error = 0
    _, h = freqs(numerator, denominator, worN=np.logspace(-1, 4, FREQ_STEPS))
    for i in range(len(h)):
        e = (abs(h[i]) - abs(desired_h[i])) ** 2
        # e = (abs(desired_h[i]) - abs(h[i])) ** 2
        error += e
    return error**0.5


if __name__ == '__main__':
    # Initialize Particles
    for i in range(POPULATION):
        # position_num.append(np.random.uniform(MIN_POSITION, MAX_POSITION, N_COEFICIENTS))
        position_den.append(np.random.uniform(MIN_POSITION, MAX_POSITION, N_COEFICIENTS))
        
        # velocity_num.append(np.random.uniform(MIN_VEL, MAX_VEL + 1, N_COEFICIENTS))
        velocity_den.append(np.random.uniform(MIN_VEL, MAX_VEL, N_COEFICIENTS))
        
        # particle_best_num.append(position_num[i])
        # particle_best_num.append([100, 1000])
        particle_best_den.append(position_den[i])

        if i == 0:
            # global_best_num = position_num[0]
            global_best_den = position_den[0]
        elif calculateError(position_den[i]) < calculateError(global_best_den):
            # global_best_num = position_num[i]
            global_best_den = position_den[i]

    # Run PSO
    for i in range(ITERATIONS):
        if (i + 1) % 50 == 0:
            print('{} Iterations.'.format(i + 1))

        for j in range(POPULATION):
            # Update Velocity
            # velocity_num[j] = INERTIA * velocity_num[j] + PERSONAL_WEIGHT * np.random.rand() * (particle_best_num[j] - position_num[j]) + SOCIAL_WEIGHT * np.random.rand() * (global_best_num - position_num[j])
            velocity_den[j] = INERTIA * velocity_den[j] + PERSONAL_WEIGHT * np.random.rand() * (particle_best_den[j] - position_den[j]) + SOCIAL_WEIGHT * np.random.rand() * (global_best_den - position_den[j])
            
            # clamp velocity
            # velocity_den[j] = np.clip(velocity_den[j], MIN_VEL, MAX_VEL)
            # for k in range(N_COEFICIENTS):
            #     if velocity_den[j][k] > MAX_VEL:
            #         # print('Clamping up velocity')
            #         velocity_den[j][k] = MIN_VEL/10
            #     elif velocity_den[j][k] < MIN_VEL:
            #         # print('Clamping down velocity')
            #         velocity_den[j][k] = MAX_VEL/10

            # Update Position
            # position_num[j] = position_num[j] + velocity_num[j]
            position_den[j] = position_den[j] + velocity_den[j]

            # Clamp position
            position_den[j] = np.clip(position_den[j], MIN_POSITION, MAX_POSITION)
            # for k in range(N_COEFICIENTS):
            #     if (position_den[j][k] < MIN_POSITION) or (position_den[j][k] > MAX_POSITION):
            #         # print('Clipping position.')
            #         position_den[j][k] = np.clip(position_den[j][k], MIN_POSITION, MAX_POSITION)
            #         velocity_den[j][k] = -velocity_den[j][k]

            # Update Particle Best
            currentError = calculateError(position_den[j])
            if currentError < calculateError(particle_best_den[j]):
                # particle_best_num[j] = position_num[j]
                particle_best_den[j] = position_den[j]

                # Update Global Best
                if currentError < calculateError(global_best_den):
                    print('New Global Best! Error: {}'.format(currentError))
                    # global_best_num = position_num[j]
                    global_best_den = position_den[j]

        globalError = calculateError(global_best_den)
        plt.figure(2)
        plt.ion()
        plt.show()
        plt.plot(i + 1, globalError, 'ro', markersize=5)
        plt.pause(0.001)
        
        # print(globalError)
        
        if globalError <= ERROR_THRESHOLD:
            print('Reached error threshold.')
            break
        
        # Damping Inertia
        INERTIA *= DAMPINIG

    print('Final denominator: {}'.format(global_best_den))
    final_w, final_h = freqs([810000, 0, 0], global_best_den, worN=np.logspace(-1, 4, FREQ_STEPS))

    # print(global_best_num)
    # print(global_best_den)

    plt.ioff()
    plt.figure(1)
    plt.semilogx(final_w, 20 * np.log10(abs(final_h)))
    plt.show()
    # plt.axis((0.1, 10000, -100, 20))
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude response [dB]')
    # plt.grid(which='both', axis='both')