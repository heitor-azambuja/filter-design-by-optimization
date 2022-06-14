from scipy.signal import freqs, iirfilter
import matplotlib.pyplot as plt
import numpy as np

# PSO config
ITERATIONS = 500
POPULATION = 50
PERSONAL_WEIGHT = 2
SOCIAL_WEIGHT = 2
ERROR_THRESHOLD = 0.01
INERTIA = 1

MAX_VEL = 1000
MIN_VEL = -1000
# MAX_POSITION = 1e12
# MIN_POSITION = -1e12
DAMPINIG = 0.999

# MAX_VEL = 2000
# MIN_VEL = -2000
MAX_POSITION = 1e10
MIN_POSITION = 0
# MAX_POSITION = 1e99
# MIN_POSITION = -1e99
# DAMPINIG = 1

# filter config
DESIRED_FILTER = 2  # 1 = lowpass; any = bandpass
ORDER = 4
N_COEFICIENTS = ORDER + 1
INIT_FREQ = 0.1
FINAL_FREQ = 10000
FREQ_STEPS = 1000

if DESIRED_FILTER == 1:
    print('lowpass')
    b, a = iirfilter(50, [100], analog=True, btype='lowpass')
else:
    print('bandpass')
    b, a = iirfilter(2, [100, 1000], analog=True, btype='bandpass')
print('Desired filter Numerator: {}'.format(b))
print('Desired filter Denumerator: {}'.format(a))

desired_w, desired_h = freqs(b, a, worN=np.logspace(-1, 4, FREQ_STEPS))


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


position_num = []
position_den = []
velocity_num = []
velocity_den = []
particle_best_num = []
particle_best_den = []
global_best_num = []
global_best_den = []

def calculateError(numerator, denominator):
    error = 0
    _, h = freqs(numerator, denominator, worN=np.logspace(-1, 4, FREQ_STEPS))
    for i in range(len(h)):
        # e = (abs(h[i]) - abs(desired_h[i])) ** 2
        e = (abs(desired_h[i]) - abs(h[i])) ** 2
        error += e
    return error**0.5


if __name__ == '__main__':
    # Initialize Particles
    for i in range(POPULATION):
        position_num.append(np.random.uniform(MIN_POSITION, MAX_POSITION, N_COEFICIENTS))
        position_den.append(np.random.uniform(MIN_POSITION, MAX_POSITION, N_COEFICIENTS))
        
        velocity_num.append(np.random.uniform(MIN_VEL, MAX_VEL + 1, N_COEFICIENTS))
        velocity_den.append(np.random.uniform(MIN_VEL, MAX_VEL + 1, N_COEFICIENTS))
        
        particle_best_num.append(position_num[i])
        particle_best_den.append(position_den[i])

        if i == 0:
            global_best_num = position_num[0]
            global_best_den = position_den[0]
        elif calculateError(position_num[i], position_den[i]) < calculateError(global_best_num, global_best_den):
            global_best_num = position_num[i]
            global_best_den = position_den[i]

    # Run PSO
    for i in range(ITERATIONS):
        if (i + 1) % 50 == 0:
            print('{} Iterations.'.format(i + 1))

        for j in range(POPULATION):
            # Update Velocity
            velocity_num[j] = INERTIA * velocity_num[j] + PERSONAL_WEIGHT * np.random.rand() * (particle_best_num[j] - position_num[j]) + SOCIAL_WEIGHT * np.random.rand() * (global_best_num - position_num[j])
            velocity_den[j] = INERTIA * velocity_den[j] + PERSONAL_WEIGHT * np.random.rand() * (particle_best_den[j] - position_den[j]) + SOCIAL_WEIGHT * np.random.rand() * (global_best_den - position_den[j])

            # Update Position
            position_num[j] = position_num[j] + velocity_num[j]
            position_den[j] = position_den[j] + velocity_den[j]

            # Update Particle Best
            currentError = calculateError(position_num[j], position_den[j])
            if currentError < calculateError(particle_best_num[j], particle_best_den[j]):
                particle_best_num[j] = position_num[j]
                particle_best_den[j] = position_den[j]

                # Update Global Best
                if currentError < calculateError(global_best_num, global_best_den):
                    print('New Global Best!')
                    global_best_num = position_num[j]
                    global_best_den = position_den[j]

        globalError = calculateError(global_best_num, global_best_den)    
        plt.figure(2)
        plt.ion()
        plt.show()
        plt.plot(i + 1, globalError, 'ro', markersize=5)
        plt.pause(0.001)
        
        if globalError < ERROR_THRESHOLD:
            break

        # Damping Inertia
        INERTIA *= DAMPINIG


    final_w, final_h = freqs(global_best_num, global_best_den, worN=np.logspace(-1, 4, FREQ_STEPS))

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