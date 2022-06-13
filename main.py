##  Author: Heitor Teixeira de Azambuja
##  Date:   03/05/2022

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as anim

###################################
# https://stackoverflow.com/questions/3909794/plotting-mplot3d-axes3d-xyz-surface-plot-with-log-scale ####
import matplotlib.ticker as mticker

# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation
###################################################

MAX_VEL = .4
ITERATIONS = 10
x, y = [], []
minx = None
miny = None
N = 10
inertia = 1
personalWeight = 2
socialWeight = 2
vel = []
pBest = []

# ohmegaSq = (2 * np.pi * fc) ** 2

fCut = 5000
omegaCut = 2 * np.pi * fCut
capacitorFactor = 10 ** -9

s = np.complex(0, omegaCut)


steps = 200

def fun(x, y):
    # var = ((1 / np.sqrt(1 / ohmegaSq)) * (1 / (ohmegaSq * y * 0.001))) / x
    # var = 1 / (ohmegaSq * y * cFactor * x)
    
    C = y * capacitorFactor
    L = 1 / ((omegaCut ** 2) * y * C)
    
    # sC = np.complex(0, omegaCut * C)
    # sL = np.complex(0, omegaCut * L)
    
    # rC = np.sqrt(sC ** 2)
    # rL = np.sqrt(sL ** 2)

    # angleC = np.angle(sC)
    # angleL = np.angle(sL)

    # sC = rC * np.exp(angleC * 1j)
    # sL = rL * np.exp(angleL * 1j)

    # var = (1 / x) / ((1 / x) + (1 / (s * L)) + (s * C))    
    # var = (1 / x) / ((1 / x) + (1 / (sL)) + (sC))    
    # return var

    error = 0
    currentFreq = 1
    fstep = 10000 / steps
    # Low-pass
    for i in range(steps):
        s = np.complex(0, currentFreq)
        var = x ** 2 / ((s ** 2) + ((x/y) * s) + (x ** 2))
        if currentFreq >= fCut:
            d = 0
        else:
            d = 10
        error += (d - np.abs(var)) ** 2
        currentFreq += fstep
    return np.sqrt(error)
    
    # band-pass
    # for i in range(steps):
    #     s = np.complex(0, currentFreq)
    #     var = ((x / y) * s) / ((s ** 2) + ((x / y) * s) + (x ** 2))
    #     if currentFreq <= 500 or currentFreq >= 2000:
    #         d = 0
    #     else:
    #         d = 10
    #     error += (d - np.abs(var)) ** 2
    #     currentFreq += fstep
    # # return error
    # return np.sqrt(error)

    # band-rejection
    # for i in range(steps):
    #     s = np.complex(0, currentFreq)
    #     var = ((x ** 2 + s ** 2) / ((s ** 2) + ((x / y) * s) + (x ** 2)))
    #     if currentFreq <= 400 or currentFreq >= 600:
    #         d = 10
    #     else:
    #         d = 0
    #     error += (d - np.abs(var)) ** 2
    #     currentFreq += fstep
    # return np.sqrt(error)


#  Figure Creation and initial plots
fig = plt.figure(num='3D Parabola PSO')
fig.set_size_inches(11, 6)

fig.suptitle('3D Parabola Particle swarm optimization', fontsize=14, fontweight='bold')
ax = fig.add_subplot(121, projection='3d', title='Mesh')
# x = y = np.arange(0.1, 10.0, 0.05)
# x = y = np.arange(1, 100000, 100)
x = np.arange(1, 10000, 10)
y = np.arange(0.1, 5, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

# ax.plot_surface(np.log10(X), Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

ax.set_xlabel('\u03c9' + '0')  # omega0 cutoff frequency
ax.set_ylabel('Q')  
ax.set_zlabel('error')

# ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax = fig.add_subplot(122, title='Error heat map')
# ax = plt.imshow(Z, cmap=cm.coolwarm, extent=[-4,4,-4,4])
ax = plt.imshow(Z, cmap=cm.coolwarm, extent=[0.1,10,0.1,10])
plt.colorbar(ax)
# scat = plt.scatter(x,y, c='black', s=15)
# scatMin = plt.scatter(x,y, c='r', s=15)


# #  Initialize particles animation
# def initAnim():
#     print('Initializing particles - ', end=' ')
#     global x, y, scat, scatMin, scat, scatMin, vel, minx, miny
#     x, y = [], []
    
#     scat.remove()
#     scatMin.remove()
    
#     x.append(np.random.rand(N) * 8 - 4)
#     y.append(np.random.rand(N) * 8 - 4)
    
#     minimum = 999999999999
#     minx = None
#     miny = None
#     for i in range(len(x[0])):
#         val = fun(x[0][i], y[0][i])
#         vel.append(np.random.rand(2) * MAX_VEL * 2 - MAX_VEL)
#         pBest.append(np.array([x[0][i], y[0][i]]))
#         if val < minimum:
#             minimum = val
#             minx = x[0][i]
#             miny = y[0][i]

#     print('Global minimum: ' + str(minimum))

#     scat = plt.scatter(x,y, c='black', s=15)
#     scatMin = plt.scatter(minx,miny, c='r', s=15)

#     plt.pause(.3)


# #  Compute swarm particle optimization
# def animate(i):
#     print ('Iteration ' + str(i + 1) + ' - ', end=' ')
#     global x, y, minx, miny, scat, scatMin, vel, inertia, personalWeight, socialWeight
#     minimum = 999999999999
    
#     for i in range(len(x[0])):
#         # compute new velocity
#         vel[i] = (inertia * vel[i]) + (personalWeight * np.random.rand() * (pBest[i] - np.array([x[0][i], y[0][i]]))) + (socialWeight * np.random.rand() * ([minx, miny] - np.array([x[0][i], y[0][i]])))
#         # max velocity limit
#         vel[i][0] = min(vel[i][0], MAX_VEL)
#         vel[i][1] = min(vel[i][1], MAX_VEL)
#         # min velocity limit
#         vel[i][0] = max(vel[i][0], -MAX_VEL)
#         vel[i][1] = max(vel[i][1], -MAX_VEL)
#         # compute new position
#         x[0][i] = x[0][i] + vel[i][0]
#         y[0][i] = y[0][i] + vel[i][1]
#         # check if new position is inside the domain
#         if x[0][i] > 4:
#             x[0][i] = 4
#             vel[i][0] = -vel[i][0]
#         if x[0][i] < -4:
#             x[0][i] = -4
#             vel[i][0] = -vel[i][0]
#         if y[0][i] > 4:
#             y[0][i] = 4
#             vel[i][1] = -vel[i][1]
#         if y[0][i] < -4:
#             y[0][i] = -4
#             vel[i][1] = -vel[i][1]
#         # update pBest
#         val = fun(x[0][i], y[0][i])
#         if val < fun(pBest[i][0], pBest[i][1]):
#             pBest[i] = [x[0][i], y[0][i]]
#         # update min
#         if val < minimum:
#             minimum = val
#             minx = x[0][i]
#             miny = y[0][i]
    
#     print('Global minimum: ' + str(minimum))

#     scat.remove()
#     scatMin.remove()
#     scat = plt.scatter(x,y, c='black', s=15)
#     scatMin = plt.scatter(minx,miny, c='r', s=15)


# ani = anim.FuncAnimation(fig, animate, frames=ITERATIONS, init_func=initAnim, interval=500, repeat=True) 

plt.tight_layout(w_pad=4)
plt.show()