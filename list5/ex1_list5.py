import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

def plague_model(u, t, beta, b, k):
    """
    Function that represents a system of differential equations (model)
    :param u: an array with system of differential equations
    :param t: a sequence of time points for which to solve for u
    :param beta: transmission rate
    :param b: uninfected off-spring producing rate 
    :param k: dying rate
    :return: system of differential equations as an array
    """
    S = u[0]
    I = u[1]
    return np.array([b*S-beta*I*S, beta*I*S-k*I]) 

def integrate(model, init, beta, k, b):
    """
    Function that generates the solution for a model
    :param model: system of ordinary differential equations to be solved
    :param init: initial conditions
    :param beta: transmission rate
    :param b: uninfected off-spring producing rate 
    :param k: dying rate
    :return: two vectors for system of two differential equations
    """
    t = np.linspace(0, 100, 100000)
    result = odeint(model, init, t, args=(beta, k, b))
    return result[:, 0], result[:, 1]

def draw_models(beta, b, k, S0, I0_list):
    """
    Function that generates a plot with the solution for different initial conditions
    :param beta: transmission rate
    :param b: uninfected off-spring producing rate 
    :param k: dying rate
    :param S0: initial condition
    :param I0_list: list of initial conditions
    """
    t = np.linspace(0, 100, 100000)
    N = len(I0_list)
    fig, axes = plt.subplots(1, N, figsize=(15, 4))
    for i in range(N):
        init = [S0, I0_list[i]]
        S, I = integrate(plague_model, init, beta, k, b)
        axes[i].plot(t, S, label="$S$")
        axes[i].plot(t, I, label="$I$")
        axes[i].set_title(r"$I_0 = $"+str(I0_list[i]), fontsize=16)
        axes[i].set_xlabel(r"$t$", fontsize=16)
        axes[i].set_ylabel(r"$S/I$", fontsize=16)
        axes[i].legend(loc=5, fontsize=12)
        axes[i].set_xlim([0, 20])
    plt.tight_layout()
    plt.show()
    
def phase_plane(beta, b, k, lim):
    """
    Function that generates a phase plane for the plague model
    :param beta: transmission rate
    :param b: uninfected off-spring producing rate 
    :param k: dying rate
    :param lim: vector length
    """
    X, Y  = np.meshgrid(np.linspace(0, lim, 50), np.linspace(0, lim, 50))
    t = np.linspace(0, 100, 10000)
    U, V = plague_model([X, Y], t, beta, b, k)
    plt.streamplot(X, Y, U, V, color = 'k', linewidth=0.5)
    plt.title('Phase plane')
    plt.show()

def phase_portrait(I0_list, beta, k, b, lim):
    """
    Function that generates a phase plane with solution for the plague model
    :param I0_list: list of initial conditions
    :param beta: transmission rate
    :param b: uninfected off-spring producing rate 
    :param k: dying rate
    :param lim: vector length
    """
    phase_plane(beta, b, k, lim)
    S0 = 1
    for I0 in I0_list:
        S, I = integrate(plague_model, [S0, I0], beta, k, b)
        plt.plot(S, I, c=(random.random(), random.random(), random.random()), label = r"$I_0 = $"+str(I0))
        plt.legend(loc="upper left")

#draw_models(3, 3, 3, 1, [0.1,0.5,1.3])
#phase_portrait([0.1,0.5,1.3], 3, 3, 3, 5)












