import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

def SIR_model(u, t, beta, r):
    """
    Function that represents a system of differential equations (model)
    :param u: an array with system of differential equations
    :param t: a sequence of time points for which to solve for u
    :param beta: parameter for infectivity
    :param r: recovery rate 
    :return: system of differential equations as an array
    """
    S = u[0]
    I = u[1]
    R = u[2]
    return np.array([-beta*I*S, beta*I*S-r*I, r*I])


def integrate(model, init, beta, r):
    """
    Function that generates the solution for a model
    :param model: system of ordinary differential equations to be solved
    :param init: initial conditions
    :param beta: parameter for infectivity
    :param r: recovery rate 
    :return: three vectors for set of three differential equations
    """
    t = np.linspace(0, 100, 100000)
    result = odeint(model, init, t, args=(beta, r))
    return result[:, 0], result[:, 1], result[:, 2]


def draw_models(r, S0, I0, R0, betas_list):
    """
    Function that generates a plot with the solution for different parameter for infectivity
    :param r: recovery rate 
    :param S0: initial condition
    :param I0: initial condition
    :param R0: initial condition
    :param betas_list: list of infectivity parameter
    """
    t = np.linspace(0, 100, 100000)
    N = len(betas_list)
    fig, axes = plt.subplots(1, N, figsize=(15, 4))
    for i in range(N):
        beta = betas_list[i]
        R_r = (S0*beta)/r
        init = [S0, I0, R0]
        S, I, R = integrate(SIR_model, init, beta, r)
        axes[i].plot(t, S, label="$S$")
        axes[i].plot(t, I, label="$I$")
        axes[i].plot(t, R, label="$R$")
        axes[i].set_title(r"$\beta = $"+str(beta) + ', ' +r"$R_0 = $" + str(round(R_r,2)), fontsize=16)
        axes[i].set_xlabel(r"$t$", fontsize=16)
        axes[i].set_ylabel(r"$S/I/R$", fontsize=16)
        axes[i].legend(loc=5, fontsize=12)
        axes[i].set_xlim([0, 2])
    plt.tight_layout()
    plt.show()

def phase_plane(beta, r, lim):
    """
    Function that generates a phase plane for the SIR model
    :param beta: transmission rate
    :param r: recovery rate 
    :param lim: vector length
    """
    X, Y  = np.meshgrid(np.linspace(0, lim, 50), np.linspace(0, lim, 50))
    t = np.linspace(0, 100, 100000)
    U, V, W = SIR_model([X, Y, 0], t, beta, r)
    plt.streamplot(X, Y, U, V, color = 'k', linewidth=0.5)
    plt.title('Phase plane')
    plt.show()
    
      
def phase_portrait(I0_list, beta, r, lim):
    """
    Function that generates a phase plane with solution for the SIR model
    :param I0_list: list of initial conditions
    :param beta: transmission rate
    :param r: recovery rate 
    :param lim: vector length
    """
    phase_plane(beta, r, lim)
    S0 = 1000
    R0 = 0
    for I0 in I0_list:
        S, I, R = integrate(SIR_model, [S0, I0, R0], beta, r)
        plt.plot(S, I, c=(random.random(), random.random(), random.random()), label = r"$I_0 = $"+str(I0))
        plt.legend(loc="upper right")
        plt.ylim([0, 250])
            
def get_R0(r, S0, I0, R0, betas):
    """
    Function that plots R0
    :param r: recovery rate 
    :param S0: initial condition
    :param I0: initial condition
    :param R0: initial condition
    :param betas: list of transmission rates
    """
    N = len(betas)
    R0_list = []
    R_list = []
    for i in range(N):
        beta = betas[i]
        init = [S0, I0, R0]
        S, I, R = integrate(SIR_model, init, beta, r)
        R = R[-1]
        R_list.append(R)
        R0_list.append(beta * S0/r)
    plt.plot(R0_list, R_list)
    plt.xlabel(r"$R_0$")
    plt.ylabel("number of infected people")
    plt.show()


#draw_models(5.5, 100, 1, 0, [0.01, 0.05, 0.07, 0.15])
#phase_portrait([1, 50, 100, 150, 200], 0.2, 170, 1000)
#get_R0(170, 100, 1, 0, np.linspace(0, 10, 100))