import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def random_walk(n, xlim=5, ylim=5): 
    '''
    function that returns two lists with x and y coordinates of random walk
    :param n: number of steps
    :type n: int
    :param xlim : right edge of a lattice, optional. The default is 5.
    :type xlim: int
    :param ylim : top edge of a lattice, optional. The default is 5.
    :type ylim: int
    :return: x and y coordinates lists
    '''
    x, y = [0], [0]   
    for i in range(n):
        (dx, dy) = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        xx = x[i] + dx
        yy = y[i] + dy
        while (xx>xlim or xx<-xlim) or (yy>ylim or yy<-ylim):
            (dx, dy) = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            xx = x[i] + dx
            yy = y[i] + dy                
        x.append(xx)
        y.append(yy)
    return x, y


def animation_plot(n, xlim=6, ylim=6):
    '''
    function that generates animated plot of random walk
    :param n: number of points
    :type n: int
    :param xlim : right edge of a figure box, optional. The default is 6.
    :type xlim: int
    :param ylim : top edge of a figure box, optional. The default is 6.
    :type ylim: int
    :return: animation
    '''
    def animation_frame(i):
        line.set_data(x[:i],y[:i])
        return line,
    fig, ax = plt.subplots()
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    line, = ax.plot(0, 0, color='green', lw = 2, marker='.')
    x, y = random_walk(n, xlim-1, ylim-1)
    animation = FuncAnimation(fig, animation_frame, interval=10)
    plt.grid()
    plt.show()
    return animation
    
a = animation_plot(250, 5, 5)
#a.save('animation1.gif', writer='imagemagick', fps=30)










