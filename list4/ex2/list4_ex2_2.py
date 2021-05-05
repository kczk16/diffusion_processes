import numpy as np
import matplotlib.pyplot as plt
from list4ex2_1 import random_walk 

def random_walk_mc(n):
    '''
    function that returns a list with fraction of time when the walker is in right 
    half plane and in the first quadrant
    :param n: number of steps
    :type n: int
    :return: lists with fraction time
    '''
    frac = []
    frac2 = []
    s2 = 0
    for j in range(n):
        x, y = [0], [0]
        x,y = random_walk(n)
        s = sum(i > 0 for i in x)
        frac.append(s/n)
        for f, b in zip(x, y):
            if (f>0 and b>0):
                s2 += 1
        frac2.append(s2/n)
        s2 = 0
    return frac, frac2
    
def plots(n):
    '''
    function that returns histograms for fraction time
    :param n: number of steps
    :type n: int
    :return: histograms
    '''
    frac, frac2 = random_walk_mc(n)   
    sr = np.mean(frac)
    sr2 = np.mean(frac2)    
    plt.figure(1, [9, 5])
    plt.subplot(121)
    plt.hist(frac, 15, facecolor='g')
    plt.xlabel('frac time')
    plt.ylabel('number')
    plt.title('An - x>0')
    plt.text(0.3,1, 'Average frac = %s' % (sr))
    plt.subplot(122)
    plt.hist(frac2, 15, facecolor='m')
    plt.xlabel('frac time')
    plt.ylabel('number')
    plt.title('Bn - x>0, y>0')
    plt.text(0.15,1, 'Average frac = %s' % (sr2))
        
plots(1000)
    
    
    
    
    

