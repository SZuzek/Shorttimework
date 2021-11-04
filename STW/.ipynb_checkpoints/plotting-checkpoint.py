
import matplotlib.pyplot as plt
import numpy as np



def plot_gekko(stw, xlim = None, ylim = None): 
    ''' Overview plot for fast evaluation '''
#     outside = stw.outside_option(np.array(stw.theta.value))
#     qhat = stw.outside_qhat(np.array(stw.theta.value))
    plt.figure(1) # plot results
    plt.plot(stw.theta.value,stw.sa.value,'b-',label=r'$S_A$')
    plt.plot(stw.theta.value,stw.oo.value,'b--',label=r'$S_O$')
    plt.plot(stw.theta.value,stw.q.value,'r-',label=r'$q$')
    plt.plot(stw.theta.value,stw.qhat.value,'r--',label=r'$\hat{q}$')
    plt.plot(stw.theta.value, stw. iota.value, 'g-', label = r'$\iota$')
    
    if xlim is not None:
        plt.xlim(left = xlim[0], right = xlim[1])
    
    if ylim is not None:
        plt.ylim(bottom = ylim[0], top = ylim[1])
    
    plt.legend(loc='best')
    plt.xlabel('Theta')
    plt.ylabel('Value')
    plt.show()
    
    
    
    
    
def plot_presi(stw, which, fname = None, disp = True, xlim = None, ploteff = False):
    '''
    Plots for presentations
    '''    
    plt.figure() # start new figure
    if which=="sa":
        plt.plot(stw.theta.value,stw.sa.value,'k-',label=r'$S_A$')
        plt.plot(stw.theta.value,stw.oo.value,'b--',label=r'$S_O$')
        plt.legend(loc='best')
        plt.xlabel('Theta')
        plt.ylabel('Value')
    elif which=="n":
        plt.plot(stw.theta.value,stw.q.value,'r:',label=r'$n^\star$')
        plt.plot(stw.theta.value,stw.qhat.value,'k-',label=r'$\hat{n}$')
        if ploteff:
            plt.plot(stw.theta.value,stw.qeff, 'k--',label=r'$n_{eff}$')
        plt.legend(loc='best')
        plt.xlabel('Theta')
        plt.ylabel('Hours')
    elif which=="onlynhat":
        plt.plot(stw.theta.value,stw.qhat.value,'r--',label=r'$\hat{n}$')
        plt.legend(loc='best')
        plt.xlabel('Theta')
        plt.ylabel('Hours')
    elif which=="transfers":
        transfers = np.array(stw.t.value)
        plt.plot(stw.theta.value, transfers, label = "transfers")
        plt.legend(loc='best')
        plt.xlabel('Theta')
        plt.ylabel('Transfers')
    elif which=="iota":
        plt.plot(stw.theta.value, stw.iota.value, label = "iota")
        plt.legend(loc='best')
        plt.xlabel('Theta')
        plt.ylabel('iota')
    else: pass
    
    if xlim is not None:
        plt.xlim(left = xlim[0], right = xlim[1])
    
    if disp:
        plt.show
    if fname is not None:
        plt.savefig(fname, dpi = 250)