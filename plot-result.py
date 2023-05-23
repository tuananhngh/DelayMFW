import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from pylab import cm
import pandas as pd
# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

colors = cm.get_cmap('tab10', 3)

graph_styles = ["er","grid","complete"]
delay_amount = [51, 101, 201, 301, 401, 501, 601]

path_centralized = "../result-centralized/"
def plot_compare_centralized(delay_amount, path):
    path_read = os.path.join(path, "{}-delay.jld".format(delay_amount))
    with h5py.File(path_read, "r") as f:
        dmfw = f["dmfw"][:]
        dofw = f["dofw"][:]
        bofw = f["bofw"][:]
    iteration = len(dmfw)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0,0, 1, 1])
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.plot(np.arange(1,iteration+1), np.cumsum(dmfw), label="DeLMFW",linewidth=2, color=colors(0),
            markevery=[i for i in range(iteration) if i%500==0], marker="s")
    ax.plot(np.arange(1,iteration+1), np.cumsum(dofw), label="DOFW",linewidth=2, color=colors(1), 
            markevery=[i for i in range(iteration) if i%500==0], marker=">")
    ax.plot(np.arange(1,iteration+1), np.cumsum(bofw), label="BOFW",linewidth=2, color=colors(2),
            markevery=[i for i in range(iteration) if i%500==0], marker="o")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(500))
    ax.set_xlim(-300,5000)
    ax.set_xlabel('#Iterations', labelpad=10)
    ax.set_ylabel('Cumulative Loss', labelpad=10)
    ax.legend(loc='upper left', frameon=False)
    plt.show()
    #plt.savefig("./plots/centralized/{}-delay.png".format(delay_amount), dpi=300, 
    #            transparent=False, bbox_inches='tight')
    
def compare_delay_centralized(delay_amount_list, path):
    values_dmfw = []
    values_dofw = []
    values_bofw = []
    for d in delay_amount_list:
        path_read = os.path.join(path, "{}-delay.jld".format(d))
        with h5py.File(path_read, "r") as f:
            dmfw = f["dmfw"][:]
            dofw = f["dofw"][:]
            bofw = f["bofw"][:]
        values_dmfw.append(np.cumsum(dmfw)[-1])
        values_dofw.append(np.cumsum(dofw)[-1])
        values_bofw.append(np.cumsum(bofw)[-1])
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0,0, 1, 1])
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.plot(delay_amount_list, values_dmfw, label="DeLMFW",linewidth=2, color=colors(0), marker="s")
    ax.plot(delay_amount_list, values_dofw, label="DOFW",linewidth=2, color=colors(1),marker=">")
    ax.plot(delay_amount_list, values_bofw, label="BOFW",linewidth=2, color=colors(2), marker="o")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(500))
    ax.set_xlim(0, 600)
    ax.set_xlabel('Maximum Delay', labelpad=10)
    ax.set_ylabel('Total Loss', labelpad=10)
    ax.legend(loc='center right', frameon=False)
    plt.show()
    #plt.savefig("./plots/centralized/comparison-delay.png", dpi=300, 
    #            transparent=False, bbox_inches='tight')


path_decentralized = "../result-decentralized/"

def figure_regret(data_name, path):
    path_read = os.path.join(path, data_name, "1000-11-staticlr-er.jld")
    path_opt = os.path.join(path, data_name, "1000-optimal.jld")
    with h5py.File(path_read, "r") as f:
        ddmfw = f["ddmfw"][:]
        ddsgd = f["ddsgd"][:]
    with h5py.File(path_opt, "r") as f:
        opt = f["opt"][:]

    fig2 = plt.figure(figsize=(3,3))
    ax = fig2.add_axes([0,0, 1, 1])
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.plot(np.arange(1,len(ddmfw)+1), np.cumsum(ddmfw-opt), label="De2MFW", color=colors(0), 
            marker="s", markevery=[i for i in range(len(ddmfw)) if i%100==0])
    ax.plot(np.arange(1,len(ddmfw)+1), np.cumsum(ddsgd-opt), label="DDGD", color=colors(1),
            marker=">", markevery=[i for i in range(len(ddmfw)) if i%100==0])
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
    ax.set_xlim(-50,1000)
    #ax.set_ylim(-50,1000)
    ax.set_xlabel("#Iterations", labelpad=10)
    ax.set_ylabel("Regret", labelpad=10)
    ax.legend(loc="upper left", frameon=False)
    plt.show()
    #plt.savefig("./plots/decentralized/{}-delay-regret.png".format(data_name), dpi=300,
    #            transparent=False, bbox_inches='tight')


for d in delay_amount:
    plot_compare_centralized(d, path_centralized)

compare_delay_centralized(delay_amount, path_centralized)

    
names = ["Mnist","Fashionmnist","cifar10","svhn2"]
for n in names:
    figure_regret(n, path_decentralized)
    
    
