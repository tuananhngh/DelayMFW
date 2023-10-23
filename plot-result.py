import h5py
import os
from more_itertools import last
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from pylab import cm
import pandas as pd
from pyparsing import col
# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2

colors = cm.get_cmap('tab10', 4)

graph_styles = ["er","grid","complete"]
delay_amount = [1, 21, 41, 61, 81, 101]
#delay_amount = [61, 81, 101]

path_centralized = "../result-centralized-ml/"
path_opt_centralized = "../result-decentralized/Mnist/1000-optimal.jld"

def create_csv(delay_amount, path):
    path_read = os.path.join(path, "{}-delay.jld".format(delay_amount))
    with h5py.File(path_read, "r") as f:
        dmfw = f["dmfw"][:]
        dofw = f["dofw"][:]
        bmfw = f["bmfw"][:]
    ite = np.arange(1,len(dmfw)+1)
    data_file = pd.DataFrame({"iteration":ite,"DelMFW":np.cumsum(dmfw), "DOFW":np.cumsum(dofw), "BOLD-MFW":np.cumsum(bmfw)})
    data_file.to_csv(os.path.join(path,"{}-delay-cumsum.csv".format(delay_amount)), index=False)

for d in delay_amount:
    create_csv(d, path_centralized)   
 

def plot_compare_centralized(delay_amount, path, path_opt):
    path_read = os.path.join(path, "{}-delay.jld".format(delay_amount))
    with h5py.File(path_read, "r") as f:
        dmfw = f["dmfw"][:]
        dofw = f["dofw"][:]
        bmfw = f["bmfw"][:]
    with h5py.File(path_opt, "r") as f:
        opt = f["opt"][:]
    iteration = len(dmfw)
    fig = plt.figure(figsize=(4,2))
    plt.title("Maximum Delay: {}".format(delay_amount))
    ax = fig.add_axes([0,0, 1, 1])
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.plot(np.arange(1,iteration+1), np.cumsum(dmfw), label="DeLMFW",linewidth=2, color=colors(0),
            markevery=[i for i in range(iteration) if i%100==0], marker="s")
    ax.plot(np.arange(1,iteration+1), np.cumsum(dofw), label="DOFW",linewidth=2, color=colors(1), 
            markevery=[i for i in range(iteration) if i%100==0], marker=">")
    ax.plot(np.arange(1,iteration+1), np.cumsum(bmfw), label="BOLD-MFW",linewidth=2, color=colors(3),
            markevery=[i for i in range(iteration) if i%100==0], marker="o")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(500))
    ax.set_xlim(-50,iteration)
    ax.set_xlabel('#Iterations', labelpad=10)
    ax.set_ylabel('Cumulative Loss', labelpad=10)
    ax.legend(loc='upper left', frameon=True, framealpha=1)
    plt.show()
    #plt.savefig("./plots/centralized/{}-delay.png".format(delay_amount), dpi=300, 
    #            transparent=False, bbox_inches='tight')
    
def compare_delay_centralized(delay_amount_list, path):
    values_dmfw = []
    values_dofw = []
    values_bofw = []
    values_bmfw = []
    for d in delay_amount_list:
        path_read = os.path.join(path, "{}-delay.jld".format(d))
        with h5py.File(path_read, "r") as f:
            dmfw = f["dmfw"][:]
            dofw = f["dofw"][:]
            bmfw = f["bmfw"][:]
        values_dmfw.append(np.cumsum(dmfw)[-1])
        values_dofw.append(np.cumsum(dofw)[-1])
        values_bmfw.append(np.cumsum(bmfw)[-1])
    file = pd.DataFrame({"delay":delay_amount_list, "DeLMFW":values_dmfw, "DOFW":values_dofw, "BOLD-MFW":values_bmfw})
    file.to_csv(os.path.join(path,"comparison-delay.csv"), index=False)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0,0, 1, 1])
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')
    ax.plot(delay_amount_list, values_dmfw, label="DeLMFW",linewidth=2, color=colors(0), marker="s")
    ax.plot(delay_amount_list, values_dofw, label="DOFW",linewidth=2, color=colors(1),marker=">")
    ax.plot(delay_amount_list, values_bmfw, label="BOLD-MFW",linewidth=2, color=colors(3), marker="o")
    #ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    ax.set_xticks(delay_amount)
    #ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(500))
    ax.set_xlim(0, delay_amount[-1]+1)
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
    plot_compare_centralized(d, path_centralized, path_opt_centralized)

compare_delay_centralized(delay_amount, path_centralized)

    

def plot_decentralized(path, graph_list,cumulative=True):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])

    for graph_type in graph_list:
        path_result = path+"{}.jld".format(graph_type)
        with h5py.File(path_result, "r") as f:
            ddmfw = f["ddmfw"][:]
        if cumulative:
            ddmfw_cum = np.cumsum(ddmfw)
            title = "Cumulative Regret"
        else:
            ddmfw_cum = ddmfw
            title = "Regret"
        ax.plot(np.arange(1, len(ddmfw_cum) + 1), ddmfw_cum, label=f"De2MFW-{graph_type}", marker="s", markevery=[i for i in range(len(ddmfw_cum)) if i % 100 == 0])

    ax.set_xlim(-1, len(ddmfw_cum))
    ax.set_xlabel("#Iterations", labelpad=10)
    ax.set_ylabel("Cumulative Regret", labelpad=10)
    ax.legend(loc="upper left", frameon=False)
    plt.show()
    
param = {
    "data_name": "fashionmnist",
    "iteration" : "1000",
    "delay" : "101",
    "num_agent" : "30",
    "radius":"24",
    "max_delay_agent":"501",
    "num_agent_delay":"10"
}

data_name, iteration,delay, num_agent, radius, max_delay_agent, num_agent_delay = param.values()
path_decen = "../result-decentralized-ml2/{}/{}-{}-{}-{}-".format(data_name,iteration, delay, num_agent, radius)
path_decen2 = "../result-decentralized-ml2/{}/{}-{}-{}-{}-select_{}-{}-amaxdelay-".format(data_name,iteration, delay, num_agent, radius, num_agent_delay, max_delay_agent)

graph_list = ["er","grid","complete","cycle"]
#path_opt_decen = "../result-decentralized/FashionMnist/1000-optimal.jld"    
plot_decentralized(path_decen,graph_list ,cumulative=True)

delay_list = [1, 31, 101, 501]
graph_list = ["er","grid","complete","cycle"]
path_compare = "../result-decentralized-ml2/fashionmnist/"
path_select = os.path.join(path_compare,"csv-file")

def make_csv_select(delay_list, graph_list, path):
    select_list = [2, 5, 10, 20]
    max_del_list = [101,501,1001]
    path_save = os.path.join(path,"csv-file")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    ite = np.arange(1,1001)
    s_dict = {}
    s_dict["iteration"] = ite
    for s in select_list:
        for g in graph_list:
            path_read = os.path.join(path,"1000-1-30-24-select_{}-{}-amaxdelay-{}.jld".format(s, 501, g))
            with h5py.File(path_read, "r") as f:
                ddmfw = f["ddmfw"][:]
            s_dict["select-{}-{}-Loss".format(s,g)] = ddmfw
            s_dict["select-{}-{}-cumsum".format(s,g)] = np.cumsum(ddmfw)
    data_file = pd.DataFrame(s_dict)
    data_file.to_csv(os.path.join(path_save,"select-max501-compare.csv"), index=False)
    
make_csv_select(delay_list, graph_list, path_compare)

def make_csv(delay_list, graph_list, path):
    # columns = ["iteration", "non-delay-Loss", "31-Loss", "101-Loss", "501-Loss", "non-delay-cumsum", "31-cumsum", "101-cumsum", "501-cumsum"]
    ite = np.arange(1, 1001)
    non_delay_dict = {}
    path_save = os.path.join(path,"csv-file")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    for g in graph_list:
        graph_dict = {}
        graph_dict["iteration"] = ite
        for d in delay_list:
            path_read = os.path.join(path, "1000-{}-30-24-{}.jld".format(d,g))
            with h5py.File(path_read, "r") as f:
                ddmfw = f["ddmfw"][:]
            graph_dict["{}-Loss".format(d)] = ddmfw
            graph_dict["{}-cumsum".format(d)] = np.cumsum(ddmfw) 
        data_file = pd.DataFrame(graph_dict)
        data_file.to_csv(os.path.join(path_save,"{}-compare.csv".format(g)), index=False)
    
    non_delay_dict = {}
    non_delay_dict["iteration"] = ite
    for g in graph_list:
        non_delay_path = os.path.join(path, "1000-1-30-24-{}.jld".format(g))
        with h5py.File(non_delay_path, "r") as f:
            ddmfw = f["ddmfw"][:]
        non_delay_dict["{}-Loss".format(g)] = ddmfw
        non_delay_dict["{}-cumsum".format(g)] = np.cumsum(ddmfw)
    data_file = pd.DataFrame(non_delay_dict)
    data_file.to_csv(os.path.join(path_save,"non-delay-compare.csv"), index=False)
    
    for d in delay_list:
        delay_dict = {}
        delay_dict["iteration"] = ite
        for g in graph_list:
            path_read = os.path.join(path, "1000-{}-30-24-{}.jld".format(d,g))
            with h5py.File(path_read, "r") as f:
                ddmfw = f["ddmfw"][:]
            delay_dict["{}-Loss".format(g)] = ddmfw
            delay_dict["{}-cumsum".format(g)] = np.cumsum(ddmfw)
        data_file = pd.DataFrame(delay_dict)
        data_file.to_csv(os.path.join(path_save,"{}-compare.csv".format(d)), index=False)
        
make_csv(delay_list, graph_list, path_compare)

def plot_compare_decentralized(delay_list, graph_list, path, cumsum=False):
    save_path = os.path.join(path,"./plots/")
    non_delay = pd.read_csv(os.path.join(path,"non-delay-compare.csv"))
    ite = np.arange(1,1001)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for g in graph_list:
        data = pd.read_csv(os.path.join(path,"{}-compare.csv".format(g)))
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        
        # Plot ratio of each delay with 1 delay
        for d in delay_list:
            if cumsum:
                ax.plot(np.arange(1,1001), data["{}-cumsum".format(d)]/data["1-cumsum"], label="{}-delay".format(d), marker="s", markevery=[i for i in range(1000) if i % 100 == 0])
            else:
                ax.plot(np.arange(1,1001), data["{}-Loss".format(d)]/data["1-Loss"], label="{}-delay".format(d), marker="s", markevery=[i for i in range(1000) if i % 100 == 0])
        ax.set_xlim(-1, 1000)
        ax.set_xlabel("#Iterations", labelpad=10)
        ax.set_ylabel("Ratio", labelpad=10)
        ax.set_title("Graph: {}".format(g))
        ax.legend(loc="upper left", frameon=False)
        plt.show()
        plt.savefig(os.path.join(save_path,"{}-delay-regret.pdf".format(g)), dpi=300)
    for d in delay_list:
        delay_dict = {}
        delay_dict["iteration"] = ite
        data = pd.read_csv(os.path.join(path,"{}-compare.csv".format(d)))
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        for g in graph_list:
            if cumsum:
                cumsum_ratio = data["{}-cumsum".format(g)]/non_delay["{}-cumsum".format(g)]
                delay_dict["{}-cumsum".format(g)] = cumsum_ratio
                ax.plot(np.arange(1,1001), data["{}-cumsum".format(g)]/non_delay["{}-cumsum".format(g)], label="{}-graph".format(g), marker="s", markevery=[i for i in range(1000) if i % 100 == 0])
            else:
                loss_ratio = data["{}-Loss".format(g)]/non_delay["{}-Loss".format(g)]
                delay_dict["{}-Loss".format(g)] = loss_ratio
                ax.plot(np.arange(1,1001), data["{}-Loss".format(g)]/non_delay["{}-Loss".format(g)], label="{}-graph".format(g), marker="s", markevery=[i for i in range(1000) if i % 100 == 0])

        data_delay = pd.DataFrame(delay_dict)
        data_delay.to_csv(os.path.join(path,"{}-delay-ratio.csv".format(d)), index=False)
        ax.set_xlim(-1, 1000)
        ax.set_xlabel("#Iterations", labelpad=10)
        ax.set_ylabel("Ratio", labelpad=10)
        ax.set_title("Delay: {}".format(d))
        ax.legend(loc="lower right", frameon=False)
        plt.show()
        plt.savefig(os.path.join(save_path,"{}-graph-regret.pdf".format(d)), dpi=300)
        
plot_compare_decentralized(delay_list, graph_list, path_select, cumsum=True)

def plot_select_compare(path):
    select_list = [2, 5, 10, 20]
    path_save = os.path.join(path,"./plots/")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    data = pd.read_csv(os.path.join(path,"select-max501-compare.csv"))
    for s in select_list:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        for g in graph_list:
            ax.plot(np.arange(1,1001), data["select-{}-{}-cumsum".format(s,g)], label="{}-graph".format(g), marker="s", markevery=[i for i in range(1000) if i % 100 == 0])
        ax.set_xlim(-1, 1000)
        ax.set_xlabel("#Iterations", labelpad=10)
        ax.set_ylabel("Cumulative Loss", labelpad=10)
        ax.set_title("Select: {}, Max Delay: 501".format(s))
        ax.legend(loc="lower right", frameon=False)
        plt.show()
        plt.savefig(os.path.join(path_save,"select-{}-max501-regret.pdf".format(s)), dpi=300)
        
path_select = os.path.join(path_compare,"csv-file")
plot_select_compare(path_select)

data = pd.read_csv(os.path.join(path_compare,"csv-file","select-max501-compare.csv"))
non_delay = pd.read_csv(os.path.join(path_compare,"csv-file","non-delay-compare.csv"))
cols_cumsum = [col for col in data.columns if 'cumsum' in col]
data_cumsum = data[cols_cumsum]

last_iter_non = non_delay.filter(like='cumsum').iloc[-1]
last_iter_er = data_cumsum.filter(like='er').iloc[-1]
last_iter_grid = data_cumsum.filter(like='grid').iloc[-1]
last_iter_complete = data_cumsum.filter(like='complete').iloc[-1]
last_iter_cycle = data_cumsum.filter(like='cycle').iloc[-1]

# Make dataframe for plotting
# Rename select-2-er-cumsum to 2 and so on
er = [last_iter_non["er-cumsum"]]
grid = [last_iter_non["grid-cumsum"]]
complete = [last_iter_non["complete-cumsum"]]
cycle = [last_iter_non["cycle-cumsum"]]
for i in [2,5,10,20]:
    er.append(last_iter_er["select-{}-er-cumsum".format(i)])
    grid.append(last_iter_grid["select-{}-grid-cumsum".format(i)])
    complete.append(last_iter_complete["select-{}-complete-cumsum".format(i)])
    cycle.append(last_iter_cycle["select-{}-cycle-cumsum".format(i)])

# Make dataframe for plotting
data = {"agent":[0,2,5,10,20], "er":er, "grid":grid, "complete":complete, "cycle":cycle}
data_file = pd.DataFrame(data)
data_file.to_csv(os.path.join(path_select,"compare-increase-agent.csv"), index=True)

def compute_percen_rows(data):
    pd_percen = pd.DataFrame()
    for i in range(len(data)):
        diff = data.iloc[i] - data.iloc[0]
        avg = (data.iloc[i] + data.iloc[0])/2
        percen = diff/avg * 100
        pd_percen = pd_percen.append(percen, ignore_index=True)
    return pd_percen

compute_percen_rows(data_file)