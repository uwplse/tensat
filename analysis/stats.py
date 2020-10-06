"""
This script gets the statistics of the results and plot the result figures.

Example:
    $ python stats.py --mode all_speedup

Modes:
    * all_speedup: Bar plot of speedups of the optimized graphs
    * equivalent: Get number of equivalent graphs explored
    * optimizer: Bar plot of the optimizer time
    * multi: Plot trend with iterations of multi-pattern rewrites

TODOs:
    * use some sort of python linter or format checker like flake8.

"""

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scipy
import scipy.stats

BENCHMARKS = ['nasrnn', 'bert', 'resnext50', 'nasneta', 'inceptionv3']
#BENCHMARKS = ['inceptionv3']

def get_args():
    parser = argparse.ArgumentParser(description='Analysis script, get the statistics we want')
    parser.add_argument('--mode', type=str, default='runtime',
        help='Mode of analysis')

    return parser.parse_args()

def speedup_bar(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    if benchmark == "inceptionv3":
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_2_stats.txt".format(benchmark))
    else:
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    taso_runtime_file = os.path.join(taso_root, "examples/{}_time.txt".format(taso_benchmark_name))

    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_runtimes = []
    for res in egg_results[-5:]:
        egg_runtimes.append(res['optimized'])

    with open(taso_runtime_file, 'r') as f:
        content = f.readlines()

    orig_runtimes = []
    taso_runtimes = []
    for line in content[-5:]:
        times = line.split('\t')
        orig_runtimes.append(float(times[0]))
        taso_runtimes.append(float(times[1]))

    # Get original runtime mean, TASO mean and ste, egg mean and ste
    orig_mean = np.mean(orig_runtimes)
    taso_speedup = [(orig_mean/x - 1) * 100 for x in taso_runtimes]
    egg_speedup = [(orig_mean/x - 1) * 100 for x in egg_runtimes]
    taso_mean = np.mean(taso_speedup)
    egg_mean = np.mean(egg_speedup)
    taso_ste = scipy.stats.sem(taso_speedup)
    egg_ste = scipy.stats.sem(egg_speedup)

    taso_mean_time = np.mean(taso_runtimes)

    print("{}: orig {} taso {} egg {}".format(benchmark, orig_mean, taso_mean_time, np.mean(egg_runtimes)))

    '''# Plot bar and save
    width = 0.8
    x_locs = [0, 1]
    x_locs = [a + width/2 for a in x_locs]
    colors = ['b', 'r']

    fig, ax1 = plt.subplots()
    bar_0 = ax1.bar(x_locs[0], taso_mean, width=width, yerr=taso_ste, ecolor='m', error_kw=dict(lw=5, capsize=5, capthick=3), label='TASO', color=colors[0])
    bar_1 = ax1.bar(x_locs[1], egg_mean, width=width, yerr=egg_ste, ecolor='m', error_kw=dict(lw=5, capsize=5, capthick=3), label='Tensat', color=colors[1])
    rect = bar_1.patches[0]

    speedup_ratio = egg_mean / taso_mean
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2.0, height/2.0, "{:0.1f}x".format(speedup_ratio), ha='center', va='bottom', weight='heavy')

    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True, prop={'size': 14})
    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])
    ax1.set_ylabel('Speed up percentage')
    ax1.set_xlabel(benchmark)

    fig = plt.gcf()
    fig.set_size_inches(2, 5)

    plt.savefig("{}_speedup.pdf".format(benchmark), bbox_inches='tight')

    figlegend = plt.figure(figsize=(3.0,0.5))
    figlegend.legend([bar_0, bar_1], ("TASO", "Tensat"), 'center', ncol=2, fancybox=True, shadow=True, prop={'size': 14})
    figlegend.savefig("legend.pdf")
    plt.close()'''

def speedup_bar_result(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    if benchmark == "inceptionv3":
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_2_stats.txt".format(benchmark))
    else:
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    taso_runtime_file = os.path.join(taso_root, "examples/{}_time.txt".format(taso_benchmark_name))

    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_runtimes = []
    for res in egg_results[-5:]:
        egg_runtimes.append(res['optimized'])

    with open(taso_runtime_file, 'r') as f:
        content = f.readlines()

    orig_runtimes = []
    taso_runtimes = []
    for line in content[-5:]:
        times = line.split('\t')
        orig_runtimes.append(float(times[0]))
        taso_runtimes.append(float(times[1]))

    # Get original runtime mean, TASO mean and ste, egg mean and ste
    orig_mean = np.mean(orig_runtimes)
    taso_speedup = [(orig_mean/x - 1) * 100 for x in taso_runtimes]
    egg_speedup = [(orig_mean/x - 1) * 100 for x in egg_runtimes]
    taso_mean = np.mean(taso_speedup)
    egg_mean = np.mean(egg_speedup)
    taso_ste = scipy.stats.sem(taso_speedup)
    egg_ste = scipy.stats.sem(egg_speedup)

    taso_mean_time = np.mean(taso_runtimes)

    speedup_ratio = egg_mean / taso_mean

    result = {}
    result['taso_mean'] = taso_mean
    result['taso_ste'] = taso_ste
    result['egg_mean'] = egg_mean
    result['egg_ste'] = egg_ste
    result['speedup_ratio'] = speedup_ratio

    return result

def optimizer_time_bar(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    if benchmark == "inceptionv3":
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_2_stats.txt".format(benchmark))
    else:
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    taso_stats_file = os.path.join(taso_root, "examples/{}_stats.txt".format(taso_benchmark_name))

    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_times = []
    egg_sat_times = []
    egg_ext_times = []
    for res in egg_results[-5:]:
        egg_times.append(res['extraction'] + res['saturation'])
        egg_sat_times.append(res['saturation'])
        egg_ext_times.append(res['extraction'])

    with open(taso_stats_file, 'r') as f:
        content = f.readlines()

    taso_totals = []
    taso_bests = []
    for line in content[-5:]:
        elements = line.split(' ')
        taso_totals.append(float(elements[3][:-1]))
        taso_bests.append(float(elements[1][:-1]))

    sat_time_mean = np.mean(egg_sat_times)
    ext_time_mean = np.mean(egg_ext_times)

    print("{}, sat time {}, ext time {}".format(benchmark, sat_time_mean, ext_time_mean))

    width = 0.8
    x_locs = [0, 1]
    x_locs = [a + width/2 for a in x_locs]
    colors = ['b', 'g', 'r', 'c']

    egg_time = np.mean(egg_times)
    taso_total = np.mean(taso_totals)
    taso_best = np.mean(taso_bests)

    fig, ax2 = plt.subplots()
    bar_0 = ax2.bar(x_locs[0], taso_total, width=width, label='TASO total', color=colors[0])
    bar_1 = ax2.bar(x_locs[0], taso_best, width=width, label='TASO best', color=colors[1])
    bar_2 = ax2.bar(x_locs[1], egg_time, width=width, label='Tensat', color=colors[2])

    rect = bar_2.patches[0]
    speedup_ratio = taso_total / egg_time
    speedup_ratio_best = taso_best / egg_time
    print("{} speedup best {}".format(benchmark, speedup_ratio_best))
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2.0, height, "{:0.1f}x".format(speedup_ratio), ha='center', va='bottom', weight='heavy')

    ax2.set_ylabel('Optimizer time (seconds)')
    ax2.set_xlabel(benchmark)
    fig = plt.gcf()
    fig.set_size_inches(2, 5)
    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])

    #ax2.legend(lines + lines2, labels + labels2, fontsize=10)
    #ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, prop={'size': 12})

    plt.savefig("{}_optim_time.pdf".format(benchmark), bbox_inches='tight')

    figlegend = plt.figure(figsize=(7.0,0.5))
    figlegend.legend([bar_0, bar_1, bar_2], ("TASO total", "TASO best", "Tensat"), 'center', ncol=3, fancybox=True, shadow=True, prop={'size': 14})
    figlegend.savefig("legend_overhead.pdf")

    plt.close()

def optimizer_time_result(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    if benchmark == "inceptionv3":
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_2_stats.txt".format(benchmark))
    else:
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    taso_stats_file = os.path.join(taso_root, "examples/{}_stats.txt".format(taso_benchmark_name))

    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_times = []
    egg_sat_times = []
    egg_ext_times = []
    for res in egg_results[-5:]:
        egg_times.append(res['extraction'] + res['saturation'])
        egg_sat_times.append(res['saturation'])
        egg_ext_times.append(res['extraction'])

    with open(taso_stats_file, 'r') as f:
        content = f.readlines()

    taso_totals = []
    taso_bests = []
    for line in content[-5:]:
        elements = line.split(' ')
        taso_totals.append(float(elements[3][:-1]))
        taso_bests.append(float(elements[1][:-1]))

    sat_time_mean = np.mean(egg_sat_times)
    ext_time_mean = np.mean(egg_ext_times)

    print("{}, sat time {}, ext time {}".format(benchmark, sat_time_mean, ext_time_mean))

    egg_time = np.mean(egg_times)
    taso_total = np.mean(taso_totals)
    taso_best = np.mean(taso_bests)

    speedup_ratio = taso_total / egg_time

    result = {}
    result['egg_time'] = egg_time
    result['taso_total'] = taso_total
    result['taso_best'] = taso_best
    result['speedup_ratio'] = speedup_ratio

    return result

def equivalent_graphs(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_stats_file = os.path.join(taso_root, "examples/{}_stats.txt".format(benchmark))

    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_equiv = []
    for res in egg_results[-5:]:
        egg_equiv.append(res['programs'])

    with open(taso_stats_file, 'r') as f:
        content = f.readlines()

    taso_equiv = []
    for line in content[-5:]:
        elements = line.split(' ')
        taso_equiv.append(int(elements[-1])+100)

    egg_mean = np.mean(egg_equiv)
    taso_mean = np.mean(taso_equiv)

    print("{}: egg (power of 2) {}, taso {}".format(benchmark, egg_mean, taso_mean))

def get_iter_stats(benchmark, tamago_root, iter=1):
    egg_stats_file = os.path.join(tamago_root, "tmp/{}_{}_stats.txt".format(benchmark, iter))
    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_runtimes = []
    egg_sat_times = []
    egg_ext_times = []
    egg_n_nodes = []
    for res in egg_results[-1:]:
        egg_runtimes.append(res['optimized'])
        egg_sat_times.append(res['saturation'])
        egg_ext_times.append(res['extraction'])
        egg_n_nodes.append(res['nodes'])

    mean_iter = np.mean(egg_runtimes)
    mean_sat_iter = np.mean(egg_sat_times)
    mean_ext_iter = np.mean(egg_ext_times)
    mean_nodes_iter = np.mean(egg_n_nodes)

    return (mean_iter, mean_sat_iter, mean_ext_iter, mean_nodes_iter)

def multi_trend(benchmark):
    """This function plots the trend when the number of iterations of
    multi-pattern rewrites varies.

    It plots:
        How the speedup varies
        How the optimizer time varies
        How the number of enodes in the final egraph varies

    """
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    taso_runtime_file = os.path.join(taso_root, "examples/{}_time.txt".format(benchmark))

    with open(taso_runtime_file, 'r') as f:
        content = f.readlines()

    orig_runtimes = []
    for line in content[-5:]:
        times = line.split('\t')
        orig_runtimes.append(float(times[0]))
    orig_mean = np.mean(orig_runtimes)

    # iter=1
    mean_iter_1, mean_sat_iter_1, mean_ext_iter_1, mean_nodes_iter_1 = get_iter_stats(benchmark, tamago_root, iter=1)

    # iter=2
    mean_iter_2, mean_sat_iter_2, mean_ext_iter_2, mean_nodes_iter_2 = get_iter_stats(benchmark, tamago_root, iter=2)

    # iter=3
    if benchmark == 'resnext50':
        mean_iter_3, mean_sat_iter_3, mean_ext_iter_3, mean_nodes_iter_3 = get_iter_stats(benchmark, tamago_root, iter=3)

    # The number of nodes for these three in iter 3 is manually recorded, since the ILP solver
    # times out, and the results are not saved in files
    elif benchmark == 'bert':
        mean_iter_3 = -1
        mean_nodes_iter_3 = 842044

    elif benchmark == 'nasrnn':
        mean_iter_3 = -1
        mean_nodes_iter_3 = 10177140

    elif benchmark == 'nasneta':
        mean_iter_3 = -1
        mean_nodes_iter_3 = 11114360

    # Plot runtime & optimizer time v.s. iter
    n_iter = [1,2,3]
    speedup = [orig_mean/mean_iter_1, orig_mean/mean_iter_2]
    optimizer_time = [mean_sat_iter_1+mean_ext_iter_1, mean_sat_iter_2+mean_ext_iter_2]
    if mean_iter_3 > 0:
        speedup.append(orig_mean/mean_iter_3)
        optimizer_time.append(mean_sat_iter_3+mean_ext_iter_3)

    speedup = [(i-1)*100 for i in speedup]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    color = 'tab:red'
    ax1.set_xlabel('#iter of multi pattern rewrites')
    ax1.set_ylabel('Speedup percentage', color=color)
    lns1 = ax1.plot(n_iter[:len(speedup)], speedup, marker='s', color=color, label='Speedup')

    plt.xticks(n_iter, ['{}'.format(i) for i in n_iter])

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Optimizer time (seconds)', color=color)
    lns2 = ax2.plot(n_iter[:len(speedup)], optimizer_time, marker='^', color=color, label='Optimizer time')

    if len(speedup) < 3:
        ax2.scatter(n_iter[-1], 3600, marker='x', color='b')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)

    plt.savefig("{}_trend_time.png".format(benchmark), bbox_inches='tight')
    plt.close()

    # Plot nodes v.s. iter
    n_iter = [1,2,3]
    nodes = [mean_nodes_iter_1, mean_nodes_iter_2, mean_nodes_iter_3]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    color = 'tab:green'
    ax1.set_xlabel('#iter of multi pattern rewrites')
    ax1.set_ylabel('#enodes', color=color)
    lns1 = ax1.plot(n_iter, nodes, marker='s', color=color)

    plt.xticks(n_iter, ['{}'.format(i) for i in n_iter])
    plt.savefig("{}_trend_nodes.png".format(benchmark), bbox_inches='tight')
    plt.close()

def multi_results(benchmark):
    """This function gets the results of the trend when the number of iterations of
    multi-pattern rewrites varies.

    It includes:
        How the speedup varies
        How the optimizer time varies
        How the number of enodes in the final egraph varies

    """
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    taso_runtime_file = os.path.join(taso_root, "examples/{}_time.txt".format(benchmark))

    with open(taso_runtime_file, 'r') as f:
        content = f.readlines()

    orig_runtimes = []
    for line in content[-5:]:
        times = line.split('\t')
        orig_runtimes.append(float(times[0]))
    orig_mean = np.mean(orig_runtimes)

    # iter=1
    mean_iter_1, mean_sat_iter_1, mean_ext_iter_1, mean_nodes_iter_1 = get_iter_stats(benchmark, tamago_root, iter=1)

    # iter=2
    mean_iter_2, mean_sat_iter_2, mean_ext_iter_2, mean_nodes_iter_2 = get_iter_stats(benchmark, tamago_root, iter=2)

    # iter=3
    mean_iter_3, mean_sat_iter_3, mean_ext_iter_3, mean_nodes_iter_3 = get_iter_stats(benchmark, tamago_root, iter=3)

    # Plot runtime & optimizer time v.s. iter
    speedup = [orig_mean/mean_iter_1, orig_mean/mean_iter_2]
    optimizer_time = [mean_sat_iter_1+mean_ext_iter_1, mean_sat_iter_2+mean_ext_iter_2]
    if mean_iter_3 > 0:
        speedup.append(orig_mean/mean_iter_3)
        optimizer_time.append(mean_sat_iter_3+mean_ext_iter_3)

    speedup = [(i-1)*100 for i in speedup]

    nodes = [mean_nodes_iter_1, mean_nodes_iter_2, mean_nodes_iter_3]

    result = {}
    result['speedup'] = speedup
    result['optimizer'] = optimizer_time
    result['nodes'] = nodes

    return result

def plot_speedup(args):
    plt.rcParams.update({'font.size': 18})
    for benchmark in BENCHMARKS:
        speedup_bar(benchmark)

def plot_speedup_together(args):
    plt.rcParams.update({'font.size': 18})
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = speedup_bar_result(benchmark)

    # Plot bar and save
    width = 0.8
    x_locs = [i*3 for i in range(len(BENCHMARKS))]

    colors = ['b', 'r']

    fig, ax1 = plt.subplots()
    for (i, benchmark) in enumerate(BENCHMARKS):
        x_taso = x_locs[i] + width/2
        x_egg = x_taso + 1
        result = results[benchmark]

        bar_0 = ax1.bar(x_taso, result['taso_mean'], width=width, yerr=result['taso_ste'], ecolor='m', error_kw=dict(lw=5, capsize=5, capthick=3), label='TASO', color=colors[0])
        bar_1 = ax1.bar(x_egg, result['egg_mean'], width=width, yerr=result['egg_ste'], ecolor='m', error_kw=dict(lw=5, capsize=5, capthick=3), label='Tensat', color=colors[1])
        rect = bar_1.patches[0]

        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, height+0.5, "{:0.1f}x".format(result['speedup_ratio']), ha='center', va='bottom', weight='heavy')

    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True, prop={'size': 14})
    ax1.legend((bar_0, bar_1), ("TASO", "Tensat"), loc='upper center', ncol=2, fancybox=True, shadow=True, prop={'size': 18})
    tick_locs = [x + width/2 + 0.5 for x in x_locs]
    plt.xticks(tick_locs, BENCHMARKS)
    ax1.set_ylabel('Speed up percentage')

    fig = plt.gcf()
    fig.set_size_inches(10, 12)

    plt.savefig("all_speedup.pdf", bbox_inches='tight')

    plt.close()

def get_equivalent_graphs(args):
    for benchmark in BENCHMARKS:
        equivalent_graphs(benchmark)

def plot_optimizer_time(args):
    plt.rcParams.update({'font.size': 18})
    for benchmark in BENCHMARKS:
        optimizer_time_bar(benchmark)

def optimizer_time_together(args):
    plt.rcParams.update({'font.size': 18})
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = optimizer_time_result(benchmark)

    # Plot bar and save
    width = 0.8
    x_locs = [i*3 for i in range(len(BENCHMARKS))]

    colors = ['b', 'g', 'r', 'c']

    fig, ax1 = plt.subplots()
    for (i, benchmark) in enumerate(BENCHMARKS):
        x_taso = x_locs[i] + width/2
        x_egg = x_taso + 1
        result = results[benchmark]

        bar_0 = ax1.bar(x_taso, result['taso_total'], width=width, label='TASO total', color=colors[0])
        bar_1 = ax1.bar(x_taso, result['taso_best'], width=width, label='TASO best', color=colors[1])
        bar_2 = ax1.bar(x_egg, result['egg_time'], width=width, label='Tensat', color=colors[2])

        rect = bar_2.patches[0]
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0 + 0.3, height, "{:0.1f}x".format(result['speedup_ratio']), ha='center', va='bottom', weight='heavy')

    ax1.set_yscale('log')
    ax1.set_ylim([None,5000])
    ax1.set_ylabel('Optimizer time (seconds)')


    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True, prop={'size': 14})
    ax1.legend((bar_0, bar_1, bar_2), ("TASO total", "TASO best", "Tensat"), loc='upper center', ncol=3, fancybox=True, shadow=True, prop={'size': 18})
    tick_locs = [x + width/2 + 0.5 for x in x_locs]
    plt.xticks(tick_locs, BENCHMARKS)

    fig = plt.gcf()
    fig.set_size_inches(10, 12)

    plt.savefig("all_optim_time.pdf", bbox_inches='tight')

    plt.close()


def plot_multi_trend(args):
    plt.rcParams.update({'font.size': 18})
    for benchmark in BENCHMARKS:
        multi_trend(benchmark)

def multi_trend_together(args):
    plt.rcParams.update({'font.size': 18})
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = multi_results(benchmark)

    colors = ['b', 'g', 'tab:orange', 'm', 'r', 'c', 'k']
    n_iter = [1,2,3]

    # Plot speedup
    #create a new figure with two subplots
    fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)

    #set the "zoom" or the y-limits on each subplots
    ax2.set_ylim(0,15)
    ax1.set_ylim(60,90)

    ax2.set_xlabel('#iter of multi pattern rewrites')

    for (i, benchmark) in enumerate(BENCHMARKS):
        speedup = results[benchmark]['speedup']
        ax1.plot(n_iter[:len(speedup)], speedup, marker='s', color=colors[i], label=benchmark)
        ax2.plot(n_iter[:len(speedup)], speedup, marker='s', color=colors[i], label=benchmark)
    #remove the bottom border from the top plot and the upper border from the bottom plot
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax2.set_xticks(n_iter)
    ax2.set_xticklabels(['{}'.format(i) for i in n_iter])

    handles, labels = ax2.get_legend_handles_labels()

    fig.text(0.0, 0.5, 'Speedup percentage', va='center', rotation='vertical')

    #fig.set_size_inches(4, 6)

    fig.savefig("speedup_trend.pdf", bbox_inches='tight')

    # Plot legend
    figlegend = plt.figure(figsize=(2.0,3.0))
    figlegend.legend(handles, labels, 'center', ncol=1, fancybox=True, shadow=True, prop={'size': 14})
    figlegend.savefig("legend_trend.pdf")


    # Plot optimizer time
    fig_optim, ax_optim = plt.subplots()

    for (i, benchmark) in enumerate(BENCHMARKS):
        optimizer_time = results[benchmark]['optimizer']
        lns2 = ax_optim.plot(n_iter[:len(optimizer_time)], optimizer_time, marker='s', color=colors[i], label=benchmark)
        if len(optimizer_time) < 3:
            ax_optim.scatter(n_iter[-1], 3600, marker='x', color=colors[i])

    ax_optim.set_yscale('log')
    ax_optim.set_ylabel('Optimizer time (seconds)')
    ax_optim.set_xlabel('#iter of multi pattern rewrites')

    ax_optim.set_xticks(n_iter)
    ax_optim.set_xticklabels(['{}'.format(i) for i in n_iter])

    fig_optim.savefig("optim_trend.pdf", bbox_inches='tight')

    # Plot number of nodes
    fig_nodes, ax_nodes = plt.subplots()

    for (i, benchmark) in enumerate(BENCHMARKS):
        nodes = results[benchmark]['nodes']
        lns2 = ax_nodes.plot(n_iter[:len(nodes)], nodes, marker='s', color=colors[i], label=benchmark)

    ax_nodes.set_yscale('log')
    ax_nodes.set_ylabel('#enodes')

    ax_nodes.set_xticks(n_iter)
    ax_nodes.set_xticklabels(['{}'.format(i) for i in n_iter])

    ax_nodes.set_xlabel('#iter of multi pattern rewrites')

    fig_nodes.savefig("nodes_trend.pdf", bbox_inches='tight')

    plt.close()

def main():
    # Parse arguments
    args = get_args()
    if args.mode == 'all_speedup':
        # Bar plot of speedups of the optimized graphs
        plot_speedup(args)
    elif args.mode == 'speedup_together':
        # Bar plot of speedups of the optimized graphs, together
        plot_speedup_together(args)
    elif args.mode == 'equivalent':
        # Get number of equivalent graphs explored
        get_equivalent_graphs(args)
    elif args.mode == 'optimizer':
        # Bar plot of the optimizer time
        plot_optimizer_time(args)
    elif args.mode == 'optimizer_together':
        # Bar plot of the optimizer time
        optimizer_time_together(args)
    elif args.mode == 'multi':
        # Plot trend with iterations of multi-pattern rewrites
        plot_multi_trend(args)
    elif args.mode == "multi_together":
        # Plot trend with iterations of multi-pattern rewrites, benchmarks together
        multi_trend_together(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
