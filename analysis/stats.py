from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scipy
import scipy.stats

def get_args():
    parser = argparse.ArgumentParser(description='Analysis script, get the statistics we want')
    parser.add_argument('--mode', type=str, default='runtime',
        help='Mode of analysis')
    parser.add_argument('--file', type=str, default='data.txt',
        help='File for the input data to analyze')

    return parser.parse_args()

# Plot speedup and optimizer time together in a same bar plot
def plot_runtime_and_speed(benchmark_name, result):
    width = 0.8
    x_locs = [0, 1, 2, 3, 6, 7]
    x_locs = [a + width/2 for a in x_locs]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig, ax1 = plt.subplots()
    ax1.bar(x_locs[0], result['orig_runtime'], width=width, label='Original', color=colors[0])
    ax1.bar(x_locs[1], result['taso'], width=width, label='TASO', color=colors[1])
    ax1.bar(x_locs[2], result['greedy'], width=width, label='Greedy', color=colors[2])
    ax1.bar(x_locs[3], result['ilp'], width=width, label='ILP', color=colors[3])

    runtimes = [result['orig_runtime'], result['taso'], result['greedy'], result['ilp']]

    ax1.set_ylabel('Graph runtime (milliseconds)')

    ax2 = ax1.twinx()
    ax2.bar(x_locs[4], result['taso_total_time'], width=width, label='TASO total', color=colors[4])
    ax2.bar(x_locs[4], result['taso_best_time'], width=width, label='TASO best', color=colors[5])
    ax2.bar(x_locs[5], result['ilp_time'], width=width, label='Sat.+ILP', color=colors[6])

    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])

    ax2.set_ylabel('Optimizer time (seconds)')
    ax1.set_xlabel(benchmark_name)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, prop={'size': 12})

    plt.savefig("{}.png".format(benchmark_name), bbox_inches='tight')
    plt.close()

def plot_runtime_and_speed_2(benchmark_name, result):

    width = 0.8
    x_locs = [0, 1, 2, 3]
    x_locs = [a + width/2 for a in x_locs]
    colors = ['b', 'g', 'r', 'c']

    fig, ax1 = plt.subplots()
    ax1.bar(x_locs[0], result['orig_runtime'], width=width, label='Original', color=colors[0])
    ax1.bar(x_locs[1], result['taso'], width=width, label='TASO', color=colors[1])
    ax1.bar(x_locs[2], result['greedy'], width=width, label='Sat.+Greedy', color=colors[2])
    ax1.bar(x_locs[3], result['ilp'], width=width, label='Sat.+ILP', color=colors[3])

    runtimes = [result['orig_runtime'], result['taso'], result['greedy'], result['ilp']]

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, prop={'size': 12})
    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])
    ax1.set_ylabel('Graph runtime (milliseconds)')
    ax1.set_xlabel(benchmark_name)


    plt.savefig("{}_runtime.png".format(benchmark_name), bbox_inches='tight')
    plt.close()

    x_locs = [0, 1]
    x_locs = [a + width/2 for a in x_locs]
    colors = ['b', 'g', 'r', 'c']

    fig, ax2 = plt.subplots()
    ax2.bar(x_locs[0], result['taso_total_time'], width=width, label='TASO total', color=colors[0])
    ax2.bar(x_locs[0], result['taso_best_time'], width=width, label='TASO best', color=colors[1])
    ax2.bar(x_locs[1], result['ilp_time'], width=width, label='Sat.+ILP', color=colors[2])

    ax2.set_ylabel('Optimizer time (seconds)')
    ax2.set_xlabel(benchmark_name)
    fig = plt.gcf()
    fig.set_size_inches(3, 5)
    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, prop={'size': 12})

    plt.savefig("{}_optimizer.png".format(benchmark_name), bbox_inches='tight')
    plt.close()

def runtime_stats(args):
    with open(args.file, 'r') as f:
        content = f.readlines()

    start_times = []
    ext_times = []
    for line in content:
        times = line.split('\t')
        start_times.append(float(times[0]))
        ext_times.append(float(times[1]))

    start_mean = np.mean(start_times)
    start_std = np.std(start_times)
    ext_mean = np.mean(ext_times)
    ext_std = np.std(ext_times)
    print("Start graph runtime: mean {}, std {}".format(start_mean, start_std))
    print("Extracted graph runtime: mean {}, std {}".format(ext_mean, ext_std))

def plot_bars(args):
    # Results for the spotlight talk was manually input, since we don't have the pipeline to store results then
    results = {
        "bert": {
            "orig_runtime": 1.8964,
            "taso": 1.7415,
            "greedy": 1.8903,
            "ilp": 1.7410,
            "taso_total_time": 13.98,
            "taso_best_time": 3.410,
            "ilp_time": 3.022,
        },
        "nasrnn": {
            "orig_runtime": 1.8601,
            "taso": 1.2890,
            "greedy": 1.1446,
            "ilp": 1.1106,
            "taso_total_time": 175.4, 
            "taso_best_time": 121.1,
            "ilp_time": 28.47,
        },
        "resnext50": {
            "orig_runtime": 6.0775,
            "taso": 5.8144,
            "greedy": 5.5850,
            "ilp": 5.5704,
            "taso_total_time": 25.00,
            "taso_best_time": 5.909,
            "ilp_time": 1.314,
        }
    }

    plt.rcParams.update({'font.size': 16})

    for (benchmark, result) in results.items():
        plot_runtime_and_speed_2(benchmark, result)


def speedup_bar(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_runtime_file = os.path.join(taso_root, "examples/{}_time.txt".format(benchmark))

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

    print("{}: orig {} taso {}".format(benchmark, orig_mean, taso_mean_time))

    # Plot bar and save
    width = 0.8
    x_locs = [0, 1]
    x_locs = [a + width/2 for a in x_locs]
    colors = ['b', 'r']

    fig, ax1 = plt.subplots()
    ax1.bar(x_locs[0], taso_mean, width=width, yerr=taso_ste, ecolor='m', capsize=2.0, label='TASO', color=colors[0])
    ax1.bar(x_locs[1], egg_mean, width=width, yerr=egg_ste, ecolor='m', capsize=2.0, label='Sat.+ILP', color=colors[1])

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True, prop={'size': 14})
    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])
    ax1.set_ylabel('Speed up percentage')
    ax1.set_xlabel(benchmark)

    fig = plt.gcf()
    fig.set_size_inches(2, 5)

    plt.savefig("{}_speedup.png".format(benchmark), bbox_inches='tight')
    plt.close()

def optimizer_time_bar(benchmark):
    # Read in results
    tamago_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tamago_root), "TASO")

    egg_stats_file = os.path.join(tamago_root, "tmp/{}_2_stats.txt".format(benchmark))
    taso_stats_file = os.path.join(taso_root, "examples/{}_stats.txt".format(benchmark))

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
    ax2.bar(x_locs[0], taso_total, width=width, label='TASO total', color=colors[0])
    ax2.bar(x_locs[0], taso_best, width=width, label='TASO best', color=colors[1])
    ax2.bar(x_locs[1], egg_time, width=width, label='Sat.+ILP', color=colors[2])

    ax2.set_ylabel('Optimizer time (seconds)')
    ax2.set_xlabel(benchmark)
    fig = plt.gcf()
    fig.set_size_inches(3, 5)
    plt.xticks(x_locs, ['' for _ in range(len(x_locs))])

    #ax2.legend(lines + lines2, labels + labels2, fontsize=10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, prop={'size': 12})

    plt.savefig("{}_optim_time.png".format(benchmark), bbox_inches='tight')
    plt.close()
    

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


def multi_trend(benchmark):
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
    egg_stats_file = os.path.join(tamago_root, "tmp/{}_1_stats.txt".format(benchmark))
    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_runtimes = []
    egg_sat_times = []
    egg_ext_times = []
    egg_n_nodes = []
    for res in egg_results[-5:]:
        egg_runtimes.append(res['optimized'])
        egg_sat_times.append(res['saturation'])
        egg_ext_times.append(res['extraction'])
        egg_n_nodes.append(res['nodes'])

    mean_iter_1 = np.mean(egg_runtimes)
    mean_sat_iter_1 = np.mean(egg_sat_times)
    mean_ext_iter_1 = np.mean(egg_ext_times)
    mean_nodes_iter_1 = np.mean(egg_n_nodes)

    # iter=2
    egg_stats_file = os.path.join(tamago_root, "tmp/{}_2_stats.txt".format(benchmark))
    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    egg_runtimes = []
    egg_sat_times = []
    egg_ext_times = []
    egg_n_nodes = []
    for res in egg_results[-5:]:
        egg_runtimes.append(res['optimized'])
        egg_sat_times.append(res['saturation'])
        egg_ext_times.append(res['extraction'])
        egg_n_nodes.append(res['nodes'])

    mean_iter_2 = np.mean(egg_runtimes)
    mean_sat_iter_2 = np.mean(egg_sat_times)
    mean_ext_iter_2 = np.mean(egg_ext_times)
    mean_nodes_iter_2 = np.mean(egg_n_nodes)

    # iter=3
    if benchmark == 'resnext50':
        egg_stats_file = os.path.join(tamago_root, "tmp/{}_3_stats.txt".format(benchmark))
        with open(egg_stats_file, 'r') as egg_f:
            egg_results = egg_f.readlines()

        egg_results = [json.loads(x) for x in egg_results]
        egg_runtimes = []
        egg_sat_times = []
        egg_ext_times = []
        egg_n_nodes = []
        for res in egg_results[-5:]:
            egg_runtimes.append(res['optimized'])
            egg_sat_times.append(res['saturation'])
            egg_ext_times.append(res['extraction'])
            egg_n_nodes.append(res['nodes'])

        mean_iter_3 = np.mean(egg_runtimes)
        mean_sat_iter_3 = np.mean(egg_sat_times)
        mean_ext_iter_3 = np.mean(egg_ext_times)
        mean_nodes_iter_3 = np.mean(egg_n_nodes)

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


def plot_speedup(args):
    plt.rcParams.update({'font.size': 18})
    for benchmark in ['nasrnn', 'bert', 'resnext50', 'nasneta']:
        speedup_bar(benchmark)

def get_equivalent_graphs(args):
    for benchmark in ['nasrnn', 'bert', 'resnext50', 'nasneta']:
        equivalent_graphs(benchmark)

def plot_optimizer_time(args):
    plt.rcParams.update({'font.size': 18})
    for benchmark in ['nasrnn', 'bert', 'resnext50', 'nasneta']:
        optimizer_time_bar(benchmark)

def plot_multi_trend(args):
    plt.rcParams.update({'font.size': 18})
    for benchmark in ['nasrnn', 'bert', 'resnext50', 'nasneta']:
        multi_trend(benchmark)

def main():
    # Parse arguments
    args = get_args()
    if args.mode == 'runtime':
        # Get the runtime stats (not used now)
        runtime_stats(args)
    elif args.mode == 'plot':
        # Plot results for the spotlight talk
        plot_bars(args)
    elif args.mode == 'all_speedup':
        # Bar plot of speedups of the optimized graphs
        plot_speedup(args)
    elif args.mode == 'equivalent':
        # Get number of equivalent graphs explored
        get_equivalent_graphs(args)
    elif args.mode == 'optimizer':
        # Bar plot of the optimizer time
        plot_optimizer_time(args)
    elif args.mode == 'multi':
        # Plot trend with iterations of multi-pattern rewrites
        plot_multi_trend(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()