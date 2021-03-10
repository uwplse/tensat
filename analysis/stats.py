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

BENCHMARKS = ['nasrnn', 'bert', 'resnext50', 'nasneta', 'squeezenet', 'vgg', 'inceptionv3', 'inceptionv3_2']
BENCHMARK_NAMES = ['NasRNN', 'BERT', 'ResNeXt', 'NasNet-A', 'Squeeze.', 'VGG', 'Incept.', 'Incept. k=2']
BENCHMARKS_TREND = ['nasrnn', 'bert', 'resnext50', 'nasneta', 'squeezenet', 'vgg', 'inceptionv3']
BENCHMARK_NAMES_TREND = ['NasRNN', 'BERT', 'ResNeXt', 'NasNet-A', 'Squeeze.', 'VGG', 'Incept.']
#BENCHMARKS = ['inceptionv3']

def get_args():
    parser = argparse.ArgumentParser(description='Analysis script, get the statistics we want')
    parser.add_argument('--mode', type=str, default='runtime',
        help='Mode of analysis')

    parser.add_argument('--single', action='store_true', default=False, help='Plot single trajectory')

    return parser.parse_args()

def speedup_bar_result(benchmark):
    # Read in results
    tensat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tensat_root), "TASO")

    if benchmark == "inceptionv3_2":
        egg_stats_file = os.path.join(tensat_root, "tmp/inceptionv3_2_stats.txt")
    else:
        egg_stats_file = os.path.join(tensat_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    elif benchmark == 'inceptionv3_2':
        taso_benchmark_name = 'inceptionv3'
    elif benchmark == 'vgg':
        taso_benchmark_name = 'vgg19-7'
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

    #print("{}, orig {} taso {}".format(benchmark, np.mean(orig_runtimes), np.mean(taso_runtimes)))
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

def optimizer_time_breakdown(benchmark, post_fix=''):
    # Read in results
    tensat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if benchmark == "inceptionv3_2":
        egg_stats_file = os.path.join(tensat_root, "tmp/inceptionv3_2_stats.txt")
    else:
        egg_stats_file = os.path.join(tensat_root, "tmp/{}_1_stats.txt".format(benchmark))

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

    sat_time_mean = np.mean(egg_sat_times)
    ext_time_mean = np.mean(egg_ext_times)

    print(benchmark)
    print("sat time {}, ext time {}".format(sat_time_mean, ext_time_mean))

def optimizer_time_result(benchmark, post_fix=''):
    # Read in results
    tensat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tensat_root), "TASO")

    if benchmark == "inceptionv3_2":
        egg_stats_file = os.path.join(tensat_root, "tmp/inceptionv3_2_stats.txt")
    else:
        egg_stats_file = os.path.join(tensat_root, "tmp/{}_1_stats.txt".format(benchmark))
    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    elif benchmark == 'inceptionv3_2':
        taso_benchmark_name = 'inceptionv3'
    elif benchmark == 'vgg':
        taso_benchmark_name = 'vgg19-7'
    taso_stats_file = os.path.join(taso_root, "examples/{}_stats{}.txt".format(taso_benchmark_name, post_fix))

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

    egg_time = np.mean(egg_times)
    taso_total = np.mean(taso_totals)
    taso_best = np.mean(taso_bests)

    speedup_ratio = taso_total / egg_time

    result = {}
    result['egg_time'] = egg_time
    result['taso_total'] = taso_total
    result['taso_best'] = taso_best
    result['speedup_ratio'] = speedup_ratio

    egg_sat_time = np.mean(egg_sat_times)
    egg_ext_time = np.mean(egg_ext_times)
    result['egg_sat_time'] = egg_sat_time
    result['egg_ext_time'] = egg_ext_time

    return result

def equivalent_graphs(benchmark):
    # Read in results
    tensat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tensat_root), "TASO")

    egg_stats_file = os.path.join(tensat_root, "tmp/{}_1_stats.txt".format(benchmark))
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

def get_iter_stats(benchmark, tensat_root, iter=1):
    egg_stats_file = os.path.join(tensat_root, "tmp/{}_{}_stats.txt".format(benchmark, iter))
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

def get_iter_stats_self(benchmark, tensat_root, iter=1):
    egg_stats_file = os.path.join(tensat_root, "tmp/{}_{}_stats.txt".format(benchmark, iter))
    with open(egg_stats_file, 'r') as egg_f:
        egg_results = egg_f.readlines()

    egg_results = [json.loads(x) for x in egg_results]
    orig_runtimes = []
    egg_runtimes = []
    egg_sat_times = []
    egg_ext_times = []
    egg_n_nodes = []
    for res in egg_results[-1:]:
        orig_runtimes.append(res['original'])
        egg_runtimes.append(res['optimized'])
        egg_sat_times.append(res['saturation'])
        egg_ext_times.append(res['extraction'])
        egg_n_nodes.append(res['nodes'])

    mean_orig = np.mean(orig_runtimes)
    mean_optim = np.mean(egg_runtimes)
    mean_sat_iter = np.mean(egg_sat_times)
    mean_ext_iter = np.mean(egg_ext_times)

    return (mean_orig, mean_optim, mean_sat_iter + mean_ext_iter)

def multi_results(benchmark):
    """This function gets the results of the trend when the number of iterations of
    multi-pattern rewrites varies.

    It includes:
        How the speedup varies
        How the optimizer time varies
        How the number of enodes in the final egraph varies

    """
    # Read in results
    tensat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tensat_root), "TASO")

    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    elif benchmark == 'vgg':
        taso_benchmark_name = 'vgg19-7'
    taso_runtime_file = os.path.join(taso_root, "examples/{}_time.txt".format(taso_benchmark_name))

    with open(taso_runtime_file, 'r') as f:
        content = f.readlines()

    orig_runtimes = []
    for line in content[-5:]:
        times = line.split('\t')
        orig_runtimes.append(float(times[0]))
    orig_mean = np.mean(orig_runtimes)


    # iter=0
    mean_iter_0, mean_sat_iter_0, mean_ext_iter_0, mean_nodes_iter_0 = get_iter_stats(benchmark, tensat_root, iter=0)

    # iter=1
    mean_iter_1, mean_sat_iter_1, mean_ext_iter_1, mean_nodes_iter_1 = get_iter_stats(benchmark, tensat_root, iter=1)

    # iter=2
    mean_iter_2, mean_sat_iter_2, mean_ext_iter_2, mean_nodes_iter_2 = get_iter_stats(benchmark, tensat_root, iter=2)

    # iter=3
    mean_iter_3, mean_sat_iter_3, mean_ext_iter_3, mean_nodes_iter_3 = get_iter_stats(benchmark, tensat_root, iter=3)

    # Plot runtime & optimizer time v.s. iter
    speedup = [orig_mean/mean_iter_0, orig_mean/mean_iter_1, orig_mean/mean_iter_2]
    optimizer_time = [mean_sat_iter_0+mean_ext_iter_0, mean_sat_iter_1+mean_ext_iter_1, mean_sat_iter_2+mean_ext_iter_2]
    if mean_iter_3 > 0:
        speedup.append(orig_mean/mean_iter_3)
        optimizer_time.append(mean_sat_iter_3+mean_ext_iter_3)

    speedup = [(i-1)*100 for i in speedup]

    nodes = [mean_nodes_iter_0, mean_nodes_iter_1, mean_nodes_iter_2, mean_nodes_iter_3]

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
    plt.rcParams.update({'font.size': 24})
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
        #ax1.text(rect.get_x() + rect.get_width()/2.0, height+0.5, "{:0.1f}x".format(result['speedup_ratio']), ha='center', va='bottom', weight='heavy')

    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True, prop={'size': 14})
    ax1.legend((bar_0, bar_1), ("TASO", "Tensat"), loc='upper center', ncol=2, fancybox=True, shadow=True, prop={'size': 24})
    tick_locs = [x + width/2 + 0.5 for x in x_locs]
    plt.xticks(tick_locs, BENCHMARK_NAMES)
    ax1.tick_params(axis='x', labelrotation = 20)
    ax1.set_ylabel('Speedup percentage')

    fig = plt.gcf()
    fig.set_size_inches(1.8*len(BENCHMARKS), 12)

    plt.savefig("all_speedup.pdf", bbox_inches='tight')

    plt.close()

def speedup_mean(args):
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = speedup_bar_result(benchmark)
        #print("{}: egg {} taso {}".format(benchmark, results[benchmark]['egg_mean'], results[benchmark]['taso_mean']))

    taso_speedups = [results[benchmark]['taso_mean'] for benchmark in BENCHMARKS]
    egg_speedups = [results[benchmark]['egg_mean'] for benchmark in BENCHMARKS]
    taso_speedup_mean = scipy.stats.gmean(taso_speedups)
    egg_speedup_mean = scipy.stats.gmean(egg_speedups)

    speedup_diff = [egg_speedups[i] - taso_speedups[i] for i in range(len(taso_speedups))]
    #print("Mean speedup diff: {}".format(np.mean(speedup_diff)))
    print("TASO speedup mean: {}".format(np.mean(taso_speedups)))

def get_equivalent_graphs(args):
    for benchmark in BENCHMARKS:
        equivalent_graphs(benchmark)

def time_breakdown(args):
    results = {}
    for benchmark in BENCHMARKS:
        optimizer_time_breakdown(benchmark)

def optimizer_time_together(args):
    plt.rcParams.update({'font.size': 24})
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = optimizer_time_result(benchmark)

    # Plot bar and save
    width = 0.8
    x_locs = [i*3 for i in range(len(BENCHMARKS))]

    colors = ['b', 'lightblue', 'r', 'c']

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
    ax1.legend((bar_0, bar_1, bar_2), ("TASO total", "TASO best", "Tensat"), loc='upper center', ncol=3, fancybox=True, shadow=True, prop={'size': 24})
    tick_locs = [x + width/2 + 0.5 for x in x_locs]
    plt.xticks(tick_locs, BENCHMARK_NAMES)
    ax1.tick_params(axis='x', labelrotation = 20)

    fig = plt.gcf()
    fig.set_size_inches(1.8*len(BENCHMARKS), 12)

    plt.savefig("all_optim_time.pdf", bbox_inches='tight')

    plt.close()

def optimizer_time_mean(args):
    results = {}
    ratios_taso = []
    for benchmark in BENCHMARKS:
        results[benchmark] = optimizer_time_result(benchmark, post_fix='')
        print("{}: egg {} taso {}".format(benchmark, results[benchmark]['egg_time'], results[benchmark]['taso_total']))
        time_100 = results[benchmark]['taso_total']
        results[benchmark] = optimizer_time_result(benchmark, post_fix='_k5')
        time_1k = results[benchmark]['taso_total']
        ratios_taso.append(time_1k/time_100)

    ratios = [results[benchmark]['speedup_ratio'] for benchmark in BENCHMARKS]
    ratio_mean = scipy.stats.gmean(ratios)
    print("Mean ratio: {}".format(ratio_mean))

    print("Mean taso ratio: {}".format(scipy.stats.gmean(ratios_taso)))

def traj_results(benchmark, single=False):
    """This function gets the trajectory of how speedup varies with optimization time. For Tensat, it is when the number of iterations of
    multi-pattern rewrites varies; for TASO, it is varying number of iterations.

    Return: dict with
    - 'taso': {'speedup': [speedups], 'time': [times]}
    - 'tensat': {'speedup': [speedups], 'time': [times]}
    """
    # Read in results
    tensat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taso_root = os.path.join(os.path.dirname(tensat_root), "TASO")

    taso_benchmark_name = benchmark
    if benchmark == 'nasneta':
        taso_benchmark_name = 'nasnet_a'
    elif benchmark == 'vgg':
        taso_benchmark_name = 'vgg19-7'

    if single:
        taso_iters = [1,2,3,8,10,30,100]
    else:
        taso_iters = [10, 30, 100]
    orig_times = []
    speedups = []
    optimizer_times = []
    for iter in taso_iters:
        taso_runtime_file = os.path.join(taso_root, "examples/{}_time_{}.txt".format(taso_benchmark_name, iter))

        with open(taso_runtime_file, 'r') as f:
            content = f.readlines()

        orig_runtimes = []
        optim_runtimes = []
        for line in content[-5:]:
            times = line.split('\t')
            orig_runtimes.append(float(times[0]))
            optim_runtimes.append(float(times[1]))

        orig_mean = np.mean(orig_runtimes)
        optim_mean = np.mean(optim_runtimes)

        speedup = orig_mean / optim_mean
        speedup = (speedup - 1) * 100
        speedups.append(speedup)
        orig_times.append(orig_mean)

        taso_stats_file = os.path.join(taso_root, "examples/{}_stats_{}.txt".format(taso_benchmark_name, iter))
        with open(taso_stats_file, 'r') as f:
            content = f.readlines()
        taso_totals = []
        for line in content[-5:]:
            elements = line.split(' ')
            taso_totals.append(float(elements[3][:-1]))
        time_mean = np.mean(taso_totals)
        optimizer_times.append(time_mean)

    return_dict = {}
    return_dict['taso'] = {
        'speedup': speedups,
        'time': optimizer_times,
    }

    tensat_iters = [0, 1, 2]
    tensat_speedups = []
    tensat_times = []
    for iter in tensat_iters:
        orig, optim, optim_time = get_iter_stats_self(benchmark, tensat_root, iter=iter)
        speedup = orig / optim
        speedup = (speedup - 1) * 100
        tensat_speedups.append(speedup)
        tensat_times.append(optim_time)

    return_dict['tensat'] = {
        'speedup': tensat_speedups,
        'time': tensat_times,
    }

    return return_dict

def trajectories(args):
    # Single plot with legend. Each benchmark has a color, Tensat and TASO uses different line styles
    plt.rcParams.update({'font.size': 18})
    results = {}

    if args.single:
        BENCHMARKS_TREND = ['inceptionv3']
    for benchmark in BENCHMARKS_TREND:
        results[benchmark] = traj_results(benchmark, single=args.single)

    colors = ['b', 'g', 'tab:orange', 'm', 'r', 'c', 'k']

    # Plot optimizer time
    fig, ax = plt.subplots(figsize=(5,7))

    if args.single:
        for (i, benchmark) in enumerate(BENCHMARKS_TREND):
            # TASO
            taso_speedups = results[benchmark]['taso']['speedup']
            taso_times = results[benchmark]['taso']['time']
            # Timeout at 60 sec, so represents the final part with flat line
            taso_speedups[-1] = taso_speedups[-2]
            lns = ax.plot(taso_times, taso_speedups, marker='x', color=colors[i], label='TASO')

            # tensat
            tensat_speedups = results[benchmark]['tensat']['speedup']
            tensat_times = results[benchmark]['tensat']['time']
            # Timeout at 60 sec, so represents the final part with flat line
            tensat_times.append(taso_times[-1])
            tensat_speedups.append(tensat_speedups[-1])
            lns2 = ax.plot(tensat_times, tensat_speedups, marker='s', color=colors[i+4], label="Tensat")

        #ax.set_xscale('log')
        ax.set_xlim(right=60)
        fig.text(0.0, 0.5, 'Speedup percentage', va='center', rotation='vertical')
        ax.set_xlabel('Optimizer time (seconds)')
        ax.legend()

        fig.savefig("traj.pdf", bbox_inches='tight')

    else:

        for (i, benchmark) in enumerate(BENCHMARKS_TREND):
            # TASO
            taso_speedups = results[benchmark]['taso']['speedup']
            taso_times = results[benchmark]['taso']['time']
            lns = ax.plot(taso_times, taso_speedups, marker='x', color=colors[i], label=BENCHMARK_NAMES_TREND[i])

            # tensat
            tensat_speedups = results[benchmark]['tensat']['speedup']
            tensat_times = results[benchmark]['tensat']['time']
            lns2 = ax.plot(tensat_times, tensat_speedups, marker='s', color=colors[i])

        ax.set_xscale('log')
        fig.text(0.0, 0.5, 'Speedup percentage', va='center', rotation='vertical')
        ax.set_xlabel('Optimizer time (seconds)')

        fig.savefig("traj.pdf", bbox_inches='tight')

        handles, labels = ax.get_legend_handles_labels()
        for handle in handles:
            handle.set_marker("")

        # Plot legend
        figlegend = plt.figure(figsize=(2.0,2.5))
        figlegend.legend(handles, labels, 'center', ncol=1, fancybox=True, shadow=True, prop={'size': 14})
        figlegend.savefig("legend_traj.pdf")

        marker_legend = plt.figure(figsize=(2.0, 0.7))
        handles_marker = [handles[0], handles[1]]
        handles_marker[0].set_marker("x")
        handles_marker[1].set_marker("s")
        handles_marker[0].set_color("k")
        handles_marker[1].set_color("k")
        labels_marker = ["TASO", "Tensat"]
        marker_legend.legend(handles_marker, labels_marker, 'center', ncol=1, fancybox=True, shadow=True, prop={'size': 14})
        marker_legend.savefig("legend_traj_marker.pdf")

    plt.close()

def multi_trend_together(args):
    plt.rcParams.update({'font.size': 18})
    results = {}
    for benchmark in BENCHMARKS_TREND:
        results[benchmark] = multi_results(benchmark)

    colors = ['b', 'g', 'tab:orange', 'm', 'r', 'c', 'k']
    n_iter = [0,1,2,3]

    # Plot speedup
    #create a new figure with two subplots
    fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)

    #set the "zoom" or the y-limits on each subplots
    ax2.set_ylim(0,30)
    ax1.set_ylim(60,90)

    ax2.set_xlabel('#iter of multi pattern rewrites')

    for (i, benchmark) in enumerate(BENCHMARKS_TREND):
        speedup = results[benchmark]['speedup']
        ax1.plot(n_iter[:len(speedup)], speedup, marker='s', color=colors[i], label=BENCHMARK_NAMES_TREND[i])
        ax2.plot(n_iter[:len(speedup)], speedup, marker='s', color=colors[i], label=BENCHMARK_NAMES_TREND[i])
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

    for (i, benchmark) in enumerate(BENCHMARKS_TREND):
        optimizer_time = results[benchmark]['optimizer']
        lns2 = ax_optim.plot(n_iter[:len(optimizer_time)], optimizer_time, marker='s', color=colors[i], label=BENCHMARK_NAMES_TREND[i])
        #if len(optimizer_time) < 3:
        #    ax_optim.scatter(n_iter[-1], 3600, marker='x', color=colors[i])

    ax_optim.set_yscale('log')
    ax_optim.set_ylabel('Optimizer time (seconds)')
    ax_optim.set_xlabel('#iter of multi pattern rewrites')

    ax_optim.set_xticks(n_iter)
    ax_optim.set_xticklabels(['{}'.format(i) for i in n_iter])

    fig_optim.savefig("optim_trend.pdf", bbox_inches='tight')

    # Plot number of nodes
    fig_nodes, ax_nodes = plt.subplots()

    for (i, benchmark) in enumerate(BENCHMARKS_TREND):
        nodes = results[benchmark]['nodes']
        lns2 = ax_nodes.plot(n_iter[:len(nodes)], nodes, marker='s', color=colors[i], label=BENCHMARK_NAMES_TREND[i])

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
    if args.mode == 'speedup_together':
        # Bar plot of speedups of the optimized graphs, together
        plot_speedup_together(args)
    elif args.mode == 'speedup_mean':
        speedup_mean(args)
    elif args.mode == 'equivalent':
        # Get number of equivalent graphs explored
        get_equivalent_graphs(args)
    elif args.mode == 'optim_mean':
        optimizer_time_mean(args)
    elif args.mode == 'optimizer_together':
        # Bar plot of the optimizer time
        optimizer_time_together(args)
    elif args.mode == 'breakdown':
        # Get sat and ext time breakdown
        time_breakdown(args)
    elif args.mode == "multi_together":
        # Plot trend with iterations of multi-pattern rewrites, benchmarks together
        multi_trend_together(args)
    elif args.mode == "traj":
        # Plot trajectory of speedup varying with optimization time
        trajectories(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
