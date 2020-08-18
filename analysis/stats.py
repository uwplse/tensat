from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Analysis script, get the statistics we want')
    parser.add_argument('--mode', type=str, default='runtime',
        help='Mode of analysis')
    parser.add_argument('--file', type=str, default='data.txt',
        help='File for the input data to analyze')

    return parser.parse_args()

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

    #low = min(runtimes)
    #high = max(runtimes)
    #ax1.set_ylim(low-0.5*(high-low), high+0.5*(high-low))
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
    #ax2.legend(lines + lines2, labels + labels2, fontsize=10)
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

    #low = min(runtimes)
    #high = max(runtimes)
    #ax1.set_ylim(low-0.5*(high-low), high+0.5*(high-low))
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

    #ax2.legend(lines + lines2, labels + labels2, fontsize=10)
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
    #plt.rcParams.update({'figure.autolayout': True})

    for (benchmark, result) in results.items():
        #plot_runtime_and_speed(benchmark, result)
        plot_runtime_and_speed_2(benchmark, result)




def main():
    # Parse arguments
    args = get_args()
    if args.mode == 'runtime':
        runtime_stats(args)
    elif args.mode == 'plot':
        plot_bars(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()