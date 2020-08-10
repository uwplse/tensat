from __future__ import print_function
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Analysis script, get the statistics we want')
    parser.add_argument('--mode', type=str, default='runtime',
        help='Mode of analysis')
    parser.add_argument('--file', type=str,
        help='File for the input data to analyze')

    return parser.parse_args()

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

def main():
    # Parse arguments
    args = get_args()
    if args.mode == 'runtime':
        runtime_stats(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()