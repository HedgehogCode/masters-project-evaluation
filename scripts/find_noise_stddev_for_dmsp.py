#!/usr/bin/env python

"""Tries different levels of noise for the DMSP algorithm with the ffdnet-v02-0.0-0.2 model
"""

from __future__ import print_function
import os
import sys
from multiprocessing import Process
import argparse
import itertools
from collections import namedtuple

import pandas as pd

DEBLUR_NB = 'deblur_nb'
SUPER_RES = 'super_res'

DMSP = 'dmsp'

METRICS = [
    'psnr', 'ssim', 'fsim'
]

PRIOR_NOISE_STDDEVS = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]

DEBLUR_DATASETS = [
    ('bsds500/dmsp', 'validation', 'image'),
    # ('set5', 'test', 'hr')
]
DEBLUR_NOISE_STDDEV = [0.01, 0.02, 0.03, 0.04]
DEBLUR_KERNELS = ('schelten_kernels/dmsp', 'test')
# DEBLUR_NOISE_STDDEV = [0.04]  # DEBUG

SR_DATASETS = [
    ('set5', 'test'),
    # ('set14', 'test')
]
SR_SCALES = [2, 3, 4, 5]
# SR_SCALES = [4]  # DEBUG
SR_SCALING_METHOD = 'bicubic'
SR_ANTIALIAS = True

COMMON_ARGS = ['dataset', 'split', 'model', 'task', 'metrics',
               'algorithm', 'prior_noise_stddev',
               'num_steps', 'csv_file', 'test_run']
DEBLUR_ARGS = ['kernels', 'kernels_split', 'noise_stddev', 'dataset_image_key']
SUPER_RES_ARGS = ['sr_scale', 'sr_method', 'sr_antialias']

ArgsDeblur = namedtuple('ArgsDeblur', [*COMMON_ARGS, *DEBLUR_ARGS])
ArgsSuperRes = namedtuple('ArgsSuperRes', [*COMMON_ARGS, *SUPER_RES_ARGS])
File = namedtuple('File', ['name'])


def run_evaluate(eval_args):
    import eval_image_reconstruction
    eval_image_reconstruction.main(eval_args)


def main(args):

    def csv_file(name):
        return File(name=os.path.join(args.results, f'{name}.csv'))

    model_file = args.model

    configs = []

    # Deblur Non-Blind - DMSP
    deblur_options = itertools.product(PRIOR_NOISE_STDDEVS, DEBLUR_DATASETS, DEBLUR_NOISE_STDDEV)
    for p_noise_stddev, dataset, noise_stddev in deblur_options:
        task = DEBLUR_NB
        dataset_name = dataset[0].replace('/', '-')
        num_steps = 5 if args.test_run else 300
        csv_name = f'{task}--{noise_stddev}__{dataset_name}__{p_noise_stddev:.3f}'
        configs.append(ArgsDeblur(
            dataset=dataset[0],
            split=dataset[1],
            dataset_image_key=dataset[2],
            model=model_file,
            task=task,
            algorithm=DMSP,
            metrics=METRICS,
            num_steps=num_steps,
            kernels=DEBLUR_KERNELS[0],
            kernels_split=DEBLUR_KERNELS[1],
            noise_stddev=noise_stddev,
            prior_noise_stddev=p_noise_stddev,
            csv_file=csv_file(csv_name),
            test_run=args.test_run
        ))

    # Super resolution - DMSP
    sr_options = itertools.product(PRIOR_NOISE_STDDEVS, SR_DATASETS, SR_SCALES)
    for p_noise_stddev, dataset, scale in sr_options:
        task = SUPER_RES
        dataset_name = dataset[0].replace('/', '-')
        num_steps = 5 if args.test_run else 300
        csv_name = f'{task}--x{scale}__{dataset_name}--AA__{p_noise_stddev:.3f}'
        configs.append(ArgsSuperRes(
            dataset=dataset[0],
            split=dataset[1],
            model=model_file,
            task=task,
            algorithm=DMSP,
            metrics=METRICS,
            num_steps=num_steps,
            sr_scale=scale,
            sr_method=SR_SCALING_METHOD,
            sr_antialias=SR_ANTIALIAS,
            prior_noise_stddev=p_noise_stddev,
            csv_file=csv_file(csv_name),
            test_run=args.test_run
        ))

    # Evaluate each config
    for i, eval_args in enumerate(configs):
        print(f'Evaluating config {i} of {len(configs)}...')
        if not os.path.exists(eval_args.csv_file.name) and not args.dry_run:
            print(f'Writing to file {eval_args.csv_file.name}')
            p = Process(target=run_evaluate, args=(eval_args,))
            p.start()
            p.join()
        elif args.dry_run:
            print(f'Arguments: {eval_args}')
        else:
            # Skip if the evaluation was already done
            print(f'File exists "{eval_args.csv_file.name}", skipping...')

    # Combine all csv files
    dfs = {DEBLUR_NB: [], SUPER_RES: []}
    for eval_args in configs:
        try:
            df = pd.read_csv(eval_args.csv_file.name)
        except FileNotFoundError as e:
            # Ignore the missing file if this is a dry run
            if args.dry_run:
                continue
            else:
                raise e

        task = eval_args.task
        df['task'] = task
        df['dataset'] = eval_args.dataset
        df['split'] = eval_args.split
        df['model'] = MODEL_NAME
        df['num_steps'] = eval_args.num_steps
        df['prior_noise_stddev'] = eval_args.prior_noise_stddev

        if task == DEBLUR_NB:
            df['noise_stddev'] = eval_args.noise_stddev
        elif task == SUPER_RES:
            df['sr_scale'] = eval_args.sr_scale

        dfs[task].append(df)

    for task, df_list in dfs.items():
        df = pd.concat(df_list)
        csv_file_name = csv_file(task).name
        print(f"Saving csv with columns {list(df.columns)} to {csv_file_name}...")
        if not args.dry_run:
            df.to_csv(csv_file_name)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help="Path to the model h5.", type=argparse.FileType('r'))
    parser.add_argument('results', help="Folder where the results will be written to.", type=str)
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help="Dry run: Do not exectute anything but log what would be executed.")
    parser.add_argument('--test-run', action='store_true',
                        help="Test run: Run each task only for 5 step.")
    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
