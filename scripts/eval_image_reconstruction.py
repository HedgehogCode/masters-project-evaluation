#!/usr/bin/env python
"""Evaluate image restoration models.
"""

from __future__ import print_function
import sys
import argparse
from typing import Dict, Tuple
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_datasets_bw as datasets
import dppp
from dppp.types import *

DEBLUR_NB = 'deblur_nb'
DEBLUR_NA = 'deblur_na'
SUPER_RES = 'super_res'
INPAINTING = 'inpaint'

DMSP = 'dmsp'
HQS = 'hqs'

METRICS = {
    'psnr': (dppp.psnr, 1),
    'ssim': (dppp.ssim, 1),
    'fsim': (dppp.fsim, 10**10)
}


def main(args):
    data = get_dataset(args)
    metrics = {k: METRICS[k] for k in args.metrics}
    dmsp_fn = get_reconstruction_fn(args)

    if args.test_run:
        data = data.take(6)

    # Run the evaluation
    evaluate(args, dmsp_fn, data, metrics, args.csv_file.name)

    return 0


def get_dataset(args):
    if args.task == SUPER_RES:
        return get_super_res_dataset(args)
    if args.task == DEBLUR_NB:
        return get_deblur_nb_dataset(args)


def get_super_res_dataset(args):
    # Args for the dataset builder
    builder_kwargs = {
        'resize_method': args.sr_method,
        'scale': args.sr_scale,
        'antialias': args.sr_antialias
    }

    # Load the data
    return tfds.load(name=args.dataset, split=args.split, builder_kwargs=builder_kwargs) \
        .map(datasets.map_on_dict(datasets.to_float32)) \
        .map(datasets.map_on_dict(datasets.from_255_to_1_range)) \
        .map(lambda x: ([x['lr']], x['hr'])) \
        .batch(1)


def get_deblur_nb_dataset(args):
    # Load the images and kernels and apply basic mapping
    images = tfds.load(name=args.dataset, split=args.split) \
        .map(datasets.map_on_dict(datasets.to_float32)) \
        .map(datasets.map_on_dict(datasets.from_255_to_1_range)) \
        .batch(1)
    kernels = tfds.load(name=args.kernels, split=args.kernels_split) \
        .map(datasets.crop_kernel_to_size) \
        .map(datasets.map_on_dict(datasets.to_float32)) \
        .map(datasets.map_on_dict(dppp.conv2D_filter_rgb))

    # A function to add an image to each kernel
    def add_image(image):
        def apply(x):
            return {'kernel': x['kernel'], 'image': image}
        return apply

    # A function to return a dataset of images and kernels for each image
    def add_kernels(x):
        return kernels.map(add_image(x[args.dataset_image_key]))

    # Blur and arrange for the test loop
    def blur_and_arrange(x):
        image = x['image']
        kernel = x['kernel']
        degraded = dppp.blur(image, kernel, noise_stddev=args.noise_stddev, mode='wrap')
        return (degraded, kernel, tf.constant(args.noise_stddev)), image

    return images.flat_map(add_kernels).map(blur_and_arrange)


def get_reconstruction_fn(args):
    denoiser, (denoiser_min, denoiser_max) = dppp.load_denoiser(args.model.name)

    prior_noise_stddev = args.prior_noise_stddev
    if prior_noise_stddev is not None and denoiser_min <= prior_noise_stddev <= denoiser_max:
        denoiser_stddev = prior_noise_stddev
    else:
        denoiser_stddev = denoiser_max

    task = args.task
    algo = args.algorithm

    # DMSP deblur non-blind
    if task == DEBLUR_NB and algo == DMSP:
        return partial(dppp.dmsp_deblur_nb,
                       denoiser=denoiser,
                       denoiser_stddev=denoiser_stddev,
                       num_steps=args.num_steps,
                       mode='wrap')

    # DMSP super resolution
    if task == SUPER_RES and algo == DMSP:
        return partial(dppp.dmsp_super_resolve,
                       sr_factor=args.sr_scale,
                       denoiser=denoiser,
                       denoiser_stddev=denoiser_stddev,
                       num_steps=args.num_steps)

    # HQS deblur non-blind
    if task == DEBLUR_NB and algo == HQS:
        return partial(dppp.hqs_deblur_nb,
                       denoiser=denoiser,
                       denoiser_stddev=denoiser_max,
                       num_steps=args.num_steps)

    # HQS super resolution
    if task == SUPER_RES and algo == HQS:
        return partial(dppp.hqs_super_resolve,
                       sr_factor=args.sr_scale,
                       denoiser=denoiser,
                       denoiser_stddev=denoiser_stddev,
                       num_steps=args.num_steps)

    # TODO support super resolution with a given kernel

    raise ValueError(f"The task '{args.task}' with the algorithm '{args.algorithm}' " +
                     "is not yet supported.")


def evaluate(args,
             reconstruction_fn,
             data: tf.data.Dataset,
             metrics: Dict[str, Tuple[MetricFnType, int]],
             csv_filename: str):

    # Set the CSV callback
    csv_callback, csv_string = dppp.callback_csv_metric(metrics, args.num_steps)

    # Run the reconstruction function on the whole dataset
    num_images = 0
    for d in data:
        print(f"Evaluating image {num_images + 1}...")
        args, gt_image = d

        callback = [csv_callback(image_id=str(num_images), gt_image=gt_image)]
        reconstruction_fn(*args, callbacks=callback)
        num_images += 1

    with open(csv_filename, 'wb') as file:
        file.write(csv_string.numpy())


def parse_args(arguments):
    """Parse the command line arguments."""
    # TODO these are many arguments and using this as a script is not useful anymore
    # I should create a class containing the arguments and make other scripts creating
    # the arguments.
    rm = tf.image.ResizeMethod
    sr_method_choices = [
        rm.AREA, rm.BICUBIC, rm.BILINEAR, rm.GAUSSIAN, rm.LANCZOS3, rm.LANCZOS5, rm.MITCHELLCUBIC,
        rm.NEAREST_NEIGHBOR
    ]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    def add(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    add('-d', '--dataset',
        help="Name of the dataset.",
        type=str)
    add('--split',
        help="Split of the dataset",
        choices=['train', 'test', 'validation'],
        default='test')
    add('-m', '--model',
        help="Path to the model h5.",
        type=argparse.FileType('r'))
    add('-t', '--task',
        help="The task to perform.",
        choices=[DEBLUR_NB, DEBLUR_NA, SUPER_RES, INPAINTING])
    add('-a', '--algorithm',
        help="The algorithms to use.",
        choices=[DMSP, HQS],
        default=DMSP)
    add('--metrics',
        help="List of the metrics to evaluate",
        choices=list(METRICS.keys()),
        nargs='+',
        default=list(METRICS.keys()))
    add('--num-steps',
        help="The number of steps to perform during the optimization",
        type=int,
        default=300)
    add('csv_file',
        help="Path to the csv file where the results will be written to.",
        type=argparse.FileType('w'))

    # Deblur specific
    add('-k', '--kernels',
        help="Name of the kernels dataset.",
        type=str,
        default="")
    add('--kernels-split',
        help="Split of the kernels dataset.",
        choices=['train', 'test', 'validation'],
        default='test')
    add('--noise-stddev',
        help="Standard deviation of the added gaussian noise.",
        type=float,
        default=0.01)
    add('--dataset-image-key',
        help="The key used for the image in the dataset.",
        type=str,
        default="image")

    # Super resolution specific
    add('--sr-scale',
        help="Scaling for the super resolution.",
        default=4,
        type=int)
    add('--sr-method',
        help="Downscaling method for the super resolution.",
        choices=sr_method_choices,
        default=rm.BILINEAR)
    add('--sr-antialias',
        help="Use antialias to downscale the images for super resolution.",
        default=True,
        action='store_true')

    # DMSP specific
    add('--prior-noise-stddev',
        help="The standard deviation of the noise for the prior " +
        "if the model was trained on a range of values. " +
        "Ignored if the model was trained for a specific value",
        default=None,
        type=int)

    # Special
    add('--test-run',
        help="Test run: Run only on 2 images.",
        action='store_true')
    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
