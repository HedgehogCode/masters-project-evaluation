#!/usr/bin/env python

"""Convert the pretrained DAE from the DMSP paper to keras.
Download the weights from https://github.com/siavashBigdeli/DMSP-tensorflow/blob/master/DAE_sigma11.mat
"""

from __future__ import print_function
import sys
import argparse

import scipy.io
import numpy as np
from tensorflow import keras

# Define some architecture variables
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_CONV = 20
FILTERS = 64
KERNEL_SIZE = (3, 3)


def main(args):
    model = _dae_model()
    model.summary()
    weights = scipy.io.loadmat(args.weights.name)['net']
    _load_weights(model, weights)
    model.save(args.outfile.name)


def _dae_model():
    inp = keras.layers.Input((None, None, IN_CHANNELS))
    x = inp
    for _ in range(NUM_CONV - 1):
        x = keras.layers.Conv2D(FILTERS, KERNEL_SIZE, (1, 1), 'SAME', activation='relu')(x)
    oup = keras.layers.Conv2D(OUT_CHANNELS, KERNEL_SIZE, (1, 1), 'SAME')(x) + inp
    return keras.Model(inputs=inp, outputs=oup)


def _load_weights(model, weights):
    for idx, layer in enumerate(model.layers[1:-1]):
        kernels = _get_kernels(weights, idx * 2)
        bias = _get_bias(weights, idx * 2)

        # BGR -> RGB
        if idx == 0:
            kernels = kernels[:, :, ::-1, :]
        elif idx == NUM_CONV - 1:
            kernels = kernels[:, :, :, ::-1]
            bias = bias[::-1]

        # 0-255 range to 0-1 range
        bias = bias / 255

        layer.set_weights([kernels, bias])


def _get_kernels(weights, layer_idx):
    kernels = weights[0, layer_idx]['weights'][0, 0][0, 0].astype(np.float32)
    return np.transpose(kernels, (1, 0, 2, 3))


def _get_bias(weights, layer_idx):
    bias = weights[0, layer_idx]['weights'][0, 0][0, 1].astype(np.float32)
    return np.reshape(bias, (-1, ))


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('weights', help="Path to the matlab weights file.",
                        type=argparse.FileType('r'))
    parser.add_argument('outfile', help="Output file",
                        type=argparse.FileType('w'))

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
