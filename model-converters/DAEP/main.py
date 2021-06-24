#!/usr/bin/env python

"""Convert the pretrained DAE from the DAEP paper to keras.
Download the weights from https://github.com/siavashBigdeli/DAEP/blob/master/model/DAE_sigma25.caffemodel
"""

from __future__ import print_function
import sys
import argparse

from caffe_pb2 import NetParameter
from tensorflow import keras
import numpy as np

# Define some architecture variables
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_CONV = 20
FILTERS = 64
KERNEL_SIZE = (3, 3)


def main(args):
    # Create the model
    model = _dae_model()
    model.summary()

    # Load the weights
    caffe_weights = NetParameter()
    caffe_weights.MergeFromString(args.weights.read())
    _load_weights(model, caffe_weights)

    # Save the model
    model.save(args.outfile.name)


def _dae_model():
    inp = keras.layers.Input((None, None, IN_CHANNELS))
    x = inp
    x = keras.layers.Conv2D(FILTERS, KERNEL_SIZE, (1, 1), 'SAME', name=f'conv1')(x)
    x = keras.layers.Activation('relu', name=f'relu1')(x)
    for idx in range(NUM_CONV - 2):
        x = keras.layers.Conv2D(FILTERS, KERNEL_SIZE, (1, 1), 'SAME', name=f'conv{idx+2}')(x)
        x = keras.layers.BatchNormalization(name=f'bn{idx+2}',
                                            epsilon=1e-5,
                                            scale=False,
                                            center=False)(x)
        x = keras.layers.Activation('relu', name=f'relu{idx+2}')(x)
    oup = keras.layers.Conv2D(OUT_CHANNELS, KERNEL_SIZE, (1, 1),
                              'SAME', name=f'conv{NUM_CONV}')(x) + inp
    return keras.Model(inputs=inp, outputs=oup)


def _load_weights(model, caffe_weights):
    for layer in model.layers:
        name = layer.name

        # Conv layers
        if name.startswith('conv'):
            kernels, bias = _get_weights(caffe_weights, name)

            # Fix the kernel dimensions
            kernels = np.transpose(kernels, (2, 3, 1, 0))

            # BGR -> RGB
            if name == 'conv1':
                kernels = kernels[:, :, ::-1, :]
            elif name == f'conv{NUM_CONV}':
                kernels = kernels[:, :, :, ::-1]
                bias = bias[::-1]

            # 0-255 range to 0-1 range
            bias = bias / 255

            layer.set_weights([kernels, bias])

        # Batch nomalization layers
        if name.startswith('bn'):
            mean, variance, scale_factor = _get_weights(caffe_weights, name)
            mean = mean / 255
            layer.set_weights([mean / scale_factor, variance / scale_factor])


def _get_weights(caffe_weights, layer_name):
    layer = [l for l in caffe_weights.layer if l.name == layer_name][0]
    return [np.reshape(np.array(b.data), b.shape.dim).astype(np.float32) for b in layer.blobs]


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('weights', help="Path to the caffe weights file.",
                        type=argparse.FileType('rb'))
    parser.add_argument('outfile', help="Output file",
                        type=argparse.FileType('w'))

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
