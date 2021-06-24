#!/usr/bin/env python

"""Convert the pretrained DRUNet from the Deep Plug-and-Play Image Restoration paper to keras.
Download the weights from https://github.com/cszn/DPIR/tree/master/model_zoo
"""

from __future__ import print_function
import sys
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import tensorflow as tf


def main(args):
    # Import dpir dependencies
    sys.path.append(args.dpir)
    from pytorch2keras import pytorch_to_keras
    from models.network_unet import UNetRes

    # Load the model (see https://github.com/cszn/DPIR/blob/master/main_dpir_deblur.py)
    model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode='strideconv', upsample_mode="convtranspose")
    model.load_state_dict(torch.load(args.weights.name), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False

    # Dummy input
    input_np = np.random.uniform(0, 1, (1, 4, 128, 128))
    input_var = Variable(torch.FloatTensor(input_np))

    # Convert
    k_model = pytorch_to_keras(
        model, input_var, [(4, None, None,)], verbose=True, change_ordering=True)

    # Create new model with padded input for arbitrary shapes
    inp = tf.keras.Input((None, None, 4))
    x = inp
    x, (h, w) = pad_input(x, 4)
    x = k_model(x)
    oup = x[:, :h, :w, :]

    k_model_padded = tf.keras.Model(inp, oup)

    # NOTE:
    # We do not have to multiply the input by 255 because the model only uses convolutional
    # layers without a bias. Therefore, it is scale invariant.

    # Save
    k_model_padded.save(args.outfile.name)


def pad_input(x, num_levels):
    # Pad the input to a multiple of num_levels^2
    patch_size = 2**num_levels
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    pad_h = -h % patch_size
    pad_w = -w % patch_size
    padding = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
    return tf.pad(x, padding, mode='CONSTANT'), (h, w)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dpir', help="Path to the DPIR project (https://github.com/cszn/DPI).",
                        type=str)
    parser.add_argument('weights', help="Path to the pytorch weights file.",
                        type=argparse.FileType('r'))
    parser.add_argument('outfile', help="Output file",
                        type=argparse.FileType('w'))

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
