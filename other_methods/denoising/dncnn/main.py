import os

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

import torch

from image_similarity_measures import quality_metrics

import dncnn
from models.network_dncnn import DnCNN as net
from utils import utils_image as util

###############################################################################
# METRICS TO USE
###############################################################################

def psnr(imgs_a, imgs_b):
    return tf.image.psnr(imgs_a[None, ...], imgs_b[None, ...], max_val=1)[0].numpy()


def fsim(imgs_a, imgs_b):
    a = tf.cast(imgs_a, tf.float32)
    b = tf.cast(imgs_b, tf.float32)
    fsim_val = tf.numpy_function(quality_metrics.fsim, [a, b], tf.float64)
    return tf.cast(fsim_val, tf.float32)


###############################################################################
# SOME HELPER FUNCTIONS
###############################################################################

def degrade(x, noise_stddev):
    noise = tf.random.normal(tf.shape(x), stddev=noise_stddev)
    return x + noise


def load_model(device):
    weights_path = os.path.join('KAIR', 'model_zoo', 'dncnn_color_blind.pth')
    model = net(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(weights_path), strict=True)
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)


def evaluate(dataset, dataset_name, noise_stddev, model, device):
    print(f"\n{dataset_name} Results (PSNR, FSIM):")

    psnr_sum = 0
    fsim_sum = 0
    num = 0

    for index, img in enumerate(dataset):
        noisy = degrade(img, noise_stddev)

        # Run the denoiser
        noisy_tensor = util.single2tensor4(noisy).to(device)
        denoised_tensor = model(noisy_tensor)
        denoised = util.tensor2single(denoised_tensor)

        # Compute PSNR and FSIM
        psnr_ = psnr(img, denoised)
        fsim_ = fsim(img, denoised)

        # Print for this image
        print(f"{index:4d} -- ({psnr_:.2f}, {fsim_:.4f})")

        # Update the sums
        psnr_sum += psnr_
        fsim_sum += fsim_

        num += 1

    print(f"MEAN -- ({psnr_sum/num:.2f}, {fsim_sum/num:.4f})")


###############################################################################
# SCRIPT
###############################################################################

dataset = tfds.load('bsds500/dmsp', split='validation') \
    .map(datasets.get_value('image')) \
    .map(datasets.to_float32) \
    .map(datasets.from_255_to_1_range)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)


for noise_stddev in [0.01, 0.05, 0.10, 0.20]:
    evaluate(dataset, f"BSDS - {noise_stddev}", noise_stddev, model, device)
