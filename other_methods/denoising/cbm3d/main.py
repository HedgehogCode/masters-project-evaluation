import os
import imageio
import bm3d
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from image_similarity_measures import quality_metrics


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


def evaluate(dataset, dataset_name, noise_stddev):
    print(f"\n{dataset_name} Results (PSNR, FSIM):")

    psnr_sum = 0
    fsim_sum = 0
    num = 0

    for index, img in enumerate(dataset):
        noisy = degrade(img, noise_stddev)
        denoised = bm3d.bm3d_rgb(noisy.numpy(), sigma_psd=noise_stddev)
        #  stage_arg=bm3d.BM3DStages.ALL_STAGES)

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

for noise_stddev in [0.01, 0.05, 0.10, 0.20]:
    evaluate(dataset, f"BSDS500 - {noise_stddev}", noise_stddev)
