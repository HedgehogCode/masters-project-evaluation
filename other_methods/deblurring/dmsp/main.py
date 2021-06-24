import os
import sys
import argparse

import numpy as np
import imageio

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

import dppp

from image_similarity_measures import quality_metrics

from DMSP import DMSPRestore


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


def load_model():
    return tf.saved_model.load(os.path.join('DMSP', 'DAE'))


def crop_to_valid(x, kernel):
    pad_y = np.floor(kernel.shape[0] / 2.0).astype(np.int64)
    pad_x = np.floor(kernel.shape[1] / 2.0).astype(np.int64)
    return x[pad_y:-pad_y, pad_x:-pad_x, :]


def to_uint8(x):
    return tf.cast(tf.clip_by_value(x, 0, 1) * 255, tf.uint8)


def run_dmsp(degraded, kernel, noise_level, model):
    kernel = kernel[:, :, 0, 0].numpy()
    params = {}
    params['denoiser'] = model
    params['sigma_dae'] = 11.0
    params['num_iter'] = 300
    params['mu'] = 0.9
    params['alpha'] = 0.1
    # params['gt'] = image * 255

    # Prepare degraded input
    degraded = crop_to_valid(degraded, kernel) * 255.0

    # No downscaling -> No subsampling mask
    subsampling_mask = np.ones_like(degraded)

    # running DMSP
    restored = DMSPRestore.DMSP_restore(degraded=degraded,
                                        kernel=kernel,
                                        subsampling_mask=subsampling_mask,
                                        sigma_d=noise_level * 255,
                                        params=params)

    return restored / 255.0


def get_dataset(images_name, images_split, kernels_name, kernels_split, noise_stddev, mode, image_key='image'):
    images = tfds.load(name=images_name, split=images_split) \
        .map(datasets.map_on_dict(datasets.to_float32)) \
        .map(datasets.map_on_dict(datasets.from_255_to_1_range)) \
        .batch(1)
    kernels = tfds.load(name=kernels_name, split=kernels_split) \
        .map(datasets.crop_kernel_to_size) \
        .map(datasets.map_on_dict(datasets.to_float32)) \
        .map(datasets.map_on_dict(dppp.conv2D_filter_rgb))

    def add_image(image):
        def apply(x):
            return {'kernel': x['kernel'], 'image': image}
        return apply

    def add_kernels(x):
        return kernels.map(add_image(x[image_key]))

    def blur_and_arrange(x):
        image = x['image']
        kernel = x['kernel']
        degraded = dppp.blur(
            image, kernel, noise_stddev=noise_stddev, mode=mode)
        return (degraded, kernel), image

    return images.flat_map(add_kernels).map(blur_and_arrange)


def evaluate(dataset, dataset_name, noise_stddev, model):
    print(f"\n{dataset_name} Results (PSNR, FSIM):")

    psnr_sum = 0
    fsim_sum = 0
    num = 0

    for index, ((degraded, kernel), image) in enumerate(dataset):
        degraded = degraded[0]
        image = image[0]

        # Run dmsp
        reconstructed = run_dmsp(degraded, kernel, noise_stddev, model)

        # Compute PSNR and FSIM
        image_cropped = crop_to_valid(image, kernel)
        reconstructed_cropped = crop_to_valid(reconstructed, kernel)
        psnr_ = psnr(image_cropped, reconstructed_cropped)
        fsim_ = fsim(image_cropped, reconstructed_cropped)

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


def main(args):
    mode = 'constant'
    model = load_model()

    if args.example:
        # Example mode
        noise_stddev = 0.04
        dataset = get_dataset('set14', 'test', 'schelten_kernels/dmsp',
                              'test', noise_stddev, mode, image_key='hr')

        # Get the example image
        (degraded, kernel), image = datasets.get_one_example(dataset, index=0)
        degraded = degraded[0]
        image = image[0]

        # Run dmsp
        reconstructed = run_dmsp(degraded, kernel, noise_stddev, model)

        # Compute PSNR and FSIM
        image_cropped = crop_to_valid(image, kernel)
        reconstructed_cropped = crop_to_valid(reconstructed, kernel)
        psnr_ = psnr(image_cropped, reconstructed_cropped)
        fsim_ = fsim(image_cropped, reconstructed_cropped)

        # Save the reconstructed image
        imageio.imwrite('dmsp.png', to_uint8(reconstructed))

        # Save the PSNR and FSIM to a text file
        with open('dmsp.txt', 'w') as f:
            f.write(f"PSNR: {psnr_}, FSIM: {fsim_}")

    else:
        # Normal evaluation mode
        for noise_stddev in [0.01, 0.02, 0.03, 0.04]:
            dataset = get_dataset('bsds500/dmsp', 'validation',
                                  'schelten_kernels/dmsp', 'test', noise_stddev, mode)
            evaluate(dataset, f"DMSP - {noise_stddev}", noise_stddev, model)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--example', action='store_true',
                        help="Run on the example image and store the image in `dmsp.png`.")
    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
