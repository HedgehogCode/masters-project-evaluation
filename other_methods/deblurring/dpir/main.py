import os
import sys
import argparse

import numpy as np
import imageio

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

import dppp

import torch

from image_similarity_measures import quality_metrics

import dpir
from models.network_unet import UNetRes as net
from utils import utils_model
from utils import utils_image as util
from utils import utils_pnp as pnp
from utils import utils_sisr as sr


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


def to_uint8(x):
    return tf.cast(tf.clip_by_value(x, 0, 1) * 255, tf.uint8)


# See https://github.com/cszn/DPIR/blob/master/main_dpir_deblur.py
def load_model(device):
    weights_path = os.path.join('DPIR', 'model_zoo', 'drunet_color.pth')
    model = net(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(weights_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)


# See https://github.com/cszn/DPIR/blob/master/main_dpir_deblur.py
def run_dpir(degraded, kernel, noise_level, model, device):
    k = kernel[::-1, ::-1, 0, 0]
    noise_level_img = noise_level
    noise_level_model = noise_level_img
    modelSigma1 = 49
    modelSigma2 = noise_level_model*255
    iter_num = 8
    sf = 1
    rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model),
                                     iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
    rhos = torch.tensor(rhos).to(device)
    sigmas = torch.tensor(sigmas).to(device)

    x = util.single2tensor4(degraded).to(device)

    img_L_tensor, k_tensor = util.single2tensor4(
        degraded), util.single2tensor4(np.expand_dims(k, 2))
    [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
    FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

    for i in range(iter_num):
        # Step 1
        tau = rhos[i].float().repeat(1, 1, 1, 1)
        x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

        # Step 2
        x = util.augment_img_tensor4(x, i % 8)
        x = torch.cat((x, sigmas[i].float().repeat(
            1, 1, x.shape[2], x.shape[3])), dim=1)
        x = utils_model.test_mode(
            model, x, mode=2, refield=32, min_size=256, modulo=16)

        if i % 8 == 3 or i % 8 == 5:
            x = util.augment_img_tensor4(x, 8 - i % 8)
        else:
            x = util.augment_img_tensor4(x, i % 8)

    return util.tensor2single(x)


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


def evaluate(dataset, dataset_name, noise_stddev, model, device):
    print(f"\n{dataset_name} Results (PSNR, FSIM):")

    psnr_sum = 0
    fsim_sum = 0
    num = 0

    for index, ((degraded, kernel), image) in enumerate(dataset):
        degraded = degraded[0]
        image = image[0]

        # Run dpir
        reconstructed = run_dpir(degraded, kernel, noise_stddev, model, device)

        # Compute PSNR and FSIM
        psnr_ = psnr(image, reconstructed)
        fsim_ = fsim(image, reconstructed)

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
    mode = 'wrap'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    if args.example:
        # Example mode
        noise_stddev = 0.04
        dataset = get_dataset('set14', 'test', 'schelten_kernels/dmsp',
                              'test', noise_stddev, mode, image_key='hr')

        # Get the example image
        (degraded, kernel), image = datasets.get_one_example(dataset, index=0)
        degraded = degraded[0]
        image = image[0]

        # Run dpir
        reconstructed = run_dpir(degraded, kernel, noise_stddev, model, device)

        # Compute PSNR and FSIM
        psnr_ = psnr(image, reconstructed)
        fsim_ = fsim(image, reconstructed)

        # Save the reconstructed image
        imageio.imwrite('dpir.png', to_uint8(reconstructed))

        # Save the PSNR and FSIM to a text file
        with open('dpir.txt', 'w') as f:
            f.write(f"PSNR: {psnr_}, FSIM: {fsim_}")

    else:
        # Normal evaluation mode
        for noise_stddev in [0.01, 0.02, 0.03, 0.04]:
            dataset = get_dataset('bsds500/dmsp', 'validation',
                                  'schelten_kernels/dmsp', 'test', noise_stddev, mode)
            evaluate(dataset, f"DMSP - {noise_stddev}",
                     noise_stddev, model, device)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--example', action='store_true',
                        help="Run on the example image and store the image in `dpir.png`.")
    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
