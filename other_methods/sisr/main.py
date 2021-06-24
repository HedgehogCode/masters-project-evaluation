import os
import imageio
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from image_similarity_measures import quality_metrics


###############################################################################
# METRICS TO USE
###############################################################################


def psnr(imgs_a, imgs_b):
    return tf.image.psnr(imgs_a[None, ...], imgs_b[None, ...], max_val=255)[0].numpy()


def fsim(imgs_a, imgs_b):
    a = tf.cast(imgs_a, tf.float32) / 255.0
    b = tf.cast(imgs_b, tf.float32) / 255.0
    fsim_val = tf.numpy_function(quality_metrics.fsim, [a, b], tf.float64)
    return tf.cast(fsim_val, tf.float32)


###############################################################################
# SOME HELPER FUNCTIONS
###############################################################################

def upscale_bicubic(lr, hr):
    h, w = tf.shape(hr)[0:2]
    sr = tf.image.resize(lr, (h, w), method='bicubic', antialias=True)
    sr = tf.round(sr)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.cast(sr, tf.uint8)
    return sr


def read_hr(image, dataset, method):
    return imageio.imread(os.path.join('data', f'{dataset}_{method}', f'{image}_{method}.png'))


def evaluate(dataset, names, dataset_name):
    print(f"\n{dataset_name} Results (PSNR, FSIM):")

    psnr_bicubic_sum = 0
    psnr_enhancenet_sum = 0
    psnr_srgan_sum = 0
    psnr_esrgan_sum = 0

    fsim_bicubic_sum = 0
    fsim_enhancenet_sum = 0
    fsim_srgan_sum = 0
    fsim_esrgan_sum = 0

    num = 0

    for name, d in zip(names, dataset):
        hr = d['hr']
        lr = d['lr']

        # Upscale using bicubic
        sr_bicubic = upscale_bicubic(lr, hr)

        # Load others
        sr_enhancenet = read_hr(name, dataset_name, 'EnhanceNet')
        sr_srgan = read_hr(name, dataset_name, 'SRGAN')
        sr_esrgan = read_hr(name, dataset_name, 'ESRGAN')

        # Compute PSNR
        psnr_bicubic = psnr(sr_bicubic, hr)
        psnr_enhancenet = psnr(sr_enhancenet, hr)
        psnr_srgan = psnr(sr_srgan, hr)
        psnr_esrgan = psnr(sr_esrgan, hr)

        # Compute FSIM
        fsim_bicubic = fsim(sr_bicubic, hr)
        fsim_enhancenet = fsim(sr_enhancenet, hr)
        fsim_srgan = fsim(sr_srgan, hr)
        fsim_esrgan = fsim(sr_esrgan, hr)

        # Print for this image
        print(f"{name:<10} -- " +
              f"Bicubic: ({psnr_bicubic:.2f}, {fsim_bicubic:.4f}), " +
              f"EnhanceNet: ({psnr_enhancenet:.2f}, {fsim_enhancenet:.4f}), " +
              f"SRGAN: ({psnr_srgan:.2f}, {fsim_srgan:.4f}), " +
              f"ESRGAN: ({psnr_esrgan:.2f}, {fsim_esrgan:.4f})")

        # Update the sums
        psnr_bicubic_sum += psnr_bicubic
        psnr_enhancenet_sum += psnr_enhancenet
        psnr_srgan_sum += psnr_srgan
        psnr_esrgan_sum += psnr_esrgan

        fsim_bicubic_sum += fsim_bicubic
        fsim_enhancenet_sum += fsim_enhancenet
        fsim_srgan_sum += fsim_srgan
        fsim_esrgan_sum += fsim_esrgan

        num += 1

    print(f"MEAN       -- " +
          f"Bicubic: ({psnr_bicubic_sum/num:.2f}, {fsim_bicubic_sum/num:.4f}), " +
          f"EnhanceNet: ({psnr_enhancenet_sum/num:.2f}, {fsim_enhancenet_sum/num:.4f}), " +
          f"SRGAN: ({psnr_srgan_sum/num:.2f}, {fsim_srgan_sum/num:.4f}), " +
          f"ESRGAN: ({psnr_esrgan_sum/num:.2f}, {fsim_esrgan_sum/num:.4f})")


def evaluate_bicubic(dataset, names, dataset_name):
    print(f"\n{dataset_name} Results (PSNR, FSIM):")

    psnr_bicubic_sum = 0
    fsim_bicubic_sum = 0
    num = 0

    for name, d in zip(names, dataset):
        hr = d['hr']
        lr = d['lr']

        # Upscale using bicubic
        sr_bicubic = upscale_bicubic(lr, hr)

        # Compute PSNR
        psnr_bicubic = psnr(sr_bicubic, hr)

        # Compute FSIM
        fsim_bicubic = fsim(sr_bicubic, hr)

        # Print for this image
        print(f"{name:<10} -- ({psnr_bicubic:.2f}, {fsim_bicubic:.4f})")

        # Update the sums
        psnr_bicubic_sum += psnr_bicubic
        fsim_bicubic_sum += fsim_bicubic

        num += 1

    print(
        f"MEAN       -- ({psnr_bicubic_sum/num:.2f}, {fsim_bicubic_sum/num:.4f})")


###############################################################################
# SCRIPT
###############################################################################

set5_names = ['woman', 'butterfly', 'baby', 'head', 'bird']
set14_names = ['baboon', 'man', 'comic', 'barbara', 'pepper', 'bridge', 'monarch',
               'lenna', 'flowers', 'face', 'coastguard', 'zebra', 'foreman', 'ppt3']
builder_kwargs = {
    'resize_method': 'bicubic',
    'scale': 4,
    'antialias': True
}

# Set5
set5_data = tfds.load(name="set5", split="test", builder_kwargs=builder_kwargs)
evaluate(set5_data, set5_names, "Set5")

# Set14
set14_data = tfds.load(name="set14", split="test",
                       builder_kwargs=builder_kwargs)
evaluate(set14_data, set14_names, "Set14")

# Evaluate only bicubic for different scales
for scale in [2, 3, 4, 5]:
    builder_kwargs = {
        'resize_method': 'bicubic',
        'scale': scale,
        'antialias': True
    }

    # Set5
    set5_data = tfds.load(name="set5", split="test",
                          builder_kwargs=builder_kwargs)
    evaluate_bicubic(set5_data, set5_names, f"Set5-x{scale}")

    # Set14
    set14_data = tfds.load(name="set14", split="test",
                           builder_kwargs=builder_kwargs)
    evaluate_bicubic(set14_data, set14_names, f"Set14-x{scale}")
