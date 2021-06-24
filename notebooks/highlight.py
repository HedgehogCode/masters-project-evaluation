import os
import re

import imageio
import matplotlib.pyplot as plt

PSNR_FILE_REGEX = re.compile('PSNR: (\d+\.\d+), FSIM: (0\.\d+)')


def highlight(image, region_center, region_extent, rel_width,
              place_top=False, place_right=True, place_border=0.01,
              label_border = 0.05, fontsize=18, label="(PSNR, FSIM)"):
    # Size of the figure
    figsize = (image.shape[1]/100, image.shape[0]/100 * (1 + label_border))

    fig = plt.figure(figsize=figsize)

    # Axis1 is the whole image
    ax1 = plt.axes([0, label_border, 1, 1 - label_border], frameon=False)
    ax1.imshow(image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(label, fontsize=fontsize)

    # Axis2 is the region
    rel_height = figsize[0] * rel_width / figsize[1]
    pos_x = 1 - rel_width - place_border if place_right else place_border
    pos_y = 1 - rel_height - place_border if place_top else (place_border + label_border)
    ax2 = plt.axes([pos_x, pos_y, rel_width, rel_height])
    region = image[(region_center[0] - region_extent):(region_center[0] + region_extent),
                   (region_center[1] - region_extent):(region_center[1] + region_extent), :]
    ax2.imshow(region)
    ax2.set_xticks([])
    ax2.set_yticks([])

    return fig


def highlight_and_save(folder, *args, **kwargs):
    image_paths = [f for f in os.listdir(folder) if f.endswith('.png') and not f.endswith('_hgl.png')]

    for i_path in image_paths:
        image = imageio.imread(os.path.join(folder, i_path))
        if (i_path == 'gt.png'):
            label = "(PSNR, FSIM)"
        else:
            value_path = i_path[:-4] + '.txt'
            with open(os.path.join(folder, value_path), 'r') as f:
                psnr_and_fsim_str = f.readline()
            m = PSNR_FILE_REGEX.match(psnr_and_fsim_str)
            psnr = float(m[1])
            fsim = float(m[2])
            label = f"({psnr:.2f}dB, {fsim:.4f})"

        fig = highlight(image, *args, label=label, **kwargs)
        fig.savefig(os.path.join(folder,  i_path[:-4] + '_hgl.png'), dpi=100)