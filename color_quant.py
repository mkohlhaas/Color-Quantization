#!/usr/bin/env python


import argparse
from os import path

import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color Quantization")
    _ = parser.add_argument("file_name")
    _ = parser.add_argument("num_colors", type=int)
    args = parser.parse_args()
    (cwd, extension) = path.splitext(args.file_name)
    output_filename = cwd + "_quant" + extension

    image_as_array = mpimg.imread(args.file_name)
    (h, w, c) = image_as_array.shape
    image_as_array2d = image_as_array.reshape(h * w, c)
    model = KMeans(n_clusters=args.num_colors)
    color_palette = model.fit_predict(image_as_array2d)
    rgb_codes = model.cluster_centers_.round(0).astype("uint8")
    quantized_image = np.reshape(rgb_codes[color_palette], (h, w, c))
    mpimg.imsave(output_filename, quantized_image)
