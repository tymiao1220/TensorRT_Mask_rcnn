import os
from pprint import pprint
import os.path
import sys

import numpy as np

import tensorflow as tf

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import openslide

import mrcnn.model as modellib

#import nucleus
import mask_rcnn

# Inference Configuration
config = mask_rcnn.InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir='.', config=config)

# Path to a specific weights file
# weights_path = "model-28649275/nucleus20190611T1436/mask_rcnn_nucleus_0050.h5"
weights_path = sys.argv[3]

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# Dataset directory
FILENAME = sys.argv[1]
output_path = sys.argv[2]


m = Image.open(FILENAME)
image = np.array(m)

result = model.detect([image])[0]
pr = result['masks'].sum(axis=-1).astype(np.bool).astype(np.uint8)*255
Image.fromarray(pr).save(os.path.join(output_path, '%s_label.png' % os.path.splitext(os.path.basename(FILENAME))[0]))

'''
#image = np.array(Image.open(FILENAME))[..., :3]
slide = openslide.open_slide(FILENAME)
print('input dimensions = %dx%d' % slide.dimensions[:2])
dimensions = 1024, 1024
for i in range(0, slide.dimensions[0], dimensions[0]):
    for j in range(0, slide.dimensions[1], dimensions[1]):
        image = np.array(slide.read_region((i, j), 0, dimensions))[..., :3]
        print(i, j, image.shape)
        result = model.detect([image])[0]
        pr = result['masks'].sum(axis=-1).astype(np.bool).astype(np.uint8)*255
        filename_parameters = (os.path.splitext(os.path.basename(FILENAME))[0], i, j)
        Image.fromarray(image).save(os.path.join(output_path, '%s_%d_%d_image_1.png' % filename_parameters))
        Image.fromarray(pr).save(os.path.join(output_path, '%s_%d_%d_label_1.png' % filename_parameters))
'''