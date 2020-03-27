import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# import openslide

import mrcnn.utils as utils
from mask_rcnn import InferenceConfig
import mrcnn.model as modellib

import numpy as np

from PIL import Image
import sys


Image.MAX_IMAGE_PIXELS = None

G_LOGGER = trt.Logger(trt.Logger.INFO)

trt.init_libnvinfer_plugins(G_LOGGER, '')

with open('nucleus.engine', 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

config = InferenceConfig()
def unmold_detections(detections, mrcnn_mask, original_image_shape,
                      image_shape, window):
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids] # [N, 28*28, class]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = utils.norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    return boxes, class_ids, scores, full_masks

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def inference(image, context):
    h_output_detection = np.empty((1, 4000, 6), dtype=np.float32)
    h_output_mask_sigmoid = np.empty((1, 4000, 28, 28, 2), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input_input_image = cuda.mem_alloc(image.nbytes)
    d_output_detection = cuda.mem_alloc(h_output_detection.nbytes)
    d_output_mask_sigmoid = cuda.mem_alloc(h_output_mask_sigmoid.nbytes)

    bindings = [int(d_input_input_image), int(d_output_detection), int(d_output_mask_sigmoid)]
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input_input_image, image, stream)
    # Run inference.
    context.execute(batch_size=1,bindings=bindings)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output_detection, d_output_detection, stream)
    cuda.memcpy_dtoh_async(h_output_mask_sigmoid, d_output_mask_sigmoid, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return the host output.
    return h_output_detection, h_output_mask_sigmoid

FILENAME = sys.argv[1]
output_path = sys.argv[2]

with engine.create_execution_context() as context:
    m = Image.open(FILENAME)
    image = np.array(m)
    molded_image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
    molded_image = mold_image(molded_image, config)
    molded_image = np.stack([molded_image])
    molded_image = np.ascontiguousarray(np.transpose(molded_image, (0, 3, 1, 2)), dtype=np.float32)
    detections, masks = inference(molded_image, context)
    final_rois, final_class_ids, final_scores, final_masks = \
        unmold_detections(detections[0], masks[0],
                          image.shape, molded_image.shape[::-1],
                          window)
    pr = final_masks.sum(axis=-1).astype(np.bool).astype(np.uint8)*255
    Image.fromarray(pr).save(os.path.join(output_path, '%s_label.png' % os.path.splitext(os.path.basename(FILENAME))[0]))
'''
slide = openslide.open_slide(FILENAME)
print('input dimensions = %dx%d' % slide.dimensions[:2])
dimensions = 1024, 1024

with engine.create_execution_context() as context:
    for i in range(0, slide.dimensions[0], dimensions[0]):
        for j in range(0, slide.dimensions[1], dimensions[1]):
            image = np.array(slide.read_region((i, j), 0, dimensions))[..., :3]
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            molded_image = mold_image(molded_image, config)
            molded_image = np.stack([molded_image])
            molded_image = np.ascontiguousarray(np.transpose(molded_image, (0, 3, 1, 2)), dtype=np.float32)
            # image_meta = modellib.compose_image_meta(
            #         0, image.shape, molded_image.shape, window, scale,
            #         np.zeros([config.NUM_CLASSES], dtype=np.int32))

            # image_shape = molded_image.shape

            # anchors = get_anchors(image_shape)

            detections, masks = inference(molded_image, context)
            final_rois, final_class_ids, final_scores, final_masks = \
                unmold_detections(detections[0], masks[0],
                                  molded_image.shape, molded_image.shape,
                                  window)
            pr = final_masks.sum(axis=-1).astype(np.bool).astype(np.uint8)*255
            filename_parameters = (os.path.splitext(os.path.basename(FILENAME))[0], i, j)
            Image.fromarray(pr).save(os.path.join(output_path, '%s_%d_%d_image_1.png' % filename_parameters))
            break
        break
'''