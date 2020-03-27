from keras.models import model_from_json, Model
from keras import backend as K
from keras.layers import Input, Lambda
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from mrcnn.model import *
import mrcnn.model as modellib
from mrcnn.config import Config
import sys
import os
ROOT_DIR = os.path.abspath("./")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
import argparse
import os
import uff

def parse_command_line_arguments(args=None):
    parser = argparse.ArgumentParser(prog='keras_to_trt', description='Convert trained keras .hdf5 model to trt .uff')

    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        default=None,
        required=True,
        help="The checkpoint weights file of keras model."
    )
    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        default=None,
        required=True,
        help="The path to output .uff file."
    )

    parser.add_argument(
        '-l',
        '--list-nodes',
        action='store_true',
        help="show list of nodes contained in converted pb"
    )

    parser.add_argument(
        '-p',
        '--preprocessor',
        type=str,
        default=False,
        help="The preprocess function for converting tf node to trt plugin"
    )

    return parser.parse_args(args)

# CPU_COUNT = 4
# IMAGE_COUNT = 63

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'nucleus'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # NAME = 'nucleus'

    GPU_COUNT = 1

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4  # batch size

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (IMAGE_COUNT - len(VALIDATION_IMAGE_IDS) - len(TEST_IMAGE_IDS)) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, len(VALIDATION_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = 'resnet50'

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = 'crop'
    IMAGE_MIN_DIM = 256  # 512
    ##IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 20000 # 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    #MEAN_PIXEL = np.array([-0.65858824, -0.68972549, -0.62180392])
    MEAN_PIXEL = np.array([188.58, 154.34, 182.38])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 4000 # 400

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

def main(args=None):

    K.set_image_data_format('channels_first')
    K.set_learning_phase(0)

    args = parse_command_line_arguments(args)

    model_weights_path = args.weights
    output_file_path = args.output_file
    list_nodes = args.list_nodes

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=LOG_DIR, config=config).keras_model

    model.load_weights(model_weights_path, by_name=True)


    model_A = Model(inputs=model.input, outputs=model.get_layer('mrcnn_mask').output)
    model_A.summary()

    output_nodes = ['mrcnn_detection', "mrcnn_mask/Sigmoid"]
    convert_model(model_A, output_file_path, output_nodes, preprocessor=args.preprocessor,
                  text=True, list_nodes=list_nodes)


def convert_model(inference_model, output_path, output_nodes=[], preprocessor=None, text=False,
                  list_nodes=False):
    # convert the keras model to pb
    orig_output_node_names = [node.op.name for node in inference_model.outputs]
    print("The output names of tensorflow graph nodes: {}".format(str(orig_output_node_names)))

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        orig_output_node_names)

    temp_pb_path = "./nucleus_temp.pb"
    graph_io.write_graph(constant_graph, os.path.dirname(temp_pb_path), os.path.basename(temp_pb_path),
                         as_text=False)

    predefined_output_nodes = output_nodes
    if predefined_output_nodes != []:
        trt_output_nodes = predefined_output_nodes
    else:
        trt_output_nodes = orig_output_node_names

    # convert .pb to .uff
    uff.from_tensorflow_frozen_model(
        temp_pb_path,
        output_nodes=trt_output_nodes,
        preprocessor=preprocessor,
        text=text,
        list_nodes=list_nodes,
        output_filename=output_path,
        debug_mode = False
    )
    os.remove(temp_pb_path)


if __name__ == "__main__":
    main()
