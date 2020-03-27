import os
import xml.dom.minidom

import numpy as np

import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import image

import mrcnn.config
import mrcnn.model
import mrcnn.utils

from imgaug import augmenters as iaa

TEST_IMAGE_IDS = [
    'TCGA-21-5786-01Z-00-DX1',
    'TCGA-A7-A13E-01Z-00-DX1',
    'TCGA-AY-A8YK-01A-01-TS1',
    'TCGA-B0-5698-01Z-00-DX1',
    'TCGA-G9-6356-01Z-00-DX1',
    '141549_367',
    '162438_25',
    #'162438_58',  # move to validation
    '498959_03',
    #'572123_33',  # move to validation
    '588626_0',
]

VALIDATION_IMAGE_IDS = [
#    'TCGA-E2-A1B5-01Z-00-DX1',
#    'TCGA-DK-A2I6-01A-01-TS1',
#    'TCGA-A7-A13F-01Z-00-DX1',
#    'TCGA-A7-A13F-01Z-00-DX1',
#    'TCGA-21-5786-01Z-00-DX1',
#    'TCGA-AR-A1AK-01Z-00-DX1',
#    'TCGA-E2-A14V-01Z-00-DX1',
#    '581910_8',
#    '581910_4',
#    '498959_11',
#    '498959_01',
#    '160120_152',
    'TCGA-18-5592-01Z-00-DX1',
    'TCGA-38-6178-01Z-00-DX1',
    'TCGA-AR-A1AK-01Z-00-DX1',
    'TCGA-AR-A1AS-01Z-00-DX1',
    'TCGA-E2-A1B5-01Z-00-DX1',
    'TCGA-G2-A2EK-01A-02-TSB',
    'TCGA-KB-A93J-01A-01-TS1',
    '162438_58',  # from test
    '498959_03',
    '572123_33',  # from test
    '498959_04',
    '588626_3',
]

BNS_PATH = os.path.join(os.environ.get('TMPDIR'), 'datasets', 'ToAnnotate')
MONUSEG_PATH = os.path.join(os.environ.get('TMPDIR'), 'datasets', 'MoNuSeg Training Data')

MODEL_DIR = os.path.join(os.environ.get('TMPDIR'), 'model')

CPU_COUNT = 4
IMAGE_COUNT = 63


def cpu_count():
    return CPU_COUNT

mrcnn.model.multiprocessing.cpu_count = cpu_count


class Config(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = 'nucleus'

    GPU_COUNT = 1

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4  # batch size

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (IMAGE_COUNT - len(VALIDATION_IMAGE_IDS) - len(TEST_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VALIDATION_IMAGE_IDS) // IMAGES_PER_GPU)

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
    IMAGE_MAX_DIM = 256  # 512
    #IMAGE_MIN_SCALE = 2.0

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

class InferenceConfig(Config):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


class MaskRCNN(mrcnn.model.MaskRCNN):
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
            no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = mrcnn.model.data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = mrcnn.model.data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            mrcnn.model.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            mrcnn.model.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True,
                                            period=10),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        mrcnn.model.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        mrcnn.model.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if mrcnn.model.os.name is 'nt':
            workers = 0
        else:
            workers = mrcnn.model.multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            #workers=workers,
            #use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)


def train(model, config, dataset_train, dataset_val, skip_heads=False):
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    if not skip_heads:
        print('Train network heads')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    #epochs=20,
                    epochs=40,
                    #augmentation=augmentation,
                    layers='heads')

    print('Train all layers')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                #epochs=40,
                epochs=80,
                #augmentation=augmentation,
                layers='all')

if __name__ == '__main__':
    import argparse

    import dataset


    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', help='Path to weights .h5 file')
    parser.add_argument('--skip_heads', action='store_true')
    args = parser.parse_args()


    config = Config()
    config.display()

    dataset_train = dataset.Dataset()
    dataset_train.load_bns(BNS_PATH)
    dataset_train.load_monuseg(MONUSEG_PATH)
    image_info = [i for i in dataset_train.image_info if i['id'] not in VALIDATION_IMAGE_IDS and i['id'] not in TEST_IMAGE_IDS]
    dataset_train.image_info = image_info
    dataset_train.prepare()
    dataset_train.class_names = ['BG', 'nucleus']
    dataset_train.num_classes = 2
    dataset_train.class_from_source_map['BNS.1'] = 1
    dataset_train.class_from_source_map['MoNuSeg.1'] = 1
    dataset_train.source_class_ids['BNS'] = [0, 1]
    dataset_train.source_class_ids['MoNuSeg'] = [0, 1]
    dataset_train.cache()

    dataset_val = dataset.Dataset()
    dataset_val.load_bns(BNS_PATH)
    dataset_val.load_monuseg(MONUSEG_PATH)
    image_info = [i for i in dataset_val.image_info if i['id'] in VALIDATION_IMAGE_IDS]
    dataset_val.image_info = image_info
    dataset_val.prepare()
    dataset_val.class_names = ['BG', 'nucleus']
    dataset_val.num_classes = 2
    dataset_val.class_from_source_map['BNS.1'] = 1
    dataset_val.class_from_source_map['MoNuSeg.1'] = 1
    dataset_val.source_class_ids['BNS'] = [0, 1]
    dataset_val.source_class_ids['MoNuSeg'] = [0, 1]
    dataset_val.cache()

    #with tf.device('/gpu:0'):
    model = MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    if args.weights:
        weights = args.weights
    else:
        weights = model.get_imagenet_weights()
    model.load_weights(weights, by_name=True)

    train(model, config, dataset_train, dataset_val, skip_heads=args.skip_heads)
