import os
import sys
import argparse
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


def check_args(parsed_args):
    if parsed_args.weights_path is None:
        raise ValueError(
            "weight path mising")
    if parsed_args.train_data is None:
        raise ValueError("Missing train data")
    if parsed_args.val_data is None:
        raise ValueError("Missing validation data")
    if parsed_args.output_dir is None:
        raise ValueError("Missing output directory")


def main():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--img_width', default=300, type=int)
    parser.add_argument('--img_height', default=300, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--output_dir', type=str)

    args = sys.argv[1:]
    parsed_args = parser.parse_args(args)
    check_args(parsed_args)

    img_height = parsed_args.img_height  # Height of the model input images
    img_width = parsed_args.img_width  # Width of the model input images
    img_channels = 3  # Number of color channels of the model input images
    mean_color = [123, 117,
                  104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    swap_channels = [2, 1,
                     0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    n_classes = 6  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                     1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
                   1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    scales = scales_pascal
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100,
             300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
    normalize_coords = True

    # 1: Build the Keras model.

    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)

    # 2: Load some weights into the model.

    model.load_weights(parsed_args.weights_path, by_name=True)

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    train_dataset = DataGenerator(load_images_into_memory=False,
                                  hdf5_dataset_path=parsed_args.train_data)
    val_dataset = DataGenerator(load_images_into_memory=False,
                                hdf5_dataset_path=parsed_args.val_data)
    """
    train_dataset.parse_csv(images_dir="D:/Documents/3dsMax/renderoutput/3dsmax3",
                            labels_filename="D:/Documents/3dsMax/renderoutput/3dsmax3/annotations.ssd.csv",
                            input_format=["image_name", "xmin", "ymin", "xmax", "ymax", "class_id"], include_classes="all")
    val_dataset.parse_csv(images_dir="D:/Documents/Villeroy & Boch - Subway 2.0",
                          labels_filename="D:/Documents/Villeroy & Boch - Subway 2.0/annotations.ssd.csv",
                          input_format=["image_name", "xmin", "ymin", "xmax", "ymax", "class_id"], include_classes="all")
    """
    # 3: Set the batch size.

    batch_size = parsed_args.batch_size  # Change the batch size if you like, or if you run into GPU memory issues.

    # 4: Set the image transformations for pre-processing and data augmentation options.

    # For the training generator:
    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)

    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=[ssd_data_augmentation],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size = val_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    outputdir = parsed_args.output_dir

    # Define model callbacks.

    # TODO: Set the filepath under which you want to save the model.
    if not os.path.exists(os.path.join(outputdir, "snapshots")):
        os.mkdir(os.path.join(outputdir, "snapshots"))
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(outputdir, "snapshots",
                                                             'ssd300_3dsmax3_epoch-{epoch:02d}_loss-{loss:.4f}.h5'),
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)
    # model_checkpoint.best =

    csv_logger = CSVLogger(filename=os.path.join(outputdir, "logs", 'ssd300_coco_training_log.csv'),
                           separator=',',
                           append=True)

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                    verbose=1)
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(outputdir, "logs"),
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )

    terminate_on_nan = TerminateOnNaN()

    callbacks = [model_checkpoint,
                 csv_logger,
                 learning_rate_scheduler,
                 tensorboard_callback,
                 terminate_on_nan]

    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
    initial_epoch = 0
    final_epoch = 500
    steps_per_epoch = 10000

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(val_dataset_size / batch_size),
                                  initial_epoch=initial_epoch)


if __name__ == '__main__':
    main()
