import numpy as np
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from keras import backend as K
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras.models import load_model
from keras.optimizers import Adam
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

val_dataset = DataGenerator(load_images_into_memory=False,
                            hdf5_dataset_path=None)
val_dataset.parse_csv(images_dir="D:/Documents/Villeroy & Boch - Subway 2.0", labels_filename="D:/Documents/Villeroy & Boch - Subway 2.0/annotations.ssd.csv", input_format=["image_name", "xmin", "ymin", "xmax", "ymax", "class_id"])


# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'D:/Documents/3dsMax/renderoutput/3dsmax3/snapshots/ssd300_3dsmax3_epoch-04_loss-5.5616_val_loss-5.6297.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=300, width=300)
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)


# 3: Make predictions.

y_pred = model.predict(batch_images)

# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=True,
                                   img_height=300,
                                   img_width=300)

# 5: Convert the predictions for the original image.

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[0])