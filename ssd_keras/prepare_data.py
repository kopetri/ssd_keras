import os
import sys
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "ssd_keras"

from .data_generator.object_detection_2d_data_generator import DataGenerator

dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
dataset.parse_csv(images_dir="D:/Documents/3dsMax/renderoutput/domain_randomization/test",
                  labels_filename="D:/Documents/3dsMax/renderoutput/domain_randomization/test/annotations.test.csv",
                  input_format=["image_name", "xmin", "ymin", "xmax", "ymax", "class_id"],
                  include_classes="all"
                  )
dataset.create_hdf5_dataset(file_path="D:/Documents/3dsMax/renderoutput/domain_randomization/test/data.test.h5")