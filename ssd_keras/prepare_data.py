from data_generator.object_detection_2d_data_generator import DataGenerator

dataset = DataGenerator(load_images_into_memory=False,
                              hdf5_dataset_path=None)
dataset.parse_csv(images_dir="D:/Documents/3dsMax/renderoutput",
                  labels_filename="D:/Documents/3dsMax/renderoutput/annotations.ssd.csv",
                  input_format=["image_name", "xmin", "ymin", "xmax", "ymax", "class_id"],
                  include_classes="all")
dataset.create_hdf5_dataset(file_path="D:/Documents/3dsMax/renderoutput/data.all.h5")