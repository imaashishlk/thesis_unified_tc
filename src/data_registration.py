# src/data_registration.py
# University of Stavanger
# Authors: Armin Hajar Sabri Sabri, Aashish Karki
#
# Deep Learning for Unified Table and Caption Detection in Scientific Documents
# Delivered as part of Master Thesis, Department of Electrical Engineering and Computer Science
# June 2024

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# File paths to JSON annotations for training, validation, and test datasets.
ds_to_register_train = ['../data/Phase III/whole_dataset/mask/train_whole_mask.json']
ds_to_register_val = ['../data/Phase III/whole_dataset/mask/val_whole_mask.json']
ds_to_register_test = ['../data/Phase III/whole_dataset/mask/test_whole_mask.json']

# Directories containing the corresponding image files.
train_images = "../data/Phase III/"
val_images = "../data/Phase III/"
test_images = "../data/Phase III/"

def register_datasets(train_json, val_json, test_json):
    """
    Registers datasets for training, validation, and testing with Detectron2.

    Args:
        train_json (str): Path to the training dataset JSON.
        val_json (str): Path to the validation dataset JSON.
        test_json (str): Path to the testing dataset JSON.

    Returns:
        tuple: Returns metadata and dataset dictionaries for training, validation, and test datasets.
    """
    
    # Check if datasets are already registered; if so, unregister them before re-registering.
    if 'my_dataset_train' in DatasetCatalog.list():
        MetadataCatalog.remove('my_dataset_train')
        DatasetCatalog.remove('my_dataset_train')

    if 'my_dataset_val' in DatasetCatalog.list():
        MetadataCatalog.remove('my_dataset_val')
        DatasetCatalog.remove('my_dataset_val')

    if 'my_dataset_test' in DatasetCatalog.list():
        MetadataCatalog.remove('my_dataset_test')
        DatasetCatalog.remove('my_dataset_test')

    # Register datasets using the COCO format.
    register_coco_instances("my_dataset_train", {}, train_json, train_images)
    register_coco_instances("my_dataset_val", {}, val_json, val_images)
    register_coco_instances("my_dataset_test", {}, test_json, test_images)

    # Retrieve metadata and dataset dictionaries for all datasets.
    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")
    test_metadata = MetadataCatalog.get("my_dataset_test")
    test_dataset_dicts = DatasetCatalog.get("my_dataset_test")

    return train_metadata, train_dataset_dicts, val_metadata, val_dataset_dicts, test_metadata, test_dataset_dicts