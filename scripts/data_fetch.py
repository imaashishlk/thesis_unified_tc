# src/data_fetch.py
# University of Stavanger
# Authors: Armin Hajar Sabri Sabri, Aashish Karki
#
# Deep Learning for Unified Table and Caption Detection in Scientific Documents
# Delivered as part of Master Thesis, Department of Electrical Engineering and Computer Science
# June 2024

import gdown
import zipfile
import os

os.makedirs(os.path.join('../', 'data'), exist_ok=True)
os.makedirs(os.path.join('../', 'pre-trained'), exist_ok=True)
os.makedirs(os.path.join('../', 'outputs'), exist_ok=True)
os.makedirs(os.path.join('../notebooks', 'artifacts'), exist_ok=True)
os.makedirs(os.path.join('../notebooks', 'fine-tuned-model'), exist_ok=True)

# Defined the file ID and output file name; not subject to change
# Made Public File Keys
dataset = '1fKZ14vpVmGnlK5tEaqANXFlOGM9AZJ2v'
configuration = '1hZNN7Jl5fbnoHaj3hkM3FcaJSz4Zvuhz'


datasetfile = 'dataset.zip'
configurationfile = 'configurations.zip'

# Download the files from Google Drive
gdown.download(id=dataset, output=datasetfile)
gdown.download(id=configuration, output=configurationfile)

# Folder path for extraction
extracted_folder_path_ds = '../data'
extracted_folder_path_cfgs = '../pre-trained'

# Create a directory to extract files if it doesn't exist
os.makedirs(extracted_folder_path_ds, exist_ok=True)
os.makedirs(extracted_folder_path_cfgs, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(datasetfile, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path_ds)

with zipfile.ZipFile(configurationfile, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path_cfgs)

# Delete the zip file
os.remove(datasetfile)
os.remove(configurationfile)

# List the extracted files to confirm extraction
extracted_files_ds = os.listdir(extracted_folder_path_ds)
extracted_files_cfgs = os.listdir(extracted_folder_path_cfgs)

print("Extracted files:", extracted_files_ds)
print("Extracted files:", extracted_files_cfgs)