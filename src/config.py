# src/config.py
# University of Stavanger
# Authors: Armin Hajar Sabri Sabri, Aashish Karki
#
# Deep Learning for Unified Table and Caption Detection in Scientific Documents
# Delivered as part of Master Thesis, Department of Electrical Engineering and Computer Science
# June 2024

import os

# MLFlow Configuration for user authentication
os.environ['MLFLOW_TRACKING_USERNAME'] = 'thesisuist'  # Username for MLFlow tracking server
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'API_KEY'  # API Key for MLFlow tracking server

# Dictionary to hold configuration parameters for model training and MLFlow tracking
# TBC in comments mentions the things that need to be changed. Other parameters are to be kept still since it is just a placeholder
# Variables lr, bs, itr, freeze_at can be changed to reflect the parameters
configuration = {
    'TRACKING_URI': "https://dagshub.com/thesisuist/reversed_final.mlflow",  # URI for the MLFlow tracking server (TBC)
    'EXPERIMENT_NAME': "EXPERIMENTNAME",  # Name of the experiment to log in MLFlow (TBC)
    'RUN_DESCRIPTION': "Test Run",  # Description of the run for logging (TBC)
    'RUN_NAME': "Phase X - ModelName",  # Name of the run (TBC)
    'MODEL_WEIGHTS': "../pre-trained/mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth",  # Path to pre-trained model weights (TBC)
    'CONFIG_FILE': "../pre-trained/mask_rcnn_X_101_32x8d_FPN_3x/config.yaml",  # Path to the configuration file for the model (TBC)
    'IMS_PER_BATCH': 2,  # Images per batch for training
    'BASE_LR': 0.00025,  # Base learning rate for training
    'MAX_ITER': 1000,  # Maximum number of iterations for training
    'NUM_CLASSES': 3,  # Number of classes in the dataset
    'CHECKPOINT_PERIOD': 500,  # Period for saving checkpoints
    'NUM_WORKERS': 4,  # Number of worker threads for data loading
    'TEST_EVAL_PERIOD': 100,  # Period for performing evaluations on the test set
    'RUN_ID': '',  # Placeholder for run ID if needed for resuming logs in MLFlow
    'FREEZE_AT': '',  # Placeholder for model layer freeze configuration
    'OPTIMIZER': '',  # Placeholder for optimizer configuration
    'DECAY': '',  # Placeholder for decay configuration
}


# Accessing specific configuration parameters to use in scripts
# Important parameters, should not be removed.
# Is in array and can run for multiple configurations
experiment = configuration['EXPERIMENT_NAME']  # Experiment name accessed from the configuration
lr = [0.00025]  # List containing learning rates value for experimental modifications, modifies configuration['BASE_LR']
bs = [2]  # List containing batch size values for experimental modifications, modifies configuration['IMS_PER_BATCH']
itr = [100]  # List containing iteration counts for experimental runs, modifies configuration['MAX_ITER']
freeze_at = [0]  # List containing layer stage freeze settings for model configuration, modifies configuration['FREEZE_AT']
optimizers = ['SGD']
decay = [0.0001]