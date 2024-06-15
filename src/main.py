# # src/main.py
# University of Stavanger
# Authors: Armin Hajar Sabri Sabri, Aashish Karki
#
# Deep Learning for Unified Table and Caption Detection in Scientific Documents
# Delivered as part of Master Thesis, Department of Electrical Engineering and Computer Science
# June 2024

import os
import cv2
import json
import random
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import warnings
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from modified_trainer import ModifiedTrainer
from config import configuration, lr, bs, itr, freeze_at, optimizers, decay
from data_registration import register_datasets, ds_to_register_train, ds_to_register_val, ds_to_register_test, test_images
import mlflow


# Suppress warnings to keep logs cleaner
warnings.filterwarnings("ignore")
setup_logger()


# Configure MLFlow with user credentials (KEY SHARED FOR SIMPLICITY; PRECAUTIONS APPLIED)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'thesisuist'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cd6a7567bea2de310e1d52ac2f35483fca241e95'

# Nested loops to iterate over all combinations of training configurations
for idx, possibility in enumerate(ds_to_register_train):
    for learning_rate in lr:
        configuration['BASE_LR'] = str(learning_rate)
        for batchsize in bs:
            configuration['IMS_PER_BATCH'] = str(batchsize)
            for iterations in itr:
                configuration['MAX_ITER'] = str(iterations)
                for frz in freeze_at:
                    configuration['FREEZE_AT'] = str(frz)
                    for optimizer in optimizers:
                        configuration['OPTIMIZER'] = str(optimizer)
                        for dec in decay:
                            configuration['DECAY'] = str(dec)

                            # Dynamic configuration of run name based on parameters
                            configuration['RUN_NAME'] = "{}_{}_{}_{}_{}_FREEZE_{}_{}".format(
                                configuration['EXPERIMENT_NAME'],
                                configuration['BASE_LR'],
                                configuration['IMS_PER_BATCH'],
                                configuration['MAX_ITER'],
                                configuration['OPTIMIZER'],
                                configuration['FREEZE_AT'],
                                configuration['DECAY']
                            )

                            # Registering datasets
                            dataset_name_train = ds_to_register_train[idx]
                            dataset_name_val = ds_to_register_val[idx]
                            dataset_name_test = ds_to_register_test[idx]
                            train_metadata, train_dataset_dicts, val_metadata, val_dataset_dicts, test_metadata, test_dataset_dicts = register_datasets(dataset_name_train, dataset_name_val, dataset_name_test)

                            # Configuration for Detectron2
                            cfg = get_cfg()
                            cfg.MLFLOW = CfgNode()
                            cfg.MLFLOW.EXPERIMENT_NAME = configuration['EXPERIMENT_NAME']
                            cfg.MLFLOW.RUN_DESCRIPTION = configuration['RUN_DESCRIPTION']
                            cfg.MLFLOW.RUN_NAME = configuration['RUN_NAME']
                            cfg.MLFLOW.TRACKING_URI = configuration['TRACKING_URI']

                            # Load model configuration from file
                            cfg.merge_from_file(configuration['CONFIG_FILE'])
                            cfg.DATASETS.TRAIN = ("my_dataset_train",)
                            cfg.DATASETS.TEST = ("my_dataset_val",)
                            cfg.DATALOADER.NUM_WORKERS = configuration['NUM_WORKERS']
                            cfg.MODEL.WEIGHTS = configuration['MODEL_WEIGHTS']
                            cfg.SOLVER.IMS_PER_BATCH = batchsize
                            cfg.SOLVER.BASE_LR = learning_rate
                            cfg.SOLVER.MAX_ITER = iterations
                            cfg.SOLVER.OPTIMIZER = optimizer
                            cfg.SOLVER.WEIGHT_DECAY = dec

                            cfg.TEST.EVAL_PERIOD = configuration['TEST_EVAL_PERIOD']
                            cfg.SOLVER.STEPS = []
                            cfg.MODEL.ROI_HEADS.NUM_CLASSES = configuration['NUM_CLASSES']
                            cfg.SOLVER.CHECKPOINT_PERIOD = configuration['CHECKPOINT_PERIOD']
                            cfg.MODEL.BACKBONE.FREEZE_AT = frz

                            # Verifying the backbone freezing stage
                            if cfg.MODEL.BACKBONE.FREEZE_AT != frz:
                                raise TypeError("Freezing Point Mismatched!")

                            # Setting output directory for logs and results
                            cfg.OUTPUT_DIR = os.path.join('../outputs/', configuration['RUN_NAME'])
                            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                            setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "training-log.txt"))

                            # Training the model
                            trainer = ModifiedTrainer(cfg)
                            trainer.resume_or_load(resume=False)
                            trainer.train()

                            # Logging test evaluations
                            setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "test-set-evaluation", "evaluation-log.txt"))
                            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
                            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
                            predictor = DefaultPredictor(cfg)

                            test_evaluator = COCOEvaluator("my_dataset_test", output_dir=os.path.join(cfg.OUTPUT_DIR, "test-set-evaluation"))
                            test_loader = build_detection_test_loader(cfg, "my_dataset_test")
                            test_results = inference_on_dataset(predictor.model, test_loader, test_evaluator)

                            logging.info("Test Results on test set: %s", test_results)
                            for k, v in test_results["bbox"].items():
                                mlflow.log_metric("Test Set {}".format(k), v, step=0)

                            mlflow.log_artifacts(os.path.join(cfg.OUTPUT_DIR, "test-set-evaluation"), "test-set-results")
                            mlflow.log_text(str(test_results), "test-set-results/coco-metrics.txt")

                            # Processing and saving test images with predictions
                            input_images_directory = os.path.join(test_images, "images")
                            output_directory = os.path.join(cfg.OUTPUT_DIR, "test_images")
                            os.makedirs(output_directory, exist_ok=True)

                            for image_filename in os.listdir(input_images_directory):
                                image_path = os.path.join(input_images_directory, image_filename)
                                new_im = cv2.imread(image_path)
                                outputs = predictor(new_im)

                                v = Visualizer(new_im[:, :, ::-1], metadata=train_metadata)
                                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                                result_filename = "{}".format(image_filename) + "_result.png"

                                output_path = os.path.join(output_directory, result_filename)
                                cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

                            mlflow.log_artifacts(output_directory, "test_images")

                            # Visualizing and logging loss curves
                            metrics_df = pd.read_json(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), orient="records", lines=True)
                            mdf = metrics_df.sort_values("iteration")
                            fig, ax = plt.subplots()

                            mdf1 = mdf[~mdf["total_loss"].isna()]
                            ax.plot(mdf1["iteration"], mdf1["total_loss"], c="C0", label="train")
                            if "validation_loss" in mdf.columns:
                                mdf2 = mdf[~mdf["validation_loss"].isna()]
                                ax.plot(mdf2["iteration"], mdf2["validation_loss"], c="C1", label="validation")

                            plt.yticks(np.arange(0, max(mdf1["total_loss"]), 0.25))
                            ax.legend()
                            plt.grid()
                            ax.set_title("Loss curve")

                            mlflow.log_figure(fig, 'loss_curve.png')

                            # Cleaning up configuration for serialization and logging
                            def filter_serializable(cfg_node):
                                if isinstance(cfg_node, dict):
                                    return {k: filter_serializable(v) for k, v in cfg_node.items() if isinstance(v, (dict, list, tuple, str, int, float, bool, type(None)))}
                                elif hasattr(cfg_node, "keys"):
                                    return {k: filter_serializable(v) for k, v in cfg_node.items() if isinstance(v, (dict, list, tuple, str, int, float, bool, type(None)))}
                                else:
                                    return cfg_node

                            cfg = filter_serializable(cfg)
                            cfg.pop('MLFLOW', None)

                            with open(os.path.join(cfg.OUTPUT_DIR, 'after_training_config_file', 'finalconfig.yaml'), 'w') as file:
                                yaml.dump(cfg, file)

                            mlflow.log_artifacts(os.path.join(cfg.OUTPUT_DIR, 'after_training_config_file'), "after_train_config_file")

                            # Cleaning up datasets from Detectron2's catalog
                            if 'my_dataset_train' in DatasetCatalog.list():
                                MetadataCatalog.remove('my_dataset_train')
                                DatasetCatalog.remove('my_dataset_train')

                            if 'my_dataset_val' in DatasetCatalog.list():
                                MetadataCatalog.remove('my_dataset_val')
                                DatasetCatalog.remove('my_dataset_val')

                            # Ending the MLFlow run
                            mlflow.end_run()











