# src/modified_trainer.py
# University of Stavanger
# Authors: Armin Hajar Sabri Sabri, Aashish Karki
#
# Deep Learning for Unified Table and Caption Detection in Scientific Documents
# Delivered as part of Master Thesis, Department of Electrical Engineering and Computer Science
# June 2024

import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, DatasetMapper
from mlflow_loss_eval import MLFlowLossEval
from config import configuration

class ModifiedTrainer(DefaultTrainer):
    """
    A customized trainer that extends the DefaultTrainer from Detectron2 to integrate
    MLFlow for tracking evaluations and losses during the training process.
    """
    hooks = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Creates a COCOEvaluator for evaluating the model on a COCO format dataset.

        Args:
            cfg: Configuration object from Detectron2 containing model and solver settings.
            dataset_name (str): Name of the dataset to be evaluated.
            output_folder (str, optional): Directory where evaluation outputs will be saved.
                                          If None, defaults to 'validation-set-evaluation' within OUTPUT_DIR.

        Returns:
            COCOEvaluator: An evaluator for COCO datasets.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "validation-set-evaluation")
            os.makedirs(output_folder, exist_ok=True)

        return COCOEvaluator(dataset_name, distributed=False, output_dir=output_folder)

    def build_hooks(self):
        """
        Builds and returns a list of hooks to be used in the training loop. This implementation
        adds a custom MLFlowLossEval hook to the existing hooks from the superclass to integrate
        loss evaluation and MLFlow logging during training.

        Returns:
            list: A list of hooks to be used during the training process.
        """
        self.hooks = super().build_hooks()
        self.hooks.insert(-1, MLFlowLossEval(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            ),
            self.cfg,
            configuration
        ))
        return self.hooks