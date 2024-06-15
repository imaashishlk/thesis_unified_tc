# src/mlflow_loss_eval.py
# University of Stavanger
# Authors: Armin Hajar Sabri Sabri, Aashish Karki
#
# Deep Learning for Unified Table and Caption Detection in Scientific Documents
# Delivered as part of Master Thesis, Department of Electrical Engineering and Computer Science
# June 2024

import os
import torch
import time
import datetime
import logging
import numpy as np
import mlflow
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import inference_context
from detectron2.data import build_detection_test_loader
import detectron2.utils.comm as comm
from config import configuration, lr, bs, itr, freeze_at


class MLFlowLossEval(HookBase):
    """
    Custom hook to integrate MLFlow logging with Detectron2 training lifecycle.

    Some of the functions are referenced from 
    https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-plottogether-py 

    Attributes:
        eval_period (int): Frequency of evaluations during training.
        model (Module): The neural network model.
        data_loader (DataLoader): DataLoader for evaluation.
        cfg (CfgNode): Configuration object with training parameters.
        configuration (dict): Dictionary containing configuration for MLFlow.
    """
    def __init__(self, eval_period, model, data_loader, cfg, configuration):
        super().__init__()
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.cfg = cfg.clone()

        # Set up MLFlow tracking with the specified configuration.
        mlflow.set_tracking_uri(cfg.MLFLOW.TRACKING_URI)
        mlflow.set_experiment(cfg.MLFLOW.EXPERIMENT_NAME)
        mlflow.start_run(run_name=cfg.MLFLOW.RUN_NAME)
        mlflow.set_tag("mlflow.note.content", cfg.MLFLOW.RUN_DESCRIPTION)

    def get_run_id(self):
        """
        Retrieves the current MLFlow run ID.

        Returns:
            str: The MLFlow run ID.
        """
        active_run = mlflow.active_run()
        return active_run.info.run_id

    def close_run(self):
        """
        Safely terminates the current MLFlow run.
        """
        try:
            mlflow.end_run()
            print("MLFlow Run Terminated!")
        except:
            pass

    def before_train(self):
        """
        Hook method called before the training process begins.
        Logs all configuration parameters to MLFlow.
        """
        with torch.no_grad():
            configuration['RUN_ID'] = self.get_run_id()
            for item, value in configuration.items():
                mlflow.log_param(item, value)

    def after_train(self):
        """
        Hook method called after the training process ends.
        Logs the final set of metrics and outputs to MLFlow.
        """
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1])
            
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
                
            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)

    def _do_loss_eval(self):
        """
        Evaluates the model on the data loader and logs the average loss.
        Uses warmup steps to stabilize loss calculations.

        Returns:
            list: A list of loss values for each batch.
        """
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        """
        Computes loss for a batch of data using the model.

        Args:
            data: Batch of data from the DataLoader.

        Returns:
            float: Aggregated loss for the batch.
        """
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        """
        Hook method called after each training step.
        Logs metrics to MLFlow periodically based on the training iteration.
        """
        with torch.no_grad():
            if self.trainer.iter % 100 == 0:
                latest_metrics = self.trainer.storage.latest()
                for k, v in latest_metrics.items():
                    mlflow.log_metric(key=k, value=v[0], step=v[1])
                
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)