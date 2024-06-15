# Deep Learning for Unified Table and Caption Detection in Scientific Documents

<p align="justify">
This repository contains the codes and algorithms developed for the Master Thesis entitled **Deep Learning for Unified Table and Caption Detection in Scientific Documents**. The thesis was presented as a requirement for Master's Thesis at the University of Stavanger, Spring Semester, 2024.
</p>
<p align="justify">
The repository implements fine-tuning of the pre-trained models in developing deep learning strategy for the unified table and caption detection.
</p>

## Project Directory
```plaintext
project_name/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_registration.py
│   ├── mlflow_loss_eval.py
│   ├── modified_trainer.py
│   ├── main.py
│   ├── requirements.txt
│   └── training.sh
│
├── scripts/
│   └── data_fetch.py
│
├── example_images/
│   ├── example-1.jpg
│   ├── example-2.jpg
│   ├── example-3.jpg
│   └── example-4.jpg
│
└── README.md


```


## Overview
 <p align="justify">
        This project is designed to utilize Detectron2 for object detection and segmentation tasks. It incorporates MLFlow for experiment tracking and evaluation. The main components of the project are structured in the src directory, with separate files for configuration, dataset registration, custom trainers, and the main execution script. In addition, it is important to install Detectron2. The guide is mentioned <a href="https://detectron2.readthedocs.io/en/latest/tutorials/install.html">here</a>.
    </p>

## Running the Code
<p align="justify">
Before running the code, it is important to run the file <code>data_fetch.py</code> under the <code>scripts</code> folder. This file would create the necessary folder structures and download the dataset with the pre-trained model configuration file and the weights respectively.
</p>

To run the project, navigate to the src directory and execute the main script:

```plaintext

python src/main.py

```

To run with SLURM, which is often done in this project, execute the training script:

```plaintext

sbatch src/training.sh

```
Refer to the _training.sh_ file for detailed explanations.

Ensure you have the necessary datasets and configurations in place as downloaded by the _data_fetch.py_ file under _scripts_.

## Configuration
<p align="justify">
The configuration details are managed in <code>src/config.py</code>. Adjust the configuration parameters such as <code>EXPERIMENT_NAME</code>, <code>MODEL_WEIGHTS</code>, <code>BASE_LR</code>, <code>MAX_ITER</code>, etc., according to your requirements.
</p>


## Dataset Registration
<p align="justify">
Datasets are registered using COCO format. The dataset registration logic is handled in <code>src/data_registration.py</code>. Ensure your datasets are in the correct format and paths are correctly specified in the configuration. The defaults in the file should give an overview of running.
</p>


## Custom Trainer and MLFlow Integration
<p align="justify">
The custom trainer <code>ModifiedTrainer</code> and the MLFlow evaluation hook <code>MLFlowLossEval</code> are defined in <code>src/modified_trainer.py</code> and <code>src/mlflow_loss_eval.py</code> respectively. These integrate with Detectron2’s training loop and log metrics, parameters, and artifacts to MLFlow.
</p>


## Jupyter Notebooks
<p align="justify">
The Jupyter Notebooks under <code>notebooks</code> show examples of plots and running the fine-tuned models. Specifically, the <code>layout_detected_results</code> notebook demonstrates the use of pre-trained and fine-tuned models to run it with sample images to create the bounding boxes with the optimal model. The pre-trained files are already downloaded with <code>data_fetch.py</code> while the fine-tuned models will be automatically fetched by the notebook while running the code. This is handled by the DagsHub API Key and MLFLow Artifacts to download and render the outputs.
</p>


## Acknowledgements
<p align="justify">
We extend our heartfelt gratitude to our supervisor, Antorweep Chakravorty at UiS, for his invaluable guidance, support, and mentorship throughout this research. His expertise, dedication, and commitment to excellence have been instrumental in shaping the direction and success of our thesis.
</p>

## Authors
<p align="justify">
Armin Hajar Sabri Sabri <br />
Aashish Karki
</p>


### `requirements.txt`
```plaintext
layoutparser==0.3.4
matplotlib==3.7.1
mlflow==2.13.2
numpy==1.23.5
opencv_python_headless==4.9.0.80
pandas==1.5.3
pycocotools==2.0.7
PyYAML==6.0.1
torch==2.3.1
git+https://github.com/facebookresearch/detectron2
```
