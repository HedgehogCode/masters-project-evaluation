# Master's Project Evaluation

This repository contains code to reproduce the results of my Master's Project.

## Getting Started

Use conda to create the required Python environment and activate it:
```
$ conda env create -f environment.yml
$ conda activate masters-proj-eval
```

## Scripts

* [`scripts/eval_image_reconstruction.py`](scripts/eval_image_reconstruction.py): Evaluate non-blind deblurring or single image super-resolution on one dataset for one model. See `--help` for parameters.
* [`scripts/eval_image_reconstruction_all.py`](scripts/eval_image_reconstruction_all.py): Evaluate nb deblurring and sisr on a set of datasets and a folder of models. See `--help` for parameters.
* [`scripts/find_noise_stddev_for_dmsp.py`](scripts/find_noise_stddev_for_dmsp.py): Evaluate DMSP on a range of different noise stddev values used for the prior. See `--help` for parameters.

NOTE: For now, use the [`evaluate.py` script in HedghogCode/denosing-gan](https://github.com/HedgehogCode/denoising-gan/blob/master/evaluate.py) for denoising evaluation. An adapted version of this script will be added to this repository soon.


## Notebooks

The folder [`notebooks/`](notebooks/) contains Jupyter notebooks.

* `visualize_*.ipynb`: Runs reconstruction on one image and saves the results such that they can be included in the report.
* `evaluate_*.ipynb`: Reads the results from the appropriate scripts and formats them in a table.


## Other Methods

The folder [`other_methods/`](other_methods/) contains code to reproduce the results reported in the report for other methods. See the `README.md` file in the appropriate folder for instructions.


## Model Converters

The folder [`model_converters/`](model-converters/) contains scripts for converting existing pretrained models to TensorFlow h5 models.
