"""
This package implements PhotCalib to calibrate CaHK photometric data from the Prisitne survey.

Content
-------

The package mainly contains:
  calib                     return the calibration model 
  applycalib                return the calibrated values 
"""
from .deform_norm import deform
from .training import TrainingModule
from .make_plots import make_diagnostic_plots
from .apply_calib import generate_newcat
from .set_argparse import argparse_train_model, argparse_apply_model


