PhotCalib
-----------

Calibration tool for Prisitine CaHK data. For more details see Martin, Starkenburg & Yuan 2023



Installation
----------------

Before installation, make sure you have pytorch or install essential dependencies :

.. code::

  pip install -r requirements.txt



* Installation from PyPI

.. code::

   pip install photcalib

* Manual installation

download the repository and run the setup

.. code::

   python setup.py install      

Getting started 
----------------

.. code::

   cd examples
 

To Calibrate CaHK data using the trained run model

.. code::

   python calib_raw.py 25Am02 25Am02  -D cpu

The first run name represents the run to train the calibration model.

The second run name represents the run to be calibrated.

* D (str) is the device (default:cpu, gpu) to run pytorch. If gpu is chosen, it will use the first graphic card cuda:0 by default.

To train calibration run model:
.. code::

   python calib_mod.py 25Am02

Spceify the run for calibration using the input file as examples/data/inputs_run.npy, which is generated using Prepare_input.ipynb

Specify the device and training parameters (the followings are the default settings)

.. code::

   python calib_mod.py 17Am05 -D cpu -lr 1e-6 -n 400 -mom 0.9 -thr 1e-2

* D (str) is the device (default:cpu, mps, gpu) to run pytorch. 
* n (int) is the number of training epochs.
* lr (float) is the initial learning rate, which will decrease using scheduler (see details at https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau).
* thr (float) is the threshold of the scheduler.
* mom (float) is the momentum of gradient descent.

Citing this work
----------------

**Martin, F.N.**, **Starkenburg, E.**, & **Yuan, Z.** et al. (A&A 2024) https://www.aanda.org/articles/aa/full_html/2024/12/aa47633-23/aa47633-23.html