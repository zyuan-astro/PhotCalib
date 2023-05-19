"""
handeling all the argparse setup
"""

import argparse

_epilog = "Contact authors: Zhen Yuan (zhen.yuan@astro.unistra.fr)"
# _igrins_version = '1.0.0'
def argparse_train_model():
    """Take care of all the argparse stuff."""
    parser = argparse.ArgumentParser(
        prog        = 'PhotCalib - train calibration model',
        description = '''
        Train the run models in the main survey. \n
        The normal run models are
        '15Am01' '15Am02' '15Am03' '15Am04' '15Am05' '15Am06' '16Am02' '16Am03' \n
 '16Am04' '16Am05' '16Bm04' '16Bm06' '16Bm07' '17Am02' '17Am03' '17Am04' \n
 '17Am05' '17Am07' '17Bm01' '17Bm02' '17Bm03' '17Bm04' '17Bm05' '17Bm06' \n
 '18Am01' '18Am03' '18Am04' '18Am05' '18Am06' '18Bm01' '18Bm02' '18Bm03' \n
 '18Bm04' '18Bm05' '18Bm06' '18Bm07' '19Am02' '19Am03' '19Bm02' '19Bm03' \n
 '20Am01' '20Am02' '20Am03' '20Am04' '20Am05' '20Am06' '20Bm01' '20Bm02' \n
 '20Bm03' '20Bm04' '20Bm05' '20Bm06' '21Am01' '21Am02' '21Am03' '21Am04' \n
 '21Am05' '21Am06' '21Bm01' '21Bm02' '21Bm06' '22Am01' '22Am02' '22Am03' \n
 '22Am05' '22Am06' '22Am07' '22Bm01' '22Bm02'
        ''', epilog = _epilog
        )
    parser.add_argument(
        "run", action="store", help="Enter the run to train",
        type=str
        )
    parser.add_argument(
        "-D", dest="device", action="store", help="Enter the device for pytorch, default = cpu",
        type=str
        )
    parser.add_argument(
        "-lr", dest="lr", action="store", help="learning rate, default = 1e-6",
        type=float
        )
    parser.add_argument(
        "-n", dest="N_epochs", action="store", help="epochs, default = 400",
        type=int
        )
    parser.add_argument(
        "-mom", dest="momentum", action="store", help="momentun, default = 0.9",
        type=float
        )
    parser.add_argument(
        "-thr", dest="thr", action="store", help="threshold, default = 1e-2",
        type=float
        )
    return parser.parse_args()
    
def argparse_apply_model():
    """Take care of all the argparse stuff."""
    parser = argparse.ArgumentParser(
        prog        = 'PhotCalib - apply calibration model to raw data',
        description = '''
        Use the run models trained from the main survey to calibrate new data. \n
        The available run models are
        '15Am01' '15Am02' '15Am03' '15Am04' '15Am05' '15Am06' '16Am02' '16Am03' \n
 '16Am04' '16Am05' '16Bm04' '16Bm06' '16Bm07' '17Am02' '17Am03' '17Am04' \n
 '17Am05' '17Am07' '17Bm01' '17Bm02' '17Bm03' '17Bm04' '17Bm05' '17Bm06' \n
 '18Am01' '18Am03' '18Am04' '18Am05' '18Am06' '18Bm01' '18Bm02' '18Bm03' \n
 '18Bm04' '18Bm05' '18Bm06' '18Bm07' '19Am02' '19Am03' '19Bm02' '19Bm03' \n
 '20Am01' '20Am02' '20Am03' '20Am04' '20Am05' '20Am06' '20Bm01' '20Bm02' \n
 '20Bm03' '20Bm04' '20Bm05' '20Bm06' '21Am01' '21Am02' '21Am03' '21Am04' \n
 '21Am05' '21Am06' '21Bm01' '21Bm02' '21Bm06' '22Am01' '22Am02' '22Am03' \n
 '22Am05' '22Am06' '22Am07' '22Bm01' '22Bm02'
        ''', epilog = _epilog
        )
    parser.add_argument(
        "run", action="store", help="Enter the run used for calibration",
        type=str
        )
    parser.add_argument(
        "input", action="store", 
        help="The raw datafile to be calibrated",
        type=str
        )
    parser.add_argument(
        "-D", dest="device", action="store", help="Enter the device for pytorch, default = 'cpu'",
        type=str
        )
    
    
    return parser.parse_args()