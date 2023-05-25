import os
# DATA_ROOT = r'D:\mag_track'
DATA_ROOT = r'/home/gaofei/BiRing/Data'

PROCESSED_RECORDINGS_DIR = os.path.join(DATA_ROOT, "recordings")

RECORDINGS_DIR = os.path.join(DATA_ROOT, "recordings")
# RECORDINGS_NATIVE_DIR = os.path.join(DATA_ROOT, "recordings_native")
RECORDINGS_NATIVE_DIR = os.path.join(DATA_ROOT)
# RECORDINGS_NATIVE_DIR = os.path.join(DATA_ROOT, "recordings_interference")
PROCESSED_RECORDINGS_DIR = os.path.join(DATA_ROOT, "processed")
PREDICTIONS_DIR = os.path.join(DATA_ROOT, "predictions")
PREDICTION_MATLAB_DIR = os.path.join(DATA_ROOT, "matlab")
TABFINDER_DIR = os.path.join(DATA_ROOT, "tabfinder")
MODEL_DIR = os.path.join(DATA_ROOT, "model")
BAUD = 460800
PORT = "COM19"
DATA_ROOT="./"