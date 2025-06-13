#version ==> 0.0.1-2025.6

import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Union
from urllib.parse import unquote

import cv2
import numpy as np
import torch
import tqdm

from framework import __version__
from framework.utils.patches import imread, imshow, imwrite, torch_load, torch_save

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

#Other Constants
ARGV = sys.argv or [', ']
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
ASSETS = ROOT / 'assets' #default images
ASSETS_URL = " " #default download link
DEFAULT_CFG_PATH = ROOT / 'cfg/default.yaml'
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
AUTOINSTALL = str(os.getenv('AUTOINSTALL', True)).lower() == 'true'
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' if VERBOSE else None
LOGGING_NAME = 'framework'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in {'Darwin', 'Linux', 'Windows'})
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
ARM64 = platform.machine() in ('arm64', 'aarch64')
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = importlib.metadata.version('torchvision')
IS_VSCODE = os.environ.get('TERM_PROGRAM', False) == 'vscode'
RKNN_CHIPS = frozenset(
    {
        'rk3588', 'rk3576', 'rk3566', 'rk3568', 'rk3562', 'rv1103', 'rv1106', 'rv1103b', 'rv1106b', 'rk2118'
    }
)
HELP_MSG = """
        Use the Python SDK:
            from framework import YOLO

            # Load a model
            model = YOLO("yolo11n.yaml")  # build a new model from scratch
            model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    
            # Use the model
            results = model.train(data="coco8.yaml", epochs=3)  # train the model
            results = model.val()  # evaluate model performance on the validation set
            results = model(imagepath)  # predict on an image
            success = model.export(format="onnx")  # export the model to ONNX format
            
        Use the command line interface (CLI):
            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify, pose, obb]
                    MODE (required) is one of [train, val, predict, export, track, benchmark]
                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.
                    
        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

        - Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

        - Run special commands:
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg
        """

#Settings and Environment Variables
torch.set_printoptions(linewidth=320, precision=4, profile='default')
np.set_printoptions(linewidth=320, formatter=dict(float_kind='{:11.5g}'.format))
cv2.setNumThreads(0) #prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS) # NumExpr max threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress verbose TF compiler warnings in Colab
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR' #  suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ['KINETO_LOG_LEVEL'] = '5' # suppress verbose PyTorch profiler output when computing FLOPs

if TQDM_RICH := str(os.getenv('YOLO_TQDM_RICH', False)).lower() == 'true':
    from tqdm import rich

class TQDM(rich.tqdm if TQDM_RICH else tqdm.tqdm):
    '''
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings and provides
    additional customization options for projects. The progress bar is automatically disabled when
    VERBOSE is False or when explicitly disabled.

     Attributes:
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not
            explicitly set.

    Methods:
        __init__: Initialize the TQDM object with custom settings.
        __iter__: Return self as iterator to satisfy Iterable interface.

    Examples:
        >>> from framework.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass
    '''
    def __init__(self, *args, **kwargs):
        '''
         Notes:
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.
        '''
        warnings.filterwarnings('ignore', category=tqdm.TqdmExperimentalWarning) # suppress tqdm.rich warning
        kwargs['disable'] = not VERBOSE or kwargs.get('disable', False)
        kwargs.setdefault('bar_format', TQDM_BAR_FORMAT)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return super().__iter__()

class DataExportMixin:
    '''
    Mixin class for exporting validation metrics or prediction results in various formats.

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Pandas
    DataFrame, CSV, XML, HTML, JSON and SQLite (SQL).

    Methods:
        to_df: Convert summary to a Pandas DataFrame.
        to_csv: Export results as a CSV string.
        to_xml: Export results as an XML string (requires `lxml`).
        to_html: Export results as an HTML table.
        to_json: Export results as a JSON string.
        tojson: Deprecated alias for `to_json()`.
        to_sql: Export results to an SQLite database.

    Examples:
        >>> model = YOLO("yolo11n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
        >>> results.to_sql(table_name="yolo_results")
    '''
    def to_df(self, normalize=False, decimals=5):
        '''
        Create a pandas DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.
        '''
        import pandas as pd

        return pd.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        '''
         Args:
           normalize (bool, optional): Normalize numeric values.
           decimals (int, optional): Decimal precision.
        '''
        return self.to_df(normalize=normalize, decimals=decimals).to_csv()

    def to_xml(self, normalize=False, decimals=5):
        '''
        Args:
           normalize (bool, optional): Normalize numeric values.
           decimals (int, optional): Decimal precision.

        Notes:
            Requires `lxml` package to be installed.
        '''

