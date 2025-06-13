#version ==> 0.0.1-2025.6

import functools
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch

from framework.utils import ( ARM64,
                              ASSETS,
                              AUTOINSTALL,
                              IS_COLAB,
                              IS_GIT_DIR,
                              IS_JETSON,
                              IS_KAGGLE,
                              IS_PIP_PACKAGE,
                              LINUX,
                              LOGGER,
                              MACOS,
                              ONLINE,
                              PYTHON_VERSION,
                              RKNN_CHIPS,
                              ROOT,
                              TORCHVISION_VERSION,
                              USER_CONFIG_DIR,
                              WINDOWS,
                              Retry,
                              ThreadingLocked,
                              TryExcept,
                              clean_url,
                              colorstr,
                              downloads,
                              is_github_action_running,
                              url2file,
                              )

def parse_requirements(file_path = ROOT.parent / 'requirements.txt', package = ''):
    '''
     Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file.

    Returns:
        requirements (List[SimpleNamespace]): List of parsed requirements as SimpleNamespace objects with `name` and
            `specifier` attributes.
    '''
    if package:
        requires = [x for x in metadata.distribution(package).requires if 'extra == ' not in x]
        '''
        The metadata.distribution() function retrieves the metadata for a specific distribution (package), 
        and the .requires() method on the resulting object returns a list of strings representing the package's dependencies. 
        
        classical output:
            ['pytest (>=3.0.0) ; extra == "test"', 'pytest-cov ; extra == "test"']
        '''
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith('#'):
            line = line.partition('#')[0].strip() # ignore inline comments
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name = match[1], specifier=match[2].strip() if match[2] else ''))

    return requirements

@functools.lru_cache  # stores the results,  so can return the cached result instead of recomputing it
def parse_version(version = '0.0.0') -> tuple:
    '''
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version, i.e. (2, 0, 1)
    '''
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3])) # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(f"failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0
    