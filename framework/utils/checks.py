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
from torch.optim.optimizer import required

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


def is_ascii(s) -> bool:
    '''
    Check if a string is composed of only ASCII characters.

    function:
        all(): Return True if bool(x) is True for all values x in the iterable. If the iterable is empty, return True.
        ord(): Return the Unicode code point for a one-character string.
    '''
    return all(ord(c) < 128 for c in str(s))


def check_imgsz(imgsz, stride = 32, min_dim = 1, max_dim = 2, floor = 0):
    '''
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | List[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int] | int): Updated image size.
    '''
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)
    # Convert stride to integer if it is a tensor

    ## Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str): # # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
        '''
        function:
            imgsz.isnumeric(): Return True if the string is a numeric string, False otherwise.
            eval(imgsz): Evaluate the given source in the context of globals and locals.
        '''
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}."
            f"Valid imgsz type are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640, 640]'"
        )

    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]

    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
    #math.ceil(x / stride): Return the ceiling of x / stride as an Integral.

    if sz != imgsz:
        LOGGER.warning(f"imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) ==1 else sz

    return sz


@functools.lru_cache
def check_uv():
    '''Check if uv package manager is installed and can run successfully.'''
    try:
        return subprocess.run(['uv', '-V'], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@functools.lru_cache
def check_version(
        current: str = '0.0.0',
        required: str = '0.0.0',
        name: str = 'version',
        hard: bool = False,
        verbose: bool = False,
        msg: str = '',
    ) -> bool:
    '''
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.
        msg (str): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Examples:
        Check if current version is exactly 22.04
        >>> check_version(current="22.04", required="==22.04")

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04
        >>> check_version(current="22.04", required="<=22.04")

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")
    '''
    if not current:
        LOGGER.warning(f"invaild check_version({current}, {required}) requested, please check values")
        return True
    elif not current[0].isdigit(): #current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False
    if not required:
        return True

    if 'sys_platform' in required and ( # # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and 'linux' not in required)
        or (MACOS and 'macos' not in required and 'darwin' not in required)
    ):
        return True

    op = ''
    version = ''
    result = True
    c = parse_version(current) # # '1.2.3' -> (1, 2, 3)
    for r in required.strip(',').split(','):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups() # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = '>='
        v = parse_version(version)

        if op == '==' and c != v:
            result = False
        elif op == '!=' and c == v:
            result = False
        elif op == '>=' and not (c >= v):
            result = False
        elif op == '<=' and not (c <= v):
            result = False
        elif op == '>' and not (c > v):
            result = False
        elif op == '<' and not (c < v):
            result = False

    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name = ''):
    '''
    Returns:
        (str): The latest version of the package.
    '''
    import requests

    try:
        requests.packages.urllib3.disable_warnings() # Disable the InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()['info']['version']
    except Exception:
        return None


def check_pip_update_available():
    '''
    Check if a new version of package is available on PyPI.

    Returns:
        (bool): True if an update is available, False otherwise.
    '''
    if ONLINE and IS_PIP_PACKAGE:
        try:
            from framework import __version__

            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}>"):
                LOGGER.info(
                    f"New version available"
                    f"Update please"
                )
                return True
        except Exception:
            pass
    return False


@ThreadingLocked
@functools.lru_cache
def check_font(font = "Arial.ttf"):
    '''
    Find font locally or download to user's configuration directory if it does not already exist.

    Returns:
        (Path): Resolved font file path.
    '''
    from matplotlib import font_manager

    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # Download to USER_CONFIG_DIR if missing
    url = f"https://.../.../{name}"
    if downloads.is_url(url, check = True):
        downloads.safe_download(url = url, file = file)
        return file


def check_python(mininum: str = '3.8.0', hard: bool = True, verbose: bool = False):
    '''
    Args:
        minimum (str): Required minimum version of python.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.

    Returns:
        (bool): Whether the installed Python version meets the minimum constraints.
    '''
    return check_version(PYTHON_VERSION, mininum, name = 'Python', hard=hard, verbose=verbose)









