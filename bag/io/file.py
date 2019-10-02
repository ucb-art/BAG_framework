# -*- coding: utf-8 -*-

"""This module handles file related IO.
"""
from typing import Dict, Any

import os
import tempfile
import time
import pkg_resources
import codecs
import string

import yaml
import pickle

from .common import bag_encoding, bag_codec_error


class Pickle:
    """
    A global class for reading and writing Pickle format.
    """
    @staticmethod
    def save(obj, file, **kwargs) -> None:
        with open(file, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file, **kwargs):
        with open(file, 'rb') as f:
            return pickle.load(f)


class Yaml:
    """
    A global class for reading and writing yaml format
    For backward compatibility some module functions may overlap with this.
    """
    @staticmethod
    def save(obj, file, **kwargs) -> None:
        with open(file, 'w') as f:
            yaml.dump(obj, f)

    @staticmethod
    def load(file, **kwargs):
        with open(file, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)


def open_file(fname, mode):
    """Opens a file with the correct encoding interface.

    Use this method if you need to have a file handle.

    Parameters
    ----------
    fname : string
        the file name.
    mode : string
        the mode, either 'r', 'w', or 'a'.

    Returns
    -------
    file_obj : file
        a file objects that reads/writes string with the BAG system encoding.
    """
    if mode != 'r' and mode != 'w' and mode != 'a':
        raise ValueError("Only supports 'r', 'w', or 'a' mode.")
    return open(fname, mode, encoding=bag_encoding, errors=bag_codec_error)


def read_file(fname):
    """Read the given file and return content as string.

    Parameters
    ----------
    fname : string
        the file name.

    Returns
    -------
    content : unicode
        the content as a unicode string.
    """
    with open_file(fname, 'r') as f:
        content = f.read()
    return content


def readlines_iter(fname):
    """Iterate over lines in a file.

    Parameters
    ----------
    fname : string
        the file name.

    Yields
    ------
    line : unicode
        a line in the file.
    """
    with open_file(fname, 'r') as f:
        for line in f:
            yield line


def read_yaml_env(fname):
    # type: (str) -> Dict[str, Any]
    """Parse YAML file with environment variable substitution.

    Parameters
    ----------
    fname : str
        yaml file name.

    Returns
    -------
    table : Dict[str, Any]
        the yaml file as a dictionary.
    """
    content = read_file(fname)
    # substitute environment variables
    content = string.Template(content).substitute(os.environ)
    return yaml.load(content, Loader=yaml.Loader)


def read_yaml(fname):
    """Read the given file using YAML.

    Parameters
    ----------
    fname : string
        the file name.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    with open_file(fname, 'r') as f:
        content = yaml.load(f, Loader=yaml.Loader)

    return content


def read_resource(package, fname):
    """Read the given resource file and return content as string.

    Parameters
    ----------
    package : string
        the package name.
    fname : string
        the resource file name.

    Returns
    -------
    content : unicode
        the content as a unicode string.
    """
    raw_content = pkg_resources.resource_string(package, fname)
    return raw_content.decode(encoding=bag_encoding, errors=bag_codec_error)


def write_file(fname, content, append=False, mkdir=True):
    """Writes the given content to file.

    Parameters
    ----------
    fname : string
        the file name.
    content : unicode
        the unicode string to write to file.
    append : bool
        True to append instead of overwrite.
    mkdir : bool
        If True, will create parent directories if they don't exist.
    """
    if mkdir:
        fname = os.path.abspath(fname)
        dname = os.path.dirname(fname)
        os.makedirs(dname, exist_ok=True)

    mode = 'a' if append else 'w'
    with open_file(fname, mode) as f:
        f.write(content)


def make_temp_dir(prefix, parent_dir=None):
    """Create a new temporary directory.

    Parameters
    ----------
    prefix : string
        the directory prefix.
    parent_dir : string
        the parent directory.
    """
    prefix += time.strftime("_%Y%m%d_%H%M%S")
    parent_dir = parent_dir or tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=prefix, dir=parent_dir)


def open_temp(**kwargs):
    """Opens a new temporary file for writing with unicode interface.

    Parameters
    ----------
    **kwargs
        the tempfile keyword arguments.  See documentation for
        :func:`tempfile.NamedTemporaryFile`.

    Returns
    -------
    file : file
        the opened file that accepts unicode input.
    """
    timestr = time.strftime("_%Y%m%d_%H%M%S")
    if 'prefix' in kwargs:
        kwargs['prefix'] += timestr
    else:
        kwargs['prefix'] = timestr
    temp = tempfile.NamedTemporaryFile(**kwargs)
    return codecs.getwriter(bag_encoding)(temp, errors=bag_codec_error)
