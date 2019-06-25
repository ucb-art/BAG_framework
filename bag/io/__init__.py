# -*- coding: utf-8 -*-

"""This package provides all IO related functionalities for BAG.

Most importantly, this module sorts out all the bytes v.s. unicode differences
and simplifies writing python2/3 compatible code.
"""

from .common import fix_string, to_bytes, set_encoding, get_encoding, \
    set_error_policy, get_error_policy
from .sim_data import load_sim_results, save_sim_results, load_sim_file
from .file import read_file, read_resource, read_yaml, readlines_iter, \
    write_file, make_temp_dir, open_temp, open_file

from . import process

__all__ = ['fix_string', 'to_bytes', 'set_encoding', 'get_encoding',
           'set_error_policy', 'get_error_policy',
           'load_sim_results', 'save_sim_results', 'load_sim_file',
           'read_file', 'read_resource', 'read_yaml', 'readlines_iter',
           'write_file', 'make_temp_dir', 'open_temp', 'open_file',
           ]
