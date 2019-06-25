# -*- coding: utf-8 -*-

"""This module handles exporting schematic/layout from Virtuoso.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any

import os
from abc import ABC

from ..io import write_file, open_temp
from .base import SubProcessChecker

if TYPE_CHECKING:
    from .base import ProcInfo


class VirtuosoChecker(SubProcessChecker, ABC):
    """the base Checker class for Virtuoso.

    This class implement layout/schematic export procedures.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory.
    max_workers : int
        maximum number of parallel processes.
    cancel_timeout : float
        timeout for cancelling a subprocess.
    source_added_file : str
        file to include for schematic export.
    """

    def __init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file):
        # type: (str, int, float, str) -> None
        SubProcessChecker.__init__(self, tmp_dir, max_workers, cancel_timeout)
        self._source_added_file = source_added_file

    def setup_export_layout(self, lib_name, cell_name, out_file, view_name='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        out_file = os.path.abspath(out_file)

        run_dir = os.path.dirname(out_file)
        out_name = os.path.basename(out_file)
        log_file = os.path.join(run_dir, 'layout_export.log')

        os.makedirs(run_dir, exist_ok=True)

        # fill in stream out configuration file.
        content = self.render_file_template('layout_export_config.txt',
                                            dict(
                                                lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                run_dir=run_dir,
                                            ))

        with open_temp(prefix='stream_template', dir=run_dir, delete=False) as config_file:
            config_fname = config_file.name
            config_file.write(content)

        # run strmOut
        cmd = ['strmout', '-templateFile', config_fname]

        return cmd, log_file, None, os.environ['BAG_WORK_DIR']

    def setup_export_schematic(self, lib_name, cell_name, out_file, view_name='schematic',
                               params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        out_file = os.path.abspath(out_file)

        run_dir = os.path.dirname(out_file)
        out_name = os.path.basename(out_file)
        log_file = os.path.join(run_dir, 'schematic_export.log')

        # fill in stream out configuration file.
        content = self.render_file_template('si_env.txt',
                                            dict(
                                                lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                source_added_file=self._source_added_file,
                                                run_dir=run_dir,
                                            ))

        # create configuration file.
        config_fname = os.path.join(run_dir, 'si.env')
        write_file(config_fname, content)

        # run command
        cmd = ['si', run_dir, '-batch', '-command', 'netlist']

        return cmd, log_file, None, os.environ['BAG_WORK_DIR']
