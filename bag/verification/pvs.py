# -*- coding: utf-8 -*-

"""This module implements LVS/RCX using PVS/QRC and stream out from Virtuoso.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Sequence, Tuple

import os
import time

import yaml

from ..io import read_file, open_temp, readlines_iter, fix_string
from .virtuoso import VirtuosoChecker

if TYPE_CHECKING:
    from .base import FlowInfo


# noinspection PyUnusedLocal
def _all_pass(retcode, log_file):
    return True


# noinspection PyUnusedLocal
def lvs_passed(retcode, log_file):
    # type: (int, str) -> Tuple[bool, str]
    """Check if LVS passed

    Parameters
    ----------
    retcode : int
        return code of the LVS process.
    log_file : str
        log file name.

    Returns
    -------
    success : bool
        True if LVS passed.
    log_file : str
        the log file name.
    """
    if not os.path.isfile(log_file):
        return False, ''

    cmd_output = read_file(log_file)
    test_str = '# Run Result             : MATCH'
    return test_str in cmd_output, log_file


# noinspection PyUnusedLocal
def rcx_passed(retcode, log_file):
    """Check if RCX passed.

    Parameters
    ----------
    retcode : int
        return code of the RCX process.
    log_file : str
        log file name.

    Returns
    -------
    netlist : str
        netlist file name.
    log_file : str
        the log file name.
    """
    if not os.path.isfile(log_file):
        return None, ''

    cmd_output = read_file(log_file)
    test_str = 'INFO (LBRCXM-708): *****  Quantus QRC terminated normally  *****'
    if test_str in cmd_output:
        return '', log_file
    else:
        return None, ''


class PVS(VirtuosoChecker):
    """A subclass of VirtuosoChecker that uses PVS/QRC for verification.

    Parameters
    ----------
    tmp_dir : string
        temporary directory to save files in.
    lvs_run_dir : string
        the LVS run directory.
    lvs_runset : string
        the LVS runset filename.
    lvs_rule_file : string
        the LVS rule filename.
    rcx_runset : string
        the RCX runset filename.
    source_added_file : string
        the source.added file location.  Environment variable is supported.
        Default value is '$DK/Calibre/lvs/source.added'.
    """

    def __init__(self, tmp_dir, lvs_run_dir, lvs_runset, lvs_rule_file, rcx_runset,
                 source_added_file='$DK/Calibre/lvs/source.added', **kwargs):

        max_workers = kwargs.get('max_workers', None)
        cancel_timeout = kwargs.get('cancel_timeout_ms', None)
        if cancel_timeout is not None:
            cancel_timeout /= 1e3

        VirtuosoChecker.__init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file)

        self.default_rcx_params = kwargs.get('rcx_params', {})
        self.default_lvs_params = kwargs.get('lvs_params', {})
        self.lvs_run_dir = os.path.abspath(lvs_run_dir)
        self.lvs_runset = lvs_runset
        self.lvs_rule_file = lvs_rule_file
        self.rcx_runset = rcx_runset

    def get_rcx_netlists(self, lib_name, cell_name):
        # type: (str, str) -> List[str]
        # PVS generate schematic cellviews directly.
        return []

    def setup_lvs_flow(self, lib_name, cell_name, sch_view='schematic', lay_view='layout',
                       params=None, **kwargs):
        # type: (str, str, str, str, Optional[Dict[str, Any]], Any) -> Sequence[FlowInfo]

        run_dir = os.path.join(self.lvs_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        lay_file = os.path.join(run_dir, 'layout.gds')
        sch_file = os.path.join(run_dir, 'schematic.net')

        # add schematic/layout export to flow
        flow_list = []

        # Check if gds layout is provided
        gds_layout_path = kwargs.pop('gds_layout_path', None)

        # If not provided the gds layout, need to export layout
        if not gds_layout_path:
            cmd, log, env, cwd = self.setup_export_layout(lib_name, cell_name, lay_file, lay_view, None)
            flow_list.append((cmd, log, env, cwd, _all_pass))
        # If provided gds layout, do not export layout, just copy gds
        else:
            if not os.path.exists(gds_layout_path):
                raise ValueError(f'gds_layout_path does not exist: {gds_layout_path}')
            with open_temp(prefix='copy', dir=run_dir, delete=True) as f:
                copy_log_file = f.name
            copy_cmd = ['cp', gds_layout_path, os.path.abspath(lay_file)]
            flow_list.append((copy_cmd, copy_log_file, None, None, _all_pass))

        cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file, sch_view,
                                                         None)
        flow_list.append((cmd, log, env, cwd, _all_pass))

        lvs_params_actual = self.default_lvs_params.copy()
        if params is not None:
            lvs_params_actual.update(params)

        with open_temp(prefix='lvsLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name

        # generate new runset
        runset_content = self.modify_lvs_runset(run_dir, cell_name, lvs_params_actual)

        # save runset
        with open_temp(dir=run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['pvs', '-perc', '-lvs', '-qrc_data', '-control', runset_fname,
               '-gds', lay_file, '-layout_top_cell', cell_name,
               '-source_cdl', sch_file, '-source_top_cell', cell_name,
               self.lvs_rule_file,
               ]

        flow_list.append((cmd, log_file, None, run_dir, lvs_passed))

        return flow_list

    def setup_rcx_flow(self, lib_name, cell_name, sch_view='schematic', lay_view='layout',
                       params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> Sequence[FlowInfo]

        # update default RCX parameters.
        rcx_params_actual = self.default_rcx_params.copy()
        if params is not None:
            rcx_params_actual.update(params)

        run_dir = os.path.join(self.lvs_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        with open_temp(prefix='rcxLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name

        # generate new runset
        runset_content = self.modify_rcx_runset(run_dir, lib_name, cell_name, lay_view,
                                                rcx_params_actual)

        # save runset
        with open_temp(dir=run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['qrc', '-cmd', runset_fname]

        # NOTE: qrc needs to be run in the current working directory (virtuoso directory),
        # because it needs to access cds.lib
        return [(cmd, log_file, None, os.environ['BAG_WORK_DIR'], rcx_passed)]

    def modify_lvs_runset(self, run_dir, cell_name, lvs_params):
        # type: (str, str, Dict[str, Any]) -> str
        """Modify the given LVS runset file.

        Parameters
        ----------
        run_dir : str
            the run directory.
        cell_name : str
            the cell name.
        lvs_params : Dict[str, Any]
            override LVS parameters.

        Returns
        -------
        content : str
            the new runset content.
        """
        # convert runset content to dictionary
        lvs_options = {}
        for line in readlines_iter(self.lvs_runset):
            key, val = line.split(' ', 1)
            # remove semicolons
            val = val.strip().rstrip(';')
            if key in lvs_options:
                lvs_options[key].append(val)
            else:
                lvs_options[key] = [val]

        # get results_db file name
        results_db = os.path.join(run_dir, '%s.erc_errors.ascii' % cell_name)
        # override parameters
        lvs_options['lvs_report_file'] = ['"%s.rep"' % cell_name]
        lvs_options['report_summary'] = ['-erc "%s.sum" -replace' % cell_name]
        lvs_options['results_db'] = ['-erc "%s" -ascii' % results_db]
        lvs_options['mask_svdb_dir'] = ['"%s"' % os.path.join(run_dir, 'svdb')]

        lvs_options.update(lvs_params)
        content_list = []
        for key, val_list in lvs_options.items():
            for v in val_list:
                content_list.append('%s %s;\n' % (key, v))

        return ''.join(content_list)

    def modify_rcx_runset(self, run_dir, lib_name, cell_name, lay_view, rcx_params):
        # type: (str, str, str, str, Dict[str, Any]) -> str
        """Modify the given QRC options.

        Parameters
        ----------
        run_dir : str
            the run directory.
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        lay_view : str
            the layout view.
        rcx_params : Dict[str, Any]
            override RCX parameters.

        Returns
        -------
        content : str
            the new runset content.
        """
        data_dir = os.path.join(run_dir, 'svdb')
        # wait 10 seconds to see if not finding directory is just a network drive problem
        query_timeout = 10.0
        tstart = time.time()
        elapsed = 0.0
        while not os.path.isdir(data_dir) and elapsed < query_timeout:
            time.sleep(0.1)
            elapsed = time.time() - tstart
        if not os.path.isdir(data_dir):
            raise ValueError('cannot find directory %s.  Did you run PVS first?' % data_dir)

        # load default rcx options
        content = read_file(self.rcx_runset)
        rcx_options = yaml.load(content)

        # setup inputs/outputs
        rcx_options['input_db']['design_cell_name'] = '{} {} {}'.format(cell_name, lay_view,
                                                                        lib_name)
        rcx_options['input_db']['run_name'] = cell_name
        rcx_options['input_db']['directory_name'] = data_dir
        rcx_options['output_db']['cdl_out_map_directory'] = run_dir
        rcx_options['output_setup']['directory_name'] = data_dir
        rcx_options['output_setup']['temporary_directory_name'] = cell_name

        # override parameters
        for key, val in rcx_options.items():
            if key in rcx_params:
                val.update(rcx_params[key])

        # convert dictionary to QRC command file format.
        content_list = []
        for key, options in rcx_options.items():
            content_list.append('%s \\' % key)
            for k, v in options.items():
                v = fix_string(v)
                if isinstance(v, str):
                    # add quotes around string
                    v = '"{}"'.format(v)
                content_list.append('    -%s %s \\' % (k, v))

            # remove line continuation backslash from last option
            content_list[-1] = content_list[-1][:-2]

        return '\n'.join(content_list)
