# -*- coding: utf-8 -*-

"""This module implements LVS/RCX using ICV and stream out from Virtuoso.
"""

from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Any, Sequence

import os

from .virtuoso import VirtuosoChecker
from ..io import read_file, open_temp

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
    dirname = os.path.dirname(log_file)
    cell_name = os.path.basename(dirname)
    lvs_error_file = os.path.join(dirname, cell_name + '.LVS_ERRORS')

    # append error file at the end of log file
    with open(log_file, 'a') as logf:
        with open(lvs_error_file, 'r') as errf:
            for line in errf:
                logf.write(line)

    if not os.path.isfile(log_file):
        return False, ''

    cmd_output = read_file(log_file)
    test_str = 'Final comparison result:PASS'

    return test_str in cmd_output, log_file


class ICV(VirtuosoChecker):
    """A subclass of VirtuosoChecker that uses ICV for verification.

    Parameters
    ----------
    tmp_dir : string
        temporary directory to save files in.
    lvs_run_dir : str
        the LVS run directory.
    lvs_runset : str
        the LVS runset filename.
    rcx_run_dir : str
        the RCX run directory.
    rcx_runset : str
        the RCX runset filename.
    source_added_file : str
        the source.added file location.  Environment variable is supported.
        Default value is '$DK/Calibre/lvs/source.added'.
    rcx_mode : str
        the RC extraction mode.  Defaults to 'starrc'.
    """

    def __init__(self, tmp_dir, lvs_run_dir, lvs_runset, rcx_run_dir, rcx_runset,
                 source_added_file='$DK/Calibre/lvs/source.added', rcx_mode='pex',
                 **kwargs):

        max_workers = kwargs.get('max_workers', None)
        cancel_timeout = kwargs.get('cancel_timeout_ms', None)
        rcx_params = kwargs.get('rcx_params', {})
        lvs_params = kwargs.get('lvs_params', {})
        rcx_link_files = kwargs.get('rcx_link_files', None)
        lvs_link_files = kwargs.get('lvs_link_files', None)

        if cancel_timeout is not None:
            cancel_timeout /= 1e3

        VirtuosoChecker.__init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file)

        self.default_rcx_params = rcx_params
        self.default_lvs_params = lvs_params
        self.lvs_run_dir = os.path.abspath(lvs_run_dir)
        self.lvs_runset = lvs_runset
        self.lvs_link_files = lvs_link_files
        self.rcx_run_dir = os.path.abspath(rcx_run_dir)
        self.rcx_runset = rcx_runset
        self.rcx_link_files = rcx_link_files
        self.rcx_mode = rcx_mode

    def get_rcx_netlists(self, lib_name, cell_name):
        # type: (str, str) -> List[str]
        """Returns a list of generated extraction netlist file names.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name

        Returns
        -------
        netlists : List[str]
            a list of generated extraction netlist file names.  The first index is the main netlist.
        """
        # PVS generate schematic cellviews directly.
        if self.rcx_mode == 'starrc':
            return ['%s.spf' % cell_name]
        else:
            pass

    def setup_lvs_flow(self, lib_name, cell_name, sch_view='schematic', lay_view='layout',
                       params=None, **kwargs):
        # type: (str, str, str, str, Optional[Dict[str, Any]], Any) -> Sequence[FlowInfo]

        run_dir = os.path.join(self.lvs_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        lay_file, sch_file = self._get_lay_sch_files(run_dir)

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

        cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file, sch_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))

        lvs_params_actual = self.default_lvs_params.copy()
        if params is not None:
            lvs_params_actual.update(params)

        with open_temp(prefix='lvsLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name

        # set _drPROCESS
        dr_process_str = '_drPROCESS=' + lvs_params_actual['_drPROCESS']

        cmd = ['icv', '-D', dr_process_str, '-i', lay_file, '-s', sch_file, '-sf', 'SPICE',
               '-f', 'GDSII', '-c', cell_name, '-vue', '-I']
        for f in self.lvs_link_files:
            cmd.append(f)

        flow_list.append((cmd, log_file, None, run_dir, lvs_passed))
        return flow_list

    def setup_rcx_flow(self, lib_name, cell_name, sch_view='schematic', lay_view='layout',
                       params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> Sequence[FlowInfo]

        # update default RCX parameters.
        rcx_params_actual = self.default_rcx_params.copy()
        if params is not None:
            rcx_params_actual.update(params)

        run_dir = os.path.join(self.rcx_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        lay_file, sch_file = self._get_lay_sch_files(run_dir)
        with open_temp(prefix='rcxLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name
        flow_list = []
        cmd, log, env, cwd = self.setup_export_layout(lib_name, cell_name, lay_file, lay_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))
        cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file, sch_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))

        if self.rcx_mode == 'starrc':
            # first: run Extraction LVS
            lvs_params_actual = self.default_lvs_params.copy()

            dr_process_str = '_drPROCESS=' + lvs_params_actual['_drPROCESS']

            cmd = ['icv', '-D', '_drRCextract', '-D', dr_process_str, '-D', '_drICFOAlayers',
                   '-i', lay_file, '-s', sch_file, '-sf', 'SPICE', '-f', 'GDSII',
                   '-c', cell_name, '-I']
            for f in self.lvs_link_files:
                cmd.append(f)

            # hack the environment variables to make sure $PWD is the same as current working directory
            env_copy = os.environ.copy()
            env_copy['PWD'] = run_dir
            flow_list.append((cmd, log_file, env_copy, run_dir, lvs_passed))

            # second: setup CCP
            # make symlinks
            if self.rcx_link_files:
                for source_file in self.rcx_link_files:
                    targ_file = os.path.join(run_dir, os.path.basename(source_file))
                    if not os.path.exists(targ_file):
                        os.symlink(source_file, targ_file)

            # generate new cmd for StarXtract
            cmd_content, result = self.modify_starrc_cmd(run_dir, lib_name, cell_name,
                                                         rcx_params_actual, sch_file)

            # save cmd for StarXtract
            with open_temp(dir=run_dir, delete=False) as cmd_file:
                cmd_fname = cmd_file.name
                cmd_file.write(cmd_content)

            cmd = ['StarXtract', '-clean', cmd_fname]
        else:
            pass

        # noinspection PyUnusedLocal
        def rcx_passed(retcode, log_fname):
            dirname = os.path.dirname(log_fname)
            cell_name = os.path.basename(dirname)
            results_file = os.path.join(dirname, cell_name + '.RESULTS')

            # append error file at the end of log file
            with open(log_fname, 'a') as logf:
                with open(results_file, 'r') as errf:
                    for line in errf:
                        logf.write(line)

            if not os.path.isfile(log_fname):
                return None, ''

            cmd_output = read_file(log_fname)
            test_str = 'DRC and Extraction Results: CLEAN'

            if test_str in cmd_output:
                return results_file, log_fname
            else:
                return None, log_fname

        flow_list.append((cmd, log_file, None, run_dir, rcx_passed))
        return flow_list

    @classmethod
    def _get_lay_sch_files(cls, run_dir):
        lay_file = os.path.join(run_dir, 'layout.gds')
        sch_file = os.path.join(run_dir, 'schematic.net')
        return lay_file, sch_file

    def modify_starrc_cmd(self, run_dir, lib_name, cell_name, starrc_params, sch_file):
        # type: (str, str, str, Dict[str, Any], str) -> Tuple[str, str]
        """Modify the cmd file.

        Parameters
        ----------
        run_dir : str
            the run directory.
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        starrc_params : Dict[str, Any]
            override StarRC parameters.
        sch_file : str
            the schematic netlist

        Returns
        -------
        starrc_cmd : str
            the new StarXtract cmd file.
        output_name : str
            the extracted netlist file.
        """
        output_name = '%s.spf' % cell_name
        if 'CDSLIBPATH' in os.environ:
            cds_lib_path = os.path.abspath(os.path.join(os.environ['CDSLIBPATH'], 'cds.lib'))
        else:
            cds_lib_path = os.path.abspath('./cds.lib')
        content = self.render_string_template(read_file(self.rcx_runset),
                                              dict(
                                                  cell_name=cell_name,
                                                  extract_type=starrc_params['extract'].get('type'),
                                                  netlist_format=starrc_params.get('netlist_format',
                                                                                   'SPF'),
                                                  sch_file=sch_file,
                                                  cds_lib=cds_lib_path,
                                                  lib_name=lib_name,
                                                  run_dir=run_dir,
                                              ))
        return content, os.path.join(run_dir, output_name)
