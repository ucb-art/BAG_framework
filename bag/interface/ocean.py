# -*- coding: utf-8 -*-

"""This module implements bag's interaction with an ocean simulator.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional

import os

import bag.io
from .simulator import SimProcessManager

if TYPE_CHECKING:
    from .simulator import ProcInfo


class OceanInterface(SimProcessManager):
    """This class handles interaction with Ocean simulators.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir, sim_config):
        # type: (str, Dict[str, Any]) -> None
        """Initialize a new SkillInterface object.
        """
        SimProcessManager.__init__(self, tmp_dir, sim_config)

    def format_parameter_value(self, param_config, precision):
        # type: (Dict[str, Any], int) -> str
        """Format the given parameter value as a string.

        To support both single value parameter and parameter sweeps, each parameter value is
        represented as a string instead of simple floats.  This method will cast a parameter
        configuration (which can either be a single value or a sweep) to a
        simulator-specific string.

        Parameters
        ----------
        param_config: Dict[str, Any]
            a dictionary that describes this parameter value.

            4 formats are supported.  This is best explained by example.

            single value:
            dict(type='single', value=1.0)

            sweep a given list of values:
            dict(type='list', values=[1.0, 2.0, 3.0])

            linear sweep with inclusive start, inclusive stop, and step size:
            dict(type='linstep', start=1.0, stop=3.0, step=1.0)

            logarithmic sweep with given number of points per decade:
            dict(type='decade', start=1.0, stop=10.0, num=10)

        precision : int
            the parameter value precision.

        Returns
        -------
        param_str : str
            a string representation of param_config
        """

        fmt = '%.{}e'.format(precision)
        swp_type = param_config['type']
        if swp_type == 'single':
            return fmt % param_config['value']
        elif swp_type == 'list':
            return ' '.join((fmt % val for val in param_config['values']))
        elif swp_type == 'linstep':
            syntax = '{From/To}Linear:%s:%s:%s{From/To}' % (fmt, fmt, fmt)
            return syntax % (param_config['start'], param_config['step'], param_config['stop'])
        elif swp_type == 'decade':
            syntax = '{From/To}Decade:%s:%s:%s{From/To}' % (fmt, '%d', fmt)
            return syntax % (param_config['start'], param_config['num'], param_config['stop'])
        else:
            raise Exception('Unsupported param_config: %s' % param_config)

    def _get_ocean_info(self, save_dir, script_fname, log_fname):
        """Private helper function that launches ocean process."""
        # get the simulation command.
        sim_kwargs = self.sim_config['kwargs']
        ocn_cmd = sim_kwargs['command']
        env = sim_kwargs.get('env', None)
        cwd = sim_kwargs.get('cwd', None)
        sim_cmd = [ocn_cmd, '-nograph', '-replay', script_fname, '-log', log_fname]

        if cwd is None:
            # set working directory to BAG_WORK_DIR if None
            cwd = os.environ['BAG_WORK_DIR']

        # create empty log file to make sure it exists.
        return sim_cmd, log_fname, env, cwd, save_dir

    def setup_sim_process(self, lib, cell, outputs, precision, sim_tag):
        # type: (str, str, Dict[str, str], int, Optional[str]) -> ProcInfo

        sim_tag = sim_tag or 'BagSim'
        job_options = self.sim_config['job_options']
        init_file = self.sim_config['init_file']
        view = self.sim_config['view']
        state = self.sim_config['state']

        # format job options as skill list of string
        job_opt_str = "'( "
        for key, val in job_options.items():
            job_opt_str += '"%s" "%s" ' % (key, val)
        job_opt_str += " )"

        # create temporary save directory and log/script names
        save_dir = bag.io.make_temp_dir(prefix='%s_data' % sim_tag, parent_dir=self.tmp_dir)
        log_fname = os.path.join(save_dir, 'ocn_output.log')
        script_fname = os.path.join(save_dir, 'run.ocn')

        # setup ocean simulation script
        script = self.render_file_template('run_simulation.ocn',
                                           dict(
                                               lib=lib,
                                               cell=cell,
                                               view=view,
                                               state=state,
                                               init_file=init_file,
                                               save_dir=save_dir,
                                               precision=precision,
                                               sim_tag=sim_tag,
                                               outputs=outputs,
                                               job_opt_str=job_opt_str,
                                           ))
        bag.io.write_file(script_fname, script)

        return self._get_ocean_info(save_dir, script_fname, log_fname)

    def setup_load_process(self, lib, cell, hist_name, outputs, precision):
        # type: (str, str, str, Dict[str, str], int) -> ProcInfo

        init_file = self.sim_config['init_file']
        view = self.sim_config['view']

        # create temporary save directory and log/script names
        save_dir = bag.io.make_temp_dir(prefix='%s_data' % hist_name, parent_dir=self.tmp_dir)
        log_fname = os.path.join(save_dir, 'ocn_output.log')
        script_fname = os.path.join(save_dir, 'run.ocn')

        # setup ocean load script
        script = self.render_file_template('load_results.ocn',
                                           dict(
                                               lib=lib,
                                               cell=cell,
                                               view=view,
                                               init_file=init_file,
                                               save_dir=save_dir,
                                               precision=precision,
                                               hist_name=hist_name,
                                               outputs=outputs,
                                           ))
        bag.io.write_file(script_fname, script)

        # launch ocean
        return self._get_ocean_info(save_dir, script_fname, log_fname)
