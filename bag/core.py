# -*- coding: utf-8 -*-

"""This is the core bag module.
"""

from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional, Union, Type, Sequence, TypeVar

import os
import importlib
import cProfile
import pstats
from pathlib import Path

# noinspection PyPackageRequirements

from .interface import ZMQDealer
from .interface.database import DbAccess
from .design import ModuleDB, SchInstance
from .layout.routing import RoutingGrid
from .layout.template import TemplateDB
from .layout.core import DummyTechInfo
from .io import read_file, sim_data, read_yaml_env
from .concurrent.core import batch_async_task

if TYPE_CHECKING:
    from .interface.simulator import SimAccess
    from .layout.template import TemplateBase
    from .layout.core import TechInfo
    from .design.module import Module
    from .simulation.core_v2 import TestbenchManager, MeasurementManager

    ModuleType = TypeVar('ModuleType', bound=Module)
    TemplateType = TypeVar('TemplateType', bound=TemplateBase)


def _get_config_file_abspath(fname):
    """Get absolute path of configuration file using BAG_WORK_DIR environment variable."""
    fname = os.path.basename(fname)
    if 'BAG_WORK_DIR' not in os.environ:
        raise ValueError('Environment variable BAG_WORK_DIR not defined')

    work_dir = os.environ['BAG_WORK_DIR']
    if not os.path.isdir(work_dir):
        raise ValueError('$BAG_WORK_DIR = %s is not a directory' % work_dir)

    # read port number
    fname = os.path.join(work_dir, fname)
    if not os.path.isfile(fname):
        raise ValueError('Cannot find file: %s' % fname)
    return fname


def _get_port_number(port_file):
    # type: (str) -> Tuple[Optional[int], str]
    """Read the port number from the given port file.

    Parameters
    ----------
    port_file : str
        a file containing the communication port number.

    Returns
    -------
    port : Optional[int]
        the port number if reading is successful.
    msg : str
        Empty string on success, the error message on failure.
    """
    try:
        port_file = _get_config_file_abspath(port_file)
    except ValueError as err:
        return None, str(err)

    port = int(read_file(port_file))
    return port, ''


def _import_class_from_str(class_str):
    # type: (str) -> Type
    """Given a Python class string, convert it to the Python class.

    Parameters
    ----------
    class_str : str
        a Python class string/

    Returns
    -------
    py_class : class
        a Python class.
    """
    sections = class_str.split('.')

    module_str = '.'.join(sections[:-1])
    class_str = sections[-1]
    modul = importlib.import_module(module_str)
    return getattr(modul, class_str)


class Testbench(object):
    """A class that represents a testbench instance.

    Parameters
    ----------
    sim : :class:`bag.interface.simulator.SimAccess`
        The SimAccess instance used to issue simulation commands.
    db : :class:`bag.interface.database.DbAccess`
        The DbAccess instance used to update testbench schematic.
    lib : str
        testbench library.
    cell : str
        testbench cell.
    parameters : Dict[str, str]
        the simulation parameter dictionary.  The values are string representation
        of actual parameter values.
    env_list : Sequence[str]
        list of defined simulation environments.
    default_envs : Sequence[str]
        the selected simulation environments.
    outputs : Dict[str, str]
        default output expressions

    Attributes
    ----------
    lib : str
        testbench library.
    cell : str
        testbench cell.
    save_dir : str
        directory containing the last simulation data.
    """

    def __init__(self,  # type: Testbench
                 sim,  # type: SimAccess
                 db,  # type: DbAccess
                 lib,  # type: str
                 cell,  # type: str
                 parameters,  # type: Dict[str, str]
                 env_list,  # type: Sequence[str]
                 default_envs,  # type: Sequence[str]
                 outputs,  # type: Dict[str, str]
                 ):
        # type: (...) -> None
        """Create a new testbench instance.
        """
        self.sim = sim
        self.db = db
        self.lib = lib
        self.cell = cell
        self.parameters = parameters
        self.env_parameters = {}
        self.env_list = env_list
        self.sim_envs = default_envs
        self.config_rules = {}
        self.outputs = outputs
        self.save_dir = None

    def get_defined_simulation_environments(self):
        # type: () -> Sequence[str]
        """Return a list of defined simulation environments"""
        return self.env_list

    def get_current_simulation_environments(self):
        # type: () -> Sequence[str]
        """Returns a list of simulation environments this testbench will simulate."""
        return self.sim_envs

    def add_output(self, var, expr):
        # type: (str, str) -> None
        """Add an output expression to be recorded and exported back to python.

        Parameters
        ----------
        var : str
            output variable name.
        expr : str
            the output expression.
        """
        if var in sim_data.illegal_var_name:
            raise ValueError('Variable name %s is illegal.' % var)
        self.outputs[var] = expr

    def set_parameter(self, name, val, precision=6):
        # type: (str, Union[int, float], int) -> None
        """Sets the value of the given simulation parameter.

        Parameters
        ----------
        name : str
            parameter name.
        val : Union[int, float]
            parameter value
        precision : int
            the parameter value will be rounded to this precision.
        """
        param_config = dict(type='single', value=val)
        if isinstance(val, str):
            self.parameters[name] = val
        else:
            self.parameters[name] = self.sim.format_parameter_value(param_config, precision)

    def set_env_parameter(self, name, val_list, precision=6):
        # type: (str, Sequence[float], int) -> None
        """Configure the given parameter to have different value across simulation environments.

        Parameters
        ----------
        name : str
            the parameter name.
        val_list : Sequence[float]
            the parameter values for each simulation environment.  the order of the simulation
            environments can be found in self.sim_envs
        precision : int
            the parameter value will be rounded to this precision.
        """
        if len(self.sim_envs) != len(val_list):
            raise ValueError('env parameter must have %d values.' % len(self.sim_envs))

        default_val = None
        for env, val in zip(self.sim_envs, val_list):
            if env not in self.env_parameters:
                cur_dict = {}
                self.env_parameters[env] = cur_dict
            else:
                cur_dict = self.env_parameters[env]

            param_config = dict(type='single', value=val)
            cur_val = self.sim.format_parameter_value(param_config, precision)
            if default_val is None:
                default_val = cur_val
            cur_dict[name] = self.sim.format_parameter_value(param_config, precision)
        self.parameters[name] = default_val

    def set_sweep_parameter(self, name, precision=6, **kwargs):
        # type: (str, int, **Any) -> None
        """Set to sweep the given parameter.

        To set the sweep values directly:

        tb.set_sweep_parameter('var', values=[1.0, 5.0, 10.0])

        To set a linear sweep with start/stop/step (inclusive start and stop):

        tb.set_sweep_parameter('var', start=1.0, stop=9.0, step=4.0)

        To set a logarithmic sweep with points per decade (inclusive start and stop):

        tb.set_sweep_parameter('var', start=1.0, stop=10.0, num_decade=3)

        Parameters
        ----------
        name : str
            parameter name.
        precision : int
            the parameter value will be rounded to this precision.
        **kwargs : Any
            the sweep parameters.  Refer to the above for example calls.
        """
        if 'values' in kwargs:
            param_config = dict(type='list', values=kwargs['values'])
        elif 'start' in kwargs and 'stop' in kwargs:
            start = kwargs['start']
            stop = kwargs['stop']
            if 'step' in kwargs:
                step = kwargs['step']
                param_config = dict(type='linstep', start=start, stop=stop, step=step)
            elif 'num_decade' in kwargs:
                num = kwargs['num_decade']
                param_config = dict(type='decade', start=start, stop=stop, num=num)
            else:
                raise Exception('Unsupported sweep arguments: %s' % kwargs)
        else:
            raise Exception('Unsupported sweep arguments: %s' % kwargs)

        self.parameters[name] = self.sim.format_parameter_value(param_config, precision)

    def set_simulation_environments(self, env_list):
        # type: (Sequence[str]) -> None
        """Enable the given list of simulation environments.

        If more than one simulation environment is specified, then a sweep
        will be performed.

        Parameters
        ----------
        env_list : Sequence[str]
        """
        self.sim_envs = env_list

    def set_simulation_view(self, lib_name, cell_name, sim_view):
        # type: (str, str, str) -> None
        """Set the simulation view of the given design.

        For simulation, each design may have multiple views, such as schematic,
        veriloga, extracted, etc.  This method lets you choose which view to
        use for netlisting.  the given design can be the top level design or
        an intermediate instance.

        Parameters
        ----------
        lib_name : str
            design library name.
        cell_name : str
            design cell name.
        sim_view : str
            the view to simulate with.
        """
        key = '%s__%s' % (lib_name, cell_name)
        self.config_rules[key] = sim_view

    def update_testbench(self):
        # type: () -> None
        """Commit the testbench changes to the CAD database.
        """
        config_list = []
        for key, view in self.config_rules.items():
            lib, cell = key.split('__')
            config_list.append([lib, cell, view])

        env_params = []
        for env in self.sim_envs:
            if env in self.env_parameters:
                val_table = self.env_parameters[env]
                env_params.append(list(val_table.items()))
        self.db.update_testbench(self.lib, self.cell, self.parameters, self.sim_envs, config_list,
                                 env_params)

    def run_simulation(self, precision=6, sim_tag=None):
        # type: (int, Optional[str]) -> Optional[str]
        """Run simulation.

        Parameters
        ----------
        precision : int
            the floating point number precision.
        sim_tag : Optional[str]
            optional description for this simulation run.

        Returns
        -------
        value : Optional[str]
            the save directory path.  If simulation is cancelled, return None.
        """
        coro = self.async_run_simulation(precision=precision, sim_tag=sim_tag)
        batch_async_task([coro])
        return self.save_dir

    def load_sim_results(self, hist_name, precision=6):
        # type: (str, int) -> Optional[str]
        """Load previous simulation data.

        Parameters
        ----------
        hist_name : str
            the simulation history name.
        precision : int
            the floating point number precision.

        Returns
        -------
        value : Optional[str]
            the save directory path.  If result loading is cancelled, return None.
        """
        coro = self.async_load_results(hist_name, precision=precision)
        batch_async_task([coro])
        return self.save_dir

    async def async_run_simulation(self,
                                   precision: int = 6,
                                   sim_tag: Optional[str] = None) -> str:
        """A coroutine that runs the simulation.

        Parameters
        ----------
        precision : int
            the floating point number precision.
        sim_tag : Optional[str]
            optional description for this simulation run.

        Returns
        -------
        value : str
            the save directory path.
        """
        self.save_dir = None
        self.save_dir = await self.sim.async_run_simulation(self.lib, self.cell, self.outputs,
                                                            precision=precision, sim_tag=sim_tag)
        return self.save_dir

    async def async_load_results(self, hist_name: str, precision: int = 6) -> str:
        """A coroutine that loads previous simulation data.

        Parameters
        ----------
        hist_name : str
            the simulation history name.
        precision : int
            the floating point number precision.

        Returns
        -------
        value : str
            the save directory path.
        """
        self.save_dir = None
        self.save_dir = await self.sim.async_load_results(self.lib, self.cell, hist_name,
                                                          self.outputs, precision=precision)
        return self.save_dir


def create_tech_info(bag_config_path=None):
    # type: (Optional[str]) -> TechInfo
    """Create TechInfo object."""
    if bag_config_path is None:
        if 'BAG_CONFIG_PATH' not in os.environ:
            raise Exception('BAG_CONFIG_PATH not defined.')
        bag_config_path = os.environ['BAG_CONFIG_PATH']

    bag_config = read_yaml_env(bag_config_path)
    tech_params = read_yaml_env(bag_config['tech_config_path'])
    if 'class' in tech_params:
        tech_cls = _import_class_from_str(tech_params['class'])
        tech_info = tech_cls(tech_params)
    else:
        # just make a default tech_info object as place holder.
        print('*WARNING*: No TechInfo class defined.  Using a dummy version.')
        tech_info = DummyTechInfo(tech_params)

    return tech_info


class BagProject(object):
    """The main bag controller class.

    This class mainly stores all the user configurations, and issue
    high level bag commands.

    Parameters
    ----------
    bag_config_path : Optional[str]
        the bag configuration file path.  If None, will attempt to read from
        environment variable BAG_CONFIG_PATH.
    port : Optional[int]
        the BAG server process port number.  If not given, will read from port file.

    Attributes
    ----------
    bag_config : Dict[str, Any]
        the BAG configuration parameters dictionary.
    tech_info : bag.layout.core.TechInfo
        the BAG process technology class.
    """

    def __init__(self, bag_config_path=None, port=None):
        # type: (Optional[str], Optional[int]) -> None
        if bag_config_path is None:
            if 'BAG_CONFIG_PATH' not in os.environ:
                raise Exception('BAG_CONFIG_PATH not defined.')
            bag_config_path = os.environ['BAG_CONFIG_PATH']

        self.bag_config = read_yaml_env(bag_config_path)
        bag_tmp_dir = os.environ.get('BAG_TEMP_DIR', None)

        # get port files
        if port is None:
            socket_config = self.bag_config['socket']
            if 'port_file' in socket_config:
                port, msg = _get_port_number(socket_config['port_file'])
                if msg:
                    print('*WARNING* %s' % msg)

        # create ZMQDealer object
        dealer_kwargs = {}
        dealer_kwargs.update(self.bag_config['socket'])
        del dealer_kwargs['port_file']

        # create TechInfo instance
        self.tech_info = create_tech_info(bag_config_path=bag_config_path)

        # create design module database.
        try:
            lib_defs_file = _get_config_file_abspath(self.bag_config['lib_defs'])
        except ValueError:
            lib_defs_file = ''
        sch_exc_libs = self.bag_config['database']['schematic']['exclude_libraries']
        self.dsn_db = ModuleDB(lib_defs_file, self.tech_info, sch_exc_libs, prj=self)

        if port is not None:
            # make DbAccess instance.
            dealer = ZMQDealer(port, **dealer_kwargs)
            db_cls = _import_class_from_str(self.bag_config['database']['class'])
            self.impl_db = db_cls(dealer, bag_tmp_dir, self.bag_config['database'])
            self._default_lib_path = self.impl_db.default_lib_path
        else:
            self.impl_db = None  # type: Optional[DbAccess]
            self._default_lib_path = DbAccess.get_default_lib_path(self.bag_config['database'])

        # make SimAccess instance.
        sim_cls = _import_class_from_str(self.bag_config['simulation']['class'])
        self.sim = sim_cls(bag_tmp_dir, self.bag_config['simulation'])  # type: SimAccess

    @property
    def default_lib_path(self):
        # type: () -> str
        return self._default_lib_path

    def close_bag_server(self):
        # type: () -> None
        """Close the BAG database server."""
        if self.impl_db is not None:
            self.impl_db.close()
            self.impl_db = None

    def close_sim_server(self):
        # type: () -> None
        """Close the BAG simulation server."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def import_design_library(self, lib_name):
        # type: (str) -> None
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        new_lib_path = self.bag_config['new_lib_path']
        self.impl_db.import_design_library(lib_name, self.dsn_db, new_lib_path)

    def get_cells_in_library(self, lib_name):
        # type: (str) -> Sequence[str]
        """Get a list of cells in the given library.

        Returns an empty list if the given library does not exist.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        cell_list : Sequence[str]
            a list of cells in the library
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.get_cells_in_library(lib_name)

    def make_template_db(self, impl_lib, grid_specs, use_cybagoa=True, gds_lay_file='',
                         cache_dir=''):
        # type: (str, Dict[str, Any], bool, str, str) -> TemplateDB
        """Create and return a new TemplateDB instance.

        Parameters
        ----------
        impl_lib : str
            the library name to put generated layouts in.
        grid_specs : Dict[str, Any]
            the routing grid specification dictionary.
        use_cybagoa : bool
            True to enable cybagoa acceleration if available.
        gds_lay_file : str
            the GDS layout information file.
        cache_dir : str
            the cache directory name.
        """
        layers = grid_specs['layers']
        widths = grid_specs['widths']
        spaces = grid_specs['spaces']
        bot_dir = grid_specs['bot_dir']
        width_override = grid_specs.get('width_override', None)

        routing_grid = RoutingGrid(self.tech_info, layers, spaces, widths, bot_dir,
                                   width_override=width_override)
        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=use_cybagoa,
                         gds_lay_file=gds_lay_file, cache_dir=cache_dir, prj=self)

        return tdb

    def generate_cell(self,  # type: BagProject
                      specs,  # type: Dict[str, Any]
                      temp_cls=None,  # type: Optional[Type[TemplateType]]
                      gen_lay=True,  # type: bool
                      gen_sch=False,  # type: bool
                      run_lvs=False,  # type: bool
                      run_rcx=False,  # type: bool
                      use_cybagoa=True,  # type: bool
                      debug=False,  # type: bool
                      profile_fname='',  # type: str
                      use_cache=False,  # type: bool
                      save_cache=False,  # type: bool
                      **kwargs,
                      ):
        # type: (...) -> Optional[Union[pstats.Stats, Dict[str, Any]]]
        """Generate layout/schematic of a given cell from specification file.

        Parameters
        ----------
        specs : Dict[str, Any]
            the specification dictionary.
        temp_cls : Optional[Type[TemplateType]]
            the TemplateBase subclass to instantiate
            if not provided, it will be imported from lay_class entry in specs dictionary.
        gen_lay : bool
            True to generate layout.
        gen_sch : bool
            True to generate schematics.
        run_lvs : bool
            True to run LVS.
        run_rcx : bool
            True to run RCX.
        use_cybagoa : bool
            True to enable cybagoa acceleration if available.
        debug : bool
            True to print debug messages.
        profile_fname : str
            If not empty, profile layout generation, and save statistics to this file.
        use_cache : bool
            True to use cached layouts.
        save_cache : bool
            True to save instances in this template to cache.
        **kwargs :
            Additional optional arguments.

        Returns
        -------
        result: Optional[Union[pstats.Stats, Dict[str, Any]]]
            If profiling is enabled, result will be the statistics object.
            If the last thing done is layout or schematic, result will contain sch_params
            If the last thing done is lvs, in case of failure result will
            contain lvs log file in a dictionary, otherwise None
            If the last thing done is rcx, in case of failure result will
            contain rcx log file in a dictionary, otherwise None
        """
        prefix = kwargs.get('prefix', '')
        suffix = kwargs.get('suffix', '')

        grid_specs = specs['routing_grid']
        impl_lib = specs['impl_lib']
        impl_cell = specs['impl_cell']
        lay_str = specs.get('lay_class', '')
        sch_lib = specs.get('sch_lib', '')
        sch_cell = specs.get('sch_cell', '')
        params = specs['params']
        gds_lay_file = specs.get('gds_lay_file', '')
        cache_dir = specs.get('cache_dir', '')

        if temp_cls is None and lay_str:
            temp_cls = _import_class_from_str(lay_str)

        has_lay = temp_cls is not None
        if gen_lay and not has_lay:
            raise ValueError('layout_class is not specified')

        if use_cache:
            db_cache_dir = specs.get('cache_dir', '')
        else:
            db_cache_dir = ''

        result_pstat = None
        if has_lay:
            temp_db = self.make_template_db(impl_lib, grid_specs, use_cybagoa=use_cybagoa,
                                            gds_lay_file=gds_lay_file, cache_dir=db_cache_dir)

            name_list = [impl_cell]
            print('computing layout...')
            if profile_fname:
                profiler = cProfile.Profile()
                profiler.runcall(temp_db.new_template, params=params, temp_cls=temp_cls,
                                 debug=False)
                profiler.dump_stats(profile_fname)
                result_pstat = pstats.Stats(profile_fname).strip_dirs()

            temp = temp_db.new_template(params=params, temp_cls=temp_cls, debug=debug)
            print('computation done.')
            temp_list = [temp]

            if save_cache and cache_dir:
                master_list = [inst.master for inst in temp.instance_iter()]
                print('saving layouts to cache...')
                temp_db.save_to_cache(master_list, cache_dir, debug=debug)
                print('saving done.')

            if gen_lay:
                print('creating layout...')
                temp_db.batch_layout(self, temp_list, name_list, debug=debug)
                print('layout done.')

            sch_params = temp.sch_params
        else:
            sch_params = params

        if gen_sch:
            dsn = self.create_design_module(lib_name=sch_lib, cell_name=sch_cell)
            print('computing schematic...')
            dsn.design(**sch_params)
            print('creating schematic...')
            dsn.implement_design(impl_lib, top_cell_name=impl_cell, prefix=prefix,
                                 suffix=suffix)
            print('schematic done.')

        result = sch_params
        lvs_passed = False
        if run_lvs:
            print('running lvs...')
            lvs_passed, lvs_log = self.run_lvs(impl_lib, impl_cell, gds_lay_file=gds_lay_file)
            print('LVS log: %s' % lvs_log)
            if lvs_passed:
                print('LVS passed!')
                result = dict(log='')
            else:
                print('LVS failed...')
                result = dict(log=lvs_log)

        if run_rcx and ((run_lvs and lvs_passed) or not run_lvs):
            print('running rcx...')
            rcx_passed, rcx_log = self.run_rcx(impl_lib, impl_cell)
            print('RCX log: %s' % rcx_log)
            if rcx_passed:
                print('RCX passed!')
                result = dict(log='')
            else:
                print('RCX failed...')
                result = dict(log=rcx_log)

        if result_pstat:
            return result_pstat
        return result

    def simulate_cell(self,
                      specs: Dict[str, Any],
                      gen_cell: bool = True,
                      gen_wrapper: bool = True,
                      gen_tb: bool = True,
                      load_results: bool = False,
                      extract: bool = False,
                      run_sim: bool = True) -> Optional[Dict[str, Any]]:
        """
        Runs a minimum executable parts of the Testbench Manager flow selectively according to
        a spec dictionary.

        For example you can set the flags to generate a new cell, but since wrapper and test bench
        exist, maybe you want to skip those, and run the simulation in the end. Maybe you
        already created the cell all the way up to test bench level, and now you only need to
        run simulation.

        This function only works with Testbench Managers written in format of
        simulation.core_v2.TestbenchManager

        Parameters
        ----------
        specs:
            Dictionary of specifications
            Some non-obvious conventions:
            - if contains tbm_specs keyword, simulation is ran through testbench manager v2,
            otherwise there should be a sim_params entry that specifies the simulation.
            - Wrapper is assumed to be in the specs dictionary, if it is generated outside of
            this function, gen_wrapper should be False.
        gen_cell:
            True to call generate_cell on specs
        gen_wrapper:
            True to generate Wrapper. Currently only one top-level wrapper is supported.
        gen_tb:
            True to generate test bench. If test bench is created, this flag can be set to False.
        load_results:
            True to skip simulation and load the results.
        extract:
            False to skip layout generation and only simulate schematic
        run_sim:
            True to run simulation. If the purpose of calling this function is just to generate
            some part of simulation flow to debug, this flag can be set to False.
        Returns
        -------
        results: Optional[Dict[str, Any]]
            if run_sim/load_results = True, contains simulations results, otherwise it's None.
        """

        impl_lib = specs['impl_lib']
        impl_cell = specs['impl_cell']
        root_dir = Path(specs['root_dir'])

        if gen_cell and not load_results:
            print('generating cell ...')
            self.generate_cell(specs,
                               gen_lay=extract,
                               gen_sch=True,
                               run_lvs=extract,
                               run_rcx=extract,
                               use_cybagoa=True)
            print('cell generated.')

        # if testbench manager v2 found use that instead of interpreting simulation directly
        tbm_specs = specs.get('tbm_specs', None)
        if tbm_specs:
            tbm_cls_str = tbm_specs['tbm_cls']
            tbm_cls = _import_class_from_str(tbm_cls_str)
            tbm: TestbenchManager = tbm_cls(root_dir)
            sim_view_list = tbm_specs.get('sim_view_list', [])
            if not sim_view_list:
                # TODO: Is netlist always the right keyword?
                view_name = tbm_specs.get('view_name', 'netlist' if extract else 'schematic')
                sim_view_list.append((impl_cell, view_name))
            sim_envs = tbm_specs['sim_envs']

            if load_results:
                return tbm.load_results(impl_cell, tbm_specs)
            results = tbm.simulate(bprj=self,
                                   impl_lib=impl_lib,
                                   impl_cell=impl_cell,
                                   sim_view_list=sim_view_list,
                                   env_list=sim_envs,
                                   tb_dict=tbm_specs,
                                   wrapper_dict=None,
                                   gen_tb=gen_tb,
                                   gen_wrapper=gen_wrapper,
                                   run_sim=run_sim)
            return results

        sim_params = specs.get('sim_params', None)
        wrapper = sim_params.get('wrapper', None)

        has_wrapper = wrapper is not None
        if gen_wrapper and not has_wrapper:
            raise ValueError('must provide a wrapper in sim_params')

        wrapped_cell = ''
        if has_wrapper:
            wrapper_lib = wrapper['wrapper_lib']
            wrapper_cell = wrapper['wrapper_cell']
            wrapper_params = wrapper.get('params', {})
            wrapper_suffix = wrapper.get('wrapper_suffix', '')
            if not wrapper_suffix:
                wrapper_suffix = f'{wrapper_cell}'
            wrapped_cell = f'{impl_cell}_{wrapper_suffix}'

        if gen_wrapper and not gen_tb:
            raise ValueError('generated a new wrapper, therefore gen_tb should also be true')

        tb_lib = sim_params['tb_lib']
        tb_cell = sim_params['tb_cell']
        tb_params = sim_params.get('tb_params', {})
        tb_suffix = sim_params.get('tb_suffix', '')
        if not tb_suffix:
            tb_suffix = f'{tb_cell}'
        tb_name = f'{impl_cell}_{tb_suffix}'

        tb_fname = root_dir / Path(tb_name, f'{tb_name}.hdf5')

        if load_results:
            print("loading results ...")
            if tb_fname.exists():
                return sim_data.load_sim_file(tb_fname)
            raise ValueError(f'simulation results does not exist in {str(tb_fname)}')

        if gen_wrapper and has_wrapper:
            print('generating wrapper ...')
            # noinspection PyUnboundLocalVariable
            master = self.create_design_module(lib_name=wrapper_lib, cell_name=wrapper_cell)
            # noinspection PyUnboundLocalVariable
            master.design(dut_lib=impl_lib, dut_cell=impl_cell, **wrapper_params)
            master.implement_design(impl_lib, wrapped_cell)
            print('wrapper generated.')

        if gen_tb:
            print('generating testbench ...')
            tb_master = self.create_design_module(tb_lib, tb_cell)
            dut_cell = wrapped_cell if has_wrapper else impl_cell
            tb_master.design(dut_lib=impl_lib, dut_cell=dut_cell, **tb_params)
            tb_master.implement_design(impl_lib, tb_name)
            print('testbench generated.')

        if run_sim:
            print('seting up ADEXL ...')
            # TODO: when running simulations directly (not through tb_manager), sim_view_list is
            #  not supported
            # TODO: netlist might not be always the right keyword.
            # something like:
            # view_name = self.get_proper_view_name if extract else 'netlist'

            sim_view_list = sim_params.get('sim_view_list', [])
            if not sim_view_list:
                # TODO: Is netlist always the right keyword?
                view_name = sim_params.get('view_name', 'netlist' if extract else 'schematic')
                sim_view_list.append((impl_cell, view_name))

            sim_envs = sim_params['sim_envs']
            sim_swp_params = sim_params.get('sim_swp_params', {})
            sim_vars = sim_params.get('sim_vars', {})
            sim_outputs = sim_params.get('sim_outputs', {})

            tb = self.configure_testbench(impl_lib, tb_name)

            # set simulation variables
            for key, val in sim_vars.items():
                tb.set_parameter(key, val)

            # set sweep parameters
            for key, val in sim_swp_params.items():
                tb.set_sweep_parameter(key, values=val)

            # set the simulation outputs
            for key, val in sim_outputs.items():
                tb.add_output(key, val)

            # change the view_name (netlist or schematic)
            for cell, view in sim_view_list.items():
                tb.set_simulation_view(impl_lib, cell, view)

            tb.set_simulation_environments(sim_envs)
            tb.update_testbench()
            print('setup completed.')
            print('running simulation ...')
            tb.run_simulation()
            print('simulation done.')
            print('loading results ...')
            results = sim_data.load_sim_results(tb.save_dir)
            print('results loaded.')
            print('saving results into hdf5')
            sim_data.save_sim_results(results, tb_fname)
            print('results saved.')
            return results

    def measure_cell(self,
                     specs: Dict[str, Any],
                     gen_cell: bool = True,
                     gen_wrapper: bool = True,
                     gen_tb: bool = True,
                     load_results: bool = False,
                     extract: bool = False,
                     run_sims: bool = True) -> Optional[Dict[str, Any]]:
        """
        Runs a minimum executable parts of the Measurement Manager flow selectively according to
        a spec dictionary.

        For example you can set the flags to generate a new cell, but since wrapper and test bench
        exist, maybe you want to skip those, and run the measurement in the end. Maybe you
        already created the cell all the way up to test bench level, and now you only need to
        run simulation.

        This function only works with Measurement Managers written in format of
        simulation.core_v2.MeasurementManager

        Parameters
        ----------
        specs:
            Dictionary of specifications
            Some non-obvious conventions:
            - if contains tbm_specs keyword, simulation is ran through testbench manager v2,
            otherwise there should be a sim_params entry that specifies the simulation.
            - Wrapper is assumed to be in the specs dictionary, if it is generated outside of
            this function, gen_wrapper should be False.
        gen_cell:
            True to call generate_cell on specs
        gen_wrapper:
            True to generate Wrapper. Currently only one top-level wrapper is supported.
        gen_tb:
            True to generate test bench. If test bench is created, this flag can be set to False.
        load_results:
            True to skip simulation and load the results.
        extract:
            False to skip layout generation and only simulate schematic
        run_sims:
            True to run simulations. If the purpose of calling this function is just to generate
            some part of simulation flow to debug, this flag can be set to False.
        Returns
        -------
        results: Optional[Dict[str, Any]]
            if run_sim/load_results = True, contains measurement results, otherwise it's None.
        """

        impl_lib = specs['impl_lib']
        impl_cell = specs['impl_cell']
        root_dir = Path(specs['root_dir'])

        if gen_cell and not load_results:
            print('generating cell ...')
            self.generate_cell(specs,
                               gen_lay=extract,
                               gen_sch=True,
                               run_lvs=extract,
                               run_rcx=extract,
                               use_cybagoa=True)
            print('cell generated.')

        mm_specs = specs['mm_specs']
        mm_cls_str = mm_specs['mm_cls']
        mm_cls = _import_class_from_str(mm_cls_str)
        mm: MeasurementManager = mm_cls(root_dir, mm_specs)
        return mm.measure(self, impl_lib, impl_cell, load_results=load_results,
                          gen_wrapper=gen_wrapper, gen_tb=gen_tb, run_sims=run_sims)

    def create_library(self, lib_name, lib_path=''):
        # type: (str, str) -> None
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : str
            the library name.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.create_library(lib_name, lib_path=lib_path)

    # noinspection PyUnusedLocal
    def create_design_module(self, lib_name, cell_name, **kwargs):
        # type: (str, str, **Any) -> SchInstance
        """Create a new top level design module for the given schematic template

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        **kwargs : Any
            optional parameters.

        Returns
        -------
        dsn : SchInstance
            a configurable schematic instance of the given schematic generator.
        """
        return SchInstance(self.dsn_db, lib_name, cell_name, 'XTOP', static=False)

    def new_schematic_instance(self, lib_name='', cell_name='', params=None, sch_cls=None,
                               debug=False, **kwargs):
        # type: (str, str, Dict[str, Any], Type[ModuleType], bool, **Any) -> SchInstance
        """Create a new schematic instance

        This method is the schematic equivalent of TemplateDB's new_template() method.
        By default, we assume the design() function is used to set the schematic parameters.
        If you use another function (such as design_specs()), then you should specify
        an optional parameter design_fun equal to the name of that function.

        Parameters
        ----------
        lib_name : str
            schematic library name.
        cell_name : str
            schematic name
        params : Dict[str, Any]
            the parameter dictionary.
        sch_cls : Type[TemplateType]
            the schematic generator class to instantiate.
        debug : bool
            True to print debug messages.
        **kwargs : Any
            optional parameters.

        Returns
        -------
        dsn : SchInstance
            a schematic instance of the given schematic generator.
        """
        design_fun = kwargs.get('design_fun', 'design')
        master = self.dsn_db.new_master(lib_name, cell_name, gen_cls=sch_cls, params=params,
                                        debug=debug, design_args=None, design_fun=design_fun)

        return SchInstance(self.dsn_db, lib_name, cell_name, 'XTOP', static=False,
                           master=master)

    def clear_schematic_database(self):
        # type: () -> None
        """Reset schematic database."""
        self.dsn_db.clear()

    def instantiate_schematic(self, lib_name, content_list, lib_path=''):
        # type: (str, Sequence[Any], str) -> None
        """Create the given schematic contents in CAD database.

        NOTE: this is BAG's internal method.  TO create schematics, call batch_schematic() instead.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the schematic instances.
        content_list : Sequence[Any]
            list of schematics to create.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        self.impl_db.instantiate_schematic(lib_name, content_list, lib_path=lib_path)

    def batch_schematic(self,  # type: BagProject
                        lib_name,  # type: str
                        sch_inst_list,  # type: Sequence[SchInstance]
                        name_list=None,  # type: Optional[Sequence[Optional[str]]]
                        prefix='',  # type: str
                        suffix='',  # type: str
                        debug=False,  # type: bool
                        rename_dict=None,  # type: Optional[Dict[str, str]]
                        ):
        # type: (...) -> None
        """create all the given schematics in CAD database.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the schematic instances.
        sch_inst_list : Sequence[SchInstance]
            list of SchInstance objects.
        name_list : Optional[Sequence[Optional[str]]]
            list of master cell names.  If not given, default names will be used.
        prefix : str
            prefix to add to cell names.
        suffix : str
            suffix to add to cell names.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        """
        master_list = [inst.master for inst in sch_inst_list]

        self.dsn_db.cell_prefix = prefix
        self.dsn_db.cell_suffix = suffix
        self.dsn_db.instantiate_masters(master_list, name_list=name_list, lib_name=lib_name,
                                        debug=debug, rename_dict=rename_dict)

    def configure_testbench(self, tb_lib, tb_cell):
        # type: (str, str) -> Testbench
        """Update testbench state for the given testbench.

        This method fill in process-specific information for the given testbench, then returns
        a testbench object which you can use to control simulation.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.

        Returns
        -------
        tb : :class:`bag.core.Testbench`
            the :class:`~bag.core.Testbench` instance.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')
        if self.sim is None:
            raise Exception('SimAccess is not set up.')

        c, clist, params, outputs = self.impl_db.configure_testbench(tb_lib, tb_cell)
        return Testbench(self.sim, self.impl_db, tb_lib, tb_cell, params, clist, [c], outputs)

    def load_testbench(self, tb_lib, tb_cell):
        # type: (str, str) -> Testbench
        """Loads a testbench from the database.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.

        Returns
        -------
        tb : :class:`bag.core.Testbench`
            the :class:`~bag.core.Testbench` instance.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')
        if self.sim is None:
            raise Exception('SimAccess is not set up.')

        cur_envs, all_envs, params, outputs = self.impl_db.get_testbench_info(tb_lib, tb_cell)
        return Testbench(self.sim, self.impl_db, tb_lib, tb_cell, params, all_envs,
                         cur_envs, outputs)

    def instantiate_layout_pcell(self, lib_name, cell_name, inst_lib, inst_cell, params,
                                 pin_mapping=None, view_name='layout'):
        # type: (str, str, str, str, Dict[str, Any], Optional[Dict[str, str]], str) -> None
        """Create a layout cell with a single pcell instance.

        Parameters
        ----------
        lib_name : str
            layout library name.
        cell_name : str
            layout cell name.
        inst_lib : str
            pcell library name.
        inst_cell : str
            pcell cell name.
        params : Dict[str, Any]
            the parameter dictionary.
        pin_mapping: Optional[Dict[str, str]]
            the pin renaming dictionary.
        view_name : str
            layout view name, default is "layout".
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        pin_mapping = pin_mapping or {}
        self.impl_db.instantiate_layout_pcell(lib_name, cell_name, view_name,
                                              inst_lib, inst_cell, params, pin_mapping)

    def instantiate_layout(self, lib_name, view_name, via_tech, layout_list):
        # type: (str, str, str, Sequence[Any]) -> None
        """Create a batch of layouts.

        Parameters
        ----------
        lib_name : str
            layout library name.
        view_name : str
            layout view name.
        via_tech : str
            via technology name.
        layout_list : Sequence[Any]
            a list of layouts to create
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        self.impl_db.instantiate_layout(lib_name, view_name, via_tech, layout_list)

    def release_write_locks(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        self.impl_db.release_write_locks(lib_name, cell_view_list)

    def run_lvs(self,  # type: BagProject
                lib_name,  # type: str
                cell_name,  # type: str
                **kwargs
                ):
        # type: (...) -> Tuple[bool, str]
        """Run LVS on the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        coro = self.impl_db.async_run_lvs(lib_name, cell_name, **kwargs)
        results = batch_async_task([coro])
        if results is None or isinstance(results[0], Exception):
            return False, ''
        return results[0]

    def run_rcx(self,  # type: BagProject
                lib_name,  # type: str
                cell_name,  # type: str
                **kwargs
                ):
        # type: (...) -> Tuple[Union[bool, Optional[str]], str]
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If create_schematic is True, this method will run RCX, then if it succeeds,
        create a schematic of the extracted netlist in the database.  It then returns
        a boolean value which will be True if RCX succeeds.

        If create_schematic is False, this method will run RCX, then return a string
        which is the extracted netlist filename. If RCX failed, None will be returned
        instead.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
            override RCX parameter values.
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : Union[bool, str]
            The return value, as described.
        log_fname : str
            name of the RCX log file.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        create_schematic = kwargs.get('create_schematic', True)

        coro = self.impl_db.async_run_rcx(lib_name, cell_name, **kwargs)
        results = batch_async_task([coro])
        if results is None or isinstance(results[0], Exception):
            if create_schematic:
                return False, ''
            else:
                return None, ''
        return results[0]

    def export_layout(self, lib_name, cell_name, out_file, **kwargs):
        # type: (str, str, str, **Any) -> str
        """export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        log_fname : str
            log file name.  Empty if task cancelled.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        coro = self.impl_db.async_export_layout(lib_name, cell_name, out_file, **kwargs)
        results = batch_async_task([coro])
        if results is None or isinstance(results[0], Exception):
            return ''
        return results[0]

    def batch_export_layout(self, info_list):
        # type: (Sequence[Tuple[Any, ...]]) -> Optional[Sequence[str]]
        """Export layout of all given cells

        Parameters
        ----------
        info_list:
            list of cell information.  Each element is a tuple of:

            lib_name : str
                library name.
            cell_name : str
                cell name.
            out_file : str
                layout output file name.
            view_name : str
                layout view name.  Optional.
            params : Optional[Dict[str, Any]]
                optional export parameter values.

        Returns
        -------
        results : Optional[Sequence[str]]
            If task is cancelled, return None.  Otherwise, this is a
            list of log file names.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        coro_list = [self.impl_db.async_export_layout(*info) for info in info_list]
        temp_results = batch_async_task(coro_list)
        if temp_results is None:
            return None
        return ['' if isinstance(val, Exception) else val for val in temp_results]

    async def async_run_lvs(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """A coroutine for running LVS.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.
            LVS parameters should be specified as lvs_params.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return await self.impl_db.async_run_lvs(lib_name, cell_name, **kwargs)

    async def async_run_rcx(self,  # type: BagProject
                            lib_name: str,
                            cell_name: str,
                            **kwargs
                            ) -> Tuple[Union[bool, Optional[str]], str]:
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If create_schematic is True, this method will run RCX, then if it succeeds,
        create a schematic of the extracted netlist in the database.  It then returns
        a boolean value which will be True if RCX succeeds.

        If create_schematic is False, this method will run RCX, then return a string
        which is the extracted netlist filename. If RCX failed, None will be returned
        instead.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
            override RCX parameter values.
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : Union[bool, str]
            The return value, as described.
        log_fname : str
            name of the RCX log file.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return await self.impl_db.async_run_rcx(lib_name, cell_name, **kwargs)

    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view=None, **kwargs):
        # type: (str, str, str, Optional[str], **Any) -> None
        """Create a schematic from a netlist.

        This is mainly used to create extracted schematic from an extracted netlist.

        Parameters
        ----------
        netlist : str
            the netlist file name.
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : Optional[str]
            schematic view name.  The default value is implemendation dependent.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.create_schematic_from_netlist(netlist, lib_name, cell_name,
                                                          sch_view=sch_view, **kwargs)

    def create_verilog_view(self, verilog_file, lib_name, cell_name, **kwargs):
        # type: (str, str, str, **Any) -> None
        """Create a verilog view for mix-signal simulation.

        Parameters
        ----------
        verilog_file : str
            the verilog file name.
        lib_name : str
            library name.
        cell_name : str
            cell name.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        verilog_file = os.path.abspath(verilog_file)
        if not os.path.isfile(verilog_file):
            raise ValueError('%s is not a file.' % verilog_file)

        return self.impl_db.create_verilog_view(verilog_file, lib_name, cell_name, **kwargs)
