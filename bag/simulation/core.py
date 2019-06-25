# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, List, Iterable, Sequence

import abc
import importlib
import itertools
import os

import yaml

from bag import float_to_si_string
from bag.io import read_yaml, open_file, load_sim_results, save_sim_results, load_sim_file
from bag.layout import RoutingGrid, TemplateDB
from bag.concurrent.core import batch_async_task
from bag import BagProject

if TYPE_CHECKING:
    import numpy as np
    from bag.core import Testbench


class TestbenchManager(object, metaclass=abc.ABCMeta):
    """A class that creates and setups up a testbench for simulation, then save the result.

    This class is used by MeasurementManager to run simulations.

    Parameters
    ----------
    data_fname : str
        Simulation data file name.
    tb_name : str
        testbench name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        testbench specs.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    """
    def __init__(self,
                 data_fname,  # type: str
                 tb_name,  # type: str
                 impl_lib,  # type: str
                 specs,  # type: Dict[str, Any]
                 sim_view_list,  # type: Sequence[Tuple[str, str]]
                 env_list,  # type: Sequence[str]
                 ):
        # type: (...) -> None
        self.data_fname = os.path.abspath(data_fname)
        self.tb_name = tb_name
        self.impl_lib = impl_lib
        self.specs = specs
        self.sim_view_list = sim_view_list
        self.env_list = env_list

    @abc.abstractmethod
    def setup_testbench(self, tb):
        # type: (Testbench) -> None
        """Configure the simulation state of the given testbench.

        No need to call update_testbench(), set_simulation_environments(), and
        set_simulation_view().  These are called for you.

        Parameters
        ----------
        tb : Testbench
            the simulation Testbench instance.
        """
        pass

    async def setup_and_simulate(self, prj: BagProject,
                                 sch_params: Dict[str, Any]) -> Dict[str, Any]:
        if sch_params is None:
            print('loading testbench %s' % self.tb_name)
            tb = prj.load_testbench(self.impl_lib, self.tb_name)
        else:
            print('Creating testbench %s' % self.tb_name)
            tb = self._create_tb_schematic(prj, sch_params)

        print('Configuring testbench %s' % self.tb_name)
        tb.set_simulation_environments(self.env_list)
        self.setup_testbench(tb)
        for cell_name, view_name in self.sim_view_list:
            tb.set_simulation_view(self.impl_lib, cell_name, view_name)
        tb.update_testbench()

        # run simulation and save/return raw result
        print('Simulating %s' % self.tb_name)
        save_dir = await tb.async_run_simulation()
        print('Finished simulating %s' % self.tb_name)
        results = load_sim_results(save_dir)
        save_sim_results(results, self.data_fname)
        return results

    @classmethod
    def record_array(cls, output_dict, data_dict, arr, arr_name, sweep_params):
        # type: (Dict[str, Any], Dict[str, Any], np.ndarray, str, List[str]) -> None
        """Add the given numpy array into BAG's data structure dictionary.

        This method adds the given numpy array to output_dict, and make sure
        sweep parameter information are treated properly.

        Parameters
        ----------
        output_dict : Dict[str, Any]
            the output dictionary.
        data_dict : Dict[str, Any]
            the raw simulation data dictionary.
        arr : np.ndarray
            the numpy array to record.
        arr_name : str
            name of the given numpy array.
        sweep_params : List[str]
            a list of sweep parameters for thhe given array.
        """
        if 'sweep_params' in output_dict:
            swp_info = output_dict['sweep_params']
        else:
            swp_info = {}
            output_dict['sweep_params'] = swp_info

        # record sweep parameters information
        for var in sweep_params:
            if var not in output_dict:
                output_dict[var] = data_dict[var]
        swp_info[arr_name] = sweep_params
        output_dict[arr_name] = arr

    def _create_tb_schematic(self, prj, sch_params):
        # type: (BagProject, Dict[str, Any]) -> Testbench
        """Helper method to create a testbench schematic.

        Parmaeters
        ----------
        prj : BagProject
            the BagProject instance.
        sch_params : Dict[str, Any]
            the testbench schematic parameters dictionary.

        Returns
        -------
        tb : Testbench
            the simulation Testbench instance.
        """
        tb_lib = self.specs['tb_lib']
        tb_cell = self.specs['tb_cell']
        tb_sch = prj.create_design_module(tb_lib, tb_cell)
        tb_sch.design(**sch_params)
        tb_sch.implement_design(self.impl_lib, top_cell_name=self.tb_name)

        return prj.configure_testbench(self.impl_lib, self.tb_name)


class MeasurementManager(object, metaclass=abc.ABCMeta):
    """A class that handles circuit performance measurement.

    This class handles all the steps needed to measure a specific performance
    metric of the device-under-test.  This may involve creating and simulating
    multiple different testbenches, where configuration of successive testbenches
    depends on previous simulation results. This class reduces the potentially
    complex measurement tasks into a few simple abstract methods that designers
    simply have to implement.

    Parameters
    ----------
    data_dir : str
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    wrapper_lookup : Dict[str, str]
        the DUT wrapper cell name lookup table.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    """
    def __init__(self,  # type: MeasurementManager
                 data_dir,  # type: str
                 meas_name,  # type: str
                 impl_lib,  # type: str
                 specs,  # type: Dict[str, Any]
                 wrapper_lookup,  # type: Dict[str, str]
                 sim_view_list,  # type: Sequence[Tuple[str, str]]
                 env_list,  # type: Sequence[str]
                 ):
        # type: (...) -> None
        self.data_dir = os.path.abspath(data_dir)
        self.impl_lib = impl_lib
        self.meas_name = meas_name
        self.specs = specs
        self.wrapper_lookup = wrapper_lookup
        self.sim_view_list = sim_view_list
        self.env_list = env_list

        os.makedirs(self.data_dir, exist_ok=True)

    @abc.abstractmethod
    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return ''

    # noinspection PyUnusedLocal
    def get_testbench_info(self,  # type: MeasurementManager
                           state,  # type: str
                           prev_output,  # type: Optional[Dict[str, Any]]
                           ):
        # type: (...) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]
        """Get information about the next testbench.

        Override this method to perform more complex operations.

        Parameters
        ----------
        state : str
            the current FSM state.
        prev_output : Optional[Dict[str, Any]]
            the previous post-processing output.

        Returns
        -------
        tb_name : str
            cell name of the next testbench.  Should incorporate self.meas_name to avoid
            collision with testbench for other designs.
        tb_type : str
            the next testbench type.
        tb_specs : str
            the testbench specification dictionary.
        tb_params : Optional[Dict[str, Any]]
            the next testbench schematic parameters.  If we are reusing an existing
            testbench, this should be None.
        """
        tb_type = state
        tb_name = self.get_testbench_name(tb_type)
        tb_specs = self.get_testbench_specs(tb_type).copy()
        tb_params = self.get_default_tb_sch_params(tb_type)

        return tb_name, tb_type, tb_specs, tb_params

    @abc.abstractmethod
    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], TestbenchManager) -> Tuple[bool, str, Dict[str, Any]]
        """Process simulation output data.

        Parameters
        ----------
        state : str
            the current FSM state
        data : Dict[str, Any]
            simulation data dictionary.
        tb_manager : TestbenchManager
            the testbench manager object.

        Returns
        -------
        done : bool
            True if this measurement is finished.
        next_state : str
            the next FSM state.
        output : Dict[str, Any]
            a dictionary containing post-processed data.
        """
        return False, '', {}

    def get_testbench_name(self, tb_type):
        # type: (str) -> str
        """Returns a default testbench name given testbench type."""
        return '%s_TB_%s' % (self.meas_name, tb_type)

    async def async_measure_performance(self,
                                        prj: BagProject,
                                        load_from_file: bool = False) -> Dict[str, Any]:
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        prj : BagProject
            the BagProject instance.
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.

        Returns
        -------
        output : Dict[str, Any]
            the last dictionary returned by process_output().
        """
        cur_state = self.get_initial_state()
        prev_output = None
        done = False

        while not done:
            # create and setup testbench
            tb_name, tb_type, tb_specs, tb_sch_params = self.get_testbench_info(cur_state,
                                                                                prev_output)

            tb_package = tb_specs['tb_package']
            tb_cls_name = tb_specs['tb_class']
            tb_module = importlib.import_module(tb_package)
            tb_cls = getattr(tb_module, tb_cls_name)
            raw_data_fname = os.path.join(self.data_dir, '%s.hdf5' % cur_state)

            tb_manager = tb_cls(raw_data_fname, tb_name, self.impl_lib, tb_specs,
                                self.sim_view_list, self.env_list)

            if load_from_file:
                print('Measurement %s in state %s, '
                      'load sim data from file.' % (self.meas_name, cur_state))
                if os.path.isfile(raw_data_fname):
                    cur_results = load_sim_file(raw_data_fname)
                else:
                    print('Cannot find data file, simulating...')
                    cur_results = await tb_manager.setup_and_simulate(prj, tb_sch_params)
            else:
                cur_results = await tb_manager.setup_and_simulate(prj, tb_sch_params)

            # process and save simulation data
            print('Measurement %s in state %s, '
                  'processing data from %s' % (self.meas_name, cur_state, tb_name))
            done, next_state, prev_output = self.process_output(cur_state, cur_results, tb_manager)
            with open_file(os.path.join(self.data_dir, '%s.yaml' % cur_state), 'w') as f:
                yaml.dump(prev_output, f)

            cur_state = next_state

        return prev_output

    def get_state_output(self, state):
        # type: (str) -> Dict[str, Any]
        """Get the post-processed output of the given state."""
        with open_file(os.path.join(self.data_dir, '%s.yaml' % state), 'r') as f:
            return yaml.load(f)

    def get_testbench_specs(self, tb_type):
        # type: (str) -> Dict[str, Any]
        """Helper method to get testbench specifications."""
        return self.specs['testbenches'][tb_type]

    def get_default_tb_sch_params(self, tb_type):
        # type: (str) -> Dict[str, Any]
        """Helper method to return a default testbench schematic parameters dictionary.

        This method loads default values from specification file, the fill in dut_lib
        and dut_cell for you.

        Parameters
        ----------
        tb_type : str
            the testbench type.

        Returns
        -------
        sch_params : Dict[str, Any]
            the default schematic parameters dictionary.
        """
        tb_specs = self.get_testbench_specs(tb_type)
        wrapper_type = tb_specs['wrapper_type']

        if 'sch_params' in tb_specs:
            tb_params = tb_specs['sch_params'].copy()
        else:
            tb_params = {}

        tb_params['dut_lib'] = self.impl_lib
        tb_params['dut_cell'] = self.wrapper_lookup[wrapper_type]
        return tb_params


class DesignManager(object):
    """A class that manages instantiating design instances and running simulations.

    This class provides various methods to allow you to sweep design parameters
    and generate multiple instances at once.  It also provides methods for running
    simulations and helps you interface with TestbenchManager instances.

    Parameters
    ----------
    prj : Optional[BagProject]
        The BagProject instance.
    spec_file : str
        the specification file name or the data directory.
    """

    def __init__(self, prj, spec_file):
        # type: (Optional[BagProject], str) -> None
        self.prj = prj
        self._specs = None

        if os.path.isfile(spec_file):
            self._specs = read_yaml(spec_file)
            self._root_dir = os.path.abspath(self._specs['root_dir'])
        elif os.path.isdir(spec_file):
            self._root_dir = os.path.abspath(spec_file)
            self._specs = read_yaml(os.path.join(self._root_dir, 'specs.yaml'))
        else:
            raise ValueError('%s is neither data directory or specification file.' % spec_file)

        self._swp_var_list = tuple(sorted(self._specs['sweep_params'].keys()))

    @classmethod
    def load_state(cls, prj, root_dir):
        # type: (BagProject, str) -> DesignManager
        """Create the DesignManager instance corresponding to data in the given directory."""
        return cls(prj, root_dir)

    @classmethod
    def get_measurement_name(cls, dsn_name, meas_type):
        # type: (str, str) -> str
        """Returns the measurement name.

        Parameters
        ----------
        dsn_name : str
            design cell name.
        meas_type : str
            measurement type.

        Returns
        -------
        meas_name : str
            measurement name
        """
        return '%s_MEAS_%s' % (dsn_name, meas_type)

    @classmethod
    def get_wrapper_name(cls, dut_name, wrapper_name):
        # type: (str, str) -> str
        """Returns the wrapper cell name corresponding to the given DUT."""
        return '%s_WRAPPER_%s' % (dut_name, wrapper_name)

    @property
    def specs(self):
        # type: () -> Dict[str, Any]
        """Return the specification dictionary."""
        return self._specs

    @property
    def swp_var_list(self):
        # type: () -> Tuple[str, ...]
        return self._swp_var_list

    async def extract_design(self, lib_name: str, dsn_name: str,
                             rcx_params: Optional[Dict[str, Any]]) -> None:
        """A coroutine that runs LVS/RCX on a given design.

        Parameters
        ----------
        lib_name : str
            library name.
        dsn_name : str
            design cell name.
        rcx_params : Optional[Dict[str, Any]]
            extraction parameters dictionary.
        """
        print('Running LVS on %s' % dsn_name)
        lvs_passed, lvs_log = await self.prj.async_run_lvs(lib_name, dsn_name)
        if not lvs_passed:
            raise ValueError('LVS failed for %s.  Log file: %s' % (dsn_name, lvs_log))

        print('LVS passed on %s' % dsn_name)
        print('Running RCX on %s' % dsn_name)
        rcx_passed, rcx_log = await self.prj.async_run_rcx(lib_name, dsn_name,
                                                           rcx_params=rcx_params)
        if not rcx_passed:
            raise ValueError('RCX failed for %s.  Log file: %s' % (dsn_name, rcx_log))
        print('RCX passed on %s' % dsn_name)

    async def verify_design(self, lib_name: str, dsn_name: str,
                            load_from_file: bool = False) -> None:
        """Run all measurements on the given design.

        Parameters
        ----------
        lib_name : str
            library name.
        dsn_name : str
            design cell name.
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.
        """
        meas_list = self.specs['measurements']
        summary_fname = self.specs['summary_fname']
        view_name = self.specs['view_name']
        env_list = self.specs['env_list']
        wrapper_list = self.specs['dut_wrappers']

        wrapper_lookup = {'': dsn_name}
        for wrapper_config in wrapper_list:
            wrapper_type = wrapper_config['name']
            wrapper_lookup[wrapper_type] = self.get_wrapper_name(dsn_name, wrapper_type)

        result_summary = {}
        dsn_data_dir = os.path.join(self._root_dir, dsn_name)
        for meas_specs in meas_list:
            meas_type = meas_specs['meas_type']
            meas_package = meas_specs['meas_package']
            meas_cls_name = meas_specs['meas_class']
            out_fname = meas_specs['out_fname']
            meas_name = self.get_measurement_name(dsn_name, meas_type)
            data_dir = self.get_measurement_directory(dsn_name, meas_type)

            meas_module = importlib.import_module(meas_package)
            meas_cls = getattr(meas_module, meas_cls_name)

            meas_manager = meas_cls(data_dir, meas_name, lib_name, meas_specs,
                                    wrapper_lookup, [(dsn_name, view_name)], env_list)
            print('Performing measurement %s on %s' % (meas_name, dsn_name))
            meas_res = await meas_manager.async_measure_performance(self.prj,
                                                                    load_from_file=load_from_file)
            print('Measurement %s finished on %s' % (meas_name, dsn_name))

            with open_file(os.path.join(data_dir, out_fname), 'w') as f:
                yaml.dump(meas_res, f)
            result_summary[meas_type] = meas_res

        with open_file(os.path.join(dsn_data_dir, summary_fname), 'w') as f:
            yaml.dump(result_summary, f)

    async def main_task(self, lib_name: str, dsn_name: str,
                        rcx_params: Optional[Dict[str, Any]],
                        extract: bool = True,
                        measure: bool = True,
                        load_from_file: bool = False) -> None:
        """The main coroutine."""
        if extract:
            await self.extract_design(lib_name, dsn_name, rcx_params)
        if measure:
            await self.verify_design(lib_name, dsn_name, load_from_file=load_from_file)

    def characterize_designs(self, generate=True, measure=True, load_from_file=False):
        # type: (bool, bool, bool) -> None
        """Sweep all designs and characterize them.

        Parameters
        ----------
        generate : bool
            If True, create schematic/layout and run LVS/RCX.
        measure : bool
            If True, run all measurements.
        load_from_file : bool
            If True, measurements will load existing simulation data
            instead of running simulations.
        """
        if generate:
            extract = self.specs['view_name'] != 'schematic'
            self.create_designs(extract)
        else:
            extract = False

        rcx_params = self.specs.get('rcx_params', None)
        impl_lib = self.specs['impl_lib']
        dsn_name_list = [self.get_design_name(combo_list)
                         for combo_list in self.get_combinations_iter()]

        coro_list = [self.main_task(impl_lib, dsn_name, rcx_params, extract=extract,
                                    measure=measure, load_from_file=load_from_file)
                     for dsn_name in dsn_name_list]

        results = batch_async_task(coro_list)
        if results is not None:
            for val in results:
                if isinstance(val, Exception):
                    raise val

    def get_result(self, dsn_name):
        # type: (str) -> Dict[str, Any]
        """Returns the measurement result summary dictionary.

        Parameters
        ----------
        dsn_name : str
            the design name.

        Returns
        -------
        result : Dict[str, Any]
            the result dictionary.
        """
        fname = os.path.join(self._root_dir, dsn_name, self.specs['summary_fname'])
        with open_file(fname, 'r') as f:
            summary = yaml.load(f)

        return summary

    def test_layout(self, gen_sch=True):
        # type: (bool) -> None
        """Create a test schematic and layout for debugging purposes"""

        sweep_params = self.specs['sweep_params']
        dsn_name = self.specs['dsn_basename'] + '_TEST'

        val_list = tuple((sweep_params[key][0] for key in self.swp_var_list))
        lay_params = self.get_layout_params(val_list)

        temp_db = self.make_tdb()
        print('create test layout')
        sch_params_list = self.create_dut_layouts([lay_params], [dsn_name], temp_db)

        if gen_sch:
            print('create test schematic')
            self.create_dut_schematics(sch_params_list, [dsn_name], gen_wrappers=False)
        print('done')

    def create_designs(self, create_layout):
        # type: (bool) -> None
        """Create DUT schematics/layouts.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        temp_db = self.make_tdb()

        # make layouts
        dsn_name_list, lay_params_list, combo_list_list = [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_design_name(combo_list)
            lay_params = self.get_layout_params(combo_list)
            dsn_name_list.append(dsn_name)
            lay_params_list.append(lay_params)
            combo_list_list.append(combo_list)

        if create_layout:
            print('creating all layouts.')
            sch_params_list = self.create_dut_layouts(lay_params_list, dsn_name_list, temp_db)
        else:
            print('schematic simulation, skipping layouts.')
            sch_params_list = [self.get_schematic_params(combo_list)
                               for combo_list in self.get_combinations_iter()]

        print('creating all schematics.')
        self.create_dut_schematics(sch_params_list, dsn_name_list, gen_wrappers=True)

        print('design generation done.')

    def get_swp_var_values(self, var):
        # type: (str) -> List[Any]
        """Returns a list of valid sweep variable values.

        Parameter
        ---------
        var : str
            the sweep variable name.

        Returns
        -------
        val_list : List[Any]
            the sweep values of the given variable.
        """
        return self.specs['sweep_params'][var]

    def get_combinations_iter(self):
        # type: () -> Iterable[Tuple[Any, ...]]
        """Returns an iterator of schematic parameter combinations we sweep over.

        Returns
        -------
        combo_iter : Iterable[Tuple[Any, ...]]
            an iterator of tuples of schematic parameters values that we sweep over.
        """

        swp_par_dict = self.specs['sweep_params']
        return itertools.product(*(swp_par_dict[var] for var in self.swp_var_list))

    def get_dsn_name_iter(self):
        # type: () -> Iterable[str]
        """Returns an iterator over design names.

        Returns
        -------
        dsn_name_iter : Iterable[str]
            an iterator of design names.
        """
        return (self.get_design_name(combo_list) for combo_list in self.get_combinations_iter())

    def get_measurement_directory(self, dsn_name, meas_type):
        meas_name = self.get_measurement_name(dsn_name, meas_type)
        return os.path.join(self._root_dir, dsn_name, meas_name)

    def make_tdb(self):
        # type: () -> TemplateDB
        """Create and return a new TemplateDB object.

        Returns
        -------
        tdb : TemplateDB
            the TemplateDB object.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        target_lib = self.specs['impl_lib']
        grid_specs = self.specs['routing_grid']
        layers = grid_specs['layers']
        spaces = grid_specs['spaces']
        widths = grid_specs['widths']
        bot_dir = grid_specs['bot_dir']
        width_override = grid_specs.get('width_override', None)

        routing_grid = RoutingGrid(self.prj.tech_info, layers, spaces, widths, bot_dir, width_override=width_override)
        tdb = TemplateDB('', routing_grid, target_lib, use_cybagoa=True)
        return tdb

    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values."""
        lay_params = self.specs['layout_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            lay_params[var] = val

        return lay_params

    def get_schematic_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values."""
        lay_params = self.specs['schematic_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            lay_params[var] = val

        return lay_params

    def create_dut_schematics(self, sch_params_list, cell_name_list, gen_wrappers=True):
        # type: (Sequence[Dict[str, Any]], Sequence[str], bool) -> None
        dut_lib = self.specs['dut_lib']
        dut_cell = self.specs['dut_cell']
        impl_lib = self.specs['impl_lib']
        wrapper_list = self.specs['dut_wrappers']

        inst_list, name_list = [], []
        for sch_params, cur_name in zip(sch_params_list, cell_name_list):
            dsn = self.prj.create_design_module(dut_lib, dut_cell)
            dsn.design(**sch_params)
            inst_list.append(dsn)
            name_list.append(cur_name)
            if gen_wrappers:
                for wrapper_config in wrapper_list:
                    wrapper_name = wrapper_config['name']
                    wrapper_lib = wrapper_config['lib']
                    wrapper_cell = wrapper_config['cell']
                    wrapper_params = wrapper_config['params'].copy()
                    wrapper_params['dut_lib'] = impl_lib
                    wrapper_params['dut_cell'] = cur_name
                    dsn = self.prj.create_design_module(wrapper_lib, wrapper_cell)
                    dsn.design(**wrapper_params)
                    inst_list.append(dsn)
                    name_list.append(self.get_wrapper_name(cur_name, wrapper_name))

        self.prj.batch_schematic(impl_lib, inst_list, name_list=name_list)

    def create_dut_layouts(self, lay_params_list, cell_name_list, temp_db):
        # type: (Sequence[Dict[str, Any]], Sequence[str], TemplateDB) -> Sequence[Dict[str, Any]]
        """Create multiple layouts"""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        cls_package = self.specs['layout_package']
        cls_name = self.specs['layout_class']

        lay_module = importlib.import_module(cls_package)
        temp_cls = getattr(lay_module, cls_name)

        temp_list, sch_params_list = [], []
        for lay_params in lay_params_list:
            template = temp_db.new_template(params=lay_params, temp_cls=temp_cls, debug=False)
            temp_list.append(template)
            sch_params_list.append(template.sch_params)
        temp_db.batch_layout(self.prj, temp_list, cell_name_list)
        return sch_params_list

    def get_design_name(self, combo_list):
        # type: (Sequence[Any, ...]) -> str
        """Generate cell names based on sweep parameter values."""

        name_base = self.specs['dsn_basename']
        suffix = ''
        for var, val in zip(self.swp_var_list, combo_list):
            if isinstance(val, str):
                suffix += '_%s_%s' % (var, val)
            elif isinstance(val, int):
                suffix += '_%s_%d' % (var, val)
            elif isinstance(val, float):
                suffix += '_%s_%s' % (var, float_to_si_string(val))
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name_base + suffix
