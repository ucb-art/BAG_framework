from __future__ import annotations
from typing import (
    TYPE_CHECKING, Optional, Dict, Any, Tuple, List, Iterable, Sequence, Type, cast
)

import abc
import importlib
from pathlib import Path

from ..math import float_to_si_string
from ..io.file import Yaml
from ..io.sim_data import load_sim_results, save_sim_results, load_sim_file
from ..util.immutable import ImmutableList
from ..layout.template import TemplateDB, TemplateBase
from ..design.module import Module
from ..concurrent.core import batch_async_task
from ..interface.simulator import SimAccess

if TYPE_CHECKING:
    from ..core import BagProject
    from ..core import Testbench


class TestbenchManager(abc.ABC):
    """A class that creates and setups up a testbench for simulation, then save the result.

    This class is used by MeasurementManager to run simulations.

    Parameters
    ----------
    prj : BagProject
        BagProject object
    work_dir : Path
        working directory path.
    """

    def __init__(self, prj: BagProject, work_dir: Path) -> None:
        self._prj = prj
        self._work_dir = work_dir.resolve()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._specs = None

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def specs(self):
        return self._specs

    # noinspection PyMethodMayBeStatic
    def pre_setup(self, tb_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Override to perform any operations prior to calling the setup() function.

        Parameters
        ----------
        tb_params :
            the testbench schematic parameters.  None means the previous testbench will be reused.
            This dictionary should not be modified.

        Returns
        -------
        new_params :
            the schematic parameters to use.  Could be a modified copy of the original.
        """
        return tb_params

    def setup(self, impl_lib, impl_cell, sim_view_list, env_list,
              tb_dict, wrapper_dict=None, gen_tb=True) -> Testbench:
        tb_dict = self.pre_setup(tb_dict)
        self._specs = tb_dict
        prj = self._prj

        if wrapper_dict is None:
            wrapper_dict = tb_dict.pop('wrapper', None)
        has_wrapper = wrapper_dict is not None
        wrapped_cell = ''
        if has_wrapper:
            wrapper_lib = wrapper_dict['wrapper_lib']
            wrapper_cell = wrapper_dict['wrapper_cell']
            wrapper_params = wrapper_dict.get('params', {})
            wrapper_suffix = wrapper_dict.get('wrapper_suffix', '')
            if not wrapper_suffix:
                wrapper_suffix = f'{wrapper_cell}'
            wrapped_cell = f'{impl_cell}_{wrapper_suffix}'

        tb_lib = tb_dict['tb_lib']
        tb_cell = tb_dict['tb_cell']
        tb_params = tb_dict.get('tb_params', {})
        tb_suffix = tb_dict.get('tb_suffix', '')
        if not tb_suffix:
            tb_suffix = f'{tb_cell}'
        tb_name = f'{impl_cell}_{tb_suffix}'

        if has_wrapper:
            print('generating wrapper ...')
            # noinspection PyUnboundLocalVariable
            master = prj.create_design_module(lib_name=wrapper_lib, cell_name=wrapper_cell)
            # noinspection PyUnboundLocalVariable
            master.design(dut_lib=impl_lib, dut_cell=impl_cell, **wrapper_params)
            master.implement_design(impl_lib, wrapped_cell)
            print('wrapper generated.')

        if not gen_tb:
            print(f'loading testbench {tb_name}')
            tb = prj.load_testbench(impl_lib, tb_name)
        else:
            print(f'Creating testbench {tb_name}')
            tb_master = prj.create_design_module(tb_lib, tb_cell)
            dut_cell = wrapped_cell if has_wrapper else impl_cell
            tb_master.design(dut_lib=impl_lib, dut_cell=dut_cell, **tb_params)
            tb_master.implement_design(impl_lib, tb_name)
            print('testbench generated.')
            tb = prj.configure_testbench(impl_lib, tb_name)

        print(f'Configuring testbench {tb_name}')

        sim_swp_params = tb_dict.get('sim_swp_params', {})
        sim_vars = tb_dict.get('sim_vars', {})
        sim_outputs = tb_dict.get('sim_outputs', {})

        tb.set_simulation_environments(env_list)

        for cell_name, view_name in sim_view_list:
            tb.set_simulation_view(impl_lib, cell_name, view_name)

        for key, val in sim_vars.items():
            tb.set_parameter(key, val)

        for key, val in sim_swp_params.items():
            tb.set_sweep_parameter(key, values=val)

        for key, val in sim_outputs.items():
            tb.add_output(key, val)

        tb.update_testbench()
        return tb

    async def setup_and_simulate(self, impl_lib, impl_cell, sim_view_list, env_list, tb_dict,
                                 wrapper_dict, gen_tb):
        tb: Testbench = self.setup(impl_lib=impl_lib, impl_cell=impl_cell,
                                   sim_view_list=sim_view_list, env_list=env_list,
                                   tb_dict=tb_dict, wrapper_dict=wrapper_dict, gen_tb=gen_tb)
        print('Simulating %s' % tb.cell)
        save_dir = await tb.async_run_simulation()
        print('Finished simulating %s' % tb.cell)
        results = load_sim_results(save_dir)
        save_sim_results(results, str(self.work_dir / f'{tb.cell}_data.hdf5'))
        return results

    def simulate(self, impl_lib, impl_cell, sim_view_list, env_list, tb_dict,
                 wrapper_dict=None, gen_tb=True):
        coro = self.setup_and_simulate(impl_lib=impl_lib, impl_cell=impl_cell,
                                       sim_view_list=sim_view_list, env_list=env_list,
                                       tb_dict=tb_dict, wrapper_dict=wrapper_dict, gen_tb=gen_tb)
        results = batch_async_task([coro])
        for res in results:
            if isinstance(res, Exception):
                raise res
        return results

    def load_results(self, impl_cell, tb_dict):
        tb_cell = tb_dict['tb_cell']
        tb_suffix = tb_dict.get('tb_suffix', '')
        if not tb_suffix:
            tb_suffix = f'{tb_cell}'
        tb_name = f'{impl_cell}_{tb_suffix}'
        tb_fname = self.work_dir / f'{tb_name}_data.hdf5'
        if tb_fname.exists():
            return load_sim_file(str(tb_fname))
        raise ValueError(f'simulation results does not exist in {str(tb_fname)}')


class MeasurementManager(abc.ABC):
    """A class that handles circuit performance measurement.

    This class handles all the steps needed to measure a specific performance
    metric of the device-under-test.  This may involve creating and simulating
    multiple different testbenches, where configuration of successive testbenches
    depends on previous simulation results. This class reduces the potentially
    complex measurement tasks into a few simple abstract methods that designers
    simply have to implement.

    Parameters
    ----------
    sim : SimAccess
        the simulator interface object.
    dir_path : Path
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
    precision : int
        numeric precision in simulation netlist generation.
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        self._sim = sim
        self._dir_path = dir_path.resolve()
        self._meas_name = meas_name
        self._impl_lib = impl_lib
        self._specs = specs
        self._wrapper_lookup = wrapper_lookup
        self._sim_view_list = sim_view_list
        self._env_list = env_list
        self._precision = precision

        self._dir_path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return ''

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
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
        tb_specs : Dict[str, Any]
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
    def process_output(self, state: str, data: SimData, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process simulation output data.

        Parameters
        ----------
        state : str
            the current FSM state
        data : SimData
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

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @property
    def data_dir(self) -> Path:
        return self._dir_path

    @property
    def sim_envs(self) -> Sequence[str]:
        return self._env_list

    def get_testbench_name(self, tb_type: str) -> str:
        """Returns a default testbench name given testbench type."""
        return f'{self._meas_name}_TB_{tb_type}'

    async def async_measure_performance(self, sch_db: Optional[ModuleDB], dut_cvi_list: List[Any],
                                        dut_netlist: Optional[Path], load_from_file: bool = False,
                                        gen_sch: bool = True) -> Dict[str, Any]:
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        sch_db : Optional[ModuleDB]
            the schematic database.

            if load_from_file is True, this can be None. as it will not be used unless necessary.
        dut_cvi_list : List[str]
            cv_info for DUT cell netlist

            if load_from_file is True, this will not be used unless necessary.
        dut_netlist : Optional[Path]
            netlist of DUT cell

            if load_from_file is True, this will not be used unless necessary.
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.
        gen_sch : bool
            True to create testbench schematics.

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
            work_dir = self._dir_path / cur_state
            tb_manager: TestbenchManager = tb_cls(self._sim, work_dir, tb_name, self._impl_lib,
                                                  tb_specs, self._sim_view_list, self._env_list,
                                                  precision=self._precision)

            if load_from_file:
                print(f'Measurement {self._meas_name} in state {cur_state}, '
                      'load sim data from file.')
                try:
                    cur_results = tb_manager.load_sim_data()
                except FileNotFoundError:
                    print('Cannot find data file, simulating...')
                    if sch_db is None or not dut_cvi_list or dut_netlist is None:
                        raise ValueError('Cannot create testbench as DUT netlist not given.')

                    tb_manager.setup(sch_db, tb_sch_params, dut_cv_info_list=dut_cvi_list,
                                     dut_netlist=dut_netlist, gen_sch=gen_sch)
                    await tb_manager.async_simulate()
                    cur_results = tb_manager.load_sim_data()
            else:
                tb_manager.setup(sch_db, tb_sch_params, dut_cv_info_list=dut_cvi_list,
                                 dut_netlist=dut_netlist, gen_sch=gen_sch)
                await tb_manager.async_simulate()
                cur_results = tb_manager.load_sim_data()

            # process and save simulation data
            print(f'Measurement {self._meas_name} in state {cur_state}, '
                  f'processing data from {tb_type}')
            done, next_state, prev_output = self.process_output(cur_state, cur_results, tb_manager)
            write_yaml(self._dir_path / f'{cur_state}.yaml', prev_output)

            cur_state = next_state

        write_yaml(self._dir_path / f'{self._meas_name}.yaml', prev_output)
        return prev_output

    def measure_performance(self, sch_db: Optional[ModuleDB], dut_cvi_list: List[Any],
                            dut_netlist: Optional[Path], load_from_file: bool = False,
                            gen_sch: bool = True) -> Dict[str, Any]:
        coro = self.async_measure_performance(sch_db, dut_cvi_list, dut_netlist,
                                              load_from_file=load_from_file,
                                              gen_sch=gen_sch)
        return batch_async_task([coro])[0]

    def get_state_output(self, state: str) -> Dict[str, Any]:
        """Get the post-processed output of the given state."""
        return read_yaml(self._dir_path / f'{state}.yaml')

    def get_testbench_specs(self, tb_type: str) -> Dict[str, Any]:
        """Helper method to get testbench specifications."""
        return self._specs['testbenches'][tb_type]

    def get_default_tb_sch_params(self, tb_type: str) -> Dict[str, Any]:
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
        wrapper_type = tb_specs.get('wrapper_type', '')

        if 'sch_params' in tb_specs:
            tb_params = tb_specs['sch_params'].copy()
        else:
            tb_params = {}

        tb_params['dut_lib'] = self._impl_lib
        tb_params['dut_cell'] = self._wrapper_lookup[wrapper_type]
        return tb_params
