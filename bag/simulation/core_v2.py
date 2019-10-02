from __future__ import annotations
from typing import (
    TYPE_CHECKING, Optional, Dict, Any, Type, cast
)

import abc
from pathlib import Path

from ..io.sim_data import load_sim_results, save_sim_results, load_sim_file
from ..concurrent.core import batch_async_task
from ..core import _import_class_from_str
from ..util.immutable import to_immutable

if TYPE_CHECKING:
    from ..core import BagProject
    from ..core import Testbench
    from bag.util.immutable import ImmutableType


class TestbenchManager(abc.ABC):
    """A class that creates and setups up a testbench for simulation, then save the result.

    This class is used by MeasurementManager to run simulations.

    Parameters
    ----------
    work_dir : Path
        working directory path.
    """

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir.resolve()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._specs = None

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def specs(self):
        return self._specs

    @property
    def sim_vars(self):
        return self.specs.get('sim_vars', {})

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

    def setup(self, bprj, impl_lib, impl_cell, sim_view_list, env_list,
              tb_dict, wrapper_dict=None, gen_tb=True, gen_wrapper=True) -> Testbench:
        tb_dict = self.pre_setup(tb_dict)
        self._specs = tb_dict

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

        if has_wrapper and gen_wrapper:
            # noinspection PyUnboundLocalVariable
            print(f'Generating wrapper {impl_lib}_{wrapped_cell}')
            # noinspection PyUnboundLocalVariable
            master = bprj.create_design_module(lib_name=wrapper_lib, cell_name=wrapper_cell)
            # noinspection PyUnboundLocalVariable
            master.design(dut_lib=impl_lib, dut_cell=impl_cell, **wrapper_params)
            master.implement_design(impl_lib, wrapped_cell)
            print('wrapper generated.')

        if not gen_tb:
            print(f'loading testbench {impl_lib}_{tb_name}')
            tb = bprj.load_testbench(impl_lib, tb_name)
        else:
            print(f'Generating testbench {impl_cell}_{tb_name}')
            tb_master = bprj.create_design_module(tb_lib, tb_cell)
            dut_cell = wrapped_cell if has_wrapper else impl_cell
            tb_master.design(dut_lib=impl_lib, dut_cell=dut_cell, **tb_params)
            tb_master.implement_design(impl_lib, tb_name)
            print('testbench generated.')
            tb = bprj.configure_testbench(impl_lib, tb_name)

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

    async def setup_and_simulate(self, bprj, impl_lib, impl_cell, sim_view_list, env_list, tb_dict,
                                 wrapper_dict, gen_tb, gen_wrapper, run_sim):
        tb: Testbench = self.setup(bprj, impl_lib=impl_lib, impl_cell=impl_cell,
                                   sim_view_list=sim_view_list, env_list=env_list,
                                   tb_dict=tb_dict, wrapper_dict=wrapper_dict, gen_tb=gen_tb,
                                   gen_wrapper=gen_wrapper)
        if run_sim:
            print('Simulating %s' % tb.cell)
            save_dir = await tb.async_run_simulation()
            print('Finished simulating %s' % tb.cell)
            results = load_sim_results(save_dir)
            results_dir = str(self.work_dir / impl_cell / f'{tb.cell}_data.hdf5')
            save_sim_results(results, results_dir)
            return results

    def simulate(self, bprj, impl_lib, impl_cell, sim_view_list, env_list, tb_dict,
                 wrapper_dict=None, gen_tb=True, gen_wrapper=True, run_sim=True):
        coro = self.setup_and_simulate(bprj, impl_lib=impl_lib, impl_cell=impl_cell,
                                       sim_view_list=sim_view_list, env_list=env_list,
                                       tb_dict=tb_dict, wrapper_dict=wrapper_dict, gen_tb=gen_tb,
                                       gen_wrapper=gen_wrapper, run_sim=run_sim)
        results = batch_async_task([coro])[0]
        return results

    def load_results(self, impl_cell, tb_dict):
        self._specs = tb_dict
        tb_cell = tb_dict['tb_cell']
        tb_suffix = tb_dict.get('tb_suffix', '')
        if not tb_suffix:
            tb_suffix = f'{tb_cell}'
        tb_name = f'{impl_cell}_{tb_suffix}'
        tb_fname = self.work_dir / impl_cell / f'{tb_name}_data.hdf5'
        if tb_fname.exists():
            return load_sim_file(str(tb_fname))
        raise ValueError(f'simulation results does not exist in {str(tb_fname)}')


class MeasurementManager(abc.ABC):

    def __init__(self, work_dir: Path, mm_specs: Dict[str, Any]) -> None:
        self._work_dir = work_dir
        self._specs = mm_specs

        self.tb_managers: Dict[str, TestbenchManager] = {}
        self.tb_params: Dict[str, Dict[str, Any]] = {}
        self._wrapper_lookup: Dict[ImmutableType, Dict[str, Any]] = {}

        # fill up tb_managers and tb_params
        self._prepare_tb_specs()

        self.gen_wrapper: bool = True
        self.gen_tb: bool = True
        self.run_sims: bool = True

    @property
    def specs(self):
        return self._specs

    @property
    def work_dir(self):
        return self._work_dir

    def _prepare_tb_specs(self) -> None:
        # creates testbench manager objects and fills up the mappings
        testbenches = self.specs['testbenches']
        for tb_name, tb_dict in testbenches.items():
            tbm_cls = _import_class_from_str(tb_dict['tbm_cls'])
            tbm_cls = cast(Type[TestbenchManager], tbm_cls)
            self.tb_params[tb_name] = tb_dict
            self.tb_managers[tb_name] = tbm_cls(self._work_dir)

    def _prepare_tbm_dict(self, impl_cell, tbm_dict, extract):
        # adds sim_view_list and env_list to tbm_dict if they don't exist, so that after this
        # function there must be sim_view_list and sim_envs entries in tbm_dict
        if 'sim_view_list' not in tbm_dict:
            try:
                view_name = self.specs['view_name']
                tbm_dict['sim_view_list'] = [(impl_cell, view_name)]
            except KeyError:
                default_sim_view_list = self.specs.get('sim_view_list', [])
                if not default_sim_view_list:
                    view_name = self.specs.get('view_name', 'netlist' if extract else 'schematic')
                    default_sim_view_list.append((impl_cell, view_name))
                tbm_dict['sim_view_list'] = default_sim_view_list
        if 'sim_envs' not in tbm_dict:
            try:
                default_env_list = self.specs['sim_envs']
                tbm_dict['sim_envs'] = default_env_list
            except KeyError:
                raise ValueError('Did you forget to specify simulation environment?')

    def _wrapper_exists(self, wrapper: ImmutableType) -> bool:
        # checks if the wrapper (around impl_lib, impl_cell) has been created to avoid recreation
        return wrapper in self._wrapper_lookup

    def run_tb(self, bprj, impl_lib, impl_cell, tb_name, tbm_dict=None, extract=True,
               load_results=False):
        # if tb_dict is None the default tb_dict is used
        if tbm_dict is None:
            tbm_dict = self.tb_params[tb_name]
        tb_obj: TestbenchManager = self.tb_managers[tb_name]

        if load_results:
            return tb_obj.load_results(impl_cell, tbm_dict)

        wrapper = tbm_dict['wrapper']
        wrapper_key = to_immutable(wrapper)
        gen_wrapper = not self._wrapper_exists(wrapper_key)
        gen_wrapper = self.gen_wrapper and gen_wrapper

        # inherit default sim_envs and sim_view_list from self.specs
        self._prepare_tbm_dict(impl_cell, tbm_dict, extract)

        sim_view_list = tbm_dict['sim_view_list']
        sim_envs = tbm_dict['sim_envs']

        results = tb_obj.simulate(bprj, impl_lib, impl_cell, sim_view_list=sim_view_list,
                                  env_list=sim_envs, tb_dict=tbm_dict, gen_tb=self.gen_tb,
                                  gen_wrapper=gen_wrapper, run_sim=self.run_sims)

        if not gen_wrapper:
            self._wrapper_lookup[wrapper_key] = wrapper

        return results

    @abc.abstractmethod
    def run_flow(self, bprj: BagProject, impl_lib: str, impl_cell: str,
                 load_results: bool = False) -> Any:
        """
        Defines the FSM in code rather than passing state indicators through a dictionary
        use self.run_tb to orchestrate test benches and modify their parameters if necessary

        Don't call this method directly, call measure instead

        Parameters
        ----------
        bprj: BagProject
            BagProject object
        impl_lib:
            DUT implementation library
        impl_cell
            DUT implementation cell
        load_results:
            True to load results, this is used when debugging post processing functions

        Returns
        -------
            Any post processed result, even returning nothing is also an option
        """

        raise NotImplementedError

    def measure(self, bprj: BagProject, impl_lib: str, impl_cell: str, load_results: bool = False,
                gen_wrapper: bool = True, gen_tb: bool = True, run_sims: bool = True) -> Any:
        self.gen_wrapper = gen_wrapper
        self.gen_tb = gen_tb
        self.run_sims = run_sims

        return self.run_flow(bprj, impl_lib, impl_cell, load_results)
