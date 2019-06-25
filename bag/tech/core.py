# -*- coding: utf-8 -*-

"""This module contains commonly used technology related classes and functions.
"""

import os
import abc
import itertools
from typing import List, Union, Tuple, Dict, Any, Optional, Set

import numpy as np
import h5py
import openmdao.api as omdao

from bag.core import BagProject
from ..math.interpolate import interpolate_grid
from bag.math.dfun import VectorDiffFunction, DiffFunction
from ..mdao.core import GroupBuilder
from ..io import fix_string, to_bytes
from ..simulation.core import SimulationManager


def _equal(a, b, rtol, atol):
    """Returns True if a == b.  a and b are both strings, floats or numpy arrays."""
    # python 2/3 compatibility: convert raw bytes to string
    a = fix_string(a)
    b = fix_string(b)

    if isinstance(a, str):
        return a == b
    return np.allclose(a, b, rtol=rtol, atol=atol)


def _equal_list(a, b, rtol, atol):
    """Returns True if a == b.  a and b are list of strings/floats/numpy arrays."""
    if len(a) != len(b):
        return False
    for a_item, b_item in zip(a, b):
        if not _equal(a_item, b_item, rtol, atol):
            return False
    return True


def _index_in_list(item_list, item, rtol, atol):
    """Returns index of item in item_list, with tolerance checking for floats."""
    for idx, test in enumerate(item_list):
        if _equal(test, item, rtol, atol):
            return idx
    return -1


def _in_list(item_list, item, rtol, atol):
    """Returns True if item is in item_list, with tolerance checking for floats."""
    return _index_in_list(item_list, item, rtol, atol) >= 0


class CircuitCharacterization(SimulationManager, metaclass=abc.ABCMeta):
    """A class that handles characterization of a circuit.

    This class sweeps schematic parameters and run a testbench with a single analysis.
    It will then save the simulation data in a format CharDB understands.

    For now, this class will overwrite existing data, so please backup if you need to.

    Parameters
    ----------
    prj : BagProject
        the BagProject instance.
    spec_file : str
        the SimulationManager specification file.
    tb_type : str
        the testbench type name.  The parameter dictionary corresponding to this
        testbench should have the following entries (in addition to those required
        by Simulation Manager:

        outputs :
            list of testbench output names to save.
        constants :
            constant values used to identify this simulation run.
        sweep_params:
            a dictionary from testbench parameters to (start, stop, num_points)
            sweep tuple.

    compression : str
        HDF5 compression method.
    """

    def __init__(self, prj, spec_file, tb_type, compression='gzip'):
        super(CircuitCharacterization, self).__init__(prj, spec_file)
        self._compression = compression
        self._outputs = self.specs[tb_type]['outputs']
        self._constants = self.specs[tb_type]['constants']
        self._sweep_params = self.specs[tb_type]['sweep_params']

    def record_results(self, data, tb_type, val_list):
        # type: (Dict[str, Any], str, Tuple[Any, ...]) -> None
        """Record simulation results to file.

        Override implementation in SimulationManager in order to save data
        in a format that CharDB understands.
        """
        env_list = self.specs['sim_envs']

        tb_specs = self.specs[tb_type]
        results_dir = tb_specs['results_dir']

        os.makedirs(results_dir, exist_ok=True)
        fname = os.path.join(results_dir, 'data.hdf5')

        with h5py.File(fname, 'w') as f:
            for key, val in self._constants.items():
                f.attrs[key] = val
            for key, val in self._sweep_params.items():
                f.attrs[key] = val

            for env in env_list:
                env_result, sweep_list = self._get_env_result(data, env)

                grp = f.create_group('%d' % len(f))
                for key, val in zip(self.swp_var_list, val_list):
                    grp.attrs[key] = val
                # h5py workaround: explicitly store strings as encoded unicode data
                grp.attrs['env'] = to_bytes(env)
                grp.attrs['sweep_params'] = [to_bytes(swp) for swp in sweep_list]

                for name, val in env_result.items():
                    grp.create_dataset(name, data=val, compression=self._compression)

    def get_sim_results(self, tb_type, val_list):
        # type: (str, Tuple[Any, ...]) -> Dict[str, Any]
        # TODO: implement this.
        raise NotImplementedError('not implemented yet.')

    def _get_env_result(self, sim_results, env):
        """Extract results from a given simulation environment from the given data.

        all output sweep parameter order and data shape must be the same.

        Parameters
        ----------
        sim_results : dict[string, any]
            the simulation results dictionary
        env : str
            the target simulation environment

        Returns
        -------
        results : dict[str, any]
            the results from a given simulation environment.
        sweep_list : list[str]
            a list of sweep parameter order.
        """
        if 'corner' not in sim_results:
            # no corner sweep anyways
            results = {output: sim_results[output] for output in self._outputs}
            sweep_list = sim_results['sweep_params'][self._outputs[0]]
            return results, sweep_list

        corner_list = sim_results['corner'].tolist()
        results = {}
        # we know all sweep order and shape is the same.
        test_name = self._outputs[0]
        sweep_list = list(sim_results['sweep_params'][test_name])
        shape = sim_results[test_name].shape
        # make numpy array slice index list
        index_list = [slice(0, l) for l in shape]
        if 'corner' in sweep_list:
            idx = sweep_list.index('corner')
            index_list[idx] = corner_list.index(env)
            del sweep_list[idx]

        # store outputs in results
        for output in self._outputs:
            results[output] = sim_results[output][index_list]

        return results, sweep_list


class CharDB(abc.ABC):
    """The abstract base class of a database of characterization data.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    root_dir : str
        path to the root characterization data directory.  Supports environment variables.
    constants : Dict[str, Any]
        constants dictionary.
    discrete_params : List[str]
        a list of parameters that should take on discrete values.
    init_params : Dict[str, Any]
        a dictionary of initial parameter values.  All parameters should be specified,
        and None should be used if the parameter value is not set.
    env_list : List[str]
        list of simulation environments to consider.
    update : bool
        By default, CharDB saves and load post-processed data directly.  If update is True,
        CharDB will update the post-process data from raw simulation data. Defaults to
        False.
    rtol : float
        relative tolerance used to compare constants/sweep parameters/sweep attributes.
    atol : float
        relative tolerance used to compare constants/sweep parameters/sweep attributes.
    compression : str
        HDF5 compression method.  Used only during post-processing.
    method : str
        interpolation method.
    opt_package : str
        default Python optimization package.  Supports 'scipy' or 'pyoptsparse'.  Defaults
        to 'scipy'.
    opt_method : str
        default optimization method.  Valid values depends on the optimization package.
        Defaults to 'SLSQP'.
    opt_settings : Optional[Dict[str, Any]]
        optimizer specific settings.
    """

    def __init__(self,  # type: CharDB
                 root_dir,  # type: str
                 constants,  # type: Dict[str, Any]
                 discrete_params,  # type: List[str]
                 init_params,  # type: Dict[str, Any]
                 env_list,  # type: List[str]
                 update=False,  # type: bool
                 rtol=1e-5,  # type: float
                 atol=1e-18,  # type: float
                 compression='gzip',  # type: str
                 method='spline',  # type: str
                 opt_package='scipy',  # type: str
                 opt_method='SLSQP',  # type: str
                 opt_settings=None,  # type: Optional[Dict[str, Any]]
                 **kwargs
                 ):
        # type: (...) -> None

        root_dir = os.path.abspath(os.path.expandvars(root_dir))

        if not os.path.isdir(root_dir):
            # error checking
            raise ValueError('Directory %s not found.' % root_dir)
        if 'env' in discrete_params:
            discrete_params.remove('env')

        if opt_settings is None:
            opt_settings = {}
        else:
            pass

        if opt_method == 'IPOPT' and not opt_settings:
            # set default IPOPT settings
            opt_settings['option_file_name'] = ''

        self._discrete_params = discrete_params
        self._params = init_params.copy()
        self._env_list = env_list
        self._config = dict(opt_package=opt_package,
                            opt_method=opt_method,
                            opt_settings=opt_settings,
                            rtol=rtol,
                            atol=atol,
                            method=method,
                            )

        cache_fname = self.get_cache_file(root_dir, constants)
        if not os.path.isfile(cache_fname) or update:
            sim_fname = self.get_sim_file(root_dir, constants)
            results = self._load_sim_data(sim_fname, constants, discrete_params)
            sim_data, total_params, total_values, self._constants = results
            self._data = self.post_process_data(sim_data, total_params, total_values, self._constants)

            # save to cache
            with h5py.File(cache_fname, 'w') as f:
                for key, val in self._constants.items():
                    f.attrs[key] = val
                sp_grp = f.create_group('sweep_params')
                # h5py workaround: explicitly store strings as encoded unicode data
                sp_grp.attrs['sweep_order'] = [to_bytes(swp) for swp in total_params]
                for par, val_list in zip(total_params, total_values):
                    if val_list.dtype.kind == 'U':
                        # unicode array, convert to raw bytes array
                        val_list = val_list.astype('S')
                    sp_grp.create_dataset(par, data=val_list, compression=compression)
                data_grp = f.create_group('data')
                for name, data_arr in self._data.items():
                    data_grp.create_dataset(name, data=data_arr, compression=compression)
        else:
            # load from cache
            with h5py.File(cache_fname, 'r') as f:
                self._constants = dict(iter(f.attrs.items()))
                sp_grp = f['sweep_params']
                total_params = [fix_string(swp) for swp in sp_grp.attrs['sweep_order']]
                total_values = [self._convert_hdf5_array(sp_grp[par][()]) for par in total_params]
                data_grp = f['data']
                self._data = {name: data_grp[name][()] for name in data_grp}

        # change axes location so discrete parameters are at the start of sweep_params
        env_disc_params = ['env'] + discrete_params
        for idx, dpar in enumerate(env_disc_params):
            if total_params[idx] != dpar:
                # swap
                didx = total_params.index(dpar)
                ptmp = total_params[idx]
                vtmp = total_values[idx]
                total_params[idx] = total_params[didx]
                total_values[idx] = total_values[didx]
                total_params[didx] = ptmp
                total_values[didx] = vtmp
                for key, val in self._data.items():
                    self._data[key] = np.swapaxes(val, idx, didx)

        sidx = len(self._discrete_params) + 1
        self._cont_params = total_params[sidx:]
        self._cont_values = total_values[sidx:]
        self._discrete_values = total_values[1:sidx]
        self._env_values = total_values[0]

        # get lazy function table.
        shape = [total_values[idx].size for idx in range(len(env_disc_params))]

        fun_name_iter = itertools.chain(iter(self._data.keys()), self.derived_parameters())
        # noinspection PyTypeChecker
        self._fun = {name: np.full(shape, None, dtype=object) for name in fun_name_iter}

    @staticmethod
    def _convert_hdf5_array(arr):
        # type: (np.ndarray) -> np.ndarray
        """Check if raw bytes array, if so convert to unicode array."""
        if arr.dtype.kind == 'S':
            return arr.astype('U')
        return arr

    def _load_sim_data(self,  # type: CharDB
                       fname,  # type: str
                       constants,  # type: Dict[str, Any]
                       discrete_params  # type: List[str]
                       ):
        # type: (...) -> Tuple[Dict[str, np.ndarray], List[str], List[np.ndarray], Dict[str, Any]]
        """Returns the simulation data.

        Parameters
        ----------
        fname : str
            the simulation filename.
        constants : Dict[str, Any]
            the constants dictionary.
        discrete_params : List[str]
            a list of parameters that should take on discrete values.

        Returns
        -------
        data_dict : Dict[str, np.ndarray]
            a dictionary from output name to data as numpy array.
        master_attrs : List[str]
            list of attribute name for each dimension of numpy array.
        master_values : List[np.ndarray]
            list of attribute values for each dimension.
        file_constants : Dict[str, Any]
            the constants dictionary in file.
        """
        if not os.path.exists(fname):
            raise ValueError('Simulation file %s not found.' % fname)

        rtol, atol = self.get_config('rtol'), self.get_config('atol')  # type: float

        master_attrs = None
        master_values = None
        master_dict = None
        file_constants = None
        with h5py.File(fname, 'r') as f:
            # check constants is consistent
            for key, val in constants.items():
                if not _equal(val, f.attrs[key], rtol, atol):
                    raise ValueError('sim file attr %s = %s != %s' % (key, f.attrs[key], val))

            # simple error checking.
            if len(f) == 0:
                raise ValueError('simulation file has no data.')

            # check that attributes sweep forms regular grid.
            attr_table = {}
            for gname in f:
                grp = f[gname]
                for key, val in grp.attrs.items():
                    # convert raw bytes to unicode
                    # python 2/3 compatibility: convert raw bytes to string
                    val = fix_string(val)

                    if key != 'sweep_params':
                        if key not in attr_table:
                            attr_table[key] = []
                        val_list = attr_table[key]
                        if not _in_list(val_list, val, rtol, atol):
                            val_list.append(val)

            expected_len = 1
            for val in attr_table.values():
                expected_len *= len(val)

            if expected_len != len(f):
                raise ValueError('Attributes of f does not form complete sweep. '
                                 'Expect length = %d, but actually = %d.' % (expected_len, len(f)))

            # check all discrete parameters in attribute table.
            for disc_par in discrete_params:
                if disc_par not in attr_table:
                    raise ValueError('Discrete attribute %s not found' % disc_par)

            # get attribute order
            attr_order = sorted(attr_table.keys())
            # check all non-discrete attribute value list lies on regular grid
            attr_values = [np.array(sorted(attr_table[attr])) for attr in attr_order]
            for attr, aval_list in zip(attr_order, attr_values):
                if attr not in discrete_params and attr != 'env':
                    test_vec = np.linspace(aval_list[0], aval_list[-1], len(aval_list), endpoint=True)
                    if not np.allclose(test_vec, aval_list, rtol=rtol, atol=atol):
                        raise ValueError('Attribute %s values do not lie on regular grid' % attr)

            # consolidate all data into one giant numpy array.
            # first compute numpy array shape
            test_grp = f['0']
            sweep_params = [fix_string(tmpvar) for tmpvar in test_grp.attrs['sweep_params']]

            # get constants dictionary
            file_constants = {}
            for key, val in f.attrs.items():
                if key not in sweep_params:
                    file_constants[key] = val

            master_attrs = attr_order + sweep_params
            swp_values = [np.linspace(f.attrs[var][0], f.attrs[var][1], f.attrs[var][2],
                                      endpoint=True) for var in sweep_params]  # type: List[np.array]
            master_values = attr_values + swp_values
            master_shape = [len(val_list) for val_list in master_values]
            master_index = [slice(0, n) for n in master_shape]
            master_dict = {}
            for gname in f:
                grp = f[gname]
                # get index of the current group in the giant array.
                # Note: using linear search to compute index now, but attr_val_list should be small.
                for aidx, (attr, aval_list) in enumerate(zip(attr_order, attr_values)):
                    master_index[aidx] = _index_in_list(aval_list, grp.attrs[attr], rtol, atol)

                for output in grp:
                    dset = grp[output]
                    if output not in master_dict:
                        master_dict[output] = np.empty(master_shape, dtype=dset.dtype)
                    master_dict[output][master_index] = dset

        return master_dict, master_attrs, master_values, file_constants

    def __getitem__(self, param):
        # type: (str) -> Any
        """Returns the given parameter value.

        Parameters
        ----------
        param : str
            parameter name.

        Returns
        -------
        val : Any
            parameter value.
        """
        return self._params[param]

    def __setitem__(self, key, value):
        # type: (str, Any) -> None
        """Sets the given parameter value.

        Parameters
        ----------
        key : str
            parameter name.
        value : Any
            parameter value.  None to unset.
        """
        rtol, atol = self.get_config('rtol'), self.get_config('atol')

        if key in self._discrete_params:
            if value is not None:
                idx = self._discrete_params.index(key)
                if not _in_list(self._discrete_values[idx], value, rtol, atol):
                    raise ValueError('Cannot set discrete variable %s value to %s' % (key, value))
        elif key in self._cont_params:
            if value is not None:
                idx = self._cont_params.index(key)
                val_list = self._cont_values[idx]
                if value < val_list[0] or value > val_list[-1]:
                    raise ValueError('Variable %s value %s out of bounds.' % (key, value))
        else:
            raise ValueError('Unknown variable %s.' % key)

        self._params[key] = value

    def get_config(self, name):
        # type: (str) -> Any
        """Returns the configuration value.

        Parameters
        ----------
        name : str
            configuration name.

        Returns
        -------
        val : Any
            configuration value.
        """
        return self._config[name]

    def set_config(self, name, value):
        # type: (str, Any) -> None
        """Sets the configuration value.

        Parameters
        ----------
        name : str
            configuration name.
        value : Any
            configuration value.
        """
        if name not in self._config:
            raise ValueError('Unknown configuration %s' % name)
        self._config[name] = value

    @property
    def env_list(self):
        # type: () -> List[str]
        """The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list):
        # type: (List[str]) -> None
        """Sets the list of simulation environments to consider."""
        self._env_list = new_env_list

    @classmethod
    def get_sim_file(cls, root_dir, constants):
        # type: (str, Dict[str, Any]) -> str
        """Returns the simulation data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : Dict[str, Any]
            constants dictionary.

        Returns
        -------
        fname : str
            the simulation data file name.
        """
        raise NotImplementedError('Not implemented')

    @classmethod
    def get_cache_file(cls, root_dir, constants):
        # type: (str, Dict[str, Any]) -> str
        """Returns the post-processed characterization data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : Dict[str, Any]
            constants dictionary.

        Returns
        -------
        fname : str
            the post-processed characterization data file name.
        """
        raise NotImplementedError('Not implemented')

    @classmethod
    def post_process_data(cls, sim_data, sweep_params, sweep_values, constants):
        # type: (Dict[str, np.ndarray], List[str], List[np.ndarray], Dict[str, Any]) -> Dict[str, np.ndarray]
        """Postprocess simulation data.

        Parameters
        ----------
        sim_data : Dict[str, np.ndarray]
            the simulation data as a dictionary from output name to numpy array.
        sweep_params : List[str]
            list of parameter name for each dimension of numpy array.
        sweep_values : List[np.ndarray]
            list of parameter values for each dimension.
        constants : Dict[str, Any]
            the constants dictionary.

        Returns
        -------
        data : Dict[str, np.ndarray]
            a dictionary of post-processed data.
        """
        raise NotImplementedError('Not implemented')

    @classmethod
    def derived_parameters(cls):
        # type: () -> List[str]
        """Returns a list of derived parameters."""
        return []

    @classmethod
    def compute_derived_parameters(cls, fdict):
        # type: (Dict[str, DiffFunction]) -> Dict[str, DiffFunction]
        """Compute derived parameter functions.

        Parameters
        ----------
        fdict : Dict[str, DiffFunction]
            a dictionary from core parameter name to the corresponding function.

        Returns
        -------
        deriv_dict : Dict[str, DiffFunction]
            a dictionary from derived parameter name to the corresponding function.
        """
        return {}

    def _get_function_index(self, **kwargs):
        # type: (Any) -> List[int]
        """Returns the function index corresponding to given discrete parameter values.

        simulation environment index will be set to 0

        Parameters
        ----------
        **kwargs :
            discrete parameter values.

        Returns
        -------
        fidx_list : List[int]
            the function index.
        """
        rtol, atol = self.get_config('rtol'), self.get_config('atol')

        fidx_list = [0]
        for par, val_list in zip(self._discrete_params, self._discrete_values):
            val = kwargs.get(par, self[par])
            if val is None:
                raise ValueError('Parameter %s value not specified' % par)

            val_idx = _index_in_list(val_list, val, rtol, atol)
            if val_idx < 0:
                raise ValueError('Discrete parameter %s have illegal value %s' % (par, val))
            fidx_list.append(val_idx)

        return fidx_list

    def _get_function_helper(self, name, fidx_list):
        # type: (str, Union[List[int], Tuple[int]]) -> DiffFunction
        """Helper method for get_function()

        Parameters
        ----------
        name : str
            name of the function.
        fidx_list : Union[List[int], Tuple[int]]
            function index.

        Returns
        -------
        fun : DiffFunction
            the interpolator function.
        """
        # get function table index
        fidx_list = tuple(fidx_list)
        ftable = self._fun[name]
        if ftable[fidx_list] is None:
            if name in self._data:
                # core parameter
                char_data = self._data[name]

                # get scale list and data index
                scale_list = []
                didx = list(fidx_list)  # type: List[Union[int, slice]]
                for vec in self._cont_values:
                    scale_list.append((vec[0], vec[1] - vec[0]))
                    didx.append(slice(0, vec.size))

                # make interpolator.
                cur_data = char_data[didx]
                method = self.get_config('method')
                ftable[fidx_list] = interpolate_grid(scale_list, cur_data, method=method, extrapolate=True)
            else:
                # derived parameter
                core_fdict = {fn: self._get_function_helper(fn, fidx_list) for fn in self._data}
                deriv_fdict = self.compute_derived_parameters(core_fdict)
                for fn, deriv_fun in deriv_fdict.items():
                    self._fun[fn][fidx_list] = deriv_fun

        return ftable[fidx_list]

    def get_function(self, name, env='', **kwargs):
        # type: (str, str, **Any) -> Union[VectorDiffFunction, DiffFunction]
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.
        **kwargs : Any
            dictionary of discrete parameter values.

        Returns
        -------
        output : Union[VectorDiffFunction, DiffFunction]
            the output vector function.
        """
        fidx_list = self._get_function_index(**kwargs)
        if not env:
            fun_list = []
            for env in self.env_list:
                occur_list = np.where(self._env_values == env)[0]
                if occur_list.size == 0:
                    raise ValueError('environment %s not found.')
                env_idx = occur_list[0]
                fidx_list[0] = env_idx
                fun_list.append(self._get_function_helper(name, fidx_list))
            return VectorDiffFunction(fun_list)
        else:
            occur_list = np.where(self._env_values == env)[0]
            if occur_list.size == 0:
                raise ValueError('environment %s not found.')
            env_idx = occur_list[0]
            fidx_list[0] = env_idx
            return self._get_function_helper(name, fidx_list)

    def get_fun_sweep_params(self):
        # type: () -> Tuple[List[str], List[Tuple[float, float]]]
        """Returns interpolation function sweep parameter names and values.

        Returns
        -------
        sweep_params : List[str]
            list of parameter names.
        sweep_range : List[Tuple[float, float]]
            list of parameter range
        """
        return self._cont_params, [(vec[0], vec[-1]) for vec in self._cont_values]

    def _get_fun_arg(self, **kwargs):
        # type: (Any) -> np.ndarray
        """Make numpy array of interpolation function arguments."""
        val_list = []
        for par in self._cont_params:
            val = kwargs.get(par, self[par])
            if val is None:
                raise ValueError('Parameter %s value not specified.' % par)
            val_list.append(val)

        return np.array(val_list)

    def query(self, **kwargs):
        # type: (Any) -> Dict[str, np.ndarray]
        """Query the database for the values associated with the given parameters.

        All parameters must be specified.

        Parameters
        ----------
        **kwargs :
            parameter values.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        results = {}
        arg = self._get_fun_arg(**kwargs)
        for name in self._data:
            fun = self.get_function(name, **kwargs)
            results[name] = fun(arg)

        for var in itertools.chain(self._discrete_params, self._cont_params):
            results[var] = kwargs.get(var, self[var])

        results.update(self.compute_derived_parameters(results))

        return results

    def minimize(self,  # type: CharDB
                 objective,  # type: str
                 define=None,  # type: List[Tuple[str, int]]
                 cons=None,  # type: Dict[str, Dict[str, float]]
                 vector_params=None,  # type: Set[str]
                 debug=False,  # type: bool
                 **kwargs
                 ):
        # type: (...) -> Dict[str, Union[np.ndarray, float]]
        """Find operating point that minimizes the given objective.

        Parameters
        ----------
        objective : str
            the objective to minimize.  Must be a scalar.
        define : List[Tuple[str, int]]
            list of expressions to define new variables.  Each
            element of the list is a tuple of string and integer.  The string
            contains a python assignment that computes the variable from
            existing ones, and the integer indicates the variable shape.

            Note that define can also be used to enforce relationships between
            existing variables.  Using transistor as an example, defining
            'vgs = vds' will force the vgs of vds of the transistor to be
            equal.
        cons : Dict[str, Dict[str, float]]
            a dictionary from variable name to constraints of that variable.
            see OpenMDAO documentations for details on constraints.
        vector_params : Set[str]
            set of input variables that are vector instead of scalar.  An input
            variable is a vector if it can change across simulation environments.
        debug : bool
            True to enable debugging messages.  Defaults to False.
        **kwargs :
            known parameter values.

        Returns
        -------
        results : Dict[str, Union[np.ndarray, float]]
            the results dictionary.
        """
        cons = cons or {}
        fidx_list = self._get_function_index(**kwargs)
        builder = GroupBuilder()

        params_ranges = dict(zip(self._cont_params,
                                 ((vec[0], vec[-1]) for vec in self._cont_values)))
        # add functions
        fun_name_iter = itertools.chain(iter(self._data.keys()), self.derived_parameters())
        for name in fun_name_iter:
            fun_list = []
            for idx, env in enumerate(self.env_list):
                fidx_list[0] = idx
                fun_list.append(self._get_function_helper(name, fidx_list))

            builder.add_fun(name, fun_list, self._cont_params, params_ranges,
                            vector_params=vector_params)

        # add expressions
        for expr, ndim in define:
            builder.add_expr(expr, ndim)

        # update input bounds from constraints
        input_set = builder.get_inputs()
        var_list = builder.get_variables()

        for name in input_set:
            if name in cons:
                setup = cons[name]
                if 'equals' in setup:
                    eq_val = setup['equals']
                    builder.set_input_limit(name, equals=eq_val)
                else:
                    vmin = vmax = None
                    if 'lower' in setup:
                        vmin = setup['lower']
                    if 'upper' in setup:
                        vmax = setup['upper']
                    builder.set_input_limit(name, lower=vmin, upper=vmax)

        # build the group and make the problem
        grp, input_bounds = builder.build()

        top = omdao.Problem()
        top.root = grp

        opt_package = self.get_config('opt_package')  # type: str
        opt_settings = self.get_config('opt_settings')

        if opt_package == 'scipy':
            driver = top.driver = omdao.ScipyOptimizer()
            print_opt_name = 'disp'
        elif opt_package == 'pyoptsparse':
            driver = top.driver = omdao.pyOptSparseDriver()
            print_opt_name = 'print_results'
        else:
            raise ValueError('Unknown optimization package: %s' % opt_package)

        driver.options['optimizer'] = self.get_config('opt_method')
        driver.options[print_opt_name] = debug
        driver.opt_settings.update(opt_settings)

        # add constraints
        constants = {}
        for name, setup in cons.items():
            if name not in input_bounds:
                # add constraint
                driver.add_constraint(name, **setup)

        # add inputs
        for name in input_set:
            eq_val, lower, upper, ndim = input_bounds[name]
            val = kwargs.get(name, self[name])  # type: float
            if val is None:
                val = eq_val
            comp_name = 'comp__%s' % name
            if val is not None:
                val = np.atleast_1d(np.ones(ndim) * val)
                constants[name] = val
                top.root.add(comp_name, omdao.IndepVarComp(name, val=val), promotes=[name])
            else:
                avg = (lower + upper) / 2.0
                span = upper - lower
                val = np.atleast_1d(np.ones(ndim) * avg)
                top.root.add(comp_name, omdao.IndepVarComp(name, val=val), promotes=[name])
                driver.add_desvar(name, lower=lower, upper=upper, adder=-avg, scaler=1.0 / span)
                # driver.add_desvar(name, lower=lower, upper=upper)

        # add objective and setup
        driver.add_objective(objective)
        top.setup(check=debug)

        # somehow html file is not viewable.
        if debug:
            omdao.view_model(top, outfile='CharDB_debug.html')

        # set constants
        for name, val in constants.items():
            top[name] = val

        top.run()

        results = {var: kwargs.get(var, self[var]) for var in self._discrete_params}
        for var in var_list:
            results[var] = top[var]

        return results
