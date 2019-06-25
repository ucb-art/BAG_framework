# -*- coding: utf-8 -*-

"""This module handles simulation data related IO.

Note : when reading data files, we use Numpy to handle the encodings,
so BAG encoding settings will not apply.
"""

import os
import glob

import numpy as np
import h5py

from .common import bag_encoding, bag_codec_error

illegal_var_name = ['sweep_params']


class SweepArray(np.ndarray):
    """Subclass of numpy array that adds sweep parameters attribute.
    """

    def __new__(cls, data, sweep_params=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(data).view(cls)
        # add the new attribute to the created instance
        obj.sweep_params = sweep_params
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.sweep_params = getattr(obj, 'sweep_params', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(SweepArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.sweep_params,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):
        self.sweep_params = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        # noinspection PyArgumentList
        super(SweepArray, self).__setstate__(state[0:-1])


def _get_sweep_params(fname):
    """Parse the sweep information file and reverse engineer sweep parameters.

    Parameters
    ----------
    fname : str
        the sweep information file name.

    Returns
    -------
    swp_list : list[str]
        list of sweep parameter names.  index 0 is the outer-most loop.
    values_list : list[list[float or str]]
        list of values list for each sweep parameter.
    """
    mat = np.genfromtxt(fname, dtype=np.unicode_)
    header = mat[0, :]
    data = mat[1:, :]

    # eliminate same data
    idx_list = []
    for idx in range(len(header)):
        bool_vec = data[:, idx] == data[0, idx]  # type: np.ndarray
        if not np.all(bool_vec):
            idx_list.append(idx)

    header = header[idx_list]
    data = data[:, idx_list]
    # find the first index of last element of each column.
    last_first_idx = [np.where(data[:, idx] == data[-1, idx])[0][0] for idx in range(len(header))]
    # sort by first index of last element; the column where the last element
    # appears the earliest is the inner most loop.
    order_list = np.argsort(last_first_idx)  # type: np.ndarray

    # get list of values
    values_list = []
    skip_len = 1
    for idx in order_list:
        end_idx = last_first_idx[idx] + 1
        values = data[0:end_idx:skip_len, idx]
        if header[idx] != 'corner':
            values = values.astype(np.float)
        skip_len *= len(values)
        values_list.append(values)

    swp_list = header[order_list][::-1].tolist()
    values_list.reverse()
    return swp_list, values_list


def load_sim_results(save_dir):
    """Load exported simulation results from the given directory.

    Parameters
    ----------
    save_dir : str
        the save directory path.

    Returns
    -------
    results : dict[str, any]
        the simulation data dictionary.

        most keys in result is either a sweep parameter or an output signal.
        the values are the corresponding data as a numpy array.  In addition,
        results has a key called 'sweep_params', which contains a dictionary from
        output signal name to a list of sweep parameters of that output.

    """
    if not save_dir:
        return None

    results = {}
    sweep_params = {}

    # load sweep parameter values
    top_swp_list, values_list = _get_sweep_params(os.path.join(save_dir, 'sweep.info'))
    top_shape = []
    for swp, values in zip(top_swp_list, values_list):
        results[swp] = values
        top_shape.append(len(values))

    for swp_name in glob.glob(os.path.join(save_dir, '*.sweep')):
        base_name = os.path.basename(swp_name).split('.')[0]
        data_name = os.path.join(save_dir, '%s.data' % base_name)
        try:
            data_arr = np.loadtxt(data_name)
        except ValueError:
            # try loading complex
            data_arr = np.loadtxt(data_name, dtype=complex)

        # get sweep parameter names
        with open(swp_name, 'r', encoding='utf-8') as f:
            swp_list = [str(line.strip()) for line in f]

        # make a copy of master sweep list and sweep shape
        cur_swp_list = list(top_swp_list)
        cur_shape = list(top_shape)

        for swp in swp_list:
            if swp not in results:
                fname = os.path.join(save_dir, '%s.info' % swp)
                results[swp] = np.loadtxt(fname)

            # if sweep has more than one element.
            if results[swp].shape:
                cur_swp_list.append(swp)
                cur_shape.append(results[swp].shape[0])

        # sanity check
        if base_name in results:
            raise Exception('Error: output named %s already in results' % base_name)

        # reshape data array
        data_arr = data_arr.reshape(cur_shape)
        results[base_name] = SweepArray(data_arr, cur_swp_list)
        # record sweep parameters for this data
        sweep_params[base_name] = cur_swp_list

    if 'sweep_params' in results:
        raise Exception('illegal output name: sweep_params')

    results['sweep_params'] = sweep_params

    return results


def save_sim_results(results, fname, compression='gzip'):
    """Saves the given simulation results dictionary as a HDF5 file.

    Parameters
    ----------
    results : dict[string, any]
        the results dictionary.
    fname : str
        the file to save results to.
    compression : str
        HDF5 compression method.  Defaults to 'gzip'.
    """
    # create directory if it didn't exist.
    fname = os.path.abspath(fname)
    dir_name = os.path.dirname(fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    sweep_info = results['sweep_params']
    with h5py.File(fname, 'w') as f:
        for name, swp_vars in sweep_info.items():
            # store data
            data = np.asarray(results[name])
            if not data.shape:
                dset = f.create_dataset(name, data=data)
            else:
                dset = f.create_dataset(name, data=data, compression=compression)
            # h5py workaround: need to explicitly store unicode
            dset.attrs['sweep_params'] = [swp.encode(encoding=bag_encoding, errors=bag_codec_error)
                                          for swp in swp_vars]

            # store sweep parameter values
            for var in swp_vars:
                if var not in f:
                    swp_data = results[var]
                    if np.issubdtype(swp_data.dtype, np.unicode_):
                        # we need to explicitly encode unicode strings to bytes
                        swp_data = [v.encode(encoding=bag_encoding, errors=bag_codec_error) for v in swp_data]

                    f.create_dataset(var, data=swp_data, compression=compression)


def load_sim_file(fname):
    """Read simulation results from HDF5 file.

    Parameters
    ----------
    fname : str
        the file to read.

    Returns
    -------
    results : dict[str, any]
        the result dictionary.
    """
    if not os.path.isfile(fname):
        raise ValueError('%s is not a file.' % fname)

    results = {}
    sweep_params = {}
    with h5py.File(fname, 'r') as f:
        for name in f:
            dset = f[name]
            dset_data = dset[()]
            if np.issubdtype(dset.dtype, np.bytes_):
                # decode byte values to unicode arrays
                dset_data = np.array([v.decode(encoding=bag_encoding, errors=bag_codec_error) for v in dset_data])

            if 'sweep_params' in dset.attrs:
                cur_swp = [swp.decode(encoding=bag_encoding, errors=bag_codec_error)
                           for swp in dset.attrs['sweep_params']]
                results[name] = SweepArray(dset_data, cur_swp)
                sweep_params[name] = cur_swp
            else:
                results[name] = dset_data

    results['sweep_params'] = sweep_params
    return results
