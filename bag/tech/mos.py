# -*- coding: utf-8 -*-

"""This module contains transistor characterization and optimization related classes.
"""

import os

# import pyoptsparse
import numpy as np

from .core import CharDB


class MosCharDB(CharDB):
    """The mosfet characterization database.

    This class holds transistor characterization data and provides useful query methods.

    Parameters
    ----------
    root_dir : str
        path to the root characterization data directory.
    mos_type : str
        the transistor type.  Either 'pch' or 'nch'.
    discrete_params : list[str]
        a list of parameters that should take on discrete values instead of being interpolated.
        Usually intent, length, or transistor width (for finfets).
    env_list : list[str]
        list of simulation environments to consider.
    update : bool
        True to update post-processed data from raw simulation data.
    intent : str or None
        the threshold flavor name.
    l : float or None
        the channel length, in meters.
    w : int or float or None
        the transistor width, in meters or number of fins.
    vgs : float or None
        the Vgs voltage.
    vds : float or None
        the Vds voltage.
    vbs : float or None
        the Vbs voltage.
    **kwargs :
        additional characterization database parameters.  See documentation for CharDB.
    """

    _raw_data_names = ['ids', 'y11', 'y12', 'y13', 'y21', 'y22', 'y23', 'y31', 'y32', 'y33']

    def __init__(self, root_dir, mos_type, discrete_params, env_list,
                 intent=None, l=None, w=None, vgs=None, vds=None, vbs=None,
                 **kwargs):
        constants = dict(mos_type=mos_type)
        init_params = dict(intent=intent, l=l, w=w, vgs=vgs, vds=vds, vbs=vbs)
        CharDB.__init__(self, root_dir, constants, discrete_params, init_params, env_list, **kwargs)

    @classmethod
    def get_sim_file(cls, root_dir, constants):
        """Returns the simulation data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : dict[string, any]
            constants dictionary.

        Returns
        -------
        fname : str
            the simulation data file name.
        """
        return os.path.join(root_dir, '%s.hdf5' % constants['mos_type'])

    @classmethod
    def get_cache_file(cls, root_dir, constants):
        """Returns the post-processed characterization data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : dict[string, any]
            constants dictionary.

        Returns
        -------
        fname : str
            the post-processed characterization data file name.
        """
        return os.path.join(root_dir, '%s__%s.hdf5' % (constants['mos_type'], cls.__name__))

    @classmethod
    def post_process_data(cls, sim_data, sweep_params, sweep_values, constants):
        """Postprocess simulation data.

        Parameters
        ----------
        sim_data : dict[string, np.array]
            the simulation data as a dictionary from output name to numpy array.
        sweep_params : list[str]
            list of parameter name for each dimension of numpy array.
        sweep_values : list[numpy.array]
            list of parameter values for each dimension.
        constants : dict[string, any]
            the constants dictionary.

        Returns
        -------
        data : dict[str, np.array]
            a dictionary of post-processed data.
        """
        # compute small signal parameters
        w = 2 * np.pi * constants['char_freq']
        fg = constants['fg']

        ids = sim_data['ids']
        gm = (sim_data['y21'].real - sim_data['y31'].real) / 2.0
        gds = (sim_data['y22'].real - sim_data['y32'].real) / 2.0
        gb = (sim_data['y33'].real - sim_data['y23'].real) / 2.0 - gm - gds

        cgd = -0.5 / w * (sim_data['y12'].imag + sim_data['y21'].imag)
        cgs = -0.5 / w * (sim_data['y13'].imag + sim_data['y31'].imag)
        cds = -0.5 / w * (sim_data['y23'].imag + sim_data['y32'].imag)
        cgb = sim_data['y11'].imag / w - cgd - cgs
        cdb = sim_data['y22'].imag / w - cds - cgd
        csb = sim_data['y33'].imag / w - cgs - cds

        ss_data = dict(
            ids=ids / fg,
            gm=gm / fg,
            gds=gds / fg,
            gb=gb / fg,
            cgd=cgd / fg,
            cgs=cgs / fg,
            cds=cds / fg,
            cgb=cgb / fg,
            cdb=cdb / fg,
            csb=csb / fg,
        )

        return ss_data

    @classmethod
    def derived_parameters(cls):
        """Returns a list of derived parameters."""
        return ['cgg', 'cdd', 'css', 'cbb', 'vstar', 'gain', 'ft']

    @classmethod
    def compute_derived_parameters(cls, fdict):
        """Compute derived parameter functions.

        Parameters
        ----------
        fdict : dict[string, bag.math.dfun.DiffFunction]
            a dictionary from core parameter name to the corresponding function.

        Returns
        -------
        deriv_dict : dict[str, bag.math.dfun.DiffFunction]
            a dictionary from derived parameter name to the corresponding function.
        """
        cgg = fdict['cgd'] + fdict['cgs'] + fdict['cgb']
        return dict(
            cgg=cgg,
            cdd=fdict['cgd'] + fdict['cds'] + fdict['cdb'],
            css=fdict['cgs'] + fdict['cds'] + fdict['csb'],
            cbb=fdict['cgb'] + fdict['cdb'] + fdict['csb'],
            vstar=2.0 * (fdict['ids'] / fdict['gm']),
            gain=fdict['gm'] / fdict['gds'],
            ft=fdict['gm'] / (2.0 * np.pi * cgg),
        )


class MosCharGDDB(CharDB):
    """The mosfet characterization database.

    This class holds transistor characterization data and provides useful query methods.

    Parameters
    ----------
    root_dir : str
        path to the root characterization data directory.
    mos_type : str
        the transistor type.  Either 'pch' or 'nch'.
    discrete_params : list[str]
        a list of parameters that should take on discrete values instead of being interpolated.
        Usually intent, length, or transistor width (for finfets).
    env_list : list[str]
        list of simulation environments to consider.
    update : bool
        True to update post-processed data from raw simulation data.
    intent : str or None
        the threshold flavor name.
    l : float or None
        the channel length, in meters.
    w : int or float or None
        the transistor width, in meters or number of fins.
    vgs : float or None
        the Vgs voltage.
    vds : float or None
        the Vds voltage.
    vbs : float or None
        the Vbs voltage.
    **kwargs :
        additional characterization database parameters.  See documentation for CharDB.
    """

    _raw_data_names = ['ids', 'y11', 'y12', 'y21', 'y22']

    def __init__(self, root_dir, mos_type, discrete_params, env_list,
                 intent=None, l=None, w=None, vgs=None, vds=None,
                 **kwargs):
        constants = dict(mos_type=mos_type)
        init_params = dict(intent=intent, l=l, w=w, vgs=vgs, vds=vds)
        CharDB.__init__(self, root_dir, constants, discrete_params, init_params, env_list, **kwargs)

    @classmethod
    def get_sim_file(cls, root_dir, constants):
        """Returns the simulation data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : dict[string, any]
            constants dictionary.

        Returns
        -------
        fname : str
            the simulation data file name.
        """
        return os.path.join(root_dir, '%s.hdf5' % constants['mos_type'])

    @classmethod
    def get_cache_file(cls, root_dir, constants):
        """Returns the post-processed characterization data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : dict[string, any]
            constants dictionary.

        Returns
        -------
        fname : str
            the post-processed characterization data file name.
        """
        return os.path.join(root_dir, '%s__%s.hdf5' % (constants['mos_type'], cls.__name__))

    @classmethod
    def post_process_data(cls, sim_data, sweep_params, sweep_values, constants):
        """Postprocess simulation data.

        Parameters
        ----------
        sim_data : dict[string, np.array]
            the simulation data as a dictionary from output name to numpy array.
        sweep_params : list[str]
            list of parameter name for each dimension of numpy array.
        sweep_values : list[numpy.array]
            list of parameter values for each dimension.
        constants : dict[string, any]
            the constants dictionary.

        Returns
        -------
        data : dict[str, np.array]
            a dictionary of post-processed data.
        """
        # compute small signal parameters
        w = 2 * np.pi * constants['char_freq']
        fg = constants['fg']

        ids = sim_data['ids']
        gm = sim_data['y21'].real
        gds = sim_data['y22'].real

        cgd = -0.5 / w * (sim_data['y12'].imag + sim_data['y21'].imag)
        cgs = sim_data['y11'].imag / w - cgd
        cds = sim_data['y22'].imag / w - cgd

        ss_data = dict(
            ids=ids / fg,
            gm=gm / fg,
            gds=gds / fg,
            cgd=cgd / fg,
            cgs=cgs / fg,
            cds=cds / fg,
        )

        return ss_data

    @classmethod
    def derived_parameters(cls):
        """Returns a list of derived parameters."""
        return ['cgg', 'cdd', 'vstar', 'gain', 'ft']

    @classmethod
    def compute_derived_parameters(cls, fdict):
        """Compute derived parameter functions.

        Parameters
        ----------
        fdict : dict[string, bag.math.dfun.DiffFunction]
            a dictionary from core parameter name to the corresponding function.

        Returns
        -------
        deriv_dict : dict[str, bag.math.dfun.DiffFunction]
            a dictionary from derived parameter name to the corresponding function.
        """
        cgg = fdict['cgd'] + fdict['cgs']
        return dict(
            cgg=cgg,
            cdd=fdict['cgd'] + fdict['cds'],
            vstar=2.0 * (fdict['ids'] / fdict['gm']),
            gain=fdict['gm'] / fdict['gds'],
            ft=fdict['gm'] / (2.0 * np.pi * cgg),
        )
