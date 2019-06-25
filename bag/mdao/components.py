# -*- coding: utf-8 -*-

"""This module defines various OpenMDAO component classes.
"""

import numpy as np
import openmdao.api as omdao


class VecFunComponent(omdao.Component):
    """A component based on a list of functions.

    A component that evaluates multiple functions on the given inputs, then
    returns the result as an 1D array.  Each of the inputs may be a scalar or
    a vector with the same size as the output.  If a vector input is given,
    each function will use a different element of the vector.

    Parameters
    ----------
    output_name : str
        output name.
    fun_list : list[bag.math.dfun.DiffFunction]
        list of interpolator functions, one for each dimension.
    params : list[str]
        list of parameter names.  Parameter names may repeat, in which case the
        same parameter will be used for multiple arguments of the function.
    vector_params : set[str]
        set of parameters that are vector instead of scalar.  If a parameter
        is a vector, it will be the same size as the output, and each function
        only takes in the corresponding element of the parameter.
    """

    def __init__(self, output_name, fun_list, params,
                 vector_params=None):
        omdao.Component.__init__(self)

        vector_params = vector_params or set()

        self._output = output_name
        self._out_dim = len(fun_list)
        self._in_dim = len(params)
        self._params = params
        self._unique_params = {}
        self._fun_list = fun_list

        for par in params:
            adj = par in vector_params
            shape = self._out_dim if adj else 1

            if par not in self._unique_params:
                # linear check, but small list so should be fine.
                self.add_param(par, val=np.zeros(shape))
                self._unique_params[par] = len(self._unique_params), adj

        # construct chain rule jacobian matrix
        self._chain_jacobian = np.zeros((self._in_dim, len(self._unique_params)))
        for idx, par in enumerate(params):
            self._chain_jacobian[idx, self._unique_params[par][0]] = 1

        self.add_output(output_name, val=np.zeros(self._out_dim))

    def __call__(self, **kwargs):
        """Evaluate on the given inputs.

        Parameters
        ----------
        kwargs : dict[str, np.array or float]
            the inputs as a dictionary.

        Returns
        -------
        out : np.array
            the output array.
        """
        tmp = {}
        self.solve_nonlinear(kwargs, tmp)
        return tmp[self._output]

    def _get_inputs(self, params):
        """Given parameter values, construct inputs for functions.

        Parameters
        ----------
        params : VecWrapper, optional
            VecWrapper containing parameters. (p)

        Returns
        -------
        ans : list[list[float]]
            input lists.
        """
        ans = np.empty((self._out_dim, self._in_dim))
        for idx, name in enumerate(self._params):
            ans[:, idx] = params[name]
        return ans

    def solve_nonlinear(self, params, unknowns, resids=None):
        """Compute the output parameter.

        Parameters
        ----------
        params : VecWrapper, optional
            VecWrapper containing parameters. (p)

        unknowns : VecWrapper, optional
            VecWrapper containing outputs and states. (u)

        resids : VecWrapper, optional
            VecWrapper containing residuals. (r)
        """
        xi_mat = self._get_inputs(params)

        tmp = np.empty(self._out_dim)
        for idx in range(self._out_dim):
            tmp[idx] = self._fun_list[idx](xi_mat[idx, :])

        unknowns[self._output] = tmp

    def linearize(self, params, unknowns=None, resids=None):
        """Compute the Jacobian of the parameter.

        Parameters
        ----------
        params : VecWrapper, optional
            VecWrapper containing parameters. (p)

        unknowns : VecWrapper, optional
            VecWrapper containing outputs and states. (u)

        resids : VecWrapper, optional
            VecWrapper containing residuals. (r)
        """
        # print('rank {} computing jac for {}'.format(self.comm.rank, self._outputs))

        xi_mat = self._get_inputs(params)

        jf = np.empty((self._out_dim, self._in_dim))
        for k, fun in enumerate(self._fun_list):
            jf[k, :] = fun.jacobian(xi_mat[k, :])

        jmat = np.dot(jf, self._chain_jacobian)
        jdict = {}
        for par, (pidx, adj) in self._unique_params.items():
            tmp = jmat[:, pidx]
            if adj:
                tmp = np.diag(tmp)
            jdict[self._output, par] = tmp

        return jdict
