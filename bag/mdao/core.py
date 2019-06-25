# -*- coding: utf-8 -*-

"""This module defines core BAG openmdao classes."""

import numpy as np
import networkx as nx
import openmdao.api as omdao

import bag.util.parse

from .components import VecFunComponent


class GroupBuilder(object):
    """A class that builds new OpenMDAO groups.

    This class provides a simple interface to define new variables as function of
    other variables, and it tracks the variable dependencies using a directed
    acyclic graph.

    """

    def __init__(self):
        self._g = nx.DiGraph()
        self._input_vars = set()

    def _add_node(self, name, ndim, **kwargs):
        """Helper method to add a node and keep track of input variables."""
        self._g.add_node(name, ndim=ndim, **kwargs)
        self._input_vars.add(name)

    def _add_edge(self, parent, child):
        """Helper method to add an edge and update input variables."""
        self._g.add_edge(parent, child)
        try:
            self._input_vars.remove(child)
        except KeyError:
            pass

    def get_inputs(self):
        """Returns a set of current input variable names.

        Returns
        -------
        input_vars : set[str]
            a set of input variable names.
        """
        return self._input_vars.copy()

    def get_variables(self):
        """Returns a list of variables.

        Returns
        -------
        var_list : list[str]
            a list of variables.
        """
        return list(self._g.nodes_iter())

    def get_variable_info(self, name):
        """Returns the range and dimension of the given variable.

        Parameters
        ----------
        name : str
            variable name.

        Returns
        -------
        min : float
            minimum value.
        max : float
            maximum value.
        ndim : int
            variable dimension.
        """
        nattr = self._g.node[name]
        return nattr.copy()

    def add_fun(self, var_name, fun_list, params, param_ranges, vector_params=None):
        """Add a new variable defined by the given list of functions.

        Parameters
        ----------
        var_name : str
            variable name.
        fun_list : list[bag.math.interpolate.Interpolator]
            list of functions, one for each dimension.
        params : list[str]
            list of parameter names.  Parameter names may repeat, in which case the
            same parameter will be used for multiple arguments of the function.
        param_ranges : dict[str, (float, float)]
            a dictionary of parameter valid range.
        vector_params : set[str]
            set of parameters that are vector instead of scalar.  If a parameter
            is a vector, it will be the same size as the output, and each function
            only takes in the corresponding element of the parameter.
        """
        vector_params = vector_params or set()
        ndim = len(fun_list)

        # error checking
        for par in params:
            if par not in param_ranges:
                raise ValueError('Valid range of %s not specified.' % par)

        # add inputs
        for par, (par_min, par_max) in param_ranges.items():
            par_dim = ndim if par in vector_params else 1
            if par not in self._g:
                # add input to graph if it's not in there.
                self._add_node(par, par_dim)

            nattrs = self._g.node[par]
            if nattrs['ndim'] != par_dim:
                # error checking.
                raise ValueError('Variable %s has dimension mismatch.' % par)
            # update input range
            nattrs['min'] = max(par_min, nattrs.get('min', par_min))
            nattrs['max'] = min(par_max, nattrs.get('max', par_max))

        # add current variable
        if var_name not in self._g:
            self._add_node(var_name, ndim)

        nattrs = self._g.node[var_name]
        # error checking.
        if nattrs['ndim'] != ndim:
            raise ValueError('Variable %s has dimension mismatch.' % var_name)
        if self._g.in_degree(var_name) > 0:
            raise Exception('Variable %s already has other dependencies.' % var_name)

        nattrs['fun_list'] = fun_list
        nattrs['params'] = params
        nattrs['vec_params'] = vector_params
        for parent in param_ranges.keys():
            self._add_edge(parent, var_name)

    def add_var(self, variable, vmin, vmax, ndim=1):
        """Adds a new independent variable.

        Parameters
        ----------
        variable : str
            the variable to add
        vmin : float
            the minimum allowable value.
        vmax : float
            the maximum allowable value.
        ndim : int
            the dimension of the variable.  Defaults to 1.
        """
        if variable in self._g:
            raise Exception('Variable %s already exists.' % variable)
        self._add_node(variable, ndim, min=vmin, max=vmax)

    def set_input_limit(self, var, equals=None, lower=None, upper=None):
        """Sets the limit on the given input variable.

        Parameters
        ----------
        var : str
            name of the variable.
        equals : float or None
            if given, the equality value.
        lower : float or None
            if given, the minimum.
        upper : float or None
            if given, the maximum.
        """
        if var in self._g:
            if self._g.in_degree(var) > 0:
                raise Exception('Variable %s is not an input variable' % var)
            nattr = self._g.node[var]
            if equals is not None:
                nattr['equals'] = equals
                lower = upper = equals
            print(var, lower, upper)
            if lower is not None:
                nattr['min'] = max(nattr.get('min', lower), lower)
            if upper is not None:
                nattr['max'] = min(nattr.get('max', upper), upper)
            print(var, nattr['min'], nattr['max'])

    def add_expr(self, eqn, ndim):
        """Adds a new variable with the given expression.

        Parameters
        ----------
        eqn : str
            An equation of the form "<var> = <expr>", where var
            is the output variable name, and expr is the expression.
            All variables in expr must be already added.
        ndim : int
            the dimension of the output variable.
        """
        variable, expr = eqn.split('=', 1)
        variable = variable.strip()
        expr = expr.strip()

        if variable not in self._g:
            self._add_node(variable, ndim)
        nattrs = self._g.node[variable]
        if nattrs['ndim'] != ndim:
            raise Exception('Dimension mismatch for %s' % variable)
        if self._g.in_degree(variable) > 0:
            raise Exception('%s already depends on other variables' % variable)

        invars = bag.util.parse.get_variables(expr)
        for parent in invars:
            if parent not in self._g:
                raise Exception('Variable %s is not defined.' % parent)
            self._add_edge(parent, variable)

        nattrs['expr'] = expr

    def build(self, debug=False):
        """Returns a OpenMDAO Group from the variable graph.

        Parameters
        ----------
        debug : bool
            True to print debug messages.

        Returns
        -------
        grp : omdao.Group
            the OpenMDAO group that computes all variables.
        input_bounds : dict[str, any]
            a dictionary from input variable name to (min, max, ndim) tuple.
        """
        input_bounds = {}
        ndim_dict = {}

        if not nx.is_directed_acyclic_graph(self._g):
            raise Exception('Dependency loop detected')

        grp = omdao.Group()
        prom = ['*']
        for var in nx.topological_sort(self._g):
            nattrs = self._g.node[var]
            ndim = nattrs['ndim']
            ndim_dict[var] = ndim
            if self._g.in_degree(var) == 0:
                if debug:
                    # input variable
                    print('Input variable: %s' % var)
                # range checking
                vmin, vmax = nattrs['min'], nattrs['max']
                veq = nattrs.get('equals', None)
                if vmin > vmax:
                    raise Exception('Variable %s input range not valid.' % var)
                input_bounds[var] = veq, vmin, vmax, ndim
            else:
                init_vals = {par: np.zeros(ndim_dict[par]) for par in self._g.predecessors_iter(var)}
                comp_name = 'comp__%s' % var
                if 'expr' in nattrs:
                    eqn = '{}={}'.format(var, nattrs['expr'])
                    init_vals[var] = np.zeros(ndim)
                    # noinspection PyTypeChecker
                    grp.add(comp_name, omdao.ExecComp(eqn, **init_vals), promotes=prom)
                elif 'fun_list' in nattrs:
                    params = nattrs['params']
                    fun_list = nattrs['fun_list']
                    vec_params = nattrs['vec_params']
                    comp = VecFunComponent(var, fun_list, params, vector_params=vec_params)
                    # noinspection PyTypeChecker
                    grp.add(comp_name, comp, promotes=prom)
                else:
                    raise Exception('Unknown attributes: {}'.format(nattrs))

        return grp, input_bounds
