# -*- coding: utf-8 -*-

"""This module defines classes for computing DC operating point.
"""

from typing import Union, Dict

import scipy.sparse
import scipy.optimize
import numpy as np

from bag.tech.mos import MosCharDB


class DCCircuit(object):
    """A class that solves DC operating point of a circuit.

    Parameters
    ----------
    ndb : MosCharDB
        nmos characterization database.
    pdb : MosCharDB
        pmos characterization database.
    """

    def __init__(self, ndb, pdb):
        # type: (MosCharDB, MosCharDB) -> None
        self._n = 1
        self._ndb = ndb
        self._pdb = pdb
        self._transistors = {}
        self._node_id = {'gnd': 0, 'vss': 0, 'VSS': 0}
        self._node_name_lookup = {0: 'gnd'}
        self._node_voltage = {0: 0}

    def _get_node_id(self, name):
        # type: (str) -> int
        if name not in self._node_id:
            ans = self._n
            self._node_id[name] = ans
            self._node_name_lookup[ans] = name
            self._n += 1
            return ans
        else:
            return self._node_id[name]

    def set_voltage_source(self, node_name, voltage):
        # type: (str, float) -> None
        """
        Specify voltage the a node.

        Parameters
        ----------
        node_name : str
            the net name.
        voltage : float
            voltage of the given net.
        """
        node_id = self._get_node_id(node_name)
        self._node_voltage[node_id] = voltage

    def add_transistor(self, d_name, g_name, s_name, b_name, mos_type, intent, w, lch, fg=1):
        # type: (str, str, str, str, str, str, Union[float, int], float, int) -> None
        """Adds a small signal transistor model to the circuit.

        Parameters
        ----------
        d_name : str
            drain net name.
        g_name : str
            gate net name.
        s_name : str
            source net name.
        b_name : str
            body net name.  Defaults to 'gnd'.
        mos_type : str
            transistor type.  Either 'nch' or 'pch'.
        intent : str
            transistor threshold flavor.
        w : Union[float, int]
            transistor width.
        lch : float
            transistor channel length.
        fg : int
            transistor number of fingers.
        """
        node_d = self._get_node_id(d_name)
        node_g = self._get_node_id(g_name)
        node_s = self._get_node_id(s_name)
        node_b = self._get_node_id(b_name)

        # get existing current function.  Initalize if not found.
        ids_key = (mos_type, intent, lch)
        if ids_key in self._transistors:
            arow, acol, bdata, fg_list, ds_list = self._transistors[ids_key]
        else:
            arow, acol, bdata, fg_list, ds_list = [], [], [], [], []
            self._transistors[ids_key] = (arow, acol, bdata, fg_list, ds_list)

        # record Ai and bi data
        offset = len(fg_list) * 4
        arow.extend([offset + 1, offset + 1, offset + 2, offset + 2, offset + 3, offset + 3])
        acol.extend([node_b, node_s, node_d, node_s, node_g, node_s])
        bdata.append(w)
        fg_list.append(fg)
        ds_list.append((node_d, node_s))

    def solve(self, env, guess_dict, itol=1e-10, inorm=1e-6):
        # type: (str, Dict[str, float], float, float) -> Dict[str, float]
        """Solve DC operating point.

        Parameters
        ----------
        env : str
            the simulation environment.
        guess_dict : Dict[str, float]
            initial guess dictionary.
        itol : float
            current error tolerance.
        inorm : float
            current normalization factor.

        Returns
        -------
        op_dict : Dict[str, float]
            DC operating point dictionary.
        """
        # step 1: get list of nodes to solve
        node_list = [idx for idx in range(self._n) if idx not in self._node_voltage]
        reverse_dict = {nid: idx for idx, nid in enumerate(node_list)}
        ndim = len(node_list)

        # step 2: get Av and bv
        amatv = scipy.sparse.csr_matrix(([1] * ndim, (node_list, np.arange(ndim))), shape=(self._n, ndim))
        bmatv = np.zeros(self._n)
        for nid, val in self._node_voltage.items():
            bmatv[nid] = val

        # step 3: gather current functions, and output matrix entries
        ifun_list = []
        out_data = []
        out_row = []
        out_col = []
        out_col_cnt = 0
        for (mos_type, intent, lch), (arow, acol, bdata, fg_list, ds_list) in self._transistors.items():
            db = self._ndb if mos_type == 'nch' else self._pdb
            ifun = db.get_function('ids', env=env, intent=intent, l=lch)
            # step 3A: compute Ai and bi
            num_tran = len(fg_list)
            adata = [1, -1] * (3 * num_tran)
            amati = scipy.sparse.csr_matrix((adata, (arow, acol)), shape=(4 * num_tran, self._n))
            bmati = np.zeros(4 * num_tran)
            bmati[0::4] = bdata

            # step 3B: compute A = Ai * Av, b = Ai * bv + bi
            amat = amati.dot(amatv)
            bmat = amati.dot(bmatv) + bmati
            # record scale matrix and function.
            scale_mat = scipy.sparse.diags(fg_list) / inorm
            ifun_list.append((ifun, scale_mat, amat, bmat))
            for node_d, node_s in ds_list:
                if node_d in reverse_dict:
                    out_row.append(reverse_dict[node_d])
                    out_data.append(-1)
                    out_col.append(out_col_cnt)
                if node_s in reverse_dict:
                    out_row.append(reverse_dict[node_s])
                    out_data.append(1)
                    out_col.append(out_col_cnt)
                out_col_cnt += 1
        # construct output matrix
        out_mat = scipy.sparse.csr_matrix((out_data, (out_row, out_col)), shape=(ndim, out_col_cnt))

        # step 4: define zero function
        def zero_fun(varr):
            iarr = np.empty(out_col_cnt)
            offset = 0
            for idsf, smat, ai, bi in ifun_list:
                num_out = smat.shape[0]
                # reshape going row first instead of column
                arg = (ai.dot(varr) + bi).reshape(4, -1, order='F').T
                if idsf.ndim == 3:
                    # handle case where transistor source and body are shorted
                    tmpval = idsf(arg[:, [0, 2, 3]])
                else:
                    tmpval = idsf(arg)
                iarr[offset:offset + num_out] = smat.dot(tmpval)
                offset += num_out
            return out_mat.dot(iarr)

        # step 5: define zero function
        def jac_fun(varr):
            jarr = np.empty((out_col_cnt, ndim))
            offset = 0
            for idsf, smat, ai, bi in ifun_list:
                num_out = smat.shape[0]
                # reshape going row first instead of column
                arg = (ai.dot(varr) + bi).reshape(4, -1, order='F').T
                if idsf.ndim == 3:
                    # handle case where transistor source and body are shorted
                    tmpval = idsf.jacobian(arg[:, [0, 2, 3]])
                    # noinspection PyTypeChecker
                    tmpval = np.insert(tmpval, 1, 0.0, axis=len(tmpval.shape) - 1)
                else:
                    tmpval = idsf.jacobian(arg)
                jcur = smat.dot(tmpval)
                for idx in range(num_out):
                    # ai is sparse matrix; multiplication is matrix
                    jarr[offset + idx, :] = jcur[idx, :] @ ai[4 * idx:4 * idx + 4, :]
                offset += num_out
            return out_mat.dot(jarr)

        xguess = np.empty(ndim)
        for name, guess_val in guess_dict.items():
            xguess[reverse_dict[self._node_id[name]]] = guess_val

        result = scipy.optimize.root(zero_fun, xguess, jac=jac_fun, tol=itol / inorm, method='hybr')
        if not result.success:
            raise ValueError('solution failed.')

        op_dict = {self._node_name_lookup[nid]: result.x[idx] for idx, nid in enumerate(node_list)}
        return op_dict
