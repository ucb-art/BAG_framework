# -*- coding: utf-8 -*-

"""This module defines the differentiable function class."""

from typing import Union, List, Optional, Tuple

import abc

import numpy as np


class DiffFunction(abc.ABC):
    """An abstract class representing a differentiable scalar function.

    Supports Numpy broadcasting.  Defaults to using finite difference for derivative calculation.

    Parameters
    ----------
    input_ranges : List[Tuple[Optional[float], Optional[float]]]
        input ranges.
    delta_list : Optional[List[float]]
        a list of finite difference step size for each input.  If None,
        finite difference will be disabled.
    """

    def __init__(self, input_ranges, delta_list=None):
        # type: (List[Tuple[Optional[float], Optional[float]]], Optional[List[float]]) -> None
        # error checking
        self._ndim = len(input_ranges)
        if delta_list is not None and len(delta_list) != self._ndim:
            raise ValueError('finite difference list length inconsistent.')

        self._input_ranges = input_ranges
        self.delta_list = delta_list  # type: Optional[List[float]]

    @property
    def input_ranges(self):
        # type: () -> List[Tuple[Optional[float], Optional[float]]]
        return self._input_ranges

    @property
    def ndim(self):
        # type: () -> int
        """Number of input dimensions."""
        return self._ndim

    @abc.abstractmethod
    def __call__(self, xi):
        """Interpolate at the given coordinates.

        Numpy broadcasting rules apply.

        Parameters
        ----------
        xi : array_like
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : np.multiarray.ndarray
            The interpolated values at the given coordinates.
        """
        raise NotImplementedError('Not implemented')

    def get_input_range(self, idx):
        # type: (int) -> Tuple[Optional[float], Optional[float]]
        """Returns the input range of the given dimension."""
        return self._input_ranges[idx]

    def deriv(self, xi, j):
        """Calculate the derivative at the given coordinates with respect to input j.

        Numpy broadcasting rules apply.

        Parameters
        ----------
        xi : array_like
            The coordinates to evaluate, with shape (..., ndim)
        j : int
            input index.

        Returns
        -------
        val : np.multiarray.ndarray
            The derivatives at the given coordinates.
        """
        return self._fd(xi, j, self.delta_list[j])

    def jacobian(self, xi):
        """Calculate the Jacobian at the given coordinates.

        Numpy broadcasting rules apply.

        If finite difference step sizes are not specified,
        will call deriv() in a for loop to compute the Jacobian.

        Parameters
        ----------
        xi : array_like
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : np.multiarray.ndarray
            The Jacobian matrices at the given coordinates.
        """
        if self.delta_list:
            return self._fd_jacobian(xi, self.delta_list)
        else:
            xi = np.asarray(xi, dtype=float)
            ans = np.empty(xi.shape)
            for n in range(self.ndim):
                ans[..., n] = self.deriv(xi, n)
            return ans

    def _fd(self, xi, idx, delta):
        """Calculate the derivative along the given index using central finite difference.

        Parameters
        ----------
        xi : array_like
            The coordinates to evaluate, with shape (..., ndim)
        idx : int
            The index to calculate the derivative on.
        delta : float
            The finite difference step size.

        Returns
        -------
        val : np.multiarray.ndarray
            The derivatives at the given coordinates.
        """
        if idx < 0 or idx >= self.ndim:
            raise ValueError('Invalid derivative index: %d' % idx)

        xi = np.asarray(xi, dtype=float)
        if xi.shape[-1] != self.ndim:
            raise ValueError("The requested sample points xi have dimension %d, "
                             "but this interpolator has dimension %d" % (xi.shape[-1], self.ndim))

        # use broadcasting to evaluate two points at once
        xtest = np.broadcast_to(xi, (2,) + xi.shape).copy()
        xtest[0, ..., idx] += delta / 2.0
        xtest[1, ..., idx] -= delta / 2.0
        val = self(xtest)
        ans = (val[0] - val[1]) / delta  # type: np.ndarray

        if ans.size == 1 and not np.isscalar(ans):
            return ans[0]
        return ans

    def _fd_jacobian(self, xi, delta_list):
        """Calculate the Jacobian matrix using central finite difference.

        Parameters
        ----------
        xi : array_like
            The coordinates to evaluate, with shape (..., ndim)
        delta_list : List[float]
            list of finite difference step sizes for each input.

        Returns
        -------
        val : np.multiarray.ndarray
            The Jacobian matrices at the given coordinates.
        """
        xi = np.asarray(xi, dtype=float)
        if xi.shape[-1] != self.ndim:
            raise ValueError("The requested sample points xi have dimension %d, "
                             "but this interpolator has dimension %d" % (xi.shape[-1], self.ndim))

        # use broadcasting to evaluate all points at once
        xtest = np.broadcast_to(xi, (2 * self.ndim,) + xi.shape).copy()
        for idx, delta in enumerate(delta_list):
            xtest[2 * idx, ..., idx] += delta / 2.0
            xtest[2 * idx + 1, ..., idx] -= delta / 2.0

        val = self(xtest)
        ans = np.empty(xi.shape)
        for idx, delta in enumerate(delta_list):
            ans[..., idx] = (val[2 * idx, ...] - val[2 * idx + 1, ...]) / delta
        return ans

    def transform_input(self, amat, bmat):
        # type: (np.multiarray.ndarray, np.multiarray.ndarray) -> DiffFunction
        """Returns f(Ax + B), where f is this function and A, B are matrices.

        Parameters
        ----------
        amat : np.multiarray.ndarray
            the input transform matrix.
        bmat : np.multiarray.ndarray
            the input shift matrix.

        Returns
        -------
        dfun : DiffFunction
            a scalar differential function.
        """
        return InLinTransformFunction(self, amat, bmat)

    def __add__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, DiffFunction):
            return SumDiffFunction(self, other, f2_sgn=1.0)
        elif isinstance(other, float) or isinstance(other, int):
            return ScaleAddFunction(self, other, 1.0)
        elif isinstance(other, np.ndarray):
            return ScaleAddFunction(self, np.asscalar(other), 1.0)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __radd__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        return self.__add__(other)

    def __sub__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, DiffFunction):
            return SumDiffFunction(self, other, f2_sgn=-1.0)
        elif isinstance(other, float) or isinstance(other, int):
            return ScaleAddFunction(self, -other, 1.0)
        elif isinstance(other, np.ndarray):
            return ScaleAddFunction(self, -np.asscalar(other), 1.0)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __rsub__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, DiffFunction):
            return SumDiffFunction(other, self, f2_sgn=-1.0)
        elif isinstance(other, float) or isinstance(other, int):
            return ScaleAddFunction(self, other, -1.0)
        elif isinstance(other, np.ndarray):
            return ScaleAddFunction(self, np.asscalar(other), -1.0)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __mul__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, DiffFunction):
            return ProdFunction(self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return ScaleAddFunction(self, 0.0, other)
        elif isinstance(other, np.ndarray):
            return ScaleAddFunction(self, 0.0, np.asscalar(other))
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __rmul__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        return self.__mul__(other)

    def __pow__(self, other):
        # type: (Union[float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, float) or isinstance(other, int):
            return PwrFunction(self, other, scale=1.0)
        elif isinstance(other, np.ndarray):
            return PwrFunction(self, np.asscalar(other), scale=1.0)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __div__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, DiffFunction):
            return DivFunction(self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return ScaleAddFunction(self, 0.0, 1.0 / other)
        elif isinstance(other, np.ndarray):
            return ScaleAddFunction(self, 0.0, 1.0 / np.asscalar(other))
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __truediv__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        return self.__div__(other)

    def __rdiv__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        if isinstance(other, DiffFunction):
            return DivFunction(other, self)
        elif isinstance(other, float) or isinstance(other, int):
            return PwrFunction(self, -1.0, scale=other)
        elif isinstance(other, np.ndarray):
            return PwrFunction(self, -1.0, scale=np.asscalar(other))
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __rtruediv__(self, other):
        # type: (Union[DiffFunction, float, int, np.multiarray.ndarray]) -> DiffFunction
        return self.__rdiv__(other)

    def __neg__(self):
        # type: () -> DiffFunction
        return ScaleAddFunction(self, 0.0, -1.0)


class InLinTransformFunction(DiffFunction):
    """A DiffFunction where the input undergoes a linear transformation first.

    This function computes f(Ax + B), where A and B are matrices.

    Parameters
    ----------
    f1 : DiffFunction
        the parent function.
    amat : np.multiarray.ndarray
        the input transform matrix.
    bmat : np.multiarray.ndarray
        the input shift matrix.
    """
    def __init__(self, f1, amat, bmat):
        # type: (DiffFunction, np.multiarray.ndarray, np.multiarray.ndarray) -> None
        if amat.shape[0] != f1.ndim or bmat.shape[0] != f1.ndim:
            raise ValueError('amat/bmat number of rows must be %d' % f1.ndim)
        if len(bmat.shape) != 1:
            raise ValueError('bmat must be 1 dimension.')

        # domain of f(Ax+B) cannot be represented by input ranges.
        super(InLinTransformFunction, self).__init__([(None, None)] * amat.shape[1], delta_list=None)
        self._f1 = f1
        self._amat = amat
        self._bmat = bmat.reshape(-1, 1)

    def _get_arg(self, xi):
        xi = np.asarray(xi)
        xi_shape = xi.shape
        my_ndim = self.ndim
        if xi_shape[-1] != my_ndim:
            raise ValueError('Last dimension must have size %d' % my_ndim)

        xi = xi.reshape(-1, my_ndim)
        return (self._amat.dot(xi.T) + self._bmat).T, xi_shape

    def __call__(self, xi):
        farg, xi_shape = self._get_arg(xi)
        result = self._f1(farg)
        if np.isscalar(result):
            return result
        return result.reshape(xi_shape[:-1])

    def deriv(self, xi, j):
        jmat = self.jacobian(xi)
        return jmat[..., 0, j]

    def jacobian(self, xi):
        farg, xi_shape = self._get_arg(xi)
        jmat = self._f1.jacobian(farg).dot(self._amat)
        shape_trunc = xi_shape[:-1]  # type: Tuple[int, ...]
        return jmat.reshape(shape_trunc + (1, self.ndim))


class ScaleAddFunction(DiffFunction):
    """A DiffFunction multiply by a scalar then added to a scalar.

    Parameters
    ----------
    f1 : DiffFunction
        the first function.
    adder : float
        constant to add.
    scaler : float
        constant to multiply.
    """
    def __init__(self, f1, adder, scaler):
        # type: (DiffFunction, float, float) -> None
        DiffFunction.__init__(self, f1.input_ranges, delta_list=None)
        self._f1 = f1
        self._adder = adder
        self._scaler = scaler

    def __call__(self, xi):
        return self._f1(xi) * self._scaler + self._adder

    def deriv(self, xi, j):
        return self._f1.deriv(xi, j) * self._scaler

    def jacobian(self, xi):
        return self._f1.jacobian(xi) * self._scaler


def _intersection(*args):
    input_ranges = []
    for bound_list in zip(*args):
        lmax, umin = None, None
        for l, u in bound_list:
            if l is None:
                lmax, umin = None, None
                break
            else:
                if lmax is None:
                    lmax, umin = l, u
                else:
                    lmax = max(l, lmax)
                    umin = min(u, umin)

        input_ranges.append((lmax, umin))

    return input_ranges


class SumDiffFunction(DiffFunction):
    """Sum or Difference of two DiffFunctions

    Parameters
    ----------
    f1 : DiffFunction
        the first function.
    f2 : DiffFunction
        the second function.
    f2_sgn : float
        1 if adding, -1 if subtracting.
    """
    def __init__(self, f1, f2, f2_sgn=1.0):
        # type: (DiffFunction, DiffFunction, float) -> None
        if f1.ndim != f2.ndim:
            raise ValueError('functions dimension mismatch.')

        DiffFunction.__init__(self, _intersection(f1.input_ranges, f2.input_ranges), delta_list=None)
        self._f1 = f1
        self._f2 = f2
        self._f2_sgn = f2_sgn

    def __call__(self, xi):
        return self._f1(xi) + self._f2_sgn * self._f2(xi)

    def deriv(self, xi, j):
        return self._f1.deriv(xi, j) + self._f2_sgn * self._f2.deriv(xi, j)

    def jacobian(self, xi):
        return self._f1.jacobian(xi) + self._f2_sgn * self._f2.jacobian(xi)


class ProdFunction(DiffFunction):
    """product of two DiffFunctions

    Parameters
    ----------
    f1 : DiffFunction
        the first function.
    f2 : DiffFunction
        the second function.
    """
    def __init__(self, f1, f2):
        # type: (DiffFunction, DiffFunction) -> None
        if f1.ndim != f2.ndim:
            raise ValueError('functions dimension mismatch.')

        DiffFunction.__init__(self, _intersection(f1.input_ranges, f2.input_ranges), delta_list=None)
        self._f1 = f1
        self._f2 = f2

    def __call__(self, xi):
        return self._f1(xi) * self._f2(xi)

    def deriv(self, xi, j):
        return self._f1.deriv(xi, j) * self._f2(xi) + self._f1(xi) * self._f2.deriv(xi, j)

    def jacobian(self, xi):
        f1_val = self._f1(xi)[..., np.newaxis]
        f2_val = self._f2(xi)[..., np.newaxis]
        f1_jac = self._f1.jacobian(xi)
        f2_jac = self._f2.jacobian(xi)
        return f1_jac * f2_val + f1_val * f2_jac


class DivFunction(DiffFunction):
    """division of two DiffFunctions

    Parameters
    ----------
    f1 : DiffFunction
        the first function.
    f2 : DiffFunction
        the second function.
    """
    def __init__(self, f1, f2):
        # type: (DiffFunction, DiffFunction) -> None
        if f1.ndim != f2.ndim:
            raise ValueError('functions dimension mismatch.')

        DiffFunction.__init__(self, _intersection(f1.input_ranges, f2.input_ranges), delta_list=None)
        self._f1 = f1
        self._f2 = f2

    def __call__(self, xi):
        return self._f1(xi) / self._f2(xi)

    def deriv(self, xi, j):
        f2_val = self._f2(xi)
        return self._f1.deriv(xi, j) / f2_val - (self._f1(xi) * self._f2.deriv(xi, j) / (f2_val**2))

    def jacobian(self, xi):
        f1_val = self._f1(xi)[..., np.newaxis]
        f2_val = self._f2(xi)[..., np.newaxis]
        f1_jac = self._f1.jacobian(xi)
        f2_jac = self._f2.jacobian(xi)

        return f1_jac / f2_val - (f1_val * f2_jac) / (f2_val**2)


class PwrFunction(DiffFunction):
    """a DiffFunction raised to a power.

    Parameters
    ----------
    f : DiffFunction
        the DiffFunction.
    pwr : float
        the power.
    scale : float
        scaling factor.  Used to implement a / x.
    """
    def __init__(self, f, pwr, scale=1.0):
        # type: (DiffFunction, float, float) -> None
        DiffFunction.__init__(self, f.input_ranges, delta_list=None)
        self._f = f
        self._pwr = pwr
        self._scale = scale

    def __call__(self, xi):
        return (self._f(xi) ** self._pwr) * self._scale

    def deriv(self, xi, j):
        return (self._f(xi) ** (self._pwr - 1) * self._pwr * self._f.deriv(xi, j)) * self._scale

    def jacobian(self, xi):
        f_val = self._f(xi)[..., np.newaxis]
        f_jac = self._f.jacobian(xi)
        return (f_jac * (f_val ** (self._pwr - 1) * self._pwr)) * self._scale


class VectorDiffFunction(object):
    """A differentiable vector function.

    Parameters
    ----------
    fun_list : List[DiffFunction]
        list of interpolator functions, one for each element of the output vector.
    """

    def __init__(self, fun_list):
        # type: (List[DiffFunction]) -> None
        # error checking
        if not fun_list:
            raise ValueError('No interpolators are given.')

        self._input_ranges = _intersection(*(f.input_ranges for f in fun_list))

        self._in_dim = fun_list[0].ndim
        for fun in fun_list:
            if fun.ndim != self._in_dim:
                raise ValueError('Interpolators input dimension mismatch.')

        self._fun_list = fun_list
        self._out_dim = len(fun_list)

    @property
    def in_dim(self):
        # type: () -> int
        """Input dimension number."""
        return self._in_dim

    @property
    def out_dim(self):
        # type: () -> int
        """Output dimension number."""
        return self._out_dim

    def get_input_range(self, idx):
        # type: (int) -> Tuple[Optional[float], Optional[float]]
        """Returns the input range of the given dimension."""
        return self._input_ranges[idx]

    def __call__(self, xi):
        """Returns the output vector at the given coordinates.

        Parameters
        ----------
        xi : array-like
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        xi = np.asarray(xi, dtype=float)
        shape_trunc = xi.shape[:-1]  # type: Tuple[int, ...]
        ans = np.empty(shape_trunc + (self._out_dim, ))
        for idx in range(self._out_dim):
            ans[..., idx] = self._fun_list[idx](xi)
        return ans

    def jacobian(self, xi):
        """Calculate the Jacobian matrices of this function at the given coordinates.

        Parameters
        ----------
        xi : array-like
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The jacobian matrix at the given coordinates.
        """
        xi = np.asarray(xi, dtype=float)
        shape_trunc = xi.shape[:-1]  # type: Tuple[int, ...]
        ans = np.empty(shape_trunc + (self._out_dim, self._in_dim))
        for m in range(self._out_dim):
            ans[..., m, :] = self._fun_list[m].jacobian(xi)
        return ans

    def deriv(self, xi, i, j):
        """Compute the derivative of output i with respect to input j

        Parameters
        ----------
        xi : array-like
            The coordinates to evaluate, with shape (..., ndim)
        i : int
            output index.
        j : int
            input index.

        Returns
        -------
        val : numpy.array
            The derivatives at the given coordinates.
        """
        return self._fun_list[i].deriv(xi, j)
