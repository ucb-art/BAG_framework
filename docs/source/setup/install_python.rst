Installing Python for BAG
==========================

This section describes how to install Python for running BAG.

Installation Requirements
-------------------------

BAG is compatible with Python 3.5+ (Python 2.7+ is theoretically supported but untested), so you will need to have
Python 3.5+ installed.  For Linux/Unix systems, it is recommended to install a separate Python distribution from
the system Python.

BAG requires multiple Python packages, some of which requires compiling C++/C/Fortran extensions.  Therefore, it is
strongly recommended to download `Anaconda Python <https://www.continuum.io/downloads>`_, which provides a Python
distribution with most of the packages preinstalled.  Otherwise, please refer to documentation for each required
package for how to install/build from source.

Required Packages
-----------------
In addition to the default packages that come with Anaconda (numpy, scipy, etc.), you'll need the following additional
packages:

- `subprocess32 <https://pypi.python.org/pypi/subprocess32>`_ (Python 2 only)

  This package is a backport of Python 3.2's subprocess module to Python 2.  It is installable from ``pip``.

- `sqlitedict <https://pypi.python.org/pypi/sqlitedict>`_

  This is a dependency of OpenMDAO.  It is installable from ``pip``.

- `OpenMDAO <https://pypi.python.org/pypi/openmdao>`_

  This is a flexible optimization framework in Python developed by NASA.  It is installable from ``pip``.

- `mpich2 <https://anaconda.org/anaconda/mpich2>`_ (optional)

  This is the Message Passing Interface (MPI) library.  OpenMDAO and Pyoptsparse can optionally use this library
  for parallel computing.  You can install this package with:

  .. code-block:: bash

      > conda install mpich2

- `mpi4py <https://anaconda.org/anaconda/mpi4py>`_ (optional)

  This is the Python wrapper of ``mpich2``.  You can install this package with:

  .. code-block:: bash

      > conda install mpi4py

- `ipopt <https://anaconda.org/pkerichang/ipopt>`__ (optional)

  `Ipopt <https://projects.coin-or.org/Ipopt>`__ is a free software package for large-scale nonlinear optimization.
  This can be used to replace the default optimization solver that comes with scipy.  You can install this package with:

  .. code-block:: bash

      > conda install --channel pkerichang ipopt

- `pyoptsparse <https://anaconda.org/pkerichang/pyoptsparse>`_ (optional)

  ``pyoptsparse`` is a python package that contains a collection of optmization solvers, including a Python wrapper
  around ``Ipopt``.  You can install this package with:

  .. code-block:: bash

      > conda install --channel pkerichang pyoptsparse
