class
=====

The subclass of :ref:


.. _bag_lib_defs:

lib_defs
========

Location of the BAG design module libraries definition file.

The BAG libraries definition file is similar to the ``cds.lib`` file for Virtuoso, where it defines every design module
library and its location.  This file makes it easy to share design module libraries made by different designers.

Each line in the file contains two entries, separated by spaces.  The first entry is the name of the design module
library, and the second entry is the location of the design module library.  Environment variables may be used in this
file.

.. _bag_new_lib_path:

new_lib_path
============

Directory to put new generated design module libraries.

When you import a new schematic generator library, BAG will create a corresponding Python design module library and
define this library in the library definition file (see :ref:`bag_lib_defs`).  This field tells BAG where new design
module libraries should be created.
