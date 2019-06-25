Design Module
=============

A design module is a Python class that generates new schematics.  It computes all parameters needed to generate a
schematic from user defined specifications.  For example, a design module for an inverter needs to compute the width,
length, and threshold flavor of the NMOS and PMOS to generate a new inverter schematic.  The designer of this module can
let the user specify these parameters directly, or alternatively compute them from higher level specifications, such as
fanout, input capacitance, and leakage specs.

To create a default design module for a schematic generator, create a :class:`~bag.BagProject` instance and call
:meth:`~bag.BagProject.import_design_library` to import all schematic generators in a library from your CAD
program into Python.  The designer should then implement the three methods, :meth:`~bag.design.Module.design`,
:meth:`~bag.design.Module.get_layout_params`, and :meth:`~bag.design.Module.get_layout_pin_mapping` (The latter two are
optional if you do not use BAG to generate layout).  Once you finish the design module definition, you can create new
design module instances by calling :meth:`~bag.BagProject.create_design_module`.


The following sections describe how each of these methods should be implemented.

design()
--------

This method computes all parameters needed to generate a schematic from user defined specifications.  The input
arguments should also be specified in this method.

A design module can have multiple design methods, as long as they have difference names.  For example, You can implement
the ``design()`` method to compute parameters from high level specifications, and define a new method named
``design_override()`` that allows the user to assign parameter values directly for debugging purposes.

To enable hierarchical design, design module has a dictionary, :attr:`~bag.design.Module.instances`, that
maps children instance names to corresponding design modules, so you can simply call their
:meth:`~bag.design.Module.design` methods to set their parameters.  See :doc:`/tutorial/tutorial` for an simple example.

If you need to modify the schematic structure (such as adding more inverter buffers), you should call the corresponding
methods before calling :meth:`~bag.design.Module.design` methods of child instances, as those design module could be
changed.  The rest of this section explains how you modify the schematic.

Pin Renaming
^^^^^^^^^^^^

Most of the time, you should not rename the pin of schematic.  The only time you should rename the pin is when you have
a variable bus pin where the number of bits in the bus can change with the design.  In this case, call
:meth:`~bag.design.Module.rename_pin` to change the number of bits in the bus.  To connect/remove instances from
the added/deleted bus pins, see :ref:`instance_connection_modification`

Delete Instances
^^^^^^^^^^^^^^^^

Delete a child instance by calling :meth:`~bag.design.Module.delete_instance`.  After
this call, the corresponding value in :attr:`~bag.design.Module.instances` dictionary will become ``None``.

.. note::
    You don't have to delete 0-width or 0-finger transistors; BAG already handles that for you.

Replace Instance Master
^^^^^^^^^^^^^^^^^^^^^^^

If you have two different designs of a child instance, and you want to swap between the two designs, you can call
:meth:`~bag.design.Module.replace_instance_master` to change the instance master of a child.

.. note::
    You can replace instance masters only if the two instance masters have exactly the symbol, including pin names.

.. _instance_connection_modification:

Instance Connection Modification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Call :meth:`~bag.design.Module.reconnect_instance_terminal` to change a child instance's connection.

Arraying Child Instances
^^^^^^^^^^^^^^^^^^^^^^^^

Call :meth:`~bag.design.Module.array_instance` to array a child instance.  After this call,
:attr:`~bag.design.Module.instances` will map the child instance name to a list of design modules, one for each instance
in the array.  You can then iterate through this list and design each of the instances.  They do not need to have the
same parameter values.

Restoring to Default
^^^^^^^^^^^^^^^^^^^^

If you are using the design module in a design iteration loop, or you're using BAG interactively through the Python
console, and you want to restore a deleted/replaced/arrayed child instance to the default state, you can call
:meth:`~bag.design.Module.restore_instance`.


get_layout_params()
-------------------

This method should return a dictionary from layout parameter names to their values.  This dictionary is used to create
a layout cell that will pass LVS against the generated schematic.

get_layout_pin_mapping()
------------------------

This method should return a dictionary from layout pin names to schematic pin names.  This method exists because a
layout cell may not have the same pin names as the schematic.  If a layout pin should be left un-exported, its
corresponding value in the dictionary must be ``None``.

This dictionary only need to list the layout pins that needs to be renamed.  If no renaming is necessary, an empty
dictionary can be returned.
