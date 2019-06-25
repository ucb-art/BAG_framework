simulation
==========

This entry defines all settings related to Ocean.

simulation.class
----------------

The Python class that handles simulator interaction.  This entry is mainly to support non-Ocean simulators.  If you
use Ocean, the value must be ``bag.interface.ocean.OceanInterface``.

simulation.prompt
-----------------

The ocean prompt string.

.. _sim_init_file:

simulation.init_file
--------------------

This file will be loaded when Ocean first started up.  This allows you to configure the Ocean simulator.  If you do not want to load an initialization file, set this field to an empty string (``""``).

simulation.view
---------------

Testbench view name.  Usually ``adexl``.

simulation.state
----------------

ADE-XL setup state name.  When you run simulations from BAG, the simulation configuration will be saved to this setup
state.

simulation.update_timeout_ms
----------------------------

If simulation takes a lone time, BAG will print out a message at this time interval (in milliseconds) so you can know
if BAG is still running.

simulation.kwargs
-----------------

pexpect keyword arguments dictionary used to start the simulation.  When BAG server receive a simulation request, it
will run Ocean in a subprocess using Python pexpect module.  This entry allows you to control how pexpect starts the
Ocean subprocess.  Refer to pexpect documentation for more information.

job_options
-----------

A dictionary of job options for ADE-XL.  This entry controls whether ADE-XL runs simulations remotely or locally, and how many jobs it launches for a simulation run.  Refer to ADE-XL documentation for available options.
