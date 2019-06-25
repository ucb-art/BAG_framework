Testbench Generator
===================

A testbench generator is just a normal testbench with schematic and adexl view.  BAG will simply copy the schematic and
adexl view, and replace the device under test with the new generated schematic.  There are only 3 restrictions to the
testbench:

#. All device-under-test's (DUTs) in the testbench must have an instance name starting with ``XDUT``.  This is to inform BAG
   which child instances should be replaced.
#. The testbench must be configured to simulate with ADE-XL.  This is to make parametric/corner sweeps and monte carlo
   easier.
#. You should not define any process corners in the ADE-XL state, as BAG will load them for you.  This makes it
   possible to use the same testbench generator across different technologies.

To verify a new design, call :meth:`~bag.BagProject.create_testbench` and specify the testbench generator library/cell,
DUT library/cell, and the library to create the new testbench in.  BAG will create a :class:`~bag.core.Testbench` object
to represent this testbench.  You can then call its methods to set the parameters, process corners, or enable parametric
sweeps.  When you're done, call :meth:`~bag.core.Testbench.update_testbench` to commit the changes to Virtuoso.  If you
do not wish to run simulation in BAG, you can then open this testbench in Virtuoso and simulate it there.

If you want to start simulation from BAG and load simulation data, you need to call
:meth:`~bag.core.Testbench.add_output` method to specify which outputs to record and send back to Python.  Output
expression is a Virtuoso calculator expression.  Then, call :meth:`~bag.core.Testbench.run_simulation` to start a
simulation run.  During the simulation, you can press ``Ctrl-C`` anytime to abort simulation.  When the simulation
finish, the result directory will be saved to the attribute :attr:`~bag.core.Testbench.save_dir`, and you can call
:func:`bag.data.load_sim_results` to load the result in Python. See :doc:`/tutorial/tutorial` for an example.

Since BAG uses the ADE-XL interface to run simulation, all simulation runs will be recorded in ADE-XL's history tab, so
you can plot them in Virtuoso later for debugging purposes.  By default, all simulation runs from BAG has the ``BagSim``
history tag, but you can also specify your own tag name when you call :meth:`~bag.core.Testbench.run_simulation`.  Read
ADE-XL documentation if you want to know more about ADE-XL's history feature.
