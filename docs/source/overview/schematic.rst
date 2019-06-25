Schematic Generator
===================

A schematic generator is a schematic in your CAD program that tells BAG all the information needed to create a design.
BAG creates design modules from schematic generators, and BAG will copy and modify schematic generators to implement
new designs.

.. figure:: ./figures/gm_schematic.png
    :align: center
    :figclass: align-center

    An example schematic generator of a differential gm cell.

A schematic generator needs to follow some rules to work with BAG:

#. Instances in a schematic generator must be other schematic generators, or a cell in the ``BAG_prim`` library.
#. BAG can array any instance in a schematic generator.  That is, in the design implementation phase, BAG can
   copy/paste this instance any number of times, and modify the connections or parameters of any copy.  This is useful
   in creating array structures, such as an inverter chain with variable number of stages, or a DAC with variable
   number of bits.

   However, if you need to array an instance, its ports must be connected to wire stubs, with net labels on each of the
   wire stubs.  Also, there must be absolutely nothing to the right of the instance, since BAG will array the instance
   by copying-and-pasting to the right.  An example of an inverter buffer chain schematic generator is shown below.

    .. figure:: ./figures/inv_chain_schematic.png
        :align: center
        :figclass: align-center

        An example schematic generator of an inverter buffer chain.  Ports connected by wire stubs, nothing on the right.

#. BAG can replace the instance master of any instance.  The primary use of this is to allow the designer to change
   transistor threshold values, but this could be used for other schematic generators if implemented.  Whenever you
   switch the instance master of an instance, the symbol of the new instance must exactly match the old instance,
   including the port names.
#. Although not required, it is good practice to fill in default parameter values for all instances from the
   ``BAG_prim`` library.  This makes it so that you can simulate a schematic generator in a normal testbench, and make
   debugging easier.

