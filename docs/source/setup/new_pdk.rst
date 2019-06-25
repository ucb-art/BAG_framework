Setting up New PDK
==================

This section describes how to get BAG 2.0 to work with a new PDK.

#. Create a new technology configuration file for this PDK.  See :doc:`tech_config/tech_config` for a description of
   the technology configuration file format.

#. Create a new BAG configuration file for this PDK.  You can simply copy an existing configuration, then change the
   fields listed in :ref:`change_pdk`.

#. Create a new ``BAG_prim`` library for this PDK.  The easiest way to do this is to copy an existing ``BAG_prim``
   library, then change the underlying instances to be instances from the new PDK.  You should use the **pPar** command
   in Virtuoso to pass CDF parameters from ``BAG_prim`` instances to PDK instances.

#. Change your cds.lib to refer to the new ``BAG_prim`` library.

#. To avoid everyone having their own python design modules for BAG primitive, you should generated a global design module
   library for BAG primitives, then ask every user to include this global library in their ``bag_libs.def`` file.  To
   do so, setup a BAG workspace and execute the following commands:

    .. code-block:: python

        import bag
        prj = bag.BagProject()
        prj.import_design_library('BAG_prim')

   now copy the generate design library to a global location.
