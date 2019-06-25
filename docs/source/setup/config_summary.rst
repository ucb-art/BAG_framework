Configuration Files Summary
===========================

Although BAG has many configuration settings, most of them do not need to be changed.  This file summarizes which
settings you should modify under various use cases.

Starting New Project
--------------------

For every new project, it is a good practice to keep a set of global configuration files to make sure everyone working
on the project is simulating the same corners, running LVS and extraction with the same settings, and so on.  In this
case, you should change the following fields to point to the global configuration files:

* :ref:`sim_env_file`
* :ref:`lvs_runset`
* :ref:`rcx_runset`
* :ref:`calibre_cellmap`

Customizing Virtuoso Setups
---------------------------

If you changed your Virtuoso setup (configuration files, working directory, etc.), double check the following fields to
see if they need to be modified:

* :ref:`lvs_rundir`
* :ref:`rcx_rundir`
* :ref:`sim_init_file`

Python Design Module Customization
----------------------------------

The following fields control how BAG 2.0 finds design modules, and also where it puts new imported modules:

* :ref:`bag_lib_defs`
* :ref:`bag_new_lib_path`

.. _change_pdk:

Changing Process Technology
---------------------------

If you want to change the process technology, double check the following fields:

* :ref:`sch_tech_lib`
* :ref:`sch_exclude`
* :ref:`tb_config_libs`
* :ref:`tech_config_path`

The following fields probably won't change, but if something doesn't work it's worth to double check:

* :ref:`sch_sympin`
* :ref:`sch_ipin`
* :ref:`sch_opin`
* :ref:`sch_iopin`
* :ref:`sch_simulators`

