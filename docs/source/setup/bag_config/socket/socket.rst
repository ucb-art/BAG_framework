socket
======

This entry defines socket settings for BAG to communicate with Virtuoso.

socket.host
-----------

The host of the BAG server socket, i.e. the machine running the Virtuoso program.  usually ``localhost``.

socket.port_file
----------------

File containing socket port number for BAG server.  When Virtuoso starts the BAG server process, it finds a open port and bind the
server to this port.  It then creates a file with name in ``$BAG_WORK_DIR`` directory, and write the port number to this
file.

socket.sim_port_file
--------------------

File containing socket port number for simulation server.  When the simulation server starts, it finds a open port and bind the
server to this port.  It then creates a file with name in ``$BAG_WORK_DIR`` directory, and write the port number to this
file.


socket.log_file
---------------

Socket communication debugging log file.  All messages sent or received by BAG will be recorded in this log.

socket.pipeline
---------------

number of messages allowed in the ZMQ pipeline.  Usually you don't have to change this.
