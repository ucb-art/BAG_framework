#!/usr/bin/env bash

export PYTHONPATH="${BAG_FRAMEWORK}"

export cmd="-m bag.virtuoso run_skill_server"
export min_port=5000
export max_port=9999
export port_file="BAG_server_port.txt"
export log="skill_server.log"

export cmd="${BAG_PYTHON} ${cmd} ${min_port} ${max_port} ${port_file} ${log}"
exec $cmd
