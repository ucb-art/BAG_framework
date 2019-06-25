#!/usr/bin/env bash

# crazy black magic from:
# https://unix.stackexchange.com/questions/20880/how-can-i-use-environment-variables-in-my-shebang
# this block of code is valid in both bash and python.
# this means if this script is run under bash, it'll
# call this script again using BAG_PYTHON.  If
# this script is run under Python, this block of code
# effectively does nothing.
if "true" : '''\'
then
if [[ $BAG_PYTHON ]]; then
exec ${BAG_PYTHON} "$0" "$@"
else
echo "BAG_PYTHON environment variable is not set"
fi
exit 127
fi
'''
import os
import subprocess

import yaml


def write_to_file(fname, lines):
    with open(fname, 'w') as f:
        f.writelines((l + '\n' for l in lines))
    add_git_file(fname)


def setup_python_path(module_list):
    lines = ['# -*- coding: utf-8 -*-',
             'import os',
             'import sys',
             '',
             "sys.path.append(os.environ['BAG_FRAMEWORK'])",
             "sys.path.append(os.environ['BAG_TECH_CONFIG_DIR'])",
             ]
    template = "sys.path.append(os.path.join(os.environ['BAG_WORK_DIR'], '%s'))"
    lines.append(template % 'BAG2_TEMPLATES_EC')
    for mod_name, _ in module_list:
        lines.append(template % mod_name)

    write_to_file('bag_startup.py', lines)


def get_sch_libraries(mod_name, mod_info):
    bag_modules = mod_info.get('lib_path', 'BagModules')
    root_dir = os.path.realpath(os.path.join(mod_name, bag_modules))
    if not os.path.isdir(root_dir):
        return []
    return [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]


def setup_libs_def(module_list):
    lines = ['BAG_prim $BAG_TECH_CONFIG_DIR/DesignModules']
    template = '%s $BAG_WORK_DIR/%s/%s'
    for mod_name, mod_info in module_list:
        bag_modules = mod_info.get('lib_path', 'BagModules')
        for lib_name in get_sch_libraries(mod_name, mod_info):
            lines.append(template % (lib_name, mod_name, bag_modules))

    write_to_file('bag_libs.def', lines)


def setup_cds_lib(module_list):
    lines = ['DEFINE BAG_prim $BAG_TECH_CONFIG_DIR/BAG_prim']
    template = 'DEFINE %s $BAG_WORK_DIR/%s/%s'
    for mod_name, mod_info in module_list:
        for lib_name in get_sch_libraries(mod_name, mod_info):
            lines.append(template % (lib_name, mod_name, lib_name))

    write_to_file('cds.lib.bag', lines)


def run_command(cmd):
    timeout = 5
    proc = subprocess.Popen(cmd)
    try:
        proc.communicate()
    except KeyboardInterrupt:
        print('Ctrl-C detected, terminating')
        if proc.returncode is None:
            proc.terminate()
            print('terminating process...')
            try:
                proc.wait(timeout=timeout)
                print('process terminated')
            except subprocess.TimeoutExpired:
                proc.kill()
                print('process did not terminate, try killing...')
                try:
                    proc.wait(timeout=timeout)
                    print('process killed')
                except subprocess.TimeoutExpired:
                    print('cannot kill process...')

    if proc.returncode is None:
        raise ValueError('Ctrl-C detected, but cannot kill process')
    elif proc.returncode < 0:
        raise ValueError('process terminated with return code = %d' % proc.returncode)
    elif proc.returncode > 0:
        raise ValueError('command %s failed' % ' '.join(cmd))


def add_git_submodule(module_name, url):
    if os.path.exists(module_name):
        # skip if already exists
        return

    run_command(['git', 'submodule', 'add', url])


def add_git_file(fname):
    run_command(['git', 'add', '-f', fname])


def link_submodule(repo_path, module_name):
    if os.path.exists(module_name):
        # skip if already exists
        return

    src = os.path.join(repo_path, module_name)
    if not os.path.isdir(src):
        raise ValueError('Cannot find submodule %s in %s' % (module_name, repo_path))
    os.symlink(src, module_name)
    add_git_file(module_name)


def setup_git_submodules(module_list):
    add_git_submodule('BAG2_TEMPLATES_EC', 'git@github.com:ucb-art/BAG2_TEMPLATES_EC')

    for module_name, module_info in module_list:
        add_git_submodule(module_name, module_info['url'])


def setup_submodule_links(module_list, repo_path):
    link_submodule(repo_path, 'BAG2_TEMPLATES_EC')
    for module_name, _ in module_list:
        link_submodule(repo_path, module_name)


def run_main():
    with open('bag_submodules.yaml', 'r') as f:
        modules_info = yaml.load(f)

    module_list = [(key, modules_info[key]) for key in sorted(modules_info.keys())]

    # error checking
    bag_dir = 'BAG_framework'
    if not os.path.isdir(bag_dir):
        raise ValueError('Cannot find directory %s' % bag_dir)

    # get real absolute path of parent directory of BAG_framework
    repo_path = os.path.dirname(os.path.realpath(bag_dir))
    cur_path = os.path.realpath('.')
    if cur_path == repo_path:
        # BAG_framework is an actual directory in this repo; add dependencies as git submodules
        setup_git_submodules(module_list)
    else:
        setup_submodule_links(module_list, repo_path)

    setup_python_path(module_list)
    setup_libs_def(module_list)
    setup_cds_lib(module_list)


if __name__ == '__main__':
    run_main()
