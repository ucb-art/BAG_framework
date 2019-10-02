"""
This script removes the '#Removed by ddDeleteObj' items in cds.lib.
It is also capable of taking a list of patterns as input and delete the generated libraries that
match that pattern in glob style.
"""

from argparse import Namespace
import argparse
import re
import os
import shutil

from pathlib import Path
from bag.io.file import read_yaml_env, readlines_iter, write_file


def arg_parse() -> Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument('-rm', '--rm_patterns', nargs='+', dest='patterns', default=[],
                        help='A list of patterns to be removed from cadence library, the pattern '
                             'should be the name of the library in glob style')

    args = parser.parse_args()
    return args


def run_main(args: Namespace):
    bag_workspace_dir = Path(os.environ['BAG_WORK_DIR'])
    if 'BAG_CONFIG_PATH' not in os.environ:
        raise Exception('BAG_CONFIG_PATH not defined.')
    bag_config_path = os.environ['BAG_CONFIG_PATH']
    bag_config = read_yaml_env(bag_config_path)

    gen_libs_path = Path(bag_config['database']['default_lib_path'])
    if not gen_libs_path.exists():
        print(f'path {str(gen_libs_path)} does not exist')

    cds_lib_path = bag_workspace_dir / 'cds.lib'
    if not cds_lib_path.exists():
        print(f'path {str(cds_lib_path)} does not exist')
        return

    cds_patterns = ['#Removed by ddDeleteObj'] + [f'DEFINE {p}' for p in args.patterns]

    # clean cds.lib
    cds_lib_lines_iter = readlines_iter(str(cds_lib_path))

    new_cds_lib_content = []
    for line in cds_lib_lines_iter:
        found = False
        for p in cds_patterns:
            if re.match(p, line):
                found = True
                break
        if not found:
            new_cds_lib_content.append(line)

    write_file(str(cds_lib_path), ''.join(new_cds_lib_content))

    # clean gen_libs library names that match the pattern in args
    for p in args.patterns:
        for dir in gen_libs_path.glob(f'{p}*'):
            if dir.is_dir():
                shutil.rmtree(dir)
            else:
                print(f'path {str(dir)} is not a directory')


if __name__ == '__main__':
    args = arg_parse()
    run_main(args)
