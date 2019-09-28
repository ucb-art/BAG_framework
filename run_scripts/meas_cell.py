import argparse
from argparse import Namespace
from pathlib import Path

from bag.io.file import Pickle, Yaml
from bag.core import BagProject

io_cls_dict = {
    'pickle': Pickle,
    'yaml': Yaml,
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('specs_fname', help='specs yaml file')
    parser.add_argument('--no-cell', dest='gen_cell', action='store_false',
                        default=True, help='skip cell generation')
    parser.add_argument('--no-wrapper', dest='gen_wrapper', action='store_false',
                        default=True,  help='skip wrapper generation')
    parser.add_argument('--no-tb', dest='gen_tb', action='store_false',
                        default=True,  help='skip tb generation')
    parser.add_argument('--load', dest='load_results', action='store_true',
                        default=False,  help='skip simulation, just load the results')
    parser.add_argument('-x', '--extract', dest='extract', action='store_true',
                        default=False, help='do extracted simulation')
    parser.add_argument('--no-sim', dest='run_sim', action='store_false',
                        default=True, help='run simulation, --load has a priority over this')
    parser.add_argument('--format', default='yaml',
                        help='format of spec file (yaml, json, pickle)')
    parser.add_argument('-dump', '--dump', default='',
                        help='If given will dump output of script into that '
                             'file according to the format specified')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: Namespace):
    specs_fname = Path(args.specs_fname)
    io_cls = io_cls_dict[args.format]
    specs = io_cls.load(str(specs_fname))

    results = prj.measure_cell(specs=specs,
                               gen_cell=args.gen_cell,
                               gen_wrapper=args.gen_wrapper,
                               gen_tb=args.gen_tb,
                               load_results=args.load_results,
                               extract=args.extract,
                               run_sims=args.run_sim)

    if results is not None and args.dump:
        out_tmp_file = Path(args.dump)
        io_cls.save(results, out_tmp_file)


if __name__ == '__main__':

    args = parse_args()
    local_dict = locals()
    bprj = local_dict.get('bprj', BagProject())
    run_main(bprj, args)