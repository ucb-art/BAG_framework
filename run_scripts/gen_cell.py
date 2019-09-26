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
    parser.add_argument('--no-sch', dest='gen_sch', action='store_false',
                        default=True, help='skip schematic generation')
    parser.add_argument('--no-lay', dest='gen_lay', action='store_false',
                        default=True, help='skip layout generation')
    parser.add_argument('-v', '--lvs', action='store_true', default=False,
                        help='run lvs')
    parser.add_argument('-x', '--rcx', action='store_true', default=False,
                        help='run rcx')
    parser.add_argument('--use-cache', dest='use_cache', action='store_true',
                        default=False,
                        help='uses the cache in cache_dir')
    parser.add_argument('--save-cache', dest='save_cache', action='store_true',
                        default=False,
                        help='updates design database stored in cache_dir')
    parser.add_argument('--pre', dest='prefix', default='',
                        help='prefix used to generate all the cells')
    parser.add_argument('--suf', dest='suffix', default='',
                        help='suffix used to generate all the cells')
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

    results = prj.generate_cell(specs=specs,
                                gen_lay=args.gen_lay,
                                gen_sch=args.gen_sch,
                                run_lvs=args.lvs,
                                run_rcx=args.rcx,
                                use_cybagoa=True,
                                use_cache=args.use_cache,
                                save_cache=args.save_cache,
                                prefix=args.prefix,
                                suffix=args.suffix)

    if results is not None and args.dump:
        out_tmp_file = Path(args.dump)
        io_cls.save(results, out_tmp_file)


if __name__ == '__main__':

    args = parse_args()
    local_dict = locals()
    bprj = local_dict.get('bprj', BagProject())
    run_main(bprj, args)

