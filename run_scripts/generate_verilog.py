
import os

import yaml
from jinja2 import Environment, FileSystemLoader


def run_main():
    verilog_dir = 'verilog_models'
    cell_map_fname = 'verilog_cell_map.yaml'
    skill_read_fname = 'verilog_cell_list.txt'
    lib_name = 'AAAMODEL_QDR_HYBRID3'
    lib_loc = 'gen_libs'
    view_name = 'systemVerilog'
    model_fname = 'verilog.sv'

    with open(cell_map_fname, 'r') as f:
        cell_map = yaml.load(f)

    jinja_env = Environment(loader=FileSystemLoader(verilog_dir))

    with open(skill_read_fname, 'w') as g:
        for cell_name, fname in cell_map.items():
            root_dir = os.path.join(lib_loc, lib_name, cell_name, view_name)
            os.makedirs(root_dir, exist_ok=True)

            content = jinja_env.get_template(fname).render(cell_name=cell_name)

            with open(os.path.join(root_dir, model_fname), 'w') as f:
                f.write(content)

            g.write('%s %s %s\n' % (lib_name, cell_name, view_name))


if __name__ == '__main__':
    run_main()
