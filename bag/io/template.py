# -*- coding: utf-8 -*-

"""This module defines methods to create files from templates.
"""

from jinja2 import Environment, PackageLoader, select_autoescape


def new_template_env(parent_package, tmp_folder):
    # type: (str, str) -> Environment
    return Environment(trim_blocks=True,
                       lstrip_blocks=True,
                       keep_trailing_newline=True,
                       autoescape=select_autoescape(default_for_string=False),
                       loader=PackageLoader(parent_package, package_path=tmp_folder),
                       enable_async=False,
                       )
