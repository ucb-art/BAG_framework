# -*- coding: utf-8 -*-

"""This module defines the base of all interface classes.
"""

from typing import Dict, Any

from ..io.template import new_template_env


class InterfaceBase:
    """The base class of all interfaces.

    Provides various helper methods common to all interfaces.
    """
    def __init__(self):
        self._tmp_env = new_template_env('bag.interface', 'templates')

    def render_file_template(self, temp_name, params):
        # type: (str, Dict[str, Any]) -> str
        """Returns the rendered content from the given template file."""
        template = self._tmp_env.get_template(temp_name)
        return template.render(**params)
